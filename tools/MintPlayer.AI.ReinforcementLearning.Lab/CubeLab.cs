using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

/// <summary>
/// The cube imitation campaign (PLAN M16, `--game cube`): streams random scrambles
/// through the Kociemba oracle (one solve = a whole labeled solution path), trains the
/// two-headed <see cref="CubePolicyNet"/> supervised (CE on the next quarter-turn +
/// Huber on distance-to-go), checkpoints to the model store every eval and tracks
/// per-depth greedy/search solve rates. Resumable: the net and Adam's moments reload
/// from `cube.policy` / `cube.policy-adam`.
/// </summary>
internal static class CubeLab
{
    private const int BatchSize = 256;
    private const int SamplesPerRound = 4096;
    private static readonly int[] EvalDepths = [2, 4, 6, 8, 10, 12, 16, 20];

    public static void Run(string[] args)
    {
        double hours = 9;
        string dataDir = "data";
        ulong seed = 1;
        float learningRate = 3e-4f;
        bool evalOnly = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        var evalEvery = TimeSpan.FromMinutes(10);
        var store = new FileModelStore(dataDir);
        string logPath = Path.Combine(store.RootDirectory, "logs");
        Directory.CreateDirectory(logPath);
        string csvPath = Path.Combine(logPath, "cube-imitation.csv");
        if (!File.Exists(csvPath))
            File.AppendAllText(csvPath, "utc,solves,samples,ce,acc,huber,"
                + string.Join(',', EvalDepths.SelectMany(d => new[] { $"d{d}_greedy", $"d{d}_search" })) + "\n");

        var rng = new Xoshiro256StarStar(seed);
        CubePolicyNet net;
        using (var existing = store.TryOpenRead(CubeIds.Environment, CubeIds.Policy))
        {
            if (existing is not null)
            {
                net = CubePolicyNet.Load(existing);
                Log("resumed cube policy net from the model store");
            }
            else
            {
                net = new CubePolicyNet(new Xoshiro256StarStar(seed ^ 0xDEADBEEF));
                Log("initialized a fresh cube policy net");
            }
        }

        Adam adam;
        using (var adamState = store.TryOpenRead(CubeIds.Environment, CubeIds.PolicyAdam))
        {
            if (adamState is not null)
            {
                using var reader = new BinaryReader(adamState, System.Text.Encoding.UTF8, leaveOpen: true);
                adam = AdamCheckpoint.Read(net.Parameters(), reader);
                adam.LearningRate = learningRate;
                Log($"resumed Adam state (lr set to {learningRate:E1})");
            }
            else
            {
                adam = new Adam(net.Parameters(), learningRate);
            }
        }

        Log("warming the Kociemba tables…");
        CubeSolver.WarmUp();

        if (evalOnly)
        {
            Evaluate(net, adam: null, store, csvPath, 0, 0, 0, 0, 0);
            EvaluateGate(net);
            return;
        }

        var deadline = DateTime.UtcNow.AddHours(hours);
        var nextEval = DateTime.UtcNow + TimeSpan.FromMinutes(2); // early baseline eval
        long totalSamples = 0, totalSolves = 0;
        double windowCe = 0, windowHuber = 0, windowAcc = 0;
        long windowCount = 0;

        Log($"training until {deadline:u} (~{hours:F1} h), data dir: {store.RootDirectory}");

        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 64);
        while (DateTime.UtcNow < deadline)
        {
            samples.Clear();
            while (samples.Count < SamplesPerRound)
            {
                var path = CubeOracle.LabelScramblePath(rng);
                if (path is null) continue;
                samples.AddRange(path);
                totalSolves++;
            }
            Shuffle(samples, rng);

            for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
            {
                var (ce, huber, acc) = TrainStep(net, adam, samples, offset, BatchSize);
                windowCe += ce;
                windowHuber += huber;
                windowAcc += acc;
                windowCount++;
                totalSamples += BatchSize;
            }

            if (DateTime.UtcNow >= nextEval)
            {
                Evaluate(net, adam, store, csvPath, totalSolves, totalSamples,
                    windowCount > 0 ? windowCe / windowCount : 0,
                    windowCount > 0 ? windowAcc / windowCount : 0,
                    windowCount > 0 ? windowHuber / windowCount : 0);
                windowCe = windowHuber = windowAcc = 0;
                windowCount = 0;
                nextEval = DateTime.UtcNow + evalEvery;
            }
        }

        Evaluate(net, adam, store, csvPath, totalSolves, totalSamples,
            windowCount > 0 ? windowCe / windowCount : 0,
            windowCount > 0 ? windowAcc / windowCount : 0,
            windowCount > 0 ? windowHuber / windowCount : 0);
        Log("time budget reached — final checkpoint saved.");
    }

    private static (double Ce, double Huber, double Acc) TrainStep(
        CubePolicyNet net, Adam adam, List<CubeOracle.LabeledState> samples, int offset, int batch)
    {
        var obs = new float[batch * RubiksCubeEnv.ObservationSize];
        var weights = new float[batch * RubiksCubeEnv.ActionCount];
        var targets = new float[batch];
        for (int i = 0; i < batch; i++)
        {
            var s = samples[offset + i];
            RubiksCubeEnv.WriteObservation(FaceletCube.FromFacelets(s.Facelets),
                obs.AsSpan(i * RubiksCubeEnv.ObservationSize, RubiksCubeEnv.ObservationSize));
            weights[i * RubiksCubeEnv.ActionCount + s.Action] = 1f;
            targets[i] = s.DistanceToGo / CubePolicyNet.DistanceScale;
        }

        var (logits, value) = net.Forward(new Tensor(obs, batch, RubiksCubeEnv.ObservationSize));
        var logProbs = logits.LogSoftmax();
        var ce = logProbs.Mul(new Tensor(weights, batch, RubiksCubeEnv.ActionCount)).Sum().MulScalar(-1f / batch);
        var huber = value.Reshape(batch).HuberLoss(new Tensor(targets, batch));
        var loss = ce.Add(huber);

        adam.ZeroGrad();
        loss.Backward();
        adam.ClipGradNorm(5f);
        adam.Step();

        int correct = 0;
        for (int i = 0; i < batch; i++)
        {
            int argmax = 0;
            for (int a = 1; a < RubiksCubeEnv.ActionCount; a++)
                if (logProbs.Data[i * RubiksCubeEnv.ActionCount + a] > logProbs.Data[i * RubiksCubeEnv.ActionCount + argmax])
                    argmax = a;
            if (argmax == samples[offset + i].Action) correct++;
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    /// <summary>Per-depth greedy/search eval; checkpoints net + Adam when <paramref name="adam"/> is given.</summary>
    private static void Evaluate(CubePolicyNet net, Adam? adam, FileModelStore store, string csvPath,
        long solves, long samples, double ce, double acc, double huber)
    {
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{solves}", $"{samples}", $"{ce:F4}", $"{acc:F4}", $"{huber:F5}" };
        var report = new System.Text.StringBuilder();
        report.Append($"[eval] solves {solves:N0}, samples {samples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

        foreach (int depth in EvalDepths)
        {
            int greedySolved = 0, searchSolved = 0;
            // Small budgets on purpose: this eval is a PROGRESS TRACKER inside a
            // time-budgeted campaign (a failed full-budget search costs ~20 s; the
            // smoke run spent 40 of its 43 minutes evaluating). The pre-registered
            // gate runs with the full budget via --eval-only.
            const int episodes = 20;
            for (int episode = 0; episode < episodes; episode++)
            {
                // Fixed seeded scrambles per depth, stable across evals.
                var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { greedySolved++; searchSolved++; continue; }

                if (CubePolicySearch.GreedyRollout(net, cube).Solved) { greedySolved++; searchSolved++; }
                else if (CubePolicySearch.Solve(net, cube, maxExpansions: 2_000).Solved) searchSolved++;
            }
            cells.Add($"{greedySolved / (double)episodes:F3}");
            cells.Add($"{searchSolved / (double)episodes:F3}");
            report.Append($"d{depth}: {greedySolved}/{episodes}g {searchSolved}/{episodes}s | ");
        }

        Log(report.ToString());
        File.AppendAllText(csvPath, string.Join(',', cells) + "\n");
        if (adam is not null)
        {
            store.Save(CubeIds.Environment, CubeIds.Policy, s => net.Save(s));
            store.Save(CubeIds.Environment, CubeIds.PolicyAdam, s =>
            {
                using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
                AdamCheckpoint.Write(adam, writer);
            });
        }
    }

    /// <summary>
    /// The pre-registered M16 gate: ≥ 90% of 100 random scrambles across depths 1–10
    /// solved within 40 quarter-turns (greedy or full-budget search).
    /// </summary>
    private static void EvaluateGate(CubePolicyNet net)
    {
        int totalSolved = 0, totalGreedy = 0;
        for (int depth = 1; depth <= 10; depth++)
        {
            int greedySolved = 0, searchSolved = 0;
            for (int episode = 0; episode < 10; episode++)
            {
                var gateRng = new Xoshiro256StarStar((ulong)(900_000 + 1_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(gateRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { greedySolved++; continue; }

                if (CubePolicySearch.GreedyRollout(net, cube).Solved) greedySolved++;
                else if (CubePolicySearch.Solve(net, cube).Solved) searchSolved++;
            }
            totalGreedy += greedySolved;
            totalSolved += greedySolved + searchSolved;
            Log($"  gate depth {depth}: {greedySolved}/10 greedy, +{searchSolved} with lookahead = {greedySolved + searchSolved}/10");
        }
        Log($"gate: {totalSolved}/100 solved ({totalSolved}%, target >= 90%); greedy alone {totalGreedy}%");
    }

    private static void Shuffle<T>(IList<T> list, Xoshiro256StarStar rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static void Log(string message)
        => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}

/// <summary>Model-store ids for the cube artifacts (shared with the web app's conventions).</summary>
internal static class CubeIds
{
    public const string Environment = "cube";
    public const string Policy = "policy";
    public const string PolicyAdam = "policy-adam";
}
