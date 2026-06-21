using System.Globalization;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

/// <summary>
/// EfficientCube campaign (`--game cube-policy`): teacher-FREE self-supervised policy learning.
/// Unlike <see cref="CubeLab"/> (which imitates Kociemba and can therefore never beat it), the
/// training label is the cube's own scramble reversal — no solver in the loop — so the solver is
/// bounded only by the policy's generalization and the beam-search budget, not by any teacher.
/// Trains the two-headed <see cref="CubePolicyNet"/> supervised (CE on the reversing move + Huber
/// on path length), solves with policy beam search, and checkpoints to `cube.policy-efficient` /
/// `cube.policy-efficient-adam` (distinct ids, so the imitation net is never touched). Resumable.
/// </summary>
internal static class CubePolicyLab
{
    private const string PolicyId = "policy-efficient";
    private const string PolicyAdamId = "policy-efficient-adam";
    private const string PolicyProgressId = "policy-efficient-progress";
    private const int BatchSize = 1000;
    private const int SamplesPerRound = 50_000;
    private static readonly int[] EvalDepths = [4, 8, 12, 14, 16, 18, 20, 22, 24, 26];

    public static void Run(string[] args)
    {
        double hours = 24;
        string dataDir = "data";
        ulong seed = 1;
        float learningRate = 3e-4f;
        int width = 512;
        int maxScramble = 30;
        int beamWidth = 2_000;
        int evalEpisodes = 20;
        bool evalOnly = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--max-scramble" && i + 1 < args.Length) maxScramble = int.Parse(args[++i]);
            else if (args[i] == "--beam" && i + 1 < args.Length) beamWidth = int.Parse(args[++i]);
            else if (args[i] == "--episodes" && i + 1 < args.Length) evalEpisodes = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        using var adaptive = new AdaptiveBackend();
        Backend.Current = adaptive;
        Log($"compute backend: {adaptive.Describe()}");

        var store = new FileModelStore(dataDir);
        string logPath = Path.Combine(store.RootDirectory, "logs");
        Directory.CreateDirectory(logPath);
        string csvPath = Path.Combine(logPath, "cube-policy.csv");
        if (!File.Exists(csvPath))
            File.AppendAllText(csvPath, "utc,samples,ce,acc,huber,"
                + string.Join(',', EvalDepths.SelectMany(d => new[] { $"d{d}_greedy", $"d{d}_beam" })) + "\n");

        var rng = new Xoshiro256StarStar(seed);
        CubePolicyNet net;
        using (var existing = store.TryOpenRead(CubeIds.Environment, PolicyId))
        {
            if (existing is not null)
            {
                net = CubePolicyNet.Load(existing);
                Log($"resumed EfficientCube net '{PolicyId}' from the model store");
            }
            else
            {
                net = new CubePolicyNet(new Xoshiro256StarStar(seed ^ 0xC0FFEE), hidden: width);
                Log($"initialized a fresh EfficientCube net '{PolicyId}' (trunk width {width})");
            }
        }

        Adam adam;
        using (var adamState = store.TryOpenRead(CubeIds.Environment, PolicyAdamId))
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

        if (evalOnly)
        {
            Evaluate(net, adam: null, store, csvPath, 0, 0, 0, 0, beamWidth, evalEpisodes, adaptive, 0);
            return;
        }

        var evalEvery = TimeSpan.FromMinutes(10);
        var deadline = DateTime.UtcNow.AddHours(hours);
        var nextEval = DateTime.UtcNow + TimeSpan.FromMinutes(2); // early baseline eval
        // Persisted progress: cumulative samples (for the displayed count) and the round counter (so the
        // scramble RNG continues its stream on resume rather than regenerating the same scrambles).
        long totalSamples = 0, round = 0;
        using (var progress = store.TryOpenRead(CubeIds.Environment, PolicyProgressId))
        {
            if (progress is not null)
            {
                using var reader = new BinaryReader(progress, System.Text.Encoding.UTF8, leaveOpen: true);
                totalSamples = reader.ReadInt64();
                round = reader.ReadInt64();
                Log($"resumed progress: {totalSamples:N0} samples generated, data stream at round {round}");
            }
        }
        double windowCe = 0, windowHuber = 0, windowAcc = 0;
        long windowCount = 0;

        Log($"training until {deadline:u} (~{hours:F1} h); teacher-free (no Kociemba), "
            + $"max scramble {maxScramble}, beam {beamWidth}, data dir: {store.RootDirectory}");

        // Self-supervised data generation runs on all cores: scrambling is independent and, unlike
        // the Kociemba-imitation campaign, no solver bounds throughput — generation is nearly free.
        int generators = Math.Max(1, Environment.ProcessorCount - 2);
        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 256);
        while (DateTime.UtcNow < deadline)
        {
            samples.Clear();
            var perWorker = new List<CubeOracle.LabeledState>[generators];
            ulong roundBase = unchecked(seed + (ulong)(++round) * 1_000_003UL);
            Parallel.For(0, generators, worker =>
            {
                var workerRng = new Xoshiro256StarStar(unchecked(roundBase + 0x9E3779B97F4A7C15UL * (ulong)(worker + 1)));
                var local = new List<CubeOracle.LabeledState>(SamplesPerRound / generators + 64);
                while (local.Count < SamplesPerRound / generators)
                    local.AddRange(CubeSelfSupervised.LabelScramblePath(workerRng, maxScramble));
                perWorker[worker] = local;
            });
            foreach (var local in perWorker)
                samples.AddRange(local);
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
                Evaluate(net, adam, store, csvPath, totalSamples,
                    windowCount > 0 ? windowCe / windowCount : 0,
                    windowCount > 0 ? windowAcc / windowCount : 0,
                    windowCount > 0 ? windowHuber / windowCount : 0, beamWidth, evalEpisodes, adaptive, round);
                windowCe = windowHuber = windowAcc = 0;
                windowCount = 0;
                nextEval = DateTime.UtcNow + evalEvery;
            }
        }

        Evaluate(net, adam, store, csvPath, totalSamples,
            windowCount > 0 ? windowCe / windowCount : 0,
            windowCount > 0 ? windowAcc / windowCount : 0,
            windowCount > 0 ? windowHuber / windowCount : 0, beamWidth, evalEpisodes, adaptive, round);
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

    /// <summary>Per-depth greedy/beam solve rate (+ mean beam solution length); checkpoints when <paramref name="adam"/> is given.</summary>
    private static void Evaluate(CubePolicyNet net, Adam? adam, FileModelStore store, string csvPath,
        long samples, double ce, double acc, double huber, int beamWidth, int episodes, AdaptiveBackend adaptive, long round)
    {
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{samples}", $"{ce:F4}", $"{acc:F4}", $"{huber:F5}" };
        var report = new System.Text.StringBuilder();
        report.Append($"[eval] samples {samples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

        // Beam search runs the bulk of the net forwards, so route them through a GPU-resident DeviceMlp
        // over the policy path (weights uploaded once for this eval) when a GPU is present; CPU autograd
        // otherwise. Rebuilt each eval to snapshot the just-trained weights.
        DeviceMlp? device = adaptive.Gpu is { } gpu ? gpu.CreateResidentForward(net.PolicyAsMlp()) : null;
        Func<float[], int, float[]> beamLogits = device is not null
            ? device.Forward
            : (features, rows) =>
            {
                using (GradMode.NoGrad())
                    return net.Forward(new Tensor(features, rows, RubiksCubeEnv.ObservationSize)).Logits.Data;
            };

        try
        {
            foreach (int depth in EvalDepths)
            {
                int greedySolved = 0, beamSolved = 0, beamLen = 0;
                for (int episode = 0; episode < episodes; episode++)
                {
                    // Fixed seeded scrambles per depth, stable across evals.
                    var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
                    var cube = new FaceletCube();
                    cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                    if (cube.IsSolved) { greedySolved++; beamSolved++; continue; }

                    if (CubePolicySearch.GreedyRollout(net, cube).Solved) greedySolved++;
                    var beam = CubePolicySearch.BeamSearch(beamLogits, cube, beamWidth);
                    if (beam.Solved) { beamSolved++; beamLen += beam.Moves.Length; }
                }
                cells.Add($"{greedySolved / (double)episodes:F3}");
                cells.Add($"{beamSolved / (double)episodes:F3}");
                string lenTag = beamSolved > 0 ? $" ({beamLen / (double)beamSolved:F1}qt)" : "";
                report.Append($"d{depth}: {greedySolved}/{episodes}g {beamSolved}/{episodes}b{lenTag} | ");
            }
        }
        finally
        {
            device?.Dispose();
        }

        Log(report.ToString());
        File.AppendAllText(csvPath, string.Join(',', cells) + "\n");
        if (adam is not null)
        {
            store.Save(CubeIds.Environment, PolicyId, s => net.Save(s));
            store.Save(CubeIds.Environment, PolicyAdamId, s =>
            {
                using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
                AdamCheckpoint.Write(adam, writer);
            });
            store.Save(CubeIds.Environment, PolicyProgressId, s =>
            {
                using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
                writer.Write(samples); // cumulative samples generated
                writer.Write(round);   // data-stream round counter
            });
        }
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
