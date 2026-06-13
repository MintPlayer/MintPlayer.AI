using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// Teacher-free cube campaign (`--game cube-davi`): trains a value net by deep approximate
/// value iteration (<see cref="ValueIterationTrainer{TState}"/>) over <see cref="CubeModel"/> —
/// no Kociemba, no oracle, just the goal and a cost objective. A solve-rate curriculum grows the
/// scramble depth as the net masters each level. Runs on the <see cref="AdaptiveBackend"/>: DAVI's
/// one-step lookahead evaluates ActionCount× successors per state, so the value-net forwards are
/// large enough to land on the GPU (the small per-step train pass stays on the CPU). Resumable:
/// the value net reloads from the model store (`cube`/`value-davi`).
/// </summary>
internal static class CubeDaviLab
{
    private const int Hidden = 1024;

    public static void Run(string[] args)
    {
        double hours = 9;
        string dataDir = "data";
        ulong seed = 1;
        int width = Hidden;
        int hiddenLayers = 2;
        int maxDepthCap = 8;
        bool evalOnly = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--layers" && i + 1 < args.Length) hiddenLayers = int.Parse(args[++i]); // #3: net depth
            else if (args[i] == "--max-depth" && i + 1 < args.Length) maxDepthCap = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        using var adaptive = new AdaptiveBackend();
        Backend.Current = adaptive;
        Log($"compute backend: {adaptive.Describe()}");

        var store = new FileModelStore(dataDir);
        string logPath = Path.Combine(store.RootDirectory, "logs");
        Directory.CreateDirectory(logPath);
        string csvPath = Path.Combine(logPath, "cube-davi.csv");
        if (!File.Exists(csvPath))
            File.AppendAllText(csvPath, "utc,iterations,curriculumDepth,loss," + string.Join(',', Enumerable.Range(1, maxDepthCap).Select(d => $"d{d}")) + "\n");

        Mlp net;
        using (var existing = store.TryOpenRead(CubeIds.Environment, "value-davi"))
        {
            if (existing is not null) { net = MlpCheckpoint.Load(existing); Log($"resumed DAVI value net ({string.Join('→', net.Sizes)})"); }
            else
            {
                // #3: net shape is configurable — hiddenLayers × width. Deeper/wider raises the
                // representational ceiling (resumed nets keep their stored shape regardless).
                var sizes = new int[hiddenLayers + 2];
                sizes[0] = RubiksCubeEnv.ObservationSize;
                for (int i = 1; i <= hiddenLayers; i++) sizes[i] = width;
                sizes[^1] = 1;
                net = new Mlp(sizes, new Xoshiro256StarStar(seed ^ 0xDA71), Activation.Relu);
                Log($"fresh DAVI value net ({string.Join('→', sizes)})");
            }
        }

        var model = new CubeModel();
        var options = new ValueIterationOptions { BatchSize = 128, LearningRate = 1e-3f, DistanceScale = 1f, TargetUpdateInterval = 200 };

        // Resume the FULL training state so a restart continues seamlessly. DAVI regenerates its
        // states from scrambles (nothing to store there — regeneration is free); what a resume must
        // reuse is the LEARNED state: Adam's moments, the curriculum depth, the iteration count and
        // the sampler RNG position. Without these a restart re-warms the optimizer and re-climbs the
        // curriculum from scratch.
        Adam adam;
        int curriculumDepth = 2;
        long totalIterations = 0;
        var sampleRng = new Xoshiro256StarStar(seed);
        using (var state = store.TryOpenRead(CubeIds.Environment, "value-davi-state"))
        {
            if (state is not null)
            {
                using var reader = new BinaryReader(state, System.Text.Encoding.UTF8, leaveOpen: true);
                CheckpointFormat.ReadHeader(reader, "cube-davi-state", 1);
                curriculumDepth = reader.ReadInt32();
                totalIterations = reader.ReadInt64();
                sampleRng = CheckpointFormat.ReadRngState(reader);
                adam = AdamCheckpoint.Read(net.Parameters(), reader);
                adam.LearningRate = options.LearningRate;
                Log($"resumed training state: curriculum depth {curriculumDepth}, {totalIterations:N0} iterations done");
            }
            else
            {
                adam = new Adam(net.Parameters(), options.LearningRate);
            }
        }

        // M20 Stage 1: route DAVI's ActionCount× successor evaluation through a device-resident MLP
        // whose weights stay on the GPU across steps and re-upload only on the trainer's target sync
        // (the dominant cost is this successor batch; the small autograd train pass stays on the
        // AdaptiveBackend — CPU at this size). CPU-only machines pass null → autograd forward.
        using DeviceMlp? resident = adaptive.Gpu is { } gpu ? gpu.CreateResidentForward(net) : null;
        Log(resident is not null ? "successor evaluation: device-resident GPU forward (resident weights)" : "successor evaluation: CPU autograd forward");

        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, adam, options, resident);

        if (evalOnly)
        {
            ReportEval(trainer, csvPath, totalIterations, curriculumDepth, loss: 0, evalUpTo: maxDepthCap, maxDepthCap);
            return;
        }

        // Solve-rate curriculum: start shallow, deepen once the current level is ~mastered. The
        // value-iteration signal propagates outward from the goal, so easy depths must land first.
        FaceletCube Sample()
        {
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(sampleRng, 1 + sampleRng.NextInt(curriculumDepth), quarterTurnsOnly: true));
            return cube;
        }

        void SaveCheckpoint()
        {
            store.Save(CubeIds.Environment, "value-davi", s => MlpCheckpoint.Save(net, s));
            store.Save(CubeIds.Environment, "value-davi-state", s =>
            {
                using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
                CheckpointFormat.WriteHeader(writer, "cube-davi-state", 1);
                writer.Write(curriculumDepth);
                writer.Write(totalIterations);
                CheckpointFormat.WriteRngState(writer, sampleRng);
                AdamCheckpoint.Write(adam, writer);
            });
        }

        // Curriculum advancement: greedy solve-rate falls off with depth (greedy is myopic), so a
        // pure mastery gate caps the curriculum where greedy stalls. Advance when the frontier is
        // mostly solved OR force-advance after a stall — exposure to deeper states is what lets the
        // VALUE function (and thus value-guided search at inference) reach deep, even where greedy
        // never hits 95%.
        const double advanceThreshold = 0.6;
        const long forceAdvanceIters = 3000;
        long itersSinceAdvance = 0;

        var deadline = DateTime.UtcNow.AddHours(hours);
        float lastLoss = 0;
        Log($"training until {deadline:u} (~{hours:F1} h), data dir: {store.RootDirectory}, depth cap {maxDepthCap}");

        while (DateTime.UtcNow < deadline)
        {
            trainer.Train(Sample, iterations: 500, onIteration: (_, loss) => lastLoss = loss);
            totalIterations += 500;
            itersSinceAdvance += 500;

            // Evaluate only up to one level beyond the current curriculum — enough to decide
            // advancement, without burning time on deep failed solves the net can't reach yet.
            int evalUpTo = Math.Min(curriculumDepth + 1, maxDepthCap);
            var rates = ReportEval(trainer, csvPath, totalIterations, curriculumDepth, lastLoss, evalUpTo, maxDepthCap);
            SaveCheckpoint();

            bool mastered = rates[curriculumDepth] >= advanceThreshold;
            bool stalled = itersSinceAdvance >= forceAdvanceIters;
            if (curriculumDepth < maxDepthCap && (mastered || stalled))
            {
                curriculumDepth++;
                itersSinceAdvance = 0;
                Log($"curriculum advanced → scramble depth {curriculumDepth}{(mastered ? "" : " (forced after stall)")}");
            }
        }

        Log("time budget reached — final checkpoint saved.");
    }

    /// <summary>
    /// Per-depth greedy solve rate (20 fixed-seed scrambles each) for depths 1..<paramref name="evalUpTo"/>,
    /// logged to CSV (columns 1..<paramref name="maxDepthCap"/>, unevaluated depths left blank); returns depth→rate.
    /// </summary>
    private static Dictionary<int, double> ReportEval(
        ValueIterationTrainer<FaceletCube> trainer, string csvPath, long iterations, int curriculumDepth, float loss, int evalUpTo, int maxDepthCap)
    {
        var rates = new Dictionary<int, double>();
        var report = new System.Text.StringBuilder();
        report.Append($"[eval] iters {iterations:N0}, curr.depth {curriculumDepth}, loss {loss:F4} | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{iterations}", $"{curriculumDepth}", $"{loss:F5}" };

        for (int depth = 1; depth <= maxDepthCap; depth++)
        {
            if (depth > evalUpTo) { cells.Add(""); continue; } // beyond the curriculum frontier — skip
            int solved = 0;
            const int episodes = 20;
            for (int episode = 0; episode < episodes; episode++)
            {
                var evalRng = new Xoshiro256StarStar((ulong)(700_000 + 1_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved || trainer.Solve(cube, maxSteps: 2 * depth + 8) is not null) solved++;
            }
            double rate = solved / (double)episodes;
            rates[depth] = rate;
            cells.Add($"{rate:F3}");
            report.Append($"d{depth} {solved}/{episodes} | ");
        }

        Log(report.ToString());
        File.AppendAllText(csvPath, string.Join(',', cells) + "\n");
        return rates;
    }

    private static float[] Featurize(FaceletCube cube)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);
        return obs;
    }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
