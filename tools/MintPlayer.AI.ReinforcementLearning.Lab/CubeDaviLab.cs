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
        bool useSearch = false;       // eval via value-guided A* (else greedy)
        float searchWeight = 2f;      // f = g + weight·h; >1 reaches deeper, may be non-optimal
        int maxExpansions = 50_000;   // A* node budget per solve
        string netKind = "mlp";       // "mlp" (plain) or "residual" (M21 deep residual value net)
        int blocks = 4;               // residual block count (--net residual)
        bool batchedSearch = false;   // use batched A* (BWAS) for --search eval
        bool vsKociemba = false;      // also report Kociemba's QTM length per depth (Tier-2 gate)
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--layers" && i + 1 < args.Length) hiddenLayers = int.Parse(args[++i]); // #3: net depth
            else if (args[i] == "--blocks" && i + 1 < args.Length) blocks = int.Parse(args[++i]);
            else if (args[i] == "--net" && i + 1 < args.Length) netKind = args[++i].ToLowerInvariant();
            else if (args[i] == "--max-depth" && i + 1 < args.Length) maxDepthCap = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--search") useSearch = true;
            else if (args[i] == "--batched") batchedSearch = true;
            else if (args[i] == "--vs-kociemba") vsKociemba = true;
            else if (args[i] == "--weight" && i + 1 < args.Length) searchWeight = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--max-exp" && i + 1 < args.Length) maxExpansions = int.Parse(args[++i]);
        }
        bool residual = netKind == "residual";

        using var adaptive = new AdaptiveBackend();
        Backend.Current = adaptive;
        Log($"compute backend: {adaptive.Describe()}");

        var store = new FileModelStore(dataDir);
        string logPath = Path.Combine(store.RootDirectory, "logs");
        Directory.CreateDirectory(logPath);
        // Residual and plain nets use distinct checkpoint ids + CSVs so the two campaigns resume
        // independently and never collide on format or column count.
        string valueId = residual ? "value-davi-res" : "value-davi";
        string stateId = valueId + "-state";
        string csvPath = Path.Combine(logPath, residual ? "cube-davi-res.csv" : "cube-davi.csv");
        if (!File.Exists(csvPath))
            File.AppendAllText(csvPath, "utc,iterations,curriculumDepth,loss," + string.Join(',', Enumerable.Range(1, maxDepthCap).Select(d => $"d{d}")) + "\n");

        IValueNet net;
        using (var existing = store.TryOpenRead(CubeIds.Environment, valueId))
        {
            if (existing is not null)
            {
                net = residual ? ResidualMlpCheckpoint.Load(existing) : MlpCheckpoint.Load(existing);
                Log($"resumed DAVI value net ({DescribeNet(net)})");
            }
            else if (residual)
            {
                net = new ResidualMlp(RubiksCubeEnv.ObservationSize, width, blocks, new Xoshiro256StarStar(seed ^ 0xDA71));
                Log($"fresh DAVI value net ({DescribeNet(net)})");
            }
            else
            {
                // #3: net shape is configurable — hiddenLayers × width. Deeper/wider raises the
                // representational ceiling (resumed nets keep their stored shape regardless).
                var sizes = new int[hiddenLayers + 2];
                sizes[0] = RubiksCubeEnv.ObservationSize;
                for (int i = 1; i <= hiddenLayers; i++) sizes[i] = width;
                sizes[^1] = 1;
                net = new Mlp(sizes, new Xoshiro256StarStar(seed ^ 0xDA71), Activation.Relu);
                Log($"fresh DAVI value net ({DescribeNet(net)})");
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
        using (var state = store.TryOpenRead(CubeIds.Environment, stateId))
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

        // Route DAVI's ActionCount× successor evaluation through a device-resident forward whose
        // weights stay on the GPU across steps and re-upload only on the trainer's target sync — the
        // MLP path (M20 Stage 1) or the residual path (M20 Stage 2), both fully on-device. CPU-only
        // machines fall back to the autograd forward (null).
        ITargetForward? targetForward = adaptive.Gpu is { } gpu
            ? net switch
            {
                Mlp m => gpu.CreateResidentForward(m),
                ResidualMlp r => gpu.CreateResidentForward(r),
                _ => (ITargetForward?)null,
            }
            : null;
        using var residentDisposable = targetForward as IDisposable;
        Log(targetForward is not null
            ? "successor evaluation: device-resident GPU forward (resident weights)"
            : "successor evaluation: CPU autograd forward");

        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, adam, options, targetForward);

        if (evalOnly)
        {
            if (useSearch) Log($"eval via {(batchedSearch ? "batched " : "")}value-guided A* (weight {searchWeight}, ≤{maxExpansions:N0} expansions)");
            if (vsKociemba) { CubeSolver.WarmUp(); Log("Tier-2 gate: comparing mean QTM length vs Kociemba"); }
            ReportEval(trainer, csvPath, totalIterations, curriculumDepth, loss: 0, evalUpTo: maxDepthCap, maxDepthCap,
                useSearch, searchWeight, maxExpansions, batchedSearch, vsKociemba);
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
            store.Save(CubeIds.Environment, valueId, s =>
            {
                if (net is ResidualMlp res) ResidualMlpCheckpoint.Save(res, s);
                else MlpCheckpoint.Save((Mlp)net, s);
            });
            store.Save(CubeIds.Environment, stateId, s =>
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
        ValueIterationTrainer<FaceletCube> trainer, string csvPath, long iterations, int curriculumDepth, float loss, int evalUpTo, int maxDepthCap,
        bool useSearch = false, float searchWeight = 2f, int maxExpansions = 50_000, bool batched = false, bool vsKociemba = false)
    {
        var rates = new Dictionary<int, double>();
        var report = new System.Text.StringBuilder();
        report.Append($"[eval{(useSearch ? "/A*" : "")}] iters {iterations:N0}, curr.depth {curriculumDepth}, loss {loss:F4} | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{iterations}", $"{curriculumDepth}", $"{loss:F5}" };

        for (int depth = 1; depth <= maxDepthCap; depth++)
        {
            if (depth > evalUpTo) { cells.Add(""); continue; } // beyond the curriculum frontier — skip
            int solved = 0;
            long totalLen = 0;          // Σ net solution QTM over cubes the net solved
            long kociembaLen = 0;       // Σ Kociemba QTM over the SAME solved cubes (Tier-2 baseline)
            int beatsKociemba = 0;      // #cubes where the net's QTM ≤ Kociemba's
            const int episodes = 20;
            for (int episode = 0; episode < episodes; episode++)
            {
                var evalRng = new Xoshiro256StarStar((ulong)(700_000 + 1_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { solved++; continue; }

                var solution = useSearch
                    ? (batched ? trainer.SolveWithSearchBatched(cube, maxExpansions, searchWeight)
                               : trainer.SolveWithSearch(cube, maxExpansions, searchWeight))
                    : trainer.Solve(cube, maxSteps: 2 * depth + 8);
                if (solution is not null)
                {
                    solved++; totalLen += solution.Count;
                    if (vsKociemba)
                    {
                        int kqt = KociembaQtm(cube);
                        kociembaLen += kqt;
                        if (solution.Count <= kqt) beatsKociemba++;
                    }
                }
            }
            double rate = solved / (double)episodes;
            rates[depth] = rate;
            cells.Add($"{rate:F3}");
            // For A* the solution LENGTH matters (the shortest-move objective): show mean QTM over solves.
            string lenTag = useSearch && solved > 0 ? $" ({totalLen / (double)solved:F1}qt)" : "";
            string kTag = vsKociemba && solved > 0 ? $" [vs Koc {kociembaLen / (double)solved:F1}qt, ≤{beatsKociemba}/{solved}]" : "";
            report.Append($"d{depth} {solved}/{episodes}{lenTag} | ");
            if (useSearch) Log($"  d{depth}: {solved}/{episodes} solved{lenTag}{kTag}"); // incremental progress (A* is slow)
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

    /// <summary>Quarter-turn length of Kociemba's (half-turn-metric) solution — a "X2" face turn is 2 quarter-turns.</summary>
    private static int KociembaQtm(FaceletCube cube)
    {
        var result = CubeSolver.Solve(cube);
        if (!result.Solved) return 1000;
        int qt = 0;
        foreach (var move in result.Moves) qt += move.EndsWith('2') ? 2 : 1;
        return qt;
    }

    private static string DescribeNet(IValueNet net) => net switch
    {
        Mlp mlp => string.Join('→', mlp.Sizes),
        ResidualMlp res => $"residual {res.InputSize}→{res.Width}×{res.Blocks}blocks→1",
        _ => net.GetType().Name,
    };

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
