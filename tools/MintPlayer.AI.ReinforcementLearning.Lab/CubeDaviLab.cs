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
        long targetSamples = 0;       // --samples N: stop after N total states processed (0 = time-bounded only)
        int[]? probeOverride = null;  // --probe-depths a,b,c: BWAS capability-probe depths
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
        int batchSize = 128;          // DAVI training batch
        float learningRate = 1e-3f;   // --lr (linear-scaling rule: raise with batch)
        float epsSync = 0.06f;        // ε-loss target sync threshold (P.9); 0 disables
        int targetSyncInterval = 200; // --target-sync-interval: steps between bootstrap-target syncs (gated by ε-sync)
        float beta2 = 0.999f;         // --beta2: Adam β₂ (DeepCubeA uses 0.9999 for depth-20+ stability)
        int checkpointEvery = 1;      // --checkpoint-every N: write a checkpoint only every Nth eval (cuts I/O on slow storage)
        bool frontierBias = false;    // --frontier-bias: sample scramble depth near the curriculum frontier (Gaussian) instead of uniform
        int growToWidth = 0;          // --grow-to W: Net2WiderNet-widen the residual trunk to W once --grow-at is reached (0 = never)
        long growAtSamples = 0;       // --grow-at S: sample count at which to widen (progressive growing: train cheap narrow, widen on demand)
        double timeBudgetSec = 0;     // --time-budget S: eval-only, solve probe depths through the deployed CubeValueSearch with an S-second wall-clock budget (verifies the time-bounded web solver)
        bool valueCurve = false;      // --value-curve: eval-only, report mean predicted V(start) vs scramble depth (heuristic calibration — where V saturates is the accuracy ceiling)
        int setCurriculumDepth = 0;   // --set-curriculum-depth N: on resume, override the restored curriculum depth (consolidate the accuracy frontier instead of training capped deep states)
        double advanceRatio = 0.9;    // --advance-ratio R: curriculum value-accuracy gate — advance d→d+1 when mean V(d)/d ≥ R (replaces greedy-solve + force-advance)
        bool autoWiden = false;       // --auto-widen: when the frontier loss PLATEAUS (capacity-bound, not under-trained), widen the trunk (Net2WiderNet) automatically
        int maxWidth = 2048;          // --max-width W: cap for auto-widen (won't grow the trunk beyond this)
        long widenStallSamples = 50_000_000; // --widen-stall-samples N: loss-plateau window (no improvement) before an auto-widen fires
        bool batchedSearch = false;   // use batched A* (BWAS) for --search eval
        bool vsKociemba = false;      // also report Kociemba's QTM length per depth (Tier-2 gate)
        int evalEpisodes = 12;        // --episodes N: cubes per depth in --eval-only (fewer = faster deep probes)

        // Config precedence: in-code defaults (above) → appsettings.json "cube-davi" section → CLI flags (below).
        // The file holds the long-lived campaign config so a multi-day run resumes with just `--game cube-davi`;
        // any CLI flag still overrides it for a one-off. Only keys present in the file change anything.
        var cfg = CubeDaviConfig.Load(out string? cfgSource);
        if (cfgSource is not null) Log($"loaded cube-davi config from {cfgSource}");
        hours = cfg.Hours ?? hours;
        targetSamples = cfg.Samples ?? targetSamples;
        probeOverride = cfg.ProbeDepths ?? probeOverride;
        dataDir = cfg.Data ?? dataDir;
        seed = cfg.Seed ?? seed;
        netKind = cfg.Net?.ToLowerInvariant() ?? netKind;
        width = cfg.Width ?? width;
        hiddenLayers = cfg.Layers ?? hiddenLayers;
        blocks = cfg.Blocks ?? blocks;
        batchSize = cfg.Batch ?? batchSize;
        learningRate = cfg.Lr ?? learningRate;
        epsSync = cfg.EpsSync ?? epsSync;
        targetSyncInterval = cfg.TargetSyncInterval ?? targetSyncInterval;
        beta2 = cfg.Beta2 ?? beta2;
        checkpointEvery = Math.Max(1, cfg.CheckpointEvery ?? checkpointEvery);
        frontierBias = cfg.FrontierBias ?? frontierBias;
        growToWidth = cfg.GrowTo ?? growToWidth;
        growAtSamples = cfg.GrowAt ?? growAtSamples;
        setCurriculumDepth = cfg.SetCurriculumDepth ?? setCurriculumDepth;
        advanceRatio = cfg.AdvanceRatio ?? advanceRatio;
        autoWiden = cfg.AutoWiden ?? autoWiden;
        maxWidth = cfg.MaxWidth ?? maxWidth;
        widenStallSamples = cfg.WidenStallSamples ?? widenStallSamples;
        maxDepthCap = cfg.MaxDepth ?? maxDepthCap;
        evalOnly = cfg.EvalOnly ?? evalOnly;
        useSearch = cfg.Search ?? useSearch;
        batchedSearch = cfg.Batched ?? batchedSearch;
        vsKociemba = cfg.VsKociemba ?? vsKociemba;
        searchWeight = cfg.Weight ?? searchWeight;
        maxExpansions = cfg.MaxExpansions ?? maxExpansions;
        timeBudgetSec = cfg.TimeBudget ?? timeBudgetSec;
        valueCurve = cfg.ValueCurve ?? valueCurve;
        evalEpisodes = cfg.Episodes ?? evalEpisodes;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--samples" && i + 1 < args.Length) targetSamples = long.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--probe-depths" && i + 1 < args.Length) probeOverride = args[++i].Split(',').Select(int.Parse).ToArray();
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--layers" && i + 1 < args.Length) hiddenLayers = int.Parse(args[++i]); // #3: net depth
            else if (args[i] == "--blocks" && i + 1 < args.Length) blocks = int.Parse(args[++i]);
            else if (args[i] == "--batch" && i + 1 < args.Length) batchSize = int.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--eps-sync" && i + 1 < args.Length) epsSync = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--target-sync-interval" && i + 1 < args.Length) targetSyncInterval = int.Parse(args[++i]);
            else if (args[i] == "--beta2" && i + 1 < args.Length) beta2 = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--checkpoint-every" && i + 1 < args.Length) checkpointEvery = Math.Max(1, int.Parse(args[++i]));
            else if (args[i] == "--frontier-bias") frontierBias = true;
            else if (args[i] == "--grow-to" && i + 1 < args.Length) growToWidth = int.Parse(args[++i]);
            else if (args[i] == "--grow-at" && i + 1 < args.Length) growAtSamples = long.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--time-budget" && i + 1 < args.Length) timeBudgetSec = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--value-curve") valueCurve = true;
            else if (args[i] == "--set-curriculum-depth" && i + 1 < args.Length) setCurriculumDepth = int.Parse(args[++i]);
            else if (args[i] == "--advance-ratio" && i + 1 < args.Length) advanceRatio = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--auto-widen") autoWiden = true;
            else if (args[i] == "--max-width" && i + 1 < args.Length) maxWidth = int.Parse(args[++i]);
            else if (args[i] == "--widen-stall-samples" && i + 1 < args.Length) widenStallSamples = long.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--net" && i + 1 < args.Length) netKind = args[++i].ToLowerInvariant();
            else if (args[i] == "--max-depth" && i + 1 < args.Length) maxDepthCap = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--search") useSearch = true;
            else if (args[i] == "--batched") batchedSearch = true;
            else if (args[i] == "--vs-kociemba") vsKociemba = true;
            else if (args[i] == "--weight" && i + 1 < args.Length) searchWeight = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--max-exp" && i + 1 < args.Length) maxExpansions = int.Parse(args[++i]);
            else if (args[i] == "--episodes" && i + 1 < args.Length) evalEpisodes = int.Parse(args[++i]);
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
        // P.9 — ε-loss target sync (only advance the bootstrap target once the net has converged on it,
        // with the trainer's max-interval fallback) + a CLI-tunable LR (the bigger batch supports a higher
        // one via the linear-scaling rule).
        var options = new ValueIterationOptions
        {
            BatchSize = batchSize,
            LearningRate = learningRate,
            DistanceScale = 1f,
            TargetUpdateInterval = targetSyncInterval,
            TargetUpdateLossThreshold = epsSync,
            AdamBeta2 = beta2,
        };

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
                adam = new Adam(net.Parameters(), options.LearningRate, beta2: options.AdamBeta2);
            }
        }

        // Consolidation override: a long campaign can force-advance the curriculum to the cap while the value
        // is only accurate far shallower (the d26 push left V saturated ~14, OPTIMIZATIONS.md). Re-pinning the
        // curriculum to the accuracy frontier focuses every sample on states whose bootstrap targets are still
        // meaningful, instead of capped deep states that can't teach the net anything. Pair with --max-depth.
        if (setCurriculumDepth > 0 && curriculumDepth != setCurriculumDepth)
        {
            Log($"curriculum depth overridden {curriculumDepth} → {setCurriculumDepth} (--set-curriculum-depth)");
            curriculumDepth = setCurriculumDepth;
        }

        // Route DAVI's ActionCount× successor evaluation through a device-resident forward whose weights
        // stay on the GPU across steps and re-upload only on the trainer's target sync — the MLP path
        // (M20 Stage 1) or the residual path (M20 Stage 2). For a residual net we ALSO run the train step
        // fully on-device (M20 Stage 3: DeviceResidualTrainer — resident fwd+bwd+Adam), removing the
        // CPU-bound autograd train step. CPU-only machines fall back to the autograd path (nulls).
        // The device stack (resident successor-eval forward + resident train step) and the trainer are
        // rebuilt from the current net by BuildStack(). Held in mutable locals (not `using`) so progressive
        // growing (--grow-to) can dispose and recreate them at the new width mid-run.
        ITargetForward? targetForward = null;
        DeviceResidualTrainer? residentTrain = null;
        ValueIterationTrainer<FaceletCube> trainer = null!;
        void BuildStack()
        {
            (targetForward as IDisposable)?.Dispose();
            residentTrain?.Dispose();
            targetForward = adaptive.Gpu is { } gpu
                ? net switch
                {
                    Mlp m => gpu.CreateResidentForward(m),
                    ResidualMlp r => gpu.CreateResidentForward(r),
                    _ => (ITargetForward?)null,
                }
                : null;
            residentTrain = adaptive.Gpu is { } gpu2 && net is ResidualMlp resNet
                ? gpu2.CreateResidentTrainer(resNet, options.BatchSize, options.LearningRate, options.GradClipNorm, options.HuberDelta, options.AdamBeta2)
                : null;
            trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, adam, options, targetForward, residentTrain);
        }
        BuildStack();

        Log(residentTrain is not null
            ? "training: fully device-resident step (fwd+bwd+Adam on GPU); successor eval resident"
            : targetForward is not null
                ? "successor evaluation: device-resident GPU forward (resident weights); train step on CPU"
                : "successor/train: CPU autograd");

        try
        {

        if (evalOnly)
        {
            // Heuristic calibration: mean predicted cost-to-go V(start) per scramble depth. For random
            // scrambles V should track depth (≈ d, minus a little for cancellation); where it flattens is the
            // accuracy ceiling — the net can no longer tell a d20 cube from a d24 one, so search can't be guided
            // there. This separates "search-bound" (V still climbs, just needs more search) from "accuracy-bound
            // / under-trained at depth" (V saturated) — i.e. whether MORE TRAINING can deepen the net at all.
            if (valueCurve)
            {
                int[] depths = probeOverride ?? [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26];
                ValueCurve(trainer, depths, Math.Max(evalEpisodes, 50));
                return;
            }
            // Verify the DEPLOYED time-bounded solver (CubeValueSearch + the resident forward + the search
            // deadline) — the exact code the web `/api/cube/solve-davi` runs, with per-cube wall-clock timing.
            if (timeBudgetSec > 0)
            {
                int[] depths = probeOverride ?? [10, 12, 14, 16, 18];
                Func<FaceletCube, CubeValueSearch.SearchResult> solve = targetForward is not null
                    ? c => CubeValueSearch.Solve(targetForward.Forward, c, maxExpansions, searchWeight, TimeSpan.FromSeconds(timeBudgetSec))
                    : c => CubeValueSearch.Solve((ResidualMlp)net, c, maxExpansions, searchWeight, TimeSpan.FromSeconds(timeBudgetSec));
                VerifyTimeBudget(solve, depths, evalEpisodes, maxExpansions, searchWeight, timeBudgetSec, targetForward is not null);
                return;
            }
            if (useSearch) Log($"eval via {(batchedSearch ? "batched " : "")}value-guided A* (weight {searchWeight}, ≤{maxExpansions:N0} expansions)");
            if (vsKociemba) { CubeSolver.WarmUp(); Log("Tier-2 gate: comparing mean QTM length vs Kociemba"); }
            ReportEval(trainer, csvPath, totalIterations, curriculumDepth, loss: 0, evalUpTo: maxDepthCap, maxDepthCap,
                useSearch, searchWeight, maxExpansions, batchedSearch, vsKociemba, onlyDepths: probeOverride, episodes: evalEpisodes);
            return;
        }

        // Solve-rate curriculum: start shallow, deepen once the current level is ~mastered. The
        // value-iteration signal propagates outward from the goal, so easy depths must land first.
        // Depth sampling within [1, curriculumDepth]:
        //  - uniform (default): every depth equally likely — but easy depths converge early, so a fixed
        //    fraction of every batch keeps re-training already-solved shallow states.
        //  - --frontier-bias: triangular weighting toward the frontier (max of two uniform draws), so
        //    samples concentrate where the value signal is still moving. Opt-in: it changes the sampler's
        //    RNG draw pattern, so use it for a FRESH run, not to resume a uniform-sampled campaign.
        FaceletCube Sample()
        {
            int depth = frontierBias
                ? 1 + Math.Max(sampleRng.NextInt(curriculumDepth), sampleRng.NextInt(curriculumDepth))
                : 1 + sampleRng.NextInt(curriculumDepth);
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(sampleRng, depth, quarterTurnsOnly: true));
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

        // Curriculum advancement: VALUE-ACCURACY gate, NO forced advance (OPTIMIZATIONS.md "train outward,
        // gate on mastery"). The old rule — greedy solve-rate ≥0.6 OR force-advance every 384k samples —
        // deepened the curriculum on a timer whenever progress stalled, so it raced to the depth cap while the
        // value was only accurate far shallower; the deep bootstrap targets (1 + V(neighbour)) were then built
        // on unmastered inner shells and the value saturated (~14 moves regardless of true depth). Instead,
        // advance d→d+1 only once the frontier shell is genuinely accurate — mean predicted V at depth d is
        // within `advanceRatio` of d, so V tracks true cost-to-go there and its one-step targets for d+1 are
        // trustworthy. Greedy solve-rate is deliberately NOT the gate: it plateaus ~d10-12 (greedy is myopic)
        // even where the value is fine, so gating on it would freeze the curriculum. A stall now means "train
        // longer / add capacity" — a true ceiling signal — never "advance into garbage".
        const long stallWarnSamples = 4_000_000; // informational only: note when the frontier hasn't advanced
        long samplesSinceAdvance = 0;

        // P.8: train in larger chunks so the (CPU, GPU-idle) eval runs less often — it's pure overhead
        // that grows with curriculum depth. Eval every 1000 iters instead of 500.
        const int trainChunk = 1000;

        // The greedy in-loop eval plateaus ~depth 10 and UNDERSTATES the net (search reads it far deeper).
        // So every `probeEvery` iters run a BWAS capability probe at a few discriminating depths and log a
        // [cap] line + cap CSV — this records the true capability-over-time curve during the run.
        const long probeEvery = 15_000;
        // The cheap in-loop probe runs a small (8k-expansion) BWAS, so it only shows signal where that budget
        // can reach — keep it near the frontier (the real deep d16+ check is the heavy `--eval-only --search`).
        int[] probeDepths = probeOverride ?? [8, 10, 12, 14, 16];
        string capCsvPath = Path.Combine(logPath, residual ? "cube-davi-res-cap.csv" : "cube-davi-cap.csv");
        if (!File.Exists(capCsvPath))
            File.AppendAllText(capCsvPath, "utc,iterations," + string.Join(',', probeDepths.Select(d => $"d{d}")) + "\n");
        long nextProbe = totalIterations + probeEvery;

        var deadline = DateTime.UtcNow.AddHours(hours);
        float lastLoss = 0;
        // A campaign can be bounded by wall-clock (--hours), a total state count (--samples), or both —
        // whichever trips first. The state count resumes across sessions because totalIterations is restored,
        // so "run 1B states" can be done in chunked sessions and stops exactly once 1B total are processed.
        bool TargetReached() => targetSamples > 0 && totalIterations * (long)batchSize >= targetSamples;
        Log($"training until {deadline:u} (~{hours:F1} h){(targetSamples > 0 ? $" or {targetSamples:N0} total states ({totalIterations * (long)batchSize:N0} done)" : "")}, data dir: {store.RootDirectory}, depth cap {maxDepthCap}");
        Log($"batch {batchSize}, lr {learningRate:g}, ε-sync {epsSync:g}, target-sync every {targetSyncInterval}, β₂ {beta2:g}, eval every {trainChunk} iters, checkpoint every {checkpointEvery} eval(s), advance gate V/d ≥ {advanceRatio:F2}, BWAS cap-probe (d{string.Join(',', probeDepths)}) every {probeEvery:N0}{(frontierBias ? ", frontier-biased sampling" : "")}{(autoWiden ? $", auto-widen on loss-plateau ({widenStallSamples:N0} samples) up to width {maxWidth}" : "")}");

        int evalCount = 0;
        // Loss-plateau tracking for --auto-widen: the best (lowest) loss since the last advance/widen and when
        // it was hit. If the loss fails to improve for `widenStallSamples`, training has stopped helping at the
        // frontier — the signal that the shell is capacity-bound, not merely under-trained.
        double bestLossSinceReset = double.MaxValue;
        long samplesAtBestLoss = totalIterations * (long)batchSize;
        while (DateTime.UtcNow < deadline && !TargetReached())
        {
            trainer.Train(Sample, iterations: trainChunk, onIteration: (_, loss) => lastLoss = loss);
            totalIterations += trainChunk;
            samplesSinceAdvance += (long)trainChunk * batchSize;

            if (totalIterations >= nextProbe)
            {
                try { CapabilityProbe(trainer, capCsvPath, totalIterations, probeDepths); }
                catch (Exception ex) { Log($"[cap] probe failed (non-fatal): {ex.Message}"); }
                nextProbe = totalIterations + probeEvery;
            }

            // Evaluate only up to one level beyond the current curriculum — enough to decide
            // advancement, without burning time on deep failed solves the net can't reach yet.
            int evalUpTo = Math.Min(curriculumDepth + 1, maxDepthCap);
            ReportEval(trainer, csvPath, totalIterations, curriculumDepth, lastLoss, evalUpTo, maxDepthCap); // greedy CSV/log (not the advance gate)
            // Checkpointing dominates I/O on slow storage (weights + Adam moments, written every eval).
            // --checkpoint-every N writes only every Nth eval; the post-loop save below guarantees the
            // final state is always persisted regardless of where N lands.
            if (++evalCount % checkpointEvery == 0) SaveCheckpoint();

            // Value-accuracy gate: advance only when V at the frontier tracks true depth (mean V(d)/d ≥ ratio),
            // so d+1's bootstrap targets are trustworthy. No forced advance — a persistent stall is the honest
            // "needs more training / capacity" signal, surfaced as a note rather than papered over.
            long currentSamples = totalIterations * (long)batchSize;
            if (lastLoss < bestLossSinceReset * 0.98f) { bestLossSinceReset = lastLoss; samplesAtBestLoss = currentSamples; } // ≥2% improvement resets the plateau timer
            long lossStagnantSamples = currentSamples - samplesAtBestLoss;

            double frontierRatio = MeanValueAtDepth(trainer, curriculumDepth, episodes: 64) / curriculumDepth;
            if (curriculumDepth < maxDepthCap && frontierRatio >= advanceRatio)
            {
                curriculumDepth++;
                samplesSinceAdvance = 0;
                bestLossSinceReset = double.MaxValue; samplesAtBestLoss = currentSamples; // new shell → track its loss afresh
                Log($"curriculum advanced → scramble depth {curriculumDepth} (frontier V/d {frontierRatio:F2} ≥ {advanceRatio:F2})");
            }
            else if (autoWiden && net is ResidualMlp wNet && wNet.Width < maxWidth && lossStagnantSamples >= widenStallSamples)
            {
                // Loss has flatlined at the frontier (more training isn't helping) AND we still can't clear the
                // gate → capacity-bound. Widen the trunk (function-preserving warm start: accuracy is preserved,
                // the curriculum frontier doesn't move, so we never train inaccurate data — we just gain the
                // capacity for V to climb past the gate). On NEED (plateau), not a timer; distinct from --grow-at.
                int oldW = wNet.Width, newW = Math.Min(maxWidth, wNet.Width * 2);
                double plateauLoss = bestLossSinceReset;
                net = wNet.WidenTo(newW, new Xoshiro256StarStar(seed ^ ((ulong)newW * 0x9E3779B1u)), symmetryNoise: 1e-3f);
                adam = new Adam(net.Parameters(), options.LearningRate, beta2: options.AdamBeta2);
                BuildStack();
                bestLossSinceReset = double.MaxValue; samplesAtBestLoss = currentSamples; samplesSinceAdvance = 0;
                SaveCheckpoint();
                Log($"auto-widen {oldW}→{newW}: frontier d{curriculumDepth} loss plateaued (~{plateauLoss:F4} for {lossStagnantSamples:N0} samples) → added capacity (Net2WiderNet warm start) → {DescribeNet(net)}");
            }
            else if (curriculumDepth < maxDepthCap && samplesSinceAdvance >= stallWarnSamples)
            {
                Log($"frontier d{curriculumDepth} not yet mastered (V/d {frontierRatio:F2} < {advanceRatio:F2}) after {samplesSinceAdvance:N0} samples — needs longer training{(net is ResidualMlp nw && nw.Width < maxWidth ? " or more capacity (enable --auto-widen)" : "")} (no forced advance)");
                samplesSinceAdvance = 0; // throttle the note
            }

            // Progressive growing: train cheap at the narrow width, then widen the trunk once enough samples
            // are in (Net2WiderNet warm start — capability carries over; see ResidualMlp.WidenTo). Adam restarts
            // (its moments don't transfer through the widen) and the device stack rebuilds at the new width.
            if (growToWidth > 0 && net is ResidualMlp toGrow && toGrow.Width < growToWidth
                && totalIterations * (long)batchSize >= growAtSamples)
            {
                int oldWidth = toGrow.Width;
                net = toGrow.WidenTo(growToWidth, new Xoshiro256StarStar(seed ^ 0x67707D), symmetryNoise: 1e-3f);
                adam = new Adam(net.Parameters(), options.LearningRate, beta2: options.AdamBeta2);
                BuildStack();
                SaveCheckpoint(); // persist the widened shape immediately so a resume picks it up
                Log($"net widened {oldWidth}→{growToWidth} (Net2WiderNet warm start) at {totalIterations * (long)batchSize:N0} samples → {DescribeNet(net)}");
            }
        }

        SaveCheckpoint(); // always persist the latest state on exit (the loop may skip saves under --checkpoint-every)

        Log(TargetReached()
            ? $"target state count reached ({totalIterations * (long)batchSize:N0}) — final checkpoint saved."
            : "time budget reached — final checkpoint saved.");
        }
        finally
        {
            (targetForward as IDisposable)?.Dispose();
            residentTrain?.Dispose();
        }
    }

    /// <summary>
    /// Per-depth greedy solve rate (20 fixed-seed scrambles each) for depths 1..<paramref name="evalUpTo"/>,
    /// logged to CSV (columns 1..<paramref name="maxDepthCap"/>, unevaluated depths left blank); returns depth→rate.
    /// </summary>
    private static Dictionary<int, double> ReportEval(
        ValueIterationTrainer<FaceletCube> trainer, string csvPath, long iterations, int curriculumDepth, float loss, int evalUpTo, int maxDepthCap,
        bool useSearch = false, float searchWeight = 2f, int maxExpansions = 50_000, bool batched = false, bool vsKociemba = false,
        int[]? onlyDepths = null, int episodes = 12)
    {
        var rates = new Dictionary<int, double>();
        var report = new System.Text.StringBuilder();
        report.Append($"[eval{(useSearch ? "/A*" : "")}] iters {iterations:N0}, curr.depth {curriculumDepth}, loss {loss:F4} | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{iterations}", $"{curriculumDepth}", $"{loss:F5}" };

        for (int depth = 1; depth <= maxDepthCap; depth++)
        {
            // Skip depths beyond the curriculum frontier, or — when an explicit set is given
            // (--probe-depths in eval-only) — any depth not in it, so deep probes stay cheap.
            if (depth > evalUpTo || (onlyDepths is not null && !onlyDepths.Contains(depth))) { cells.Add(""); continue; }
            int solved = 0;
            long totalLen = 0;          // Σ net solution QTM over cubes the net solved
            long kociembaLen = 0;       // Σ Kociemba QTM over the SAME solved cubes (Tier-2 baseline)
            int beatsKociemba = 0;      // #cubes where the net's QTM ≤ Kociemba's
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

    /// <summary>
    /// BWAS capability probe at a few discriminating depths (8 cubes each, small expansion budget). The live
    /// greedy eval plateaus ~depth 10 and understates the net; this records the true solve-rate-over-time so a
    /// long run shows whether capability is still climbing. Logs a [cap] line + a cap CSV.
    /// </summary>
    private static void CapabilityProbe(ValueIterationTrainer<FaceletCube> trainer, string capCsvPath, long iterations, int[] depths)
    {
        const int episodes = 8;
        var report = new System.Text.StringBuilder($"[cap] iters {iterations:N0} (BWAS w2.5 ≤8k) | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{iterations}" };
        foreach (int depth in depths)
        {
            int solved = 0;
            long totalLen = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(900_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { solved++; continue; }
                var sol = trainer.SolveWithSearchBatched(cube, maxExpansions: 8000, weight: 2.5f);
                if (sol is not null) { solved++; totalLen += sol.Count; }
            }
            report.Append($"d{depth} {solved}/{episodes}{(solved > 0 ? $" ({totalLen / (double)solved:F1}qt)" : "")} | ");
            cells.Add($"{solved / (double)episodes:F3}");
        }
        Log(report.ToString());
        File.AppendAllText(capCsvPath, string.Join(',', cells) + "\n");
    }

    private static float[] Featurize(FaceletCube cube)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);
        return obs;
    }

    /// <summary>
    /// Mean predicted cost-to-go <c>V(start)</c> over <paramref name="episodes"/> fixed-seed random scrambles
    /// at <paramref name="depth"/> (forward passes only). The curriculum's value-accuracy gate compares
    /// <c>this / depth</c> against the advance ratio: a shell is ready to deepen once V tracks true depth there.
    /// </summary>
    private static double MeanValueAtDepth(ValueIterationTrainer<FaceletCube> trainer, int depth, int episodes)
    {
        double sum = 0;
        for (int e = 0; e < episodes; e++)
        {
            var rng = new Xoshiro256StarStar((ulong)(860_000 + 1_000 * depth + e));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            sum += trainer.Value(cube);
        }
        return sum / episodes;
    }

    /// <summary>
    /// Heuristic-calibration probe: mean predicted cost-to-go <c>V(start)</c> (in moves) over
    /// <paramref name="episodes"/> random scrambles per depth. A calibrated value tracks the scramble depth;
    /// the depth at which the mean flattens is where the net stops distinguishing harder cubes — the accuracy
    /// ceiling that gates how deep value-guided search can reach.
    /// </summary>
    private static void ValueCurve(ValueIterationTrainer<FaceletCube> trainer, int[] depths, int episodes)
    {
        Log($"value calibration: mean predicted V(start) vs scramble depth ({episodes} cubes/depth)");
        foreach (int depth in depths)
        {
            double sum = 0, sumSq = 0; float min = float.MaxValue, max = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(800_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
                float v = trainer.Value(cube);
                sum += v; sumSq += (double)v * v; min = MathF.Min(min, v); max = MathF.Max(max, v);
            }
            double mean = sum / episodes;
            double std = Math.Sqrt(Math.Max(0, sumSq / episodes - mean * mean));
            Log($"  d{depth,2}: mean V {mean,6:F2}  (std {std:F2}, range {min:F1}–{max:F1})  ratio V/d {mean / depth:F2}");
        }
    }

    /// <summary>
    /// Benchmark the deployed time-bounded solver: per depth, solve <paramref name="episodes"/> fixed-seed
    /// scrambles through <paramref name="solve"/> (the web's CubeValueSearch path) and report solve rate,
    /// mean solution length, and mean/worst wall-clock. The worst-case ms is the latency a user can actually
    /// hit — it should sit at (not far past) the budget, confirming the deadline binds before the expansion cap.
    /// </summary>
    private static void VerifyTimeBudget(
        Func<FaceletCube, CubeValueSearch.SearchResult> solve, int[] depths, int episodes,
        int maxExpansions, float weight, double seconds, bool gpu)
    {
        Log($"time-bounded solve verification ({(gpu ? "resident GPU forward" : "CPU forward")}): weight {weight:g}, ≤{maxExpansions:N0} exp ceiling, {seconds:F0}s budget, {episodes} cubes/depth");
        foreach (int depth in depths)
        {
            int solved = 0; long totalLen = 0; double totalMs = 0, worstMs = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(950_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = solve(cube);
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                totalMs += ms; worstMs = Math.Max(worstMs, ms);
                if (result.Solved) { solved++; totalLen += result.Moves.Length; }
            }
            string len = solved > 0 ? $", mean {totalLen / (double)solved:F1}qt" : "";
            Log($"  d{depth}: {solved}/{episodes} solved{len} | mean {totalMs / episodes:F0} ms, worst {worstMs:F0} ms");
        }
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
