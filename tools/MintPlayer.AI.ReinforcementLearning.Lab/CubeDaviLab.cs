using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

/// <summary>
/// `--game cube-davi` entry point: resolves the long-lived campaign config (code defaults → appsettings.json →
/// CLI flags) into a <see cref="CubeDaviSettings"/> and runs the teacher-free DAVI <see cref="CubeDaviCampaign"/>
/// on the shared <see cref="CampaignRunner"/> (PLAN M25). The loop, resume, eval cadence and checkpointing live in
/// the runner + the campaign; this entry is purely flag parsing + DI wiring. The campaign owns its two depth-column
/// CSVs, so the console-only <see cref="CampaignCli.Console"/> hook is used here (no generic metric CSV).
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
        bool frontierBias = false;    // --frontier-bias: sample scramble depth near the curriculum frontier (Gaussian) instead of uniform
        int growToWidth = 0;          // --grow-to W: Net2WiderNet-widen the residual trunk to W once --grow-at is reached (0 = never)
        long growAtSamples = 0;       // --grow-at S: sample count at which to widen (progressive growing: train cheap narrow, widen on demand)
        double timeBudgetSec = 0;     // --time-budget S: eval-only, solve probe depths through the deployed CubeValueSearch with an S-second wall-clock budget
        bool valueCurve = false;      // --value-curve: eval-only, report mean predicted V(start) vs scramble depth (heuristic calibration)
        int setCurriculumDepth = 0;   // --set-curriculum-depth N: on resume, override the restored curriculum depth (consolidate the accuracy frontier)
        double advanceRatio = 0.9;    // --advance-ratio R: curriculum value-accuracy gate — advance d→d+1 when mean V(d)/d ≥ R
        bool autoWiden = false;       // --auto-widen: when the frontier loss PLATEAUS (capacity-bound), widen the trunk (Net2WiderNet) automatically
        int maxWidth = 2048;          // --max-width W: cap for auto-widen (won't grow the trunk beyond this)
        long widenStallSamples = 50_000_000; // --widen-stall-samples N: loss-plateau window (no improvement) before an auto-widen fires
        bool batchedSearch = false;   // use batched A* (BWAS) for --search eval
        bool vsKociemba = false;      // also report Kociemba's QTM length per depth (Tier-2 gate)
        int evalEpisodes = 12;        // --episodes N: cubes per depth in --eval-only (fewer = faster deep probes)

        // Config precedence: in-code defaults (above) → appsettings.json "cube-davi" section → CLI flags (below).
        var cfg = CubeDaviConfig.Load(out string? cfgSource);
        if (cfgSource is not null) Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} loaded cube-davi config from {cfgSource}");
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
            else if (args[i] == "--layers" && i + 1 < args.Length) hiddenLayers = int.Parse(args[++i]);
            else if (args[i] == "--blocks" && i + 1 < args.Length) blocks = int.Parse(args[++i]);
            else if (args[i] == "--batch" && i + 1 < args.Length) batchSize = int.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--eps-sync" && i + 1 < args.Length) epsSync = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--target-sync-interval" && i + 1 < args.Length) targetSyncInterval = int.Parse(args[++i]);
            else if (args[i] == "--beta2" && i + 1 < args.Length) beta2 = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
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

        var settings = new CubeDaviSettings
        {
            Seed = seed,
            LogDirectory = Path.Combine(dataDir, "logs"),
            Residual = netKind == "residual",
            Width = width,
            HiddenLayers = hiddenLayers,
            Blocks = blocks,
            BatchSize = batchSize,
            LearningRate = learningRate,
            EpsSync = epsSync,
            TargetSyncInterval = targetSyncInterval,
            Beta2 = beta2,
            FrontierBias = frontierBias,
            GrowToWidth = growToWidth,
            GrowAtSamples = growAtSamples,
            SetCurriculumDepth = setCurriculumDepth,
            AdvanceRatio = advanceRatio,
            AutoWiden = autoWiden,
            MaxWidth = maxWidth,
            WidenStallSamples = widenStallSamples,
            MaxDepthCap = maxDepthCap,
            TargetSamples = targetSamples,
            ProbeOverride = probeOverride,
            UseSearch = useSearch,
            SearchWeight = searchWeight,
            MaxExpansions = maxExpansions,
            BatchedSearch = batchedSearch,
            VsKociemba = vsKociemba,
            TimeBudgetSec = timeBudgetSec,
            ValueCurve = valueCurve,
            EvalEpisodes = evalEpisodes,
        };

        // DI all the way: the model store, clock, GPU backend and CampaignRunner are resolved from the AIHost
        // container. AddGpuBackend() registers the shared AdaptiveBackend (DAVI's wide value net wins on GPU).
        var builder = AIHost.CreateBuilder(dataDir);
        builder.Services.AddGpuBackend();
        using var host = builder.Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        var backend = host.Services.GetRequiredService<AdaptiveBackend>();

        var campaign = new CubeDaviCampaign(backend, settings);
        using var viz = VizLauncher.TryStart(args, campaign, host.Services.GetRequiredService<IHostEnvironment>());
        runner.Run(campaign, store, new CampaignOptions
        {
            Duration = TimeSpan.FromHours(hours),
            EvalOnly = evalOnly,
            OnEval = CampaignCli.Console(),
        });
        SnakeLab.WaitForViewer(viz);
    }
}
