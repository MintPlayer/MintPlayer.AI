using System.Diagnostics;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// `--game snake` entry point: parses the campaign flags and runs the score-maximizing
/// <see cref="SnakeDqnCampaign"/> (PLAN M22) on the shared <see cref="CampaignRunner"/> (PLAN M25). CPU-only (the
/// 6×6 DQN net is far below the GPU routing threshold — no AddGpuBackend here). Loop, resume, eval cadence and
/// checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class SnakeLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        int trainGrid = a.Int("--train-grid", 6);
        int evalGrid = a.Int("--eval-grid", 12);
        int chunkSteps = a.Int("--chunk-steps", 5_000);
        long targetSteps = a.Long("--steps", 100_000); // the proven M22 budget (curve plateaus ~30k); --steps 0 = time-bounded only
        int evalEpisodes = a.Int("--episodes", 20);
        float learningRate = a.Flt("--lr", 5e-4f);
        float explore = a.Flt("--explore", 1.0f);      // ε-start; pass a low value (e.g. 0.2) to refine a warm-started net
        int[] hidden = a.Ints("--hidden", [128, 128]); // trunk widths for the Dueling Q-net
        double gamma = a.Dbl("--gamma", 0.99);         // discount; higher = longer planning horizon (long-snake routing)
        float stepPenalty = a.Flt("--step-penalty", -0.01f); // per-step reward; ~0 removes the safe-starvation pressure
        bool safeMask = a.Has("--safe-mask");          // forbid moves that flood-fill into too-small a region (anti-self-trap)
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");                   // progressively grow the net wider+deeper mid-training (Net2Net demo)
        int growEvery = a.Int("--grow-every", 5000);   // steps between growth steps (with --grow)

        // --search : skip training and evaluate the net-guided look-ahead planner (M34) instead of greedy Q. The net
        // is only a leaf tiebreak, so the config defaults reproduce PR #11's shipped depth-20/beam-32 sweep.
        bool search = a.Has("--search");
        string netPath = a.Str("--net", Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "snake-net.ckpt"));
        var cfg = new SnakeSearchConfig();
        cfg = cfg with
        {
            MaxDepth = a.Int("--depth", cfg.MaxDepth),
            BeamWidth = a.Int("--beam", cfg.BeamWidth),
            FoodWeight = a.Dbl("--w-food", cfg.FoodWeight),
            TrapPenalty = a.Dbl("--w-trap", cfg.TrapPenalty),
            NetWeight = a.Dbl("--w-net", cfg.NetWeight),
            SpaceWeight = a.Dbl("--w-space", cfg.SpaceWeight),
            FoodDistWeight = a.Dbl("--w-dist", cfg.FoodDistWeight),
            SpaceRatioWeight = a.Dbl("--w-ratio", cfg.SpaceRatioWeight),
        };

        // A growing run starts from the tiny first stage and adds capacity mid-training (Net2Wider/DeeperNet).
        if (grow) hidden = DqnGrowth.Start;

        if (search)
        {
            RunSearchEval(netPath, evalGrid, evalEpisodes, seed, cfg);
            return;
        }

        var options = new DqnScoreOptions
        {
            Seed = seed, ChunkSteps = chunkSteps, TargetSteps = targetSteps, EvalEpisodes = evalEpisodes,
            LearningRate = learningRate, EpsilonStart = explore, Hidden = hidden, Gamma = gamma,
            Grow = grow, GrowEvery = growEvery,
        };
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new SnakeDqnCampaign(
                trainEnv: new SnakeEnv(trainGrid, stepPenalty, safeMask),
                evalEnv: new SnakeEnv(evalGrid, stepPenalty, safeMask),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "snake-dqn.csv")));
    }

    /// <summary>
    /// Evaluates the net-guided look-ahead planner (M34) over <paramref name="episodes"/> games on a
    /// <paramref name="grid"/>×<paramref name="grid"/> board, reporting the food distribution and per-move latency
    /// (the latter is what gates the in-browser client-side director). The env runs with <c>safeMask: false</c> —
    /// the planner's survival scoring supersedes the reactive 1-ply shield.
    /// </summary>
    private static void RunSearchEval(string netPath, int grid, int episodes, ulong seed, SnakeSearchConfig cfg)
    {
        if (!File.Exists(netPath))
        {
            Console.Error.WriteLine($"Checkpoint not found: {Path.GetFullPath(netPath)} (pass --net <path>).");
            return;
        }

        var env = new SnakeEnv(grid, safeMask: false);
        using (var stream = File.OpenRead(netPath))
            env.LoadSearchNet(stream);

        Console.WriteLine($"Search eval: {episodes} episodes on {grid}×{grid}, net {Path.GetFileName(netPath)}");
        Console.WriteLine($"  config: depth={cfg.MaxDepth} beam={cfg.BeamWidth} food={cfg.FoodWeight} trap={cfg.TrapPenalty} net={cfg.NetWeight} space={cfg.SpaceWeight} dist={cfg.FoodDistWeight} ratio={cfg.SpaceRatioWeight}");

        int totalFood = 0, maxFood = 0, minFood = int.MaxValue;
        long totalMoves = 0;
        var sw = Stopwatch.StartNew();
        for (int ep = 0; ep < episodes; ep++)
        {
            env.Reset(seed + (ulong)ep);
            bool done = false;
            while (!done)
            {
                int action = env.ChooseActionSearch(cfg);
                var step = env.Step(action);
                done = step.Terminated || step.Truncated;
                totalMoves++;
            }
            int food = env.FoodEaten;
            totalFood += food;
            maxFood = Math.Max(maxFood, food);
            minFood = Math.Min(minFood, food);
            Console.WriteLine($"  ep {ep + 1,3}: food {food}");
        }
        sw.Stop();

        double meanFood = (double)totalFood / episodes;
        double msPerMove = totalMoves == 0 ? 0 : sw.Elapsed.TotalMilliseconds / totalMoves;
        Console.WriteLine($"food@{grid}: mean {meanFood:F1}  (min {minFood}, max {maxFood}, {episodes} eps)");
        Console.WriteLine($"planner latency: {msPerMove:F1} ms/move  ({totalMoves} moves, {sw.Elapsed.TotalSeconds:F1}s)");
    }
}
