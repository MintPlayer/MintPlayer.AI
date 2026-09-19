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
        var f = Parse(a);

        if (f.Search)
        {
            RunSearchEval(f.NetPath, f.EvalGrid, f.EvalEpisodes, f.Seed, f.SearchCfg);
            return;
        }

        if (f.Cycle)
        {
            RunCycleEval(f.NetPath, f.EvalGrid, f.EvalEpisodes, f.Seed, CycleConfig(a));
            return;
        }

        var options = Options(a, f.Hidden, f.Grow);
        LabHost.Run(args, f.DataDir, f.Hours, f.EvalOnly, useGpu: false,
            services => services.AddSnakeDqnCampaign(
                trainEnv: new SnakeEnv(f.TrainGrid, f.StepPenalty, f.SafeMask),
                evalEnv: new SnakeEnv(f.EvalGrid, f.StepPenalty, f.SafeMask),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(f.DataDir, "logs", "snake-dqn.csv")));
    }

    /// <summary>Everything <see cref="Run"/> reads before it dispatches to a mode: the campaign flags, the
    /// two env shapes, the checkpoint path the read-only modes load, and the look-ahead config.</summary>
    /// <remarks>
    /// M63.6: extracted from <see cref="Run"/>. <c>Hidden</c> already has the <c>--grow</c> override applied
    /// (a growing run REPLACES <c>--hidden</c> with <see cref="DqnGrowth.Start"/>), which is the one rule in
    /// this head that is not a plain flag read and was previously untestable.
    /// </remarks>
    internal sealed record Flags(
        double Hours, string DataDir, ulong Seed, int TrainGrid, int EvalGrid, int ChunkSteps, long TargetSteps,
        int EvalEpisodes, float LearningRate, float Explore, int[] Hidden, double Gamma, float StepPenalty,
        bool SafeMask, bool EvalOnly, bool Grow, int GrowEvery, bool Search, bool Cycle, string NetPath,
        SnakeSearchConfig SearchCfg);

    /// <summary>Reads the pure head of <see cref="Run"/> — no env, net or episode is touched.</summary>
    internal static Flags Parse(CliArgs a)
    {
        bool grow = a.Has("--grow");                       // grow the net wider+deeper mid-training (Net2Net demo)
        int[] hidden = a.Ints("--hidden", [128, 128]);     // trunk widths for the Dueling Q-net
        // A growing run starts from the tiny first stage and adds capacity mid-training (Net2Wider/DeeperNet).
        if (grow) hidden = DqnGrowth.Start;

        return new Flags(
            Hours: a.Dbl("--hours", 1),
            DataDir: a.Str("--data", "data"),
            Seed: a.ULong("--seed", 1),
            TrainGrid: a.Int("--train-grid", 6),
            EvalGrid: a.Int("--eval-grid", 12),
            ChunkSteps: a.Int("--chunk-steps", 5_000),
            // the proven M22 budget (curve plateaus ~30k); --steps 0 = time-bounded only
            TargetSteps: a.Long("--steps", 100_000),
            EvalEpisodes: a.Int("--episodes", 20),
            LearningRate: a.Flt("--lr", 5e-4f),
            Explore: a.Flt("--explore", 1.0f),             // ε-start; low (e.g. 0.2) to refine a warm-started net
            Hidden: hidden,
            Gamma: a.Dbl("--gamma", 0.99),                 // higher = longer planning horizon (long-snake routing)
            StepPenalty: a.Flt("--step-penalty", -0.01f),  // ~0 removes the safe-starvation pressure
            SafeMask: a.Has("--safe-mask"),                // forbid moves that flood-fill into too small a region
            EvalOnly: a.Has("--eval-only"),
            Grow: grow,
            GrowEvery: a.Int("--grow-every", 5000),        // steps between growth steps (with --grow)
            // --search : skip training and evaluate the net-guided look-ahead planner (M34) instead of greedy Q.
            Search: a.Has("--search"),
            // --cycle : skip training and evaluate the safety-cycle mode (M48) — win rate / deaths gate M48.
            Cycle: a.Has("--cycle"),
            NetPath: a.Str("--net", Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "snake-net.ckpt")),
            SearchCfg: SearchConfig(a));
    }

    /// <summary>
    /// Evaluates the safety-cycle mode (M48) over <paramref name="episodes"/> games: the M48.1 gate is
    /// <b>0 deaths and ≥95% board-full wins</b>. Episodes end by win (board full), death (must never happen), or
    /// the env's step-ceiling truncation (counted separately — a non-win, not a death).
    /// </summary>
    private static void RunCycleEval(string netPath, int grid, int episodes, ulong seed, SnakeCycleConfig cfg)
    {
        if (!File.Exists(netPath))
        {
            Console.Error.WriteLine($"Checkpoint not found: {Path.GetFullPath(netPath)} (pass --net <path>).");
            return;
        }

        var env = new SnakeEnv(grid, safeMask: false);
        using (var stream = File.OpenRead(netPath))
            env.LoadSearchNet(stream);

        Console.WriteLine($"Cycle eval: {episodes} episodes on {grid}×{grid}, net {Path.GetFileName(netPath)}");
        Console.WriteLine($"  config: net={cfg.NetWeight} progress={cfg.ProgressWeight} margin={cfg.Margin} rebuild={cfg.Rebuild}");

        int wins = 0, deaths = 0, truncations = 0, totalFood = 0, minFood = int.MaxValue;
        long totalMoves = 0, winMoves = 0;
        var sw = Stopwatch.StartNew();
        for (int ep = 0; ep < episodes; ep++)
        {
            env.Reset(seed + (ulong)ep);
            long moves = 0;
            bool terminated = false, truncated = false;
            while (!terminated && !truncated)
            {
                int action = env.ChooseActionCycle(cfg);
                var step = env.Step(action);
                terminated = step.Terminated;
                truncated = step.Truncated;
                moves++;
            }
            totalMoves += moves;
            bool win = terminated && env.Length == env.Cells;
            string outcome;
            if (win) { wins++; winMoves += moves; outcome = "WIN"; }
            else if (terminated) { deaths++; outcome = "DEATH"; }
            else { truncations++; outcome = "truncated"; }
            totalFood += env.FoodEaten;
            minFood = Math.Min(minFood, env.FoodEaten);
            Console.WriteLine($"  ep {ep + 1,3}: food {env.FoodEaten,3}  {moves,6} moves  {outcome}");
        }
        sw.Stop();

        double msPerMove = totalMoves == 0 ? 0 : sw.Elapsed.TotalMilliseconds / totalMoves;
        Console.WriteLine($"wins {wins}/{episodes} ({100.0 * wins / episodes:F0}%)  deaths {deaths}  truncations {truncations}");
        Console.WriteLine($"food@{grid}: mean {(double)totalFood / episodes:F1} (min {minFood})  steps-to-win mean {(wins == 0 ? 0 : (double)winMoves / wins):F0}");
        Console.WriteLine($"latency: {msPerMove:F2} ms/move  ({totalMoves} moves, {sw.Elapsed.TotalSeconds:F1}s)");
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

    /// <summary>
    /// The look-ahead planner's configuration (M34). Defaults reproduce PR #11's shipped depth-20/beam-32
    /// sweep, so these weights ARE the measured snake strength — extracted in M63.5 because a silent drift
    /// here was previously unassertable.
    /// </summary>
    internal static SnakeSearchConfig SearchConfig(CliArgs a)
    {
        var cfg = new SnakeSearchConfig();
        return cfg with
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
    }

    /// <summary>The safety-cycle mode's configuration (M48). <c>--no-rebuild</c> selects the M48.1
    /// fixed-cycle baseline; <c>--min-free -1</c> means half the board.</summary>
    internal static SnakeCycleConfig CycleConfig(CliArgs a)
    {
        var cfg = new SnakeCycleConfig();
        return cfg with
        {
            NetWeight = a.Dbl("--w-net", cfg.NetWeight),
            ProgressWeight = a.Dbl("--w-progress", cfg.ProgressWeight),
            Margin = a.Int("--margin", cfg.Margin),
            Rebuild = !a.Has("--no-rebuild"),
            ShortcutMinFree = a.Int("--min-free", -1),
        };
    }

    /// <summary>The DQN options this entry point's flags resolve to. <paramref name="hidden"/> and
    /// <paramref name="grow"/> are passed in because Run applies the growing-run override
    /// (<c>--grow</c> REPLACES <c>--hidden</c> with the tiny first stage) before the mode dispatch.</summary>
    internal static DqnScoreOptions Options(CliArgs a, int[] hidden, bool grow)
        => new()
        {
            Seed = a.ULong("--seed", 1),
            ChunkSteps = a.Int("--chunk-steps", 5_000),
            TargetSteps = a.Long("--steps", 100_000),   // the proven M22 budget; --steps 0 = time-bounded only
            EvalEpisodes = a.Int("--episodes", 20),
            LearningRate = a.Flt("--lr", 5e-4f),
            EpsilonStart = a.Flt("--explore", 1.0f),
            Hidden = hidden,
            Gamma = a.Dbl("--gamma", 0.99),             // higher = longer planning horizon (long-snake routing)
            Grow = grow,
            GrowEvery = a.Int("--grow-every", 5000),
        };
}
