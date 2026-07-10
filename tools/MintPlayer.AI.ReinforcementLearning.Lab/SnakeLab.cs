using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using MintPlayer.AI.ReinforcementLearning.Hosting;

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
        double hours = 1;
        string dataDir = "data";
        ulong seed = 1;
        int trainGrid = 6;
        int evalGrid = 12;
        int chunkSteps = 5_000;
        long targetSteps = 100_000; // the proven M22 budget (curve plateaus ~30k); --steps 0 = time-bounded only
        int evalEpisodes = 20;
        float learningRate = 5e-4f;
        float explore = 1.0f; // ε-start; pass a low value (e.g. 0.2) to refine a warm-started net rather than re-randomize it
        int[] hidden = [128, 128]; // --hidden 256,256 : trunk widths for the Dueling Q-net
        double gamma = 0.99;       // --gamma : discount; higher = longer planning horizon (needed for long-snake routing)
        float stepPenalty = -0.01f; // --step-penalty : per-step reward; ~0 removes the efficiency pressure that encourages safe starvation
        bool safeMask = false;     // --safe-mask : forbid moves that flood-fill into a region too small for the body (anti-self-trap shield)
        bool evalOnly = false;

        // --search : skip training and evaluate the net-guided look-ahead planner (M34) instead of greedy Q. The
        // net is only a leaf tiebreak, so the config defaults reproduce PR #11's shipped depth-20/beam-32 sweep.
        bool search = false;
        string netPath = Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "snake-net.ckpt");
        var cfg = new SnakeSearchConfig();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--search") search = true;
            else if (args[i] == "--net" && i + 1 < args.Length) netPath = args[++i];
            else if (args[i] == "--depth" && i + 1 < args.Length) cfg = cfg with { MaxDepth = int.Parse(args[++i]) };
            else if (args[i] == "--beam" && i + 1 < args.Length) cfg = cfg with { BeamWidth = int.Parse(args[++i]) };
            else if (args[i] == "--w-food" && i + 1 < args.Length) cfg = cfg with { FoodWeight = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            else if (args[i] == "--w-trap" && i + 1 < args.Length) cfg = cfg with { TrapPenalty = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            else if (args[i] == "--w-net" && i + 1 < args.Length) cfg = cfg with { NetWeight = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            else if (args[i] == "--w-space" && i + 1 < args.Length) cfg = cfg with { SpaceWeight = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            else if (args[i] == "--w-dist" && i + 1 < args.Length) cfg = cfg with { FoodDistWeight = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            else if (args[i] == "--w-ratio" && i + 1 < args.Length) cfg = cfg with { SpaceRatioWeight = double.Parse(args[++i], CultureInfo.InvariantCulture) };
        }

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--train-grid" && i + 1 < args.Length) trainGrid = int.Parse(args[++i]);
            else if (args[i] == "--eval-grid" && i + 1 < args.Length) evalGrid = int.Parse(args[++i]);
            else if (args[i] == "--chunk-steps" && i + 1 < args.Length) chunkSteps = int.Parse(args[++i]);
            else if (args[i] == "--steps" && i + 1 < args.Length) targetSteps = long.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--episodes" && i + 1 < args.Length) evalEpisodes = int.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--explore" && i + 1 < args.Length) explore = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--hidden" && i + 1 < args.Length) hidden = args[++i].Split(',').Select(int.Parse).ToArray();
            else if (args[i] == "--gamma" && i + 1 < args.Length) gamma = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--step-penalty" && i + 1 < args.Length) stepPenalty = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--safe-mask") safeMask = true;
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        if (search)
        {
            RunSearchEval(netPath, evalGrid, evalEpisodes, seed, cfg);
            return;
        }

        // DI all the way: the model store, clock and CampaignRunner are resolved from the AIHost container.
        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "snake-dqn.csv");
        runner.Run(
            new SnakeDqnCampaign(seed, trainGrid, evalGrid, chunkSteps, targetSteps, evalEpisodes, learningRate, explore, hidden, gamma, stepPenalty, safeMask),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
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
