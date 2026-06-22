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
        bool search = false;       // --search : eval with net-guided multi-ply look-ahead (SnakeSearchAgent) instead of reactive greedy
        int searchDepth = 6;       // --depth : look-ahead plies
        int searchBeam = 32;       // --beam : live nodes carried per ply
        int evalStarveCells = 2;   // --starve : eval/inference starvation window as a multiple of the board (training stays at 2)
        var sw = new SnakeSearchOptions(); // search leaf-score weights (overridable for tuning sweeps)
        float wFood = sw.FoodWeight, wTrap = sw.TrapPenalty, wNet = sw.NetWeight, wSpace = sw.SpaceWeight;
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
            else if (args[i] == "--search") search = true;
            else if (args[i] == "--depth" && i + 1 < args.Length) searchDepth = int.Parse(args[++i]);
            else if (args[i] == "--beam" && i + 1 < args.Length) searchBeam = int.Parse(args[++i]);
            else if (args[i] == "--starve" && i + 1 < args.Length) evalStarveCells = int.Parse(args[++i]);
            else if (args[i] == "--w-food" && i + 1 < args.Length) wFood = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--w-trap" && i + 1 < args.Length) wTrap = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--w-net" && i + 1 < args.Length) wNet = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--w-space" && i + 1 < args.Length) wSpace = float.Parse(args[++i], CultureInfo.InvariantCulture);
        }
        var searchOptions = new SnakeSearchOptions { FoodWeight = wFood, TrapPenalty = wTrap, NetWeight = wNet, SpaceWeight = wSpace };

        // DI all the way: the model store, clock and CampaignRunner are resolved from the AIHost container.
        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "snake-dqn.csv");
        runner.Run(
            new SnakeDqnCampaign(seed, trainGrid, evalGrid, chunkSteps, targetSteps, evalEpisodes, learningRate, explore, hidden, gamma, stepPenalty, safeMask, search, searchDepth, searchBeam, searchOptions, evalStarveCells),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
    }
}
