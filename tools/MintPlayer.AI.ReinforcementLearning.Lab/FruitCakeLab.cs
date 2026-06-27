using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Hosting;

/// <summary>
/// `--game fruitcake` entry point: parses the campaign flags and runs the score-maximizing
/// <see cref="FruitCakeDqnCampaign"/> on the shared <see cref="CampaignRunner"/>. CPU-only (the small 41→14
/// Dueling net is far below the GPU routing threshold — the cost is the physics-in-the-loop env, which is CPU).
/// Loop, resume, eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class FruitCakeLab
{
    public static void Run(string[] args)
    {
        double hours = 1;
        string dataDir = "data";
        ulong seed = 1;
        int chunkSteps = 2_000;   // drops per chunk (each drop = simulate-to-rest; far costlier than a grid step)
        long targetSteps = 0;     // 0 = time-bounded only (score-maximizing); pass --steps N for a hard drop cap
        int evalEpisodes = 10;
        float learningRate = 5e-4f;
        float explore = 1.0f;     // ε-start; pass a low value (e.g. 0.2) to refine a warm-started net
        int[] hidden = [256, 256]; // --hidden : trunk widths for the Dueling Q-net
        double gamma = 0.99;       // --gamma : discount; high for the long drop horizon (PRD bundle uses 0.997)
        int nStep = 1;             // --nstep : n-step return horizon (1 = single-step DQN)
        bool shape = false;        // --shape : enable reward shaping (tier-reached bonus + potential-based adjacency/height)
        bool evalOnly = false;
        bool noisy = false;        // --noisy : NoisyNets exploration (learned σ) instead of ε-greedy
        bool ab = false;           // --ab : head-to-head eval of --data's net vs --baseline's net (no training)
        string baselineDir = "";   // --baseline <dir> : the net to compare --data's net against
        int abEpisodes = 200;      // --ab-episodes : paired greedy games per net (averages out eval noise)
        bool searchEval = false;   // --search-eval : F1 forward-model search vs plain net greedy, on --data's net
        int depth = 2;             // --depth : search lookahead (1 or 2)
        int topK = 5;              // --topk : depth-2 first-ply expansion width
        int topK2 = 3;             // --topk2 : deeper-ply expansion width (depth 3)
        string leaf = "net";       // --leaf : search leaf value (net | height | tierpot | blend)
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--chunk-steps" && i + 1 < args.Length) chunkSteps = int.Parse(args[++i]);
            else if (args[i] == "--steps" && i + 1 < args.Length) targetSteps = long.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--episodes" && i + 1 < args.Length) evalEpisodes = int.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--explore" && i + 1 < args.Length) explore = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--hidden" && i + 1 < args.Length) hidden = args[++i].Split(',').Select(int.Parse).ToArray();
            else if (args[i] == "--gamma" && i + 1 < args.Length) gamma = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--nstep" && i + 1 < args.Length) nStep = int.Parse(args[++i]);
            else if (args[i] == "--shape") shape = true;
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--noisy") noisy = true;
            else if (args[i] == "--ab") ab = true;
            else if (args[i] == "--baseline" && i + 1 < args.Length) baselineDir = args[++i];
            else if (args[i] == "--ab-episodes" && i + 1 < args.Length) abEpisodes = int.Parse(args[++i]);
            else if (args[i] == "--search-eval") searchEval = true;
            else if (args[i] == "--depth" && i + 1 < args.Length) depth = int.Parse(args[++i]);
            else if (args[i] == "--topk" && i + 1 < args.Length) topK = int.Parse(args[++i]);
            else if (args[i] == "--topk2" && i + 1 < args.Length) topK2 = int.Parse(args[++i]);
            else if (args[i] == "--leaf" && i + 1 < args.Length) leaf = args[++i];
        }

        if (searchEval)
        {
            // F1: does forward-model search beat the plain net on max-tier? No training/host needed.
            FruitCakeSearchEval.Run(dataDir, abEpisodes, depth, topK, seedBase: 20_000, leaf, topK2);
            return;
        }

        if (ab)
        {
            // Head-to-head, no training/host needed: compare --data's net against --baseline's net.
            FruitCakeAb.Run(baselineDir, dataDir, abEpisodes, seedBase: 20_000);
            return;
        }

        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "fruitcake-dqn.csv");
        runner.Run(
            new FruitCakeDqnCampaign(seed, chunkSteps, targetSteps, evalEpisodes, learningRate, explore, hidden, gamma, noisy, nStep, shape),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
    }
}
