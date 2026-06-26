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
        double gamma = 0.99;       // --gamma : discount; high for the long drop horizon
        bool evalOnly = false;
        bool noisy = false;        // --noisy : NoisyNets exploration (learned σ) instead of ε-greedy
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
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--noisy") noisy = true;
        }

        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "fruitcake-dqn.csv");
        runner.Run(
            new FruitCakeDqnCampaign(seed, chunkSteps, targetSteps, evalEpisodes, learningRate, explore, hidden, gamma, noisy),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
    }
}
