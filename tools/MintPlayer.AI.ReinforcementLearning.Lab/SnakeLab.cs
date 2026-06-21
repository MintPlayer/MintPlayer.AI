using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
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
        bool evalOnly = false;
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
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        // DI all the way: the model store, clock and CampaignRunner are resolved from the AIHost container.
        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "snake-dqn.csv");
        runner.Run(
            new SnakeDqnCampaign(seed, trainGrid, evalGrid, chunkSteps, targetSteps, evalEpisodes, learningRate),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
    }
}
