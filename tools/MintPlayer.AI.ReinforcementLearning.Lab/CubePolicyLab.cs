using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

/// <summary>
/// `--game cube-policy` entry point: parses the campaign flags and runs the EfficientCube
/// <see cref="CubeEfficientCampaign"/> on the shared <see cref="CampaignRunner"/> (PLAN M25). Loop, resume,
/// eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class CubePolicyLab
{
    public static void Run(string[] args)
    {
        double hours = 24;
        string dataDir = "data";
        ulong seed = 1;
        float learningRate = 3e-4f;
        int width = 512;
        int maxScramble = 30;
        int beamWidth = 2_000;
        int evalEpisodes = 20;
        bool evalOnly = false;
        bool grow = false;         // --grow : progressively grow the net wider+deeper mid-training (Net2Net)
        int growEvery = 50_000;    // --grow-every : samples between growth steps (with --grow)
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--max-scramble" && i + 1 < args.Length) maxScramble = int.Parse(args[++i]);
            else if (args[i] == "--beam" && i + 1 < args.Length) beamWidth = int.Parse(args[++i]);
            else if (args[i] == "--episodes" && i + 1 < args.Length) evalEpisodes = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--grow") grow = true;
            else if (args[i] == "--grow-every" && i + 1 < args.Length) growEvery = int.Parse(args[++i]);
        }

        // DI all the way: the model store, clock, GPU backend and CampaignRunner are resolved from the AIHost
        // container. AddGpuBackend() registers the shared AdaptiveBackend (the cube nets are large enough to win on GPU).
        var builder = AIHost.CreateBuilder(dataDir);
        builder.Services.AddGpuBackend();
        using var host = builder.Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        var backend = host.Services.GetRequiredService<AdaptiveBackend>();
        string csvPath = Path.Combine(dataDir, "logs", "cube-policy.csv");

        var campaign = new CubeEfficientCampaign(backend, seed, learningRate, width, maxScramble, beamWidth, evalEpisodes, grow, growEvery);
        using var viz = VizLauncher.TryStart(args, campaign, host.Services.GetRequiredService<IHostEnvironment>());
        runner.Run(campaign, store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
        SnakeLab.WaitForViewer(viz);
    }
}
