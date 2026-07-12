using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

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

        // GPU: the cube nets are large enough to win on GPU, so the campaign runs on the AdaptiveBackend
        // (useGpu: true → LabHost registers it and this build pulls it from the container).
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: true,
            sp => new CubeEfficientCampaign(sp.GetRequiredService<AdaptiveBackend>(), seed, learningRate, width, maxScramble, beamWidth, evalEpisodes, grow, growEvery),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "cube-policy.csv")));
    }
}
