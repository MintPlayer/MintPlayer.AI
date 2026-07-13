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
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 24);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 3e-4f);
        int width = a.Int("--width", 512);
        int maxScramble = a.Int("--max-scramble", 30);
        int beamWidth = a.Int("--beam", 2_000);
        int evalEpisodes = a.Int("--episodes", 20);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");           // progressively grow the net wider+deeper mid-training (Net2Net)
        int growEvery = a.Int("--grow-every", 50_000); // samples between growth steps (with --grow)

        // GPU: the cube nets are large enough to win on GPU, so the campaign runs on the AdaptiveBackend
        // (useGpu: true → LabHost registers it and this build pulls it from the container).
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: true,
            sp => new CubeEfficientCampaign(sp.GetRequiredService<AdaptiveBackend>(), seed, learningRate, width, maxScramble, beamWidth, evalEpisodes, grow, growEvery),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "cube-policy.csv")));
    }
}
