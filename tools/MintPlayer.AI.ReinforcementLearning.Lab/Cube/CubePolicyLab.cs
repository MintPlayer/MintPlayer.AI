using MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// `--game cube-policy` entry point: parses the campaign flags and runs the EfficientCube
/// <see cref="CubeEfficientCampaign"/> on the shared <see cref="CampaignRunner"/> (PLAN M25). Loop, resume,
/// eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class CubePolicyLab
{
    public static void Run(string[] args)
    {
        var options = Parse(new CliArgs(args), out double hours, out string dataDir, out bool evalOnly);

        // GPU: the cube nets are large enough to win on GPU, so the campaign runs on the AdaptiveBackend
        // (useGpu: true → LabHost registers it and this build pulls it from the container).
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: true,
            services => services.AddCubeEfficientCampaign(options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "cube-policy.csv")));
    }

    /// <summary>The campaign options this entry point's flags resolve to (M63.5: extracted from
    /// <see cref="Run"/>, where they were only reachable by starting a real training run).</summary>
    internal static CubeEfficientOptions Parse(CliArgs a, out double hours, out string dataDir, out bool evalOnly)
    {
        hours = a.Dbl("--hours", 24);
        dataDir = a.Str("--data", "data");
        evalOnly = a.Has("--eval-only");

        return new CubeEfficientOptions
        {
            Seed = a.ULong("--seed", 1),
            LearningRate = a.Flt("--lr", 3e-4f),
            Width = a.Int("--width", 512),
            MaxScramble = a.Int("--max-scramble", 30),
            BeamWidth = a.Int("--beam", 2_000),
            EvalEpisodes = a.Int("--episodes", 20),
            Grow = a.Has("--grow"),                    // progressively grow the net wider+deeper (Net2Net)
            GrowEvery = a.Int("--grow-every", 50_000), // samples between growth steps (with --grow)
        };
    }
}
