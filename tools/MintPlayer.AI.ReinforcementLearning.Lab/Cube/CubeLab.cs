using MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// `--game cube` entry point: parses the campaign flags and runs the <see cref="CubeImitationCampaign"/>
/// (PLAN M16) on the shared <see cref="CampaignRunner"/> (PLAN M25). All console + CSV IO lives in
/// <see cref="CampaignCli"/>; the loop, resume, eval cadence and checkpointing live in the runner.
/// </summary>
internal static class CubeLab
{
    public static void Run(string[] args)
    {
        var options = Parse(new CliArgs(args), out double hours, out string dataDir, out bool evalOnly);

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddCubeImitationCampaign(options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "cube-imitation.csv")));
    }

    /// <summary>The campaign options this entry point's flags resolve to (M63.5: extracted from
    /// <see cref="Run"/>, where they were only reachable by starting a real training run).</summary>
    internal static CubeImitationOptions Parse(CliArgs a, out double hours, out string dataDir, out bool evalOnly)
    {
        hours = a.Dbl("--hours", 9);
        dataDir = a.Str("--data", "data");
        evalOnly = a.Has("--eval-only");

        return new CubeImitationOptions
        {
            Seed = a.ULong("--seed", 1),
            LearningRate = a.Flt("--lr", 3e-4f),
            Width = a.Int("--width", 512),
            Grow = a.Has("--grow"),                  // progressively grow the net wider+deeper mid-training (Net2Net)
            GrowEvery = a.Int("--grow-every", 4096), // samples between growth steps (with --grow)
        };
    }
}
