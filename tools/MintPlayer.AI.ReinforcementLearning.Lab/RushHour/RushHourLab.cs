using MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// The Lab's default `--game rushhour` entry point: parses the campaign flags and runs the
/// <see cref="RushHourImitationCampaign"/> (PLAN M16) on the shared <see cref="CampaignRunner"/> (PLAN M25). The
/// loop, resume, eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class RushHourLab
{
    public static void Run(string[] args)
    {
        var options = Parse(new CliArgs(args), out double hours, out string dataDir, out bool evalOnly);

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddRushHourImitationCampaign(options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "imitation.csv")));
    }

    /// <summary>The campaign options this entry point's flags resolve to (M63.5: extracted from
    /// <see cref="Run"/>, where they were only reachable by starting a real training run).</summary>
    internal static RushHourImitationOptions Parse(CliArgs a, out double hours, out string dataDir, out bool evalOnly)
    {
        hours = a.Dbl("--hours", 9);
        dataDir = a.Str("--data", "data");
        evalOnly = a.Has("--eval-only");

        return new RushHourImitationOptions
        {
            Seed = a.ULong("--seed", 1),
            LearningRate = a.Flt("--lr", 3e-4f),
            Grow = a.Has("--grow"),                  // progressively grow the net wider+deeper mid-training (Net2Net)
            GrowEvery = a.Int("--grow-every", 2048), // samples between growth steps (with --grow)
        };
    }
}
