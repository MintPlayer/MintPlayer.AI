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
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 9);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 3e-4f);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");           // progressively grow the net wider+deeper mid-training (Net2Net)
        int growEvery = a.Int("--grow-every", 2048); // samples between growth steps (with --grow)

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new RushHourImitationCampaign(new RushHourImitationOptions
            {
                Seed = seed, LearningRate = learningRate, Grow = grow, GrowEvery = growEvery,
            }),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "imitation.csv")));
    }
}
