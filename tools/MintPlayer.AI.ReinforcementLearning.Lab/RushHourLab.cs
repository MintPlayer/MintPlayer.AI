using System.Globalization;

/// <summary>
/// The Lab's default `--game rushhour` entry point: parses the campaign flags and runs the
/// <see cref="RushHourImitationCampaign"/> (PLAN M16) on the shared <see cref="CampaignRunner"/> (PLAN M25). The
/// loop, resume, eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class RushHourLab
{
    public static void Run(string[] args)
    {
        double hours = 9;
        string dataDir = "data";
        ulong seed = 1;
        float learningRate = 3e-4f;
        bool evalOnly = false;
        bool grow = false;         // --grow : progressively grow the net wider+deeper mid-training (Net2Net)
        int growEvery = 2048;      // --grow-every : samples between growth steps (with --grow)
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--eval-only") evalOnly = true;
            else if (args[i] == "--grow") grow = true;
            else if (args[i] == "--grow-every" && i + 1 < args.Length) growEvery = int.Parse(args[++i]);
        }

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new RushHourImitationCampaign(seed, learningRate, grow, growEvery),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "imitation.csv")));
    }
}
