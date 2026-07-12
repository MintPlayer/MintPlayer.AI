/// <summary>
/// `--game cube` entry point: parses the campaign flags and runs the <see cref="CubeImitationCampaign"/>
/// (PLAN M16) on the shared <see cref="CampaignRunner"/> (PLAN M25). All console + CSV IO lives in
/// <see cref="CampaignCli"/>; the loop, resume, eval cadence and checkpointing live in the runner.
/// </summary>
internal static class CubeLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 9);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 3e-4f);
        int width = a.Int("--width", 512);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");           // progressively grow the net wider+deeper mid-training (Net2Net)
        int growEvery = a.Int("--grow-every", 4096); // samples between growth steps (with --grow)

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new CubeImitationCampaign(seed, learningRate, width, grow, growEvery),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "cube-imitation.csv")));
    }
}

/// <summary>Model-store ids for the cube artifacts (shared with the web app's conventions).</summary>
internal static class CubeIds
{
    public const string Environment = "cube";
    public const string Policy = "policy";
    public const string PolicyAdam = "policy-adam";

    /// <summary>The (net, Adam) id pair for a given trunk width.</summary>
    internal readonly record struct NetIds(string Policy, string PolicyAdam);

    /// <summary>
    /// The shipped 512 net keeps the bare `policy` id; every other width (the M17 ladder) gets a width-tagged id
    /// so rungs never overwrite each other or the shipped net.
    /// </summary>
    public static NetIds ForWidth(int width)
        => width == 512
            ? new NetIds(Policy, PolicyAdam)
            : new NetIds($"policy-w{width}", $"policy-w{width}-adam");
}
