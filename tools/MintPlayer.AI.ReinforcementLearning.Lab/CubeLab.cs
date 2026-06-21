using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Hosting;

/// <summary>
/// `--game cube` entry point: parses the campaign flags and runs the <see cref="CubeImitationCampaign"/>
/// (PLAN M16) on the shared <see cref="CampaignRunner"/> (PLAN M25). All console + CSV IO lives in
/// <see cref="CampaignCli"/>; the loop, resume, eval cadence and checkpointing live in the runner.
/// </summary>
internal static class CubeLab
{
    public static void Run(string[] args)
    {
        double hours = 9;
        string dataDir = "data";
        ulong seed = 1;
        float learningRate = 3e-4f;
        bool evalOnly = false;
        int width = 512;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
            else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], CultureInfo.InvariantCulture);
            else if (args[i] == "--width" && i + 1 < args.Length) width = int.Parse(args[++i]);
            else if (args[i] == "--eval-only") evalOnly = true;
        }

        // DI all the way: the model store, clock and CampaignRunner are resolved from the AIHost container.
        using var host = AIHost.CreateBuilder(dataDir).Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();
        string csvPath = Path.Combine(dataDir, "logs", "cube-imitation.csv");
        runner.Run(new CubeImitationCampaign(seed, learningRate, width), store, new CampaignOptions
        {
            Duration = TimeSpan.FromHours(hours),
            EvalOnly = evalOnly,
            OnEval = CampaignCli.ConsoleAndCsv(csvPath),
        });
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
