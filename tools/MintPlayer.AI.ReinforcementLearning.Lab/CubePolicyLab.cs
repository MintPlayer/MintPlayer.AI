using System.Globalization;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

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
        }

        var store = new FileModelStore(dataDir);
        string csvPath = Path.Combine(store.RootDirectory, "logs", "cube-policy.csv");
        new CampaignRunner().Run(
            new CubeEfficientCampaign(seed, learningRate, width, maxScramble, beamWidth, evalEpisodes),
            store,
            new CampaignOptions
            {
                Duration = TimeSpan.FromHours(hours),
                EvalOnly = evalOnly,
                OnEval = CampaignCli.ConsoleAndCsv(csvPath),
            });
    }
}
