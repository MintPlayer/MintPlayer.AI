using System.Globalization;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Lab-side IO for <see cref="CampaignRunner"/> (the runner itself is IO-agnostic). Builds the
/// <see cref="CampaignOptions.OnEval"/> hook that prints each evaluation's summary to the console and appends a
/// row to a CSV whose columns are derived from the eval's <see cref="CampaignEval.Metrics"/> — so every migrated
/// game logs uniformly without the runner (in Core) ever touching the filesystem or console.
/// </summary>
internal static class CampaignCli
{
    /// <summary>
    /// An <see cref="CampaignOptions.OnEval"/> callback: console summary + a generic CSV at <paramref name="csvPath"/>
    /// (header written from the first eval's metric names; each value formatted by its <see cref="CampaignMetric.Format"/>).
    /// </summary>
    public static Action<CampaignProgress> ConsoleAndCsv(string csvPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
        bool headerWritten = File.Exists(csvPath);
        return progress =>
        {
            var metrics = progress.Eval.Metrics;
            if (!headerWritten)
            {
                File.AppendAllText(csvPath, "utc," + string.Join(',', metrics.Select(m => m.Name)) + "\n");
                headerWritten = true;
            }
            File.AppendAllText(csvPath, $"{DateTime.UtcNow:u}," +
                string.Join(',', metrics.Select(m => m.Value.ToString(m.Format ?? "G", CultureInfo.InvariantCulture))) + "\n");
            Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} [eval] {progress.Eval.Summary}");
        };
    }
}
