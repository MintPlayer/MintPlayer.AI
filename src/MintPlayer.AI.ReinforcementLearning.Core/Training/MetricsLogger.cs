using System.Globalization;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>Appends training metrics to a CSV file, one row per episode/update.</summary>
public sealed class MetricsLogger : IDisposable
{
    private readonly StreamWriter _writer;

    public MetricsLogger(string path, params string[] columns)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        _writer = new StreamWriter(path, append: false);
        _writer.WriteLine(string.Join(',', columns));
    }

    public void Log(params object[] values)
        => _writer.WriteLine(string.Join(',', values.Select(v => Convert.ToString(v, CultureInfo.InvariantCulture))));

    public void Dispose() => _writer.Dispose();
}
