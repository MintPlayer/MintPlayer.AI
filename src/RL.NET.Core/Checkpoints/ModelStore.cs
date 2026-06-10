using System.Text.RegularExpressions;

namespace RLNet.Core.Checkpoints;

/// <summary>
/// One *current* checkpoint per (environment, algorithm) pair — the persistence seam
/// behind PRD §7's "training never restarts from scratch".
/// </summary>
public interface IModelStore
{
    bool Exists(string environmentId, string algorithmId);

    /// <summary>Opens the stored checkpoint for reading, or null if none exists.</summary>
    Stream? TryOpenRead(string environmentId, string algorithmId);

    /// <summary>Saves atomically: the previous checkpoint stays intact if <paramref name="write"/> throws.</summary>
    void Save(string environmentId, string algorithmId, Action<Stream> write);

    IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List();

    bool Delete(string environmentId, string algorithmId);
}

/// <summary>
/// File-backed model store: <c>&lt;root&gt;/&lt;envId&gt;.&lt;algoId&gt;.ckpt</c>, written
/// via a temp file + rename so a crash mid-save never corrupts the current checkpoint.
/// </summary>
public sealed partial class FileModelStore(string rootDirectory) : IModelStore
{
    [GeneratedRegex("^[A-Za-z0-9_-]+$")]
    private static partial Regex ValidId();

    public string RootDirectory { get; } = Path.GetFullPath(rootDirectory);

    public string PathOf(string environmentId, string algorithmId)
        => Path.Combine(RootDirectory, $"{Validate(environmentId)}.{Validate(algorithmId)}.ckpt");

    public bool Exists(string environmentId, string algorithmId)
        => File.Exists(PathOf(environmentId, algorithmId));

    public Stream? TryOpenRead(string environmentId, string algorithmId)
    {
        string path = PathOf(environmentId, algorithmId);
        return File.Exists(path) ? File.OpenRead(path) : null;
    }

    public void Save(string environmentId, string algorithmId, Action<Stream> write)
    {
        string path = PathOf(environmentId, algorithmId);
        Directory.CreateDirectory(RootDirectory);
        string temp = path + ".tmp";
        try
        {
            using (var stream = File.Create(temp))
                write(stream);
            File.Move(temp, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List()
    {
        if (!Directory.Exists(RootDirectory)) return [];
        var entries = new List<(string, string)>();
        foreach (string file in Directory.EnumerateFiles(RootDirectory, "*.ckpt"))
        {
            string[] parts = Path.GetFileNameWithoutExtension(file).Split('.');
            if (parts.Length == 2 && ValidId().IsMatch(parts[0]) && ValidId().IsMatch(parts[1]))
                entries.Add((parts[0], parts[1]));
        }
        return entries;
    }

    public bool Delete(string environmentId, string algorithmId)
    {
        string path = PathOf(environmentId, algorithmId);
        if (!File.Exists(path)) return false;
        File.Delete(path);
        return true;
    }

    private static string Validate(string id)
        => ValidId().IsMatch(id) ? id : throw new ArgumentException(
            $"Invalid model-store id '{id}' (letters, digits, '-' and '_' only).");
}
