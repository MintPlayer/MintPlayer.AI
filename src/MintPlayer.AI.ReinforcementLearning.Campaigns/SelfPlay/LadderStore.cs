namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Persistence seam for the self-play auto-difficulty ladder (PLAN M46.4): the tier checkpoints
/// (<c>{env}.az.d{K}.ckpt</c>) plus the <c>{env}-difficulties.json</c> manifest the web app reads. The campaign
/// talks to this interface only, so ladder promotion is unit-testable against an in-memory implementation;
/// <see cref="FileLadderStore"/> is the production implementation writing into the web app's models directory
/// (atomic temp+rename, exactly the pre-seam behavior). Kept separate from <see cref="Core.Checkpoints.IModelStore"/>
/// because the ladder writes PUBLIC web assets (URL-addressed tier files + a JSON manifest) into a different
/// directory than the training data store — forcing both through one store would leak that distinction upward.
/// </summary>
public interface ILadderStore
{
    /// <summary>Writes tier <paramref name="tier"/>'s checkpoint atomically and returns its public file name
    /// (e.g. <c>chess.az.d3.ckpt</c> — the name the manifest's <c>/models/…</c> URL uses).</summary>
    string SaveTier(string environmentId, int tier, Action<Stream> write);

    /// <summary>Opens a tier checkpoint for reading, or null when absent.</summary>
    Stream? TryOpenTier(string environmentId, int tier);

    /// <summary>The highest tier stored for the environment; 0 when none (fresh ladder).</summary>
    int HighestTier(string environmentId);

    void WriteManifest(string environmentId, string json);

    string? TryReadManifest(string environmentId);
}

/// <summary>File-backed <see cref="ILadderStore"/> over the web app's models directory.</summary>
public sealed class FileLadderStore(string directory) : ILadderStore
{
    private static string TierName(string environmentId, int tier) => $"{environmentId}.az.d{tier}.ckpt";
    private string ManifestPath(string environmentId) => Path.Combine(directory, $"{environmentId}-difficulties.json");

    public string SaveTier(string environmentId, int tier, Action<Stream> write)
    {
        Directory.CreateDirectory(directory);
        string name = TierName(environmentId, tier);
        string path = Path.Combine(directory, name);
        string tmp = path + ".tmp";
        using (var fs = File.Create(tmp)) write(fs);
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
        return name;
    }

    public Stream? TryOpenTier(string environmentId, int tier)
    {
        string path = Path.Combine(directory, TierName(environmentId, tier));
        return File.Exists(path) ? File.OpenRead(path) : null;
    }

    public int HighestTier(string environmentId)
    {
        if (!Directory.Exists(directory)) return 0;
        int highest = 0;
        foreach (var f in Directory.EnumerateFiles(directory, $"{environmentId}.az.d*.ckpt"))
        {
            string name = Path.GetFileNameWithoutExtension(f); // env.az.dK
            int dot = name.LastIndexOf(".d", StringComparison.Ordinal);
            if (dot >= 0 && int.TryParse(name[(dot + 2)..], out int k) && k > highest) highest = k;
        }
        return highest;
    }

    public void WriteManifest(string environmentId, string json)
    {
        Directory.CreateDirectory(directory);
        string path = ManifestPath(environmentId);
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, json);
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
    }

    public string? TryReadManifest(string environmentId)
    {
        string path = ManifestPath(environmentId);
        return File.Exists(path) ? File.ReadAllText(path) : null;
    }
}
