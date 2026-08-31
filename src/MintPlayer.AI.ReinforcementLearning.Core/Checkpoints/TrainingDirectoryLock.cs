using System.Globalization;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// An exclusive, process-lifetime lock over a training data directory, so two training runs can never write
/// the same one.
/// <para>
/// This exists because it happened. Two Lab processes were pointed at <c>data/tet14train</c> at the same time;
/// both appended to <c>logs/*.csv</c> (leaving two header rows and interleaved steps) and both wrote
/// <c>*.ckpt</c> and <c>*-state.ckpt</c>. Nothing failed loudly — the run simply produced numbers that were a
/// blend of two policies, and they were believed and acted upon before the interleaving was noticed. A
/// resumable campaign is *especially* vulnerable: the second process reads a checkpoint the first one is
/// concurrently rewriting.
/// </para>
/// <para>
/// The lock is an OS file handle opened with <see cref="FileShare.None"/>, so it is released automatically
/// when the process exits — including a hard kill, which a PID file could not survive without going stale.
/// <see cref="FileOptions.DeleteOnClose"/> keeps the directory clean. A sibling <c>.owner</c> file records who
/// holds it, purely so the failure message can name the culprit.
/// </para>
/// </summary>
public sealed class TrainingDirectoryLock : IDisposable
{
    private const string LockFileName = ".training.lock";
    private const string OwnerFileName = ".training.owner";

    private readonly FileStream _handle;
    private readonly string _ownerPath;

    private TrainingDirectoryLock(FileStream handle, string ownerPath)
    {
        _handle = handle;
        _ownerPath = ownerPath;
    }

    /// <summary>The directory this lock covers.</summary>
    public string Directory { get; private init; } = string.Empty;

    /// <summary>
    /// Take exclusive ownership of <paramref name="dataDirectory"/> for the lifetime of this object.
    /// </summary>
    /// <exception cref="TrainingDirectoryLockedException">
    /// Another live process already owns it. The message names that process where it can be determined —
    /// the correct response is to stop it, not to retry or to point this run elsewhere and forget.
    /// </exception>
    public static TrainingDirectoryLock Acquire(string dataDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataDirectory);
        var root = Path.GetFullPath(dataDirectory);
        System.IO.Directory.CreateDirectory(root);

        var lockPath = Path.Combine(root, LockFileName);
        var ownerPath = Path.Combine(root, OwnerFileName);

        FileStream handle;
        try
        {
            handle = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None,
                bufferSize: 1, FileOptions.DeleteOnClose);
        }
        catch (IOException)
        {
            throw new TrainingDirectoryLockedException(root, DescribeOwner(ownerPath));
        }
        catch (UnauthorizedAccessException)
        {
            // Windows can surface a share violation this way when the holder opened it DeleteOnClose.
            throw new TrainingDirectoryLockedException(root, DescribeOwner(ownerPath));
        }

        TryWriteOwner(ownerPath);
        return new TrainingDirectoryLock(handle, ownerPath) { Directory = root };
    }

    private static void TryWriteOwner(string ownerPath)
    {
        // Best-effort diagnostics only: never fail a run because the breadcrumb could not be written.
        try
        {
            using var p = System.Diagnostics.Process.GetCurrentProcess();
            File.WriteAllText(ownerPath, string.Create(CultureInfo.InvariantCulture,
                $"pid={p.Id}\nstarted={DateTimeOffset.Now:O}\ncommand={Environment.CommandLine}\n"));
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static string DescribeOwner(string ownerPath)
    {
        try
        {
            return File.Exists(ownerPath) ? File.ReadAllText(ownerPath).ReplaceLineEndings(" | ").Trim() : "unknown";
        }
        catch (IOException) { return "unknown"; }
        catch (UnauthorizedAccessException) { return "unknown"; }
    }

    public void Dispose()
    {
        _handle.Dispose(); // DeleteOnClose removes the lock file
        try { File.Delete(_ownerPath); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}

/// <summary>Thrown when a training data directory is already owned by another live process.</summary>
public sealed class TrainingDirectoryLockedException(string directory, string owner)
    : InvalidOperationException(
        $"'{directory}' is already being written by another training run ({owner}). " +
        "Two runs sharing a data directory interleave their CSV logs and overwrite each other's checkpoints, " +
        "which silently corrupts BOTH runs' results. Stop the other process, or use a different --data directory.")
{
    /// <summary>The contested directory.</summary>
    public string Directory { get; } = directory;

    /// <summary>What the breadcrumb file said about the holder ("unknown" if it could not be read).</summary>
    public string Owner { get; } = owner;
}
