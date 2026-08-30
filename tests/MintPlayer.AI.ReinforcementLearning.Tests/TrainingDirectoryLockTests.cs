using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Guards the defect that prompted this type: two Lab processes were pointed at the same data directory,
/// both appended to its CSV logs and both wrote its checkpoints, and the blended numbers were believed
/// before the interleaving was noticed. Nothing failed loudly at the time — that is what makes it worth a test.
/// </summary>
public class TrainingDirectoryLockTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mp-ai-lock-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); }
        catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void SecondAcquire_OnTheSameDirectory_Throws()
    {
        using var first = TrainingDirectoryLock.Acquire(_dir);

        var ex = Assert.Throws<TrainingDirectoryLockedException>(() => TrainingDirectoryLock.Acquire(_dir));

        // The message has to be actionable: it names the directory and says what the damage would be.
        Assert.Contains(Path.GetFullPath(_dir), ex.Directory, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("interleave", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Releasing_AllowsTheNextRunToTakeIt()
    {
        using (TrainingDirectoryLock.Acquire(_dir)) { }

        // A run that ended cleanly must not leave the directory unusable.
        using var second = TrainingDirectoryLock.Acquire(_dir);
        Assert.Equal(Path.GetFullPath(_dir), second.Directory);
    }

    [Fact]
    public void DifferentDirectories_DoNotContend()
    {
        var other = _dir + "-other";
        try
        {
            using var a = TrainingDirectoryLock.Acquire(_dir);
            using var b = TrainingDirectoryLock.Acquire(other);
            Assert.NotEqual(a.Directory, b.Directory);
        }
        finally
        {
            try { if (Directory.Exists(other)) Directory.Delete(other, recursive: true); } catch (IOException) { }
        }
    }

    [Fact]
    public void Acquire_CreatesTheDirectoryIfItIsMissing()
    {
        var fresh = Path.Combine(_dir, "nested", "run");
        Assert.False(Directory.Exists(fresh));

        using var l = TrainingDirectoryLock.Acquire(fresh);
        Assert.True(Directory.Exists(fresh));
    }

    /// <summary>
    /// The lock must be an OS handle, not a PID/marker file: a killed run leaves no stale lock behind. This is
    /// the property a hand-rolled marker file would get wrong, and a killed training run is routine here (a
    /// running Lab.exe locks the build outputs, so it gets stopped to run the tests).
    /// </summary>
    [Fact]
    public void AbandonedHandle_DoesNotStrandTheDirectory()
    {
        // Simulate a process death: drop the handle without Dispose, then force finalization + collection so
        // the OS handle is released the way process exit would release it.
        AcquireAndAbandon(_dir);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        using var next = TrainingDirectoryLock.Acquire(_dir);
        Assert.Equal(Path.GetFullPath(_dir), next.Directory);
    }

    private static void AcquireAndAbandon(string dir)
    {
        _ = TrainingDirectoryLock.Acquire(dir); // deliberately not disposed
    }
}
