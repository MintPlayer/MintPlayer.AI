using System.Security.Cryptography;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The blank-slate guarantee: a run re-started at any time must reproduce itself, and an interrupted run must
/// continue identically to one that was never interrupted (PRD §7.1).
/// </summary>
/// <remarks>
/// Drives <c>TrainChunk</c> directly rather than through <c>CampaignRunner</c>, whose eval and checkpoint
/// cadence is wall-clock driven — going through the runner would make these assertions depend on machine speed.
/// </remarks>
public class BlockDudeDeterminismTests : IDisposable
{
    private readonly List<string> _directories = [];

    private static BlockDudeImitationOptions Options(ulong seed = 1) => new()
    {
        Seed = seed,
        BoardsPerRound = 2,
        SamplesPerBoard = 256,
        PinStage = 0,                  // keeps the run short; stage advance is covered by BlockDudeCurriculumTests
        GateEverySamples = long.MaxValue,
    };

    private FileModelStore NewStore()
    {
        string path = Path.Combine(Path.GetTempPath(), "bd-determinism-" + Guid.NewGuid().ToString("N")[..10]);
        Directory.CreateDirectory(path);
        _directories.Add(path);
        return new FileModelStore(path);
    }

    private static string Sha(FileModelStore store, string algorithmId)
    {
        using var stream = store.TryOpenRead(BlockDudeIds.Environment, algorithmId)
                           ?? throw new InvalidOperationException($"{algorithmId} was not written");
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    private static void Train(BlockDudeImitationCampaign campaign, int chunks)
    {
        for (int i = 0; i < chunks; i++) campaign.TrainChunk();
    }

    [Fact]
    public void TwoFreshRunsAtTheSameSeed_ProduceIdenticalCheckpoints()
    {
        var storeA = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(storeA);
            Train(campaign, 3);
            campaign.Checkpoint(storeA);
        }

        var storeB = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(storeB);
            Train(campaign, 3);
            campaign.Checkpoint(storeB);
        }

        var ids = BlockDudeIds.ForPhase(1);
        Assert.Equal(Sha(storeA, ids.Policy), Sha(storeB, ids.Policy));
        Assert.Equal(Sha(storeA, ids.State), Sha(storeB, ids.State));
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentCheckpoints()
    {
        // Guards against the tests passing because training silently does nothing.
        var storeA = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options(seed: 1)))
        {
            campaign.Resume(storeA);
            Train(campaign, 2);
            campaign.Checkpoint(storeA);
        }

        var storeB = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options(seed: 2)))
        {
            campaign.Resume(storeB);
            Train(campaign, 2);
            campaign.Checkpoint(storeB);
        }

        Assert.NotEqual(Sha(storeA, BlockDudeIds.ForPhase(1).Policy), Sha(storeB, BlockDudeIds.ForPhase(1).Policy));
    }

    [Fact]
    public void AnInterruptedRun_ContinuesIdenticallyToAnUninterruptedOne()
    {
        // THE test. Without the progress sidecar a resumed run restarts its counters, replays the boards it has
        // already trained on, and diverges - which is the latent bug this branch found in the other imitation
        // campaigns.
        var uninterrupted = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(uninterrupted);
            Train(campaign, 3);
            campaign.Checkpoint(uninterrupted);
        }

        var interrupted = NewStore();
        using (var first = new BlockDudeImitationCampaign(Options()))
        {
            first.Resume(interrupted);
            Train(first, 1);
            first.Checkpoint(interrupted);
        }
        using (var second = new BlockDudeImitationCampaign(Options()))
        {
            Assert.True(second.Resume(interrupted), "the second instance should have resumed a stored net");
            Train(second, 2);
            second.Checkpoint(interrupted);
        }

        var ids = BlockDudeIds.ForPhase(1);
        Assert.Equal(Sha(uninterrupted, ids.Policy), Sha(interrupted, ids.Policy));
        Assert.Equal(Sha(uninterrupted, ids.State), Sha(interrupted, ids.State));
    }

    [Fact]
    public void Fresh_DiscardsPriorProgress()
    {
        var store = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(store);
            Train(campaign, 2);
            campaign.Checkpoint(store);
        }

        // A --fresh run must not resume the stored net, and must land where a first-ever run would.
        using (var campaign = new BlockDudeImitationCampaign(Options() with { Fresh = true }))
        {
            Assert.False(campaign.Resume(store), "--fresh must not resume a stored net");
            Train(campaign, 2);
            campaign.Checkpoint(store);
        }
        string afterFresh = Sha(store, BlockDudeIds.ForPhase(1).Policy);

        var virgin = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(virgin);
            Train(campaign, 2);
            campaign.Checkpoint(virgin);
        }

        Assert.Equal(Sha(virgin, BlockDudeIds.ForPhase(1).Policy), afterFresh);
    }

    [Fact]
    public void AStateCheckpointFromADifferentRun_IsRefusedRatherThanBlended()
    {
        var store = NewStore();
        using (var campaign = new BlockDudeImitationCampaign(Options(seed: 1)))
        {
            campaign.Resume(store);
            Train(campaign, 2);
            campaign.Checkpoint(store);
        }

        // Same store, different seed: the fingerprint must not match, so the curriculum restarts rather than
        // silently continuing another run's schedule.
        using var other = new BlockDudeImitationCampaign(Options(seed: 99));
        other.Resume(store);
        var metrics = other.Evaluate().Metrics.ToDictionary(m => m.Name, m => m.Value);

        Assert.Equal(0, metrics["samples"]);
        Assert.Equal(0, metrics["stage"]);
    }

    [Fact]
    public void ProgressSurvivesARestart()
    {
        var store = NewStore();
        long samplesBefore;
        using (var campaign = new BlockDudeImitationCampaign(Options()))
        {
            campaign.Resume(store);
            Train(campaign, 2);
            campaign.Checkpoint(store);
            samplesBefore = (long)campaign.Evaluate().Metrics.First(m => m.Name == "samples").Value;
        }
        Assert.True(samplesBefore > 0);

        using var resumed = new BlockDudeImitationCampaign(Options());
        resumed.Resume(store);
        long samplesAfter = (long)resumed.Evaluate().Metrics.First(m => m.Name == "samples").Value;

        Assert.Equal(samplesBefore, samplesAfter);
    }

    public void Dispose()
    {
        foreach (string directory in _directories)
            try { Directory.Delete(directory, recursive: true); } catch (IOException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }
}
