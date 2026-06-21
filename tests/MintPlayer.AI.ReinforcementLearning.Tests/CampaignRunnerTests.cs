using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// CampaignRunner loop contract (PLAN M25). Deterministic + sub-second: a fake clock that only advances when the
/// fake campaign trains a chunk, so the time-budgeted loop runs an exactly predictable number of chunks/evals —
/// no wall-clock dependency, fast bucket.
/// </summary>
public class CampaignRunnerTests
{
    /// <summary>Stub store — the fake campaign's Resume/Checkpoint don't touch it; it only needs to satisfy the type.</summary>
    private sealed class StubStore : IModelStore
    {
        public bool Exists(string environmentId, string algorithmId) => false;
        public Stream? TryOpenRead(string environmentId, string algorithmId) => null;
        public void Save(string environmentId, string algorithmId, Action<Stream> write) { }
        public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List() => [];
        public bool Delete(string environmentId, string algorithmId) => false;
    }

    private sealed class FakeCampaign(Action advanceClock, int? completeAfter = null) : ITrainingCampaign
    {
        public int ResumeCalls, TrainChunkCalls, EvaluateCalls, CheckpointCalls, DisposeCalls;

        public string Environment => "fake";
        public bool Resume(IModelStore store) { ResumeCalls++; return false; }
        public long TrainChunk() { TrainChunkCalls++; advanceClock(); return TrainChunkCalls; }
        public bool IsComplete => completeAfter is { } n && TrainChunkCalls >= n;
        public CampaignEval Evaluate() { EvaluateCalls++; return new([new CampaignMetric("n", EvaluateCalls)], $"eval {EvaluateCalls}"); }
        public void Checkpoint(IModelStore store) => CheckpointCalls++;
        public void Dispose() => DisposeCalls++;
    }

    private static (FakeCampaign Campaign, List<CampaignProgress> Events) RunWith(int? completeAfter)
    {
        var now = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var step = TimeSpan.FromMinutes(1);
        var events = new List<CampaignProgress>();
        var campaign = new FakeCampaign(() => now += step, completeAfter);
        var options = new CampaignOptions
        {
            Duration = TimeSpan.FromMinutes(10),
            EvalEvery = TimeSpan.FromMinutes(3),
            FirstEvalAfter = TimeSpan.FromMinutes(2),
            Now = () => now,
            OnEval = events.Add,
        };
        CampaignRunner.Run(campaign, new StubStore(), options);
        return (campaign, events);
    }

    [Fact]
    public void Run_TimeBudgeted_TrainsAndEvaluatesOnCadenceThenFinalizes()
    {
        var (campaign, events) = RunWith(completeAfter: null);

        // Clock advances 1 min/chunk; Duration 10 min -> 10 chunks. Evals at the 2-min baseline then every 3 min
        // (chunks 2, 5, 8) + a final eval after the loop -> 4 evals/checkpoints, each paired with a checkpoint.
        Assert.Equal(1, campaign.ResumeCalls);
        Assert.Equal(10, campaign.TrainChunkCalls);
        Assert.Equal(4, campaign.EvaluateCalls);
        Assert.Equal(4, campaign.CheckpointCalls);
        Assert.Equal(4, events.Count);
        Assert.Equal(1, campaign.DisposeCalls);

        // Only the last event is final; the cadence evals are not.
        Assert.True(events[^1].IsFinal);
        Assert.All(events[..^1], e => Assert.False(e.IsFinal));
        Assert.Equal(10, events[^1].Progress);
    }

    [Fact]
    public void Run_HardStop_StopsOnIsCompleteWithoutDoubleFinalEval()
    {
        var (campaign, events) = RunWith(completeAfter: 3);

        // IsComplete fires at chunk 3 (before the next cadence eval): one cadence eval at chunk 2, one final eval
        // at chunk 3 flagged IsFinal, and NO duplicate final after the loop.
        Assert.Equal(3, campaign.TrainChunkCalls);
        Assert.Equal(2, campaign.EvaluateCalls);
        Assert.Equal(2, campaign.CheckpointCalls);
        Assert.Equal(2, events.Count);
        Assert.True(events[^1].IsFinal);
        Assert.Equal(1, campaign.DisposeCalls);
    }

    [Fact]
    public void Run_EvalOnly_EvaluatesOnceAndTrainsNothing()
    {
        var now = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var events = new List<CampaignProgress>();
        var campaign = new FakeCampaign(() => now += TimeSpan.FromMinutes(1));
        CampaignRunner.Run(campaign, new StubStore(), new CampaignOptions { EvalOnly = true, Now = () => now, OnEval = events.Add });

        Assert.Equal(1, campaign.ResumeCalls);
        Assert.Equal(0, campaign.TrainChunkCalls);
        Assert.Equal(0, campaign.CheckpointCalls);
        Assert.Equal(1, campaign.EvaluateCalls);
        Assert.Single(events);
        Assert.True(events[0].IsFinal);
        Assert.Equal(1, campaign.DisposeCalls);
    }
}
