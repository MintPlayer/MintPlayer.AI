using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M46.2 seam test: the <see cref="DqnScoreCampaign"/> spine is env-agnostic. Because the training env is a
/// constructor dependency (not `new`ed inside a subclass), the whole resume → train → checkpoint → resume contract
/// runs against a pure in-memory stub <see cref="IEnvironment{TObs,TAct}"/> — no game, no GPU, milliseconds.
/// The per-game campaigns' behavior on their REAL envs stays covered by <see cref="CampaignContractTests"/>.
/// </summary>
public class DqnScoreCampaignTests
{
    /// <summary>Deterministic 4-obs / 2-action toy: reward 1 for action 1, episode ends after 5 steps.</summary>
    private sealed class StubEnv : IEnvironment<float[], int>
    {
        private int _t;
        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 4);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(2);
        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null) { _t = 0; return (Obs(), EnvInfo.Empty); }
        public StepResult<float[]> Step(int action) { _t++; return new(Obs(), action == 1 ? 1.0 : 0.0, _t >= 5, false, EnvInfo.Empty); }
        public string RenderString() => $"t={_t}";
        private float[] Obs() => [_t / 5f, 1f - _t / 5f, 1f, 0f];
    }

    private sealed class StubDqnCampaign(IEnvironment<float[], int> env, DqnScoreOptions options)
        : DqnScoreCampaign(env, options)
    {
        public int EvaluateNetCalls;

        public override string Environment => "stub";
        protected override string StepNoun => "steps";
        protected override string GateLabel => "gate";
        protected override string DisplayName => "Stub DQN";
        protected override int ObservationSize => 4;
        protected override IReadOnlyList<string>? InputLabels => null;
        protected override IReadOnlyList<string>? OutputLabels => null;

        // Dueling: the spine's Checkpoint saves the deployable net as a DuelingQNet.
        protected override DqnOptions BaseOptions => new()
        {
            Dueling = true,
            Hidden = Options.Hidden,
            Gamma = Options.Gamma,
            LearningRate = Options.LearningRate,
            BufferCapacity = 1_000,
            BatchSize = 16,
            WarmupSteps = 32,
            TargetSyncEvery = 64,
            EvalEpisodes = 1,
        };

        protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
            => (++EvaluateNetCalls, [new CampaignMetric("stub", EvaluateNetCalls, "0")], $"stub eval {EvaluateNetCalls}");
    }

    [Fact]
    public void Spine_TrainsCheckpointsAndResumes_AgainstAStubEnv()
    {
        var dir = Directory.CreateTempSubdirectory("stub-dqn-campaign");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var options = new DqnScoreOptions { Seed = 1, ChunkSteps = 100, TargetSteps = 200, Hidden = [16, 16] };

            var c1 = new StubDqnCampaign(new StubEnv(), options);
            Assert.False(c1.Resume(store));                    // fresh
            Assert.Equal(100, c1.TrainChunk());                // one chunk against the stub
            Assert.False(c1.IsComplete);

            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "stub");
            c1.Checkpoint(store);
            c1.Dispose();
            using (var net = store.TryOpenRead("stub", "dqn")) Assert.NotNull(net);
            using (var state = store.TryOpenRead("stub", "dqn-state")) Assert.NotNull(state);

            var c2 = new StubDqnCampaign(new StubEnv(), options);
            Assert.True(c2.Resume(store));                     // resumes the stub line
            Assert.Equal(200, c2.TrainChunk());                // continues to the cap, not from zero
            Assert.True(c2.IsComplete);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
