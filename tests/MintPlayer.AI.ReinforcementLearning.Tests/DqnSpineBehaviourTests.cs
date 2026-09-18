using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.3 — the rules the DQN spine enforces that nothing tested: save-best, warm start, and the
/// absolute step cap. All four game DQN campaigns inherit these, so a regression here is a
/// regression in Snake, FruitCake, CrazyFruits and Tetris at once.
/// </summary>
public class DqnSpineBehaviourTests
{
    /// <summary>Deterministic 4-obs / 2-action toy; 5-step episodes.</summary>
    private sealed class ToyEnv : IEnvironment<float[], int>
    {
        private int _t;
        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 4);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(2);
        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null) { _t = 0; return (Obs(), EnvInfo.Empty); }
        public StepResult<float[]> Step(int action) { _t++; return new(Obs(), action == 1 ? 1.0 : 0.0, _t >= 5, false, EnvInfo.Empty); }
        public string RenderString() => $"t={_t}";
        private float[] Obs() => [_t / 5f, 1f - _t / 5f, 1f, 0f];
    }

    /// <summary>A campaign whose eval score is scripted, so save-best can be driven deliberately.</summary>
    private sealed class ScriptedCampaign(IEnvironment<float[], int> env, DqnScoreOptions options, Queue<double> gates)
        : DqnScoreCampaign(env, options)
    {
        public double LastServedGate { get; private set; }

        public override string Environment => "toy";
        protected override string StepNoun => "steps";
        protected override string GateLabel => "gate";
        protected override string DisplayName => "Toy DQN";
        protected override int ObservationSize => 4;
        protected override IReadOnlyList<string>? InputLabels => null;
        protected override IReadOnlyList<string>? OutputLabels => null;

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
        {
            // Resume() also calls EvaluateNet to seed the baseline, so an empty queue must keep serving
            // the last value rather than throwing.
            if (gates.Count > 0) LastServedGate = gates.Dequeue();
            return (LastServedGate, [new CampaignMetric("toy", LastServedGate, "0.00")], $"toy {LastServedGate:F2}");
        }
    }

    private static DqnScoreOptions Options(int chunk = 60, int target = 0)
        => new() { Seed = 3, ChunkSteps = chunk, TargetSteps = target, Hidden = [16, 16] };

    private static byte[] Read(IModelStore store, string id)
    {
        using var s = store.TryOpenRead("toy", id);
        Assert.True(s is not null, $"expected 'toy/{id}' to exist");
        using var ms = new MemoryStream();
        s!.CopyTo(ms);
        return ms.ToArray();
    }

    // ── save-best: the rule that stops a noisy eval shipping a worse net ──────────────────────

    [Fact]
    public void A_worse_eval_does_not_overwrite_the_deployable_net()
    {
        // DQN eval is noisy by design, so the deployable checkpoint is save-best: only an eval that
        // BEATS the best seen may overwrite it. Nothing tested this, and the failure mode is silent --
        // a bad draw ships and the web app serves it.
        var dir = Directory.CreateTempSubdirectory("m64-savebest");
        try
        {
            var store = new FileModelStore(dir.FullName);
            // No baseline entry: on a FRESH store Resume does not call EvaluateNet (it only seeds the
            // baseline when a net already exists), so the first value here is consumed by the first
            // Evaluate. Getting this wrong shifts every scripted gate by one.
            var gates = new Queue<double>([5.0, 3.0]);        // a good eval, then a worse one
            using var c = new ScriptedCampaign(new ToyEnv(), Options(), gates);

            c.Resume(store);
            c.TrainChunk();
            c.Evaluate();                 // gate 5.0 -> beats the 0.0 baseline, saves
            c.Checkpoint(store);
            byte[] afterGood = Read(store, "dqn");

            c.TrainChunk();
            c.Evaluate();                 // gate 3.0 -> worse, must NOT overwrite
            c.Checkpoint(store);
            byte[] afterWorse = Read(store, "dqn");

            Assert.Equal(afterGood, afterWorse);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_better_eval_does_overwrite_the_deployable_net()
    {
        // The other half: save-best must not be so conservative that nothing ever ships.
        var dir = Directory.CreateTempSubdirectory("m64-savebest-better");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var gates = new Queue<double>([1.0, 9.0]);
            using var c = new ScriptedCampaign(new ToyEnv(), Options(), gates);

            c.Resume(store);
            c.TrainChunk();
            c.Evaluate();
            c.Checkpoint(store);
            byte[] afterFirst = Read(store, "dqn");

            c.TrainChunk();
            c.Evaluate();                 // 9.0 -> new best
            c.Checkpoint(store);

            Assert.NotEqual(afterFirst, Read(store, "dqn"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_resume_state_advances_even_when_the_deployable_net_is_held_back()
    {
        // The two checkpoints have deliberately different rules: `dqn-state` ALWAYS tracks the latest
        // net so a continuation picks up where it left off, while `dqn` is save-best. Conflating them
        // would either lose progress on every non-improving chunk or ship a worse net.
        var dir = Directory.CreateTempSubdirectory("m64-state-always");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new ScriptedCampaign(new ToyEnv(), Options(), new Queue<double>([5.0, 1.0]));

            c.Resume(store);
            c.TrainChunk();
            c.Evaluate();                 // 5.0 -> ships
            c.Checkpoint(store);
            byte[] netAfterFirst = Read(store, "dqn");
            byte[] stateAfterFirst = Read(store, "dqn-state");

            c.TrainChunk();
            c.Evaluate();                 // 1.0 -> worse, held back
            c.Checkpoint(store);

            Assert.Equal(netAfterFirst, Read(store, "dqn"));            // deployable net untouched
            Assert.NotEqual(stateAfterFirst, Read(store, "dqn-state")); // progress still recorded
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the absolute step cap ─────────────────────────────────────────────────────────────────

    [Fact]
    public void The_last_chunk_stops_at_the_cap_rather_than_overrunning_it()
    {
        // MaxSteps is absolute, so a chunk that would cross TargetSteps must be truncated. Overrunning
        // means a run configured for N steps quietly trains past N.
        var dir = Directory.CreateTempSubdirectory("m64-cap");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new ScriptedCampaign(new ToyEnv(), Options(chunk: 100, target: 150), new Queue<double>([0.0]));

            c.Resume(store);

            Assert.Equal(100, c.TrainChunk());
            Assert.False(c.IsComplete);
            Assert.Equal(150, c.TrainChunk());     // 150, not 200
            Assert.True(c.IsComplete);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_zero_target_means_no_cap()
    {
        // TargetSteps = 0 is "time-bounded only" -- the runner's clock stops it, not a step count.
        var dir = Directory.CreateTempSubdirectory("m64-nocap");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new ScriptedCampaign(new ToyEnv(), Options(chunk: 60, target: 0), new Queue<double>([0.0]));

            c.Resume(store);
            c.TrainChunk();

            Assert.False(c.IsComplete);
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── warm start ────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_store_with_only_a_deployable_net_warm_starts_instead_of_training_from_random()
    {
        // The shipped checkpoint without its resume state is the normal case for a fresh machine
        // pulling models from git. Discarding those weights and starting from random would waste the
        // whole previous run while still looking like a successful resume.
        var dir = Directory.CreateTempSubdirectory("m64-warm");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new ScriptedCampaign(new ToyEnv(), Options(), new Queue<double>([0.0, 5.0])))
            {
                first.Resume(store);
                first.TrainChunk();
                first.Evaluate();
                first.Checkpoint(store);
            }
            Assert.True(store.Exists("toy", "dqn"));
            store.Delete("toy", "dqn-state");              // keep only the deployable net

            using var second = new ScriptedCampaign(new ToyEnv(), Options(), new Queue<double>([0.0]));

            Assert.True(second.Resume(store));             // warm start counts as resumed
            Assert.Equal(60, second.TrainChunk());         // and training restarts its step count
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void An_empty_store_is_not_a_resume()
    {
        var dir = Directory.CreateTempSubdirectory("m64-fresh");
        try
        {
            using var c = new ScriptedCampaign(new ToyEnv(), Options(), new Queue<double>([0.0]));

            Assert.False(c.Resume(new FileModelStore(dir.FullName)));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the metric row every training CSV column is built from ────────────────────────────────

    [Fact]
    public void The_metric_row_leads_with_the_step_count_and_ends_with_loss()
    {
        // Column ORDER is the CSV contract: a reorder silently shifts every historical log's meaning.
        var dir = Directory.CreateTempSubdirectory("m64-metrics");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new ScriptedCampaign(new ToyEnv(), Options(), new Queue<double>([0.0, 2.0]));
            c.Resume(store);
            c.TrainChunk();

            var names = c.Evaluate().Metrics.Select(m => m.Name).ToList();

            Assert.Equal("steps", names[0]);
            Assert.Equal("loss", names[^1]);
            Assert.Contains("toy", names);       // the campaign's own metric sits between them
        }
        finally { dir.Delete(recursive: true); }
    }
}
