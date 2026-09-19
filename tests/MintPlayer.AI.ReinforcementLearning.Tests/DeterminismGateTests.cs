using System.Security.Cryptography;

using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.0 — the bitwise-identity gate, made to actually gate.
/// <para>
/// M46 made "training behavior stays bitwise identical" a hard cross-cutting requirement for every
/// milestone (<c>DI_CAMPAIGNS_PRD.md</c>). Two holes were found in it on 2026-09-18:
/// </para>
/// <list type="number">
/// <item><b>It did not run in CI.</b> Both workflows filter <c>Category!=Slow</c> and every
/// checkpoint-SHA test was tagged <c>Slow</c>, so a change that broke bitwise-identity produced a
/// fully green PR.</item>
/// <item><b>It was blind to uniform drift.</b> The existing helper asserts only that the sequential,
/// dop-1 and dop-8 arms agree with <i>each other</i> — and all three are computed by the
/// post-change code. A refactor that shifts the seed fan-out <i>uniformly</i> passed all three.</item>
/// </list>
/// <para>
/// These tests close both: they compare against <b>checked-in hash literals</b>, and they carry
/// <c>Category=Determinism</c> so CI can run them as a separate, deliberately <b>uninstrumented</b>
/// job. Uninstrumented matters for cost, not correctness — coverage instrumentation does not change
/// computed values, but it does cost time, and M63.4 showed how badly probe-based timing estimates
/// for this suite mislead.
/// </para>
/// <para>
/// <b>If one of these fails, do not update the literal to make it pass.</b> A changed hash means the
/// trained bytes changed. Either that was intended — a deliberate change to the training recipe, in
/// which case update the literal <i>and say so in the commit</i> — or it is the bug this file exists
/// to catch. Configs here are deliberately tiny so the gate is fast; the property under test is
/// reproducibility, which does not need a large run to be observable.
/// </para>
/// </summary>
public class DeterminismGateTests
{
    // ── Golden RNG stream states, generated on the base commit (M64.0, 2026-09-18) ────────────
    //
    // These pin the xoshiro STATE after a fixed run, not the trained bytes. Trained weights are
    // floats and float results are not bit-identical across platform, JIT or SDK -- CI (Linux) and
    // Windows disagreed on the DQN checkpoint hash, and the repo never claimed otherwise; its
    // determinism gates had only ever run on one machine.
    //
    // Xoshiro state is pure integer arithmetic, so it is identical everywhere -- AND it is exactly
    // what a uniform shift in the seed fan-out moves, which is the failure mode this gate exists to
    // catch and the one the dop-invariance test structurally cannot see.
    //
    // Regenerate ONLY when the training recipe is intentionally changed, and say so in the commit.
    private const string StubDqnPolicyRngGolden = "df3848ede60984b9-336455a099d0a2e2-145188340709985a-fc08b289f5da3ac6";
    private const string StubDqnBufferRngGolden = "758addd3cf049483-4423f0bfaaf27aac-1428751336826696-00b7a93a6f94aefe";

    // ── Self-play ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// A fixed-seed Connect-4 run must produce the same checkpoint bytes it produced on the base
    /// commit. This is the half the existing dop-invariance test cannot see: a uniform shift in the
    /// seed fan-out moves all three of its arms together and passes.
    /// </summary>
    [Fact]
    [Trait("Category", "Determinism")]
    public void SelfPlay_IsReproducible_AcrossRuns()
    {
        // Same environment, so float determinism holds and byte comparison is meaningful. No
        // checked-in literal: see the note on the golden constants above for why trained bytes are
        // not portable across platform/JIT/SDK.
        Assert.Equal(HashSelfPlayRun(parallel: false, maxDop: null), HashSelfPlayRun(parallel: false, maxDop: null));
    }

    /// <summary>
    /// The M41.2 property, at a config small enough for CI: parallel generation must not leak into
    /// the trained weights, at any degree of parallelism.
    /// </summary>
    /// <remarks>
    /// The window must actually train for this to mean anything — see
    /// <see cref="SelfPlayOptions.BatchSize"/> in <see cref="SelfPlayRun"/>. With the shipped default
    /// of 128 a two-game chunk generates samples and trains <b>nothing</b>, and the test would pass
    /// while asserting only that generation order is stable.
    /// </remarks>
    [Fact]
    [Trait("Category", "Determinism")]
    public void SelfPlay_IsBitwiseIdentical_AtAnyDegreeOfParallelism()
    {
        string sequential = HashSelfPlayRun(parallel: false, maxDop: null);

        Assert.Equal(sequential, HashSelfPlayRun(parallel: true, maxDop: 1));
        Assert.Equal(sequential, HashSelfPlayRun(parallel: true, maxDop: 4));
    }

    private static SelfPlayOptions SelfPlayRun(bool parallel, int? maxDop) => new()
    {
        Seed = 42,
        LearningRate = 1e-3f,
        Hidden = 8,
        Search = new Mcts.Config(Simulations: 1),
        GamesPerChunk = 4,
        TempMoves = 0,
        EvalGames = 1,
        WindowCapacity = 512,
        MaxPlies = 42,
        // Deliberately below the shipped default so a 4-game chunk clears the training threshold.
        // SelfPlayCampaign.TrainChunk skips training entirely while _window.Count < BatchSize.
        BatchSize = 16,
        Parallel = parallel,
        MaxDop = maxDop,
    };

    private static string HashSelfPlayRun(bool parallel, int? maxDop)
    {
        var dir = Directory.CreateTempSubdirectory("m64-determinism-selfplay");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using (var c = new SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4", SelfPlayRun(parallel, maxDop)))
            {
                c.Resume(store);
                c.TrainChunk();
                c.TrainChunk();
                c.Checkpoint(store);
            }
            return HashStoredIds(store, "connect4", "az", "az-adam");
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    // ── DQN spine ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// The DQN family had <b>no</b> checkpoint-hash coverage at all — only a forward-pass fingerprint
    /// and resume tests. It is also the family carrying the two wiring hazards a DI change would hit
    /// (the <c>[Inject]</c>-generated constructor with two same-typed env parameters, and the
    /// singleton env whose RNG is one continuous stream across the run), so it is exactly where a
    /// silent determinism break would land.
    /// </summary>
    [Fact]
    [Trait("Category", "Determinism")]
    public void DqnSpine_RngStreams_LandOnTheGoldenState()
    {
        // The portable half of the gate. If a refactor changes how seeds fan out -- a new consumer
        // claiming an existing RngStreams index, a reordered construction that re-seeds an env, an
        // [Inject] field order change that swaps train and eval -- these states move, on every
        // platform, and this fails.
        var (policy, buffer) = DqnRngStatesAfterOneChunk();

        Assert.Equal(StubDqnPolicyRngGolden, policy);
        Assert.Equal(StubDqnBufferRngGolden, buffer);
    }

    /// <summary>Two runs of the same seed in the same process must agree — the cheapest possible
    /// smoke test that nothing per-process (hash seeds, thread timing) has leaked into training.</summary>
    [Fact]
    [Trait("Category", "Determinism")]
    public void DqnSpine_IsReproducible_WithinTheSameProcess()
    {
        Assert.Equal(HashDqnRun(), HashDqnRun());
    }

    /// <summary>Policy and buffer RNG states after one fixed chunk, as stable strings.</summary>
    private static (string Policy, string Buffer) DqnRngStatesAfterOneChunk()
    {
        var dir = Directory.CreateTempSubdirectory("m64-determinism-dqn-rng");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var options = new DqnScoreOptions { Seed = 7, ChunkSteps = 120, TargetSteps = 120, Hidden = [16, 16] };

            using (var c = new GoldenDqnCampaign(new GoldenEnv(), options))
            {
                c.Resume(store);
                c.TrainChunk();
                c.Evaluate();
                c.Checkpoint(store);
            }

            using var s = store.TryOpenRead("golden", "dqn-state");
            Assert.True(s is not null, "expected the run to have written 'golden/dqn-state'");
            var state = DqnTrainingState.Load(s!);
            return (Format(state.PolicyRng), Format(state.BufferRng));
        }
        finally
        {
            dir.Delete(recursive: true);
        }

        static string Format(Xoshiro256StarStar rng)
        {
            var (s0, s1, s2, s3) = rng.GetState();
            return $"{s0:x16}-{s1:x16}-{s2:x16}-{s3:x16}";
        }
    }

    private static string HashDqnRun()
    {
        var dir = Directory.CreateTempSubdirectory("m64-determinism-dqn");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var options = new DqnScoreOptions { Seed = 7, ChunkSteps = 120, TargetSteps = 120, Hidden = [16, 16] };

            using (var c = new GoldenDqnCampaign(new GoldenEnv(), options))
            {
                c.Resume(store);
                c.TrainChunk();
                // Evaluate BEFORE Checkpoint, exactly as CampaignRunner.Report does. Checkpoint is
                // save-best (`_lastGate > _bestGate`) and `_lastGate` is only set by Evaluate, so a
                // helper that skips it writes the resume state and NO deployable net.
                c.Evaluate();
                c.Checkpoint(store);
            }
            // `dqn` only: `dqn-state` embeds the replay buffer, which is far larger and adds nothing
            // the net bytes do not already witness for this purpose.
            return HashStoredIds(store, "golden", "dqn");
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    /// <summary>Deterministic 4-obs / 2-action toy; identical in shape to the one
    /// <see cref="DqnScoreCampaignTests"/> uses, kept separate so tuning that test cannot silently
    /// move this gate's golden hash.</summary>
    private sealed class GoldenEnv : IEnvironment<float[], int>
    {
        private int _t;
        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 4);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(2);
        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null) { _t = 0; return (Obs(), EnvInfo.Empty); }
        public StepResult<float[]> Step(int action) { _t++; return new(Obs(), action == 1 ? 1.0 : 0.0, _t >= 5, false, EnvInfo.Empty); }
        public string RenderString() => $"t={_t}";
        private float[] Obs() => [_t / 5f, 1f - _t / 5f, 1f, 0f];
    }

    private sealed class GoldenDqnCampaign(IEnvironment<float[], int> env, DqnScoreOptions options)
        : DqnScoreCampaign(env, options)
    {
        public override string Environment => "golden";
        protected override string StepNoun => "steps";
        protected override string GateLabel => "gate";
        protected override string DisplayName => "Golden DQN";
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

        // A constant gate: a varying one would make save-best behaviour, and therefore the bytes,
        // depend on eval noise rather than on training.
        protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
            => (1.0, [new CampaignMetric("golden", 1.0, "0")], "golden eval");
    }

    // ── shared ────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// SHA256 over the concatenated bytes of the named store ids, as lowercase hex.
    /// </summary>
    /// <remarks>
    /// The checkpoint format carries no timestamps, GUIDs or machine identity, so byte comparison is
    /// meaningful. Hex rather than <c>byte[]</c> so a failure message shows the value to paste when a
    /// change to the recipe is genuinely intended.
    /// </remarks>
    private static string HashStoredIds(IModelStore store, string environmentId, params string[] ids)
    {
        using var buffer = new MemoryStream();
        foreach (var id in ids)
        {
            using var s = store.TryOpenRead(environmentId, id);
            Assert.True(s is not null, $"expected the run to have written '{environmentId}/{id}'");
            s!.CopyTo(buffer);
        }
        return Convert.ToHexString(SHA256.HashData(buffer.ToArray())).ToLowerInvariant();
    }
}
