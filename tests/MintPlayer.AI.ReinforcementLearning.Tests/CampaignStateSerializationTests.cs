using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.2 — the campaign sidecar checkpoints.
/// <para>
/// Both of these are hand-rolled <b>positional</b> binary IO: fields are written and read in
/// declaration order with no names and no framing. A field added to <c>Save</c> but not to
/// <c>TryLoad</c> (or added to one in a different position) does not fail — it makes every later
/// field read as garbage. That is invisible until a long run resumes with a corrupted curriculum
/// cursor or counters, which is the expensive way to find out.
/// </para>
/// <para>
/// <c>BlockDudeTrainingState</c> is 196 lines and had <b>zero</b> coverage before this file; its
/// contents were only ever exercised through a multi-minute training run.
/// </para>
/// </summary>
public class CampaignStateSerializationTests
{
    private sealed class MemoryModelStore : IModelStore
    {
        private readonly Dictionary<(string, string), byte[]> _blobs = [];

        public bool Exists(string environmentId, string algorithmId) => _blobs.ContainsKey((environmentId, algorithmId));
        public Stream? TryOpenRead(string environmentId, string algorithmId)
            => _blobs.TryGetValue((environmentId, algorithmId), out var b) ? new MemoryStream(b, writable: false) : null;
        public void Save(string environmentId, string algorithmId, Action<Stream> write)
        {
            using var ms = new MemoryStream();
            write(ms);
            _blobs[(environmentId, algorithmId)] = ms.ToArray();
        }
        public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List() => [.. _blobs.Keys];
        public bool Delete(string environmentId, string algorithmId) => _blobs.Remove((environmentId, algorithmId));

        /// <summary>Replaces a blob with arbitrary bytes, to exercise the damaged-sidecar paths.</summary>
        public void Corrupt(string environmentId, string algorithmId, byte[] bytes)
            => _blobs[(environmentId, algorithmId)] = bytes;
    }

    // ── CampaignProgressState ─────────────────────────────────────────────────────────────────

    private static Xoshiro256StarStar Advanced(ulong seed, int draws)
    {
        var rng = new Xoshiro256StarStar(seed);
        for (int i = 0; i < draws; i++) rng.NextUInt64();
        return rng;
    }

    [Fact]
    public void Progress_round_trips_counters_and_the_metric()
    {
        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress",
            samples: 12_345, units: 67, lastMetric: 0.75, new Xoshiro256StarStar(1));

        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1);

        Assert.NotNull(loaded);
        Assert.Equal(12_345, loaded!.Samples);
        Assert.Equal(67, loaded.Units);
        Assert.Equal(0.75, loaded.LastMetric);
        Assert.Single(loaded.Rngs);
    }

    [Fact]
    public void A_restored_rng_continues_the_saved_stream_rather_than_restarting_it()
    {
        // This is the whole point of persisting RNG state: a resumed run must not replay the draws
        // the previous session already made. Advance a stream, save it, then check the restored copy
        // produces what the live one would have produced NEXT.
        var live = Advanced(seed: 99, draws: 50);
        var expectedNext = new ulong[3];
        {
            var (s0, s1, s2, s3) = live.GetState();
            var peek = new Xoshiro256StarStar(0);
            peek.SetState(s0, s1, s2, s3);
            for (int i = 0; i < expectedNext.Length; i++) expectedNext[i] = peek.NextUInt64();
        }

        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress", 0, 0, 0, live);
        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1);

        var resumed = new Xoshiro256StarStar(0);
        CampaignProgressState.RestoreInto(loaded!.Rngs[0], resumed);

        for (int i = 0; i < expectedNext.Length; i++) Assert.Equal(expectedNext[i], resumed.NextUInt64());
    }

    [Fact]
    public void Progress_round_trips_several_rngs_in_declaration_order()
    {
        // The streams are positional. If they came back in a different order, each campaign RNG would
        // silently adopt another's state.
        var a = Advanced(1, 5);
        var b = Advanced(2, 9);
        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress", 0, 0, 0, a, b);

        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 2);

        Assert.Equal(2, loaded!.Rngs.Length);
        Assert.Equal(a.GetState(), loaded.Rngs[0].GetState());
        Assert.Equal(b.GetState(), loaded.Rngs[1].GetState());
    }

    [Fact]
    public void A_missing_progress_sidecar_reads_as_start_from_zero()
    {
        var loaded = CampaignProgressState.TryLoad(new MemoryModelStore(), "env", "progress", "test-progress", rngCount: 1);

        // Null, not an exception: the file is additive and optional, so a store that predates it must
        // keep working exactly as before.
        Assert.Null(loaded);
    }

    [Fact]
    public void An_rng_count_mismatch_is_refused_and_logged()
    {
        // A campaign that gains or loses an RNG stream must not read the old file positionally and
        // hand each stream its neighbour's state.
        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress", 10, 1, 0.5, new Xoshiro256StarStar(1));

        var messages = new List<string>();
        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 2, messages.Add);

        Assert.Null(loaded);
        Assert.Contains(messages, m => m.Contains("RNG streams"));
    }

    [Fact]
    public void A_damaged_progress_sidecar_degrades_to_a_fresh_start()
    {
        // Progress is an optimisation, never correctness: a corrupt sidecar costs a replay of
        // already-seen data and must not fail a nine-hour run.
        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress", 10, 1, 0.5, new Xoshiro256StarStar(1));
        store.Corrupt("env", "progress", [1, 2, 3, 4, 5]);

        var messages = new List<string>();
        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1, messages.Add);

        Assert.Null(loaded);
        Assert.NotEmpty(messages);
    }

    [Fact]
    public void A_truncated_progress_sidecar_degrades_to_a_fresh_start()
    {
        var store = new MemoryModelStore();
        CampaignProgressState.Save(store, "env", "progress", "test-progress", 10, 1, 0.5, new Xoshiro256StarStar(1));
        using var full = store.TryOpenRead("env", "progress")!;
        var bytes = new byte[full.Length];
        full.ReadExactly(bytes);
        store.Corrupt("env", "progress", bytes[..(bytes.Length / 2)]);   // header survives, body does not

        var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1, _ => { });

        Assert.Null(loaded);
    }

    // ── BlockDudeTrainingState (196 lines, previously zero coverage) ──────────────────────────

    private static BlockDudeTrainingState PopulatedState(ulong fingerprint) => new()
    {
        ObservationSize = 64,
        ActionCount = 5,
        RunFingerprint = fingerprint,
        Stage = 3,
        StageSamples = 111,
        TotalSamples = 2222,
        SamplesAtLastGate = 1500,
        GateRates = [0.1, 0.25, -1, 0.9],
        BestStage = 2,
        BestGate = 0.83,
        BestSamples = 1900,
        BoardAttempts = 500,
        AcceptedBoards = 420,
        TruncatedBoards = 60,
        RejectedBoards = 20,
        GrowthRung = 2,
        PlateauBest = 0.77,
        PlateauEvals = 4,
        SamplesAtLastGrowth = 1234,
        ShuffleRng = Advanced(7, 11),
        GrowRng = Advanced(8, 13),
    };

    [Fact]
    public void BlockDude_state_round_trips_every_field()
    {
        // Deliberately asserts EVERY field rather than a sample. The format is positional, so a field
        // added to Save but not TryLoad shifts everything after it — and only a complete comparison
        // notices which one moved.
        ulong fp = BlockDudeTrainingState.Fingerprint(seed: 5, observationSize: 64, actionCount: 5, learningRate: 1e-3f, phase: 1);
        var original = PopulatedState(fp);

        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", original.Save);
        var loaded = BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, 64, 5);

        Assert.NotNull(loaded);
        Assert.Equal(original.ObservationSize, loaded!.ObservationSize);
        Assert.Equal(original.ActionCount, loaded.ActionCount);
        Assert.Equal(original.RunFingerprint, loaded.RunFingerprint);
        Assert.Equal(original.Stage, loaded.Stage);
        Assert.Equal(original.StageSamples, loaded.StageSamples);
        Assert.Equal(original.TotalSamples, loaded.TotalSamples);
        Assert.Equal(original.SamplesAtLastGate, loaded.SamplesAtLastGate);
        Assert.Equal(original.GateRates, loaded.GateRates);
        Assert.Equal(original.BestStage, loaded.BestStage);
        Assert.Equal(original.BestGate, loaded.BestGate);
        Assert.Equal(original.BestSamples, loaded.BestSamples);
        Assert.Equal(original.BoardAttempts, loaded.BoardAttempts);
        Assert.Equal(original.AcceptedBoards, loaded.AcceptedBoards);
        Assert.Equal(original.TruncatedBoards, loaded.TruncatedBoards);
        Assert.Equal(original.RejectedBoards, loaded.RejectedBoards);
        Assert.Equal(original.GrowthRung, loaded.GrowthRung);
        Assert.Equal(original.PlateauBest, loaded.PlateauBest);
        Assert.Equal(original.PlateauEvals, loaded.PlateauEvals);
        Assert.Equal(original.SamplesAtLastGrowth, loaded.SamplesAtLastGrowth);
        Assert.Equal(original.ShuffleRng.GetState(), loaded.ShuffleRng.GetState());
        Assert.Equal(original.GrowRng.GetState(), loaded.GrowRng.GetState());
    }

    [Fact]
    public void BlockDude_state_preserves_the_best_cursor_which_decides_what_ships()
    {
        // (BestStage, BestGate, BestSamples) is the cursor the campaign compares against to decide
        // whether to overwrite the shippable net. Losing it on resume means the next mediocre eval
        // looks like a new best.
        ulong fp = BlockDudeTrainingState.Fingerprint(1, 64, 5, 1e-3f, 1);
        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", PopulatedState(fp).Save);

        var loaded = BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, 64, 5);

        Assert.Equal(2, loaded!.BestStage);
        Assert.Equal(0.83, loaded.BestGate);
        Assert.Equal(1900, loaded.BestSamples);
    }

    [Fact]
    public void A_state_from_a_different_run_is_refused()
    {
        // The fingerprint exists so two different runs cannot blend into one checkpoint.
        ulong saved = BlockDudeTrainingState.Fingerprint(seed: 1, observationSize: 64, actionCount: 5, learningRate: 1e-3f, phase: 1);
        ulong other = BlockDudeTrainingState.Fingerprint(seed: 2, observationSize: 64, actionCount: 5, learningRate: 1e-3f, phase: 1);
        Assert.NotEqual(saved, other);

        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", PopulatedState(saved).Save);

        var messages = new List<string>();
        Assert.Null(BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", other, 64, 5, messages.Add));
        Assert.NotEmpty(messages);
    }

    [Fact]
    public void A_state_from_a_different_observation_encoding_is_refused()
    {
        // The M61 hazard: resuming into a changed observation encoding trains a net whose inputs no
        // longer mean what its weights assume.
        ulong fp = BlockDudeTrainingState.Fingerprint(1, 64, 5, 1e-3f, 1);
        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", PopulatedState(fp).Save);

        Assert.Null(BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, observationSize: 128, actionCount: 5, _ => { }));
        Assert.Null(BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, observationSize: 64, actionCount: 9, _ => { }));
    }

    [Theory]
    [InlineData(1UL, 64, 5, 1e-3f, 1)]
    public void The_fingerprint_is_stable_for_identical_inputs_and_moves_for_each_one(
        ulong seed, int obs, int actions, float lr, int phase)
    {
        ulong baseline = BlockDudeTrainingState.Fingerprint(seed, obs, actions, lr, phase);

        Assert.Equal(baseline, BlockDudeTrainingState.Fingerprint(seed, obs, actions, lr, phase));
        Assert.NotEqual(baseline, BlockDudeTrainingState.Fingerprint(seed + 1, obs, actions, lr, phase));
        Assert.NotEqual(baseline, BlockDudeTrainingState.Fingerprint(seed, obs + 1, actions, lr, phase));
        Assert.NotEqual(baseline, BlockDudeTrainingState.Fingerprint(seed, obs, actions + 1, lr, phase));
        Assert.NotEqual(baseline, BlockDudeTrainingState.Fingerprint(seed, obs, actions, lr * 2, phase));
        Assert.NotEqual(baseline, BlockDudeTrainingState.Fingerprint(seed, obs, actions, lr, phase + 1));
    }

    [Fact]
    public void A_damaged_blockdude_sidecar_degrades_to_a_fresh_start()
    {
        ulong fp = BlockDudeTrainingState.Fingerprint(1, 64, 5, 1e-3f, 1);
        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", PopulatedState(fp).Save);
        store.Corrupt("blockdude", "policy-state", [9, 9, 9]);

        Assert.Null(BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, 64, 5, _ => { }));
    }

    [Fact]
    public void An_empty_gate_rate_table_round_trips_as_empty()
    {
        // Length-prefixed array: a zero-length table is the state a run is in before its first gate,
        // and it must not read back as garbage or throw.
        ulong fp = BlockDudeTrainingState.Fingerprint(1, 64, 5, 1e-3f, 1);
        var state = PopulatedState(fp);
        state.GateRates = [];

        var store = new MemoryModelStore();
        store.Save("blockdude", "policy-state", state.Save);
        var loaded = BlockDudeTrainingState.TryLoad(store, "blockdude", "policy-state", fp, 64, 5);

        Assert.NotNull(loaded);
        Assert.Empty(loaded!.GateRates);
    }
}
