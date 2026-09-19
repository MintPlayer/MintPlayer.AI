using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M46.4 seam tests: ladder promotion runs entirely against an in-memory <see cref="ILadderStore"/> — no
/// disk. Covers the promote → manifest → resume round-trip that previously needed real files in the web models
/// dir (raw <c>File.*</c>/<c>Directory.*</c> inside the campaign): the first checkpoint promotes the Level-1
/// baseline, a resumed campaign adopts the stored champion and continues the ladder instead of restarting.
/// <para>
/// M69/B1 adds the <b>arena</b> half, which until now only the <c>Slow</c> test above reached. It needs no
/// training at all: <c>MaybePromoteDifficulty</c> runs on EVERY <c>Checkpoint</c>, and its champion-present
/// branch needs only a SECOND checkpoint on the same instance — so <c>Resume → Checkpoint → Checkpoint</c>
/// drives the whole net-vs-net arena (<c>ArenaVsNet</c> → <c>PlayArenaGame</c> → <c>ModelMoveWith</c>) in
/// milliseconds. The silent failure guarded is the ladder's central promise: <b>Level K+1 provably beats
/// Level K</b>. If the arena quietly stops playing — a gate that short-circuits, a champion snapshot that is
/// really the live net — every tier still gets written and the manifest still looks right, while the ladder
/// the browser offers as "increasingly strong" has become an arbitrary sequence of checkpoints. Hence the
/// counting game: these tests assert the arena ACTUALLY played moves, not merely that a tier appeared.
/// </para>
/// </summary>
public class SelfPlayLadderTests
{
    /// <summary>Pure in-memory <see cref="ILadderStore"/> (tiers as byte[], manifest as string).</summary>
    private sealed class MemoryLadderStore : ILadderStore
    {
        private readonly Dictionary<(string Env, int Tier), byte[]> _tiers = [];
        public string? Manifest { get; private set; }

        public string SaveTier(string environmentId, int tier, Action<Stream> write)
        {
            using var ms = new MemoryStream();
            write(ms);
            _tiers[(environmentId, tier)] = ms.ToArray();
            return $"{environmentId}.az.d{tier}.ckpt";
        }

        public Stream? TryOpenTier(string environmentId, int tier)
            => _tiers.TryGetValue((environmentId, tier), out var bytes) ? new MemoryStream(bytes) : null;

        public int HighestTier(string environmentId)
        {
            int highest = 0;
            foreach (var (env, tier) in _tiers.Keys)
                if (env == environmentId && tier > highest) highest = tier;
            return highest;
        }

        public void WriteManifest(string environmentId, string json) => Manifest = json;
        public string? TryReadManifest(string environmentId) => Manifest;
    }

    /// <summary>In-memory <see cref="IModelStore"/> so the whole test touches no disk.</summary>
    private sealed class MemoryModelStore : IModelStore
    {
        private readonly Dictionary<(string, string), byte[]> _blobs = [];
        public bool Exists(string environmentId, string algorithmId) => _blobs.ContainsKey((environmentId, algorithmId));
        public Stream? TryOpenRead(string environmentId, string algorithmId)
            => _blobs.TryGetValue((environmentId, algorithmId), out var b) ? new MemoryStream(b) : null;
        public void Save(string environmentId, string algorithmId, Action<Stream> write)
        {
            using var ms = new MemoryStream();
            write(ms);
            _blobs[(environmentId, algorithmId)] = ms.ToArray();
        }
        public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List() => [.. _blobs.Keys];
        public bool Delete(string environmentId, string algorithmId) => _blobs.Remove((environmentId, algorithmId));
    }

    /// <summary>
    /// A pass-through <see cref="Connect4Game"/> that counts how many moves have been applied. Used to prove the
    /// arena actually played: a checkpoint that promotes the BASELINE applies no moves at all, while one that
    /// goes through the champion arena applies many — so the delta across a single <c>Checkpoint</c> call
    /// separates "the arena ran" from "a tier was written".
    /// </summary>
    private sealed class CountingConnect4 : IZeroSumGame<Connect4State>
    {
        private readonly Connect4Game _inner = new();
        private int _moves;

        /// <summary>Moves applied so far (search included — the arena is the only caller here that applies any).</summary>
        public int Moves => Volatile.Read(ref _moves);

        public int PolicySize => _inner.PolicySize;
        public int ObservationSize => _inner.ObservationSize;
        public Connect4State Root(ulong? seed = null) => _inner.Root(seed);
        public IReadOnlyList<int> LegalMoves(Connect4State state) => _inner.LegalMoves(state);
        public GameResult Result(Connect4State state) => _inner.Result(state);
        public void WriteObservation(Connect4State state, Span<float> destination) => _inner.WriteObservation(state, destination);

        public Connect4State Apply(Connect4State state, int move)
        {
            Interlocked.Increment(ref _moves);
            return _inner.Apply(state, move);
        }
    }

    private static SelfPlayCampaign<Connect4State> Campaign(MemoryLadderStore ladderStore, double arenaMargin,
        IZeroSumGame<Connect4State>? game = null) =>
        new(game ?? new Connect4Game(), "connect4", new SelfPlayOptions
        {
            Seed = 1, LearningRate = 1e-3f, Hidden = 16, Search = new Mcts.Config(Simulations: 4),
            GamesPerChunk = 2, TempMoves = 2, EvalGames = 2, WindowCapacity = 1000, MaxPlies = 32,
            // Promotion thresholds: material/winRate gates out of reach; head-to-head decides via arenaMargin.
            Ladder = new LadderOptions(Dir: "unused-in-memory", PromoteMaterial: 999, PromoteMargin: 999,
                ArenaMargin: arenaMargin, ArenaGames: 2, Sims: 8, OpeningPlies: 0),
        }, ladderStore: ladderStore);

    [Fact]
    public void Checkpointing_before_the_first_evaluation_writes_a_manifest_rather_than_throwing()
    {
        // The baseline tier is promoted UNCONDITIONALLY on the first checkpoint, so a run that
        // checkpoints before it has ever evaluated carries `_lastWinRate` = NaN into the manifest.
        // NaN is not valid JSON and the default serializer throws on it, which lost the whole
        // checkpoint — the net, the optimizer and the progress sidecar, not merely the manifest.
        //
        // Every other ladder test calls Evaluate() before Checkpoint(), which is precisely why none
        // of them caught it. No TrainChunk here either: the crash needs no training, and not
        // training is what keeps this test out of the Slow bucket.
        var ladder = new MemoryLadderStore();
        var models = new MemoryModelStore();

        using var c = Campaign(ladder, arenaMargin: 999);
        Assert.False(c.Resume(models));

        c.Checkpoint(models);

        Assert.NotNull(ladder.Manifest);
        using var doc = JsonDocument.Parse(ladder.Manifest!);
        var tier = Assert.Single(doc.RootElement.EnumerateArray().ToArray());
        // `null`, not 0. Zero is a measured result — a net that genuinely never beat random — and
        // writing it here would recreate the very conflation the v2 sidecar format removed.
        Assert.Equal(JsonValueKind.Null, tier.GetProperty("winRateVsRandom").ValueKind);
    }

    [Fact]
    public void A_null_win_rate_in_a_manifest_reads_back_as_unmeasured()
    {
        // The other half of the round-trip: the reader must not call GetDouble() on a null, and a
        // resumed unmeasured tier must stay unmeasured rather than becoming a measured 0%.
        var ladder = new MemoryLadderStore();
        var models = new MemoryModelStore();

        using (var first = new SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4",
            new SelfPlayOptions
            {
                Seed = 1, LearningRate = 1e-3f, Hidden = 16, Search = new Mcts.Config(Simulations: 4),
                GamesPerChunk = 2, TempMoves = 2, EvalGames = 2, WindowCapacity = 1000, MaxPlies = 32,
                Ladder = new LadderOptions(Dir: "unused-in-memory", PromoteMaterial: 999, PromoteMargin: 999,
                    ArenaMargin: 999, ArenaGames: 2, Sims: 8, OpeningPlies: 0),
            }, ladderStore: ladder))
        {
            first.Resume(models);
            first.Checkpoint(models);
        }

        // Re-reading the manifest it just wrote must not throw.
        using var second = Campaign(ladder, arenaMargin: 999);
        Assert.True(second.Resume(models));
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Ladder_PromotesBaseline_WritesManifest_AndResumesInMemory()
    {
        var ladder = new MemoryLadderStore();
        var models = new MemoryModelStore();

        // Fresh run: the FIRST checkpoint always promotes the Level-1 baseline (no champion yet).
        using (var c1 = Campaign(ladder, arenaMargin: 999))
        {
            Assert.False(c1.Resume(models));
            c1.TrainChunk();
            c1.Evaluate();
            c1.Checkpoint(models);
        }
        Assert.Equal(1, ladder.HighestTier("connect4"));
        Assert.NotNull(ladder.Manifest);
        using (var doc = JsonDocument.Parse(ladder.Manifest!))
        {
            var tier = Assert.Single(doc.RootElement.EnumerateArray().ToArray());
            Assert.Equal("Level 1", tier.GetProperty("label").GetString());
            Assert.Equal("/models/connect4.az.d1.ckpt", tier.GetProperty("ckpt").GetString());
        }

        // Resume: the campaign adopts the stored Level-1 champion (instead of restarting the ladder) and — with
        // the head-to-head gate at 0, always satisfied — promotes Level 2 on its next checkpoint. That only
        // happens when the resume actually loaded the champion, so it proves the round-trip.
        using (var c2 = Campaign(ladder, arenaMargin: 0))
        {
            Assert.True(c2.Resume(models));
            c2.TrainChunk();
            c2.Evaluate();
            c2.Checkpoint(models);
        }
        Assert.Equal(2, ladder.HighestTier("connect4"));
        using (var doc = JsonDocument.Parse(ladder.Manifest!))
            Assert.Equal(2, doc.RootElement.GetArrayLength());
    }

    // ── M69/B1: the arena, without training ──────────────────────────────────────────────────────────────────

    [Fact]
    public void The_second_checkpoint_plays_the_champion_arena_and_promotes_Level_2()
    {
        // Resume → Checkpoint → Checkpoint on ONE instance. No TrainChunk, no Evaluate: the first checkpoint
        // promotes the unconditional baseline, which installs a champion, and the second therefore takes the
        // champion-present branch — the full net-vs-net arena. ArenaMargin 0 makes the head-to-head gate always
        // satisfied, so the promotion decision is the arena's alone (PromoteMaterial/PromoteMargin are out of
        // reach, and Connect-4 has no material notion at all).
        var ladder = new MemoryLadderStore();
        var models = new MemoryModelStore();
        var game = new CountingConnect4();

        using var c = Campaign(ladder, arenaMargin: 0, game);
        Assert.False(c.Resume(models));

        c.Checkpoint(models);
        // The baseline branch returns before ArenaVsNet, so nothing has been played yet. Asserting this is what
        // makes the delta below mean "the arena ran" rather than "something, somewhere, moved a piece".
        Assert.Equal(0, game.Moves);
        Assert.Equal(1, ladder.HighestTier("connect4"));

        c.Checkpoint(models);

        Assert.True(game.Moves > 0, "the second checkpoint must actually play the champion arena");
        Assert.Equal(2, ladder.HighestTier("connect4"));

        // The manifest carries BOTH tiers, in ladder order, pointing at DISTINCT checkpoint files. A ladder whose
        // tiers share a ckpt name is the failure the browser cannot see: two "levels" serving the same weights.
        using var doc = JsonDocument.Parse(ladder.Manifest!);
        var tiers = doc.RootElement.EnumerateArray().ToArray();
        Assert.Equal(2, tiers.Length);
        Assert.Equal("Level 1", tiers[0].GetProperty("label").GetString());
        Assert.Equal("Level 2", tiers[1].GetProperty("label").GetString());
        Assert.Equal("/models/connect4.az.d1.ckpt", tiers[0].GetProperty("ckpt").GetString());
        Assert.Equal("/models/connect4.az.d2.ckpt", tiers[1].GetProperty("ckpt").GetString());
        Assert.NotEqual(tiers[0].GetProperty("ckpt").GetString(), tiers[1].GetProperty("ckpt").GetString());

        // Both tier blobs really exist in the store, not just their names in the manifest.
        using (var d1 = ladder.TryOpenTier("connect4", 1)) Assert.NotNull(d1);
        using (var d2 = ladder.TryOpenTier("connect4", 2)) Assert.NotNull(d2);
    }

    [Fact]
    public void An_arena_the_challenger_cannot_win_leaves_the_ladder_where_it_was()
    {
        // The other half of the gate, and the one that keeps the ladder ORDERED: the arena runs on every
        // checkpoint, but an unreachable margin must NOT promote. If this ever passed, every checkpoint would
        // append a tier and "Level 5" would mean nothing more than "the fifth time we saved".
        var ladder = new MemoryLadderStore();
        var models = new MemoryModelStore();
        var game = new CountingConnect4();

        using var c = Campaign(ladder, arenaMargin: 999, game);
        c.Resume(models);

        c.Checkpoint(models);            // baseline — unconditional
        c.Checkpoint(models);            // arena runs, gate refuses

        Assert.True(game.Moves > 0, "the arena must still be PLAYED — it is the gate's evidence, not its result");
        Assert.Equal(1, ladder.HighestTier("connect4"));
        using var doc = JsonDocument.Parse(ladder.Manifest!);
        Assert.Single(doc.RootElement.EnumerateArray().ToArray());
    }
}
