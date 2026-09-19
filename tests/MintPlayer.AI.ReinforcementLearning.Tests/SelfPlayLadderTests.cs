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

    private static SelfPlayCampaign<Connect4State> Campaign(MemoryLadderStore ladderStore, double arenaMargin) =>
        new(new Connect4Game(), "connect4", new SelfPlayOptions
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
}
