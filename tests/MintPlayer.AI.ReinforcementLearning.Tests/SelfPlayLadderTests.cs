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
