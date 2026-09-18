using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.5 — the shared Campaigns helpers. Small, pure, and used by several campaigns each, so a
/// regression in one of these is a regression in every game that depends on it.
/// </summary>
public class CampaignHelperTests
{
    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "m64-helpers-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    // ── SupervisedTraining.TrainWindow — the metric means every imitation CSV column is built from ──

    [Fact]
    public void TrainWindow_means_the_samples_it_was_given()
    {
        var w = new TrainWindow();
        w.Add(1.0, 10.0, 0.5);
        w.Add(3.0, 20.0, 1.0);

        var (ce, huber, acc) = w.MeanAndReset();

        Assert.Equal(2.0, ce, 12);
        Assert.Equal(15.0, huber, 12);
        Assert.Equal(0.75, acc, 12);
    }

    [Fact]
    public void TrainWindow_resets_after_being_read()
    {
        // The window is per-chunk: not resetting would blend the previous chunk into the next one's
        // reported loss, which reads as a suspiciously smooth training curve.
        var w = new TrainWindow();
        w.Add(5, 5, 1);
        w.MeanAndReset();

        var (ce, huber, acc) = w.MeanAndReset();

        Assert.Equal(0, ce);
        Assert.Equal(0, huber);
        Assert.Equal(0, acc);
    }

    [Fact]
    public void An_empty_TrainWindow_reports_zeros_not_NaN()
    {
        // A chunk that trained nothing must not put NaN into a CSV column -- every downstream reader
        // of those logs would have to special-case it.
        var (ce, huber, acc) = new TrainWindow().MeanAndReset();

        Assert.Equal(0, ce);
        Assert.Equal(0, huber);
        Assert.Equal(0, acc);
    }

    // ── AdamState — the optimizer moments across a resume ──

    [Fact]
    public void AdamState_returns_a_fresh_optimizer_when_none_is_stored()
    {
        var store = new InMemoryStore();
        var net = new Mlp([2, 4, 2], new Xoshiro256StarStar(1));

        var adam = AdamState.LoadOrInit(store, "env", "adam", net.Parameters(), 1e-3f, _ => { });

        Assert.NotNull(adam);
    }

    [Fact]
    public void AdamState_round_trips_through_the_store()
    {
        var store = new InMemoryStore();
        var net = new Mlp([2, 4, 2], new Xoshiro256StarStar(1));
        var saved = new Adam(net.Parameters(), 1e-3f);
        AdamState.Save(store, "env", "adam", saved);

        var loaded = AdamState.LoadOrInit(store, "env", "adam", net.Parameters(), 1e-3f, _ => { });

        Assert.NotNull(loaded);
        Assert.True(store.Exists("env", "adam"));
    }

    // ── PolicyValueNetBuilders — the kind tag that decides which loader a checkpoint reaches ──

    [Fact]
    public void The_two_builders_declare_distinct_checkpoint_kinds()
    {
        // The tag is written into the checkpoint header, so a rename silently orphans every shipped
        // net and a collision would let a conv checkpoint load as an MLP.
        var mlp = new MlpNetBuilder([16, 16]);
        var conv = new ConvNetBuilder(planes: 5, boardH: 8, boardW: 8, filters: 8, blocks: 1);

        Assert.Equal("selfplay-pv", mlp.CheckpointKind);
        Assert.Equal("selfplay-pv-conv", conv.CheckpointKind);
        Assert.NotEqual(mlp.CheckpointKind, conv.CheckpointKind);
    }

    // ── FileLadderStore — the tier filenames the web app serves by URL ──

    [Fact]
    public void A_saved_tier_is_named_for_its_environment_and_index()
    {
        // The web app fetches these by path, so the naming is a published contract rather than an
        // internal detail.
        string dir = NewTempDir();
        try
        {
            var store = new FileLadderStore(dir);

            string name = store.SaveTier("connect4", 3, s => s.WriteByte(7));

            Assert.Equal("connect4.az.d3.ckpt", name);
            Assert.True(File.Exists(Path.Combine(dir, name)));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_saved_tier_reads_back()
    {
        string dir = NewTempDir();
        try
        {
            var store = new FileLadderStore(dir);
            store.SaveTier("connect4", 1, s => s.Write([1, 2, 3]));

            using var read = store.TryOpenTier("connect4", 1);

            Assert.NotNull(read);
            var buf = new byte[3];
            read!.ReadExactly(buf);
            Assert.Equal<byte[]>([1, 2, 3], buf);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Saving_a_tier_leaves_no_temp_file_behind()
    {
        // SaveTier writes to `.tmp` then moves. A leftover would eventually be served by the web app.
        string dir = NewTempDir();
        try
        {
            var store = new FileLadderStore(dir);
            store.SaveTier("connect4", 1, s => s.WriteByte(1));
            store.SaveTier("connect4", 1, s => s.WriteByte(2));   // overwrite path

            Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void HighestTier_reads_the_largest_index_back_out_of_the_filenames()
    {
        string dir = NewTempDir();
        try
        {
            var store = new FileLadderStore(dir);
            store.SaveTier("connect4", 1, s => s.WriteByte(1));
            store.SaveTier("connect4", 7, s => s.WriteByte(1));
            store.SaveTier("connect4", 3, s => s.WriteByte(1));
            store.SaveTier("othergame", 9, s => s.WriteByte(1));   // must not be counted

            Assert.Equal(7, store.HighestTier("connect4"));
            Assert.Equal(9, store.HighestTier("othergame"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void HighestTier_is_zero_for_an_empty_or_missing_directory()
    {
        string dir = NewTempDir();
        try
        {
            Assert.Equal(0, new FileLadderStore(dir).HighestTier("connect4"));
            Assert.Equal(0, new FileLadderStore(Path.Combine(dir, "does-not-exist")).HighestTier("connect4"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void The_manifest_round_trips_and_is_absent_until_written()
    {
        string dir = NewTempDir();
        try
        {
            var store = new FileLadderStore(dir);
            Assert.Null(store.TryReadManifest("connect4"));

            store.WriteManifest("connect4", """{"tiers":[1,2]}""");

            Assert.Equal("""{"tiers":[1,2]}""", store.TryReadManifest("connect4"));
        }
        finally { Directory.Delete(dir, true); }
    }

    private sealed class InMemoryStore : IModelStore
    {
        private readonly Dictionary<(string, string), byte[]> _blobs = [];
        public bool Exists(string e, string a) => _blobs.ContainsKey((e, a));
        public Stream? TryOpenRead(string e, string a) => _blobs.TryGetValue((e, a), out var b) ? new MemoryStream(b, false) : null;
        public void Save(string e, string a, Action<Stream> write)
        {
            using var ms = new MemoryStream();
            write(ms);
            _blobs[(e, a)] = ms.ToArray();
        }
        public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List() => [.. _blobs.Keys];
        public bool Delete(string e, string a) => _blobs.Remove((e, a));
    }
}
