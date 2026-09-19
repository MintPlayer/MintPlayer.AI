using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69/B5 — the <c>SelfPlayCampaign.TrainChunk</c> branches that only a non-default CONFIG reaches, driven on
/// the same tiny Connect-4 setup <see cref="SelfPlayProgressTests"/> uses (8-wide net, one simulation, a
/// two-game chunk), so each of these costs a fraction of a second.
/// <para>
/// Every one of these knobs is a production lever that is switched ON for real runs and OFF in every existing
/// test, which is exactly how they break silently:
/// </para>
/// <list type="bullet">
///   <item><c>LeafBatch &gt; 1</c> routes search through <c>Mcts.SearchBatched</c> + <c>EvaluateBatch</c> — a
///   SECOND, independent implementation of "observe, forward, masked-softmax, tanh". It is the only way the GPU
///   path keeps a device busy, and a mis-strided split there yields a net that trains on another position's
///   priors: plausible numbers, garbage learning.</item>
///   <item><c>TempMoves &gt; 0</c> turns on temperature sampling for the opening plies — the campaign's ONLY
///   source of opening variety. If its cumulative-probability walk falls through, the numerical fallback must
///   still return the most-visited legal move rather than action 0, which on Connect-4 would silently make every
///   game open in the same column.</item>
///   <item><c>WindowCapacity</c> smaller than a chunk's output exercises the rolling-window EVICTION in
///   <c>AddSample</c>. An evicted window must still be trainable; the failure mode is a window that trains on
///   nothing at all while the run reports healthy chunk counts.</item>
///   <item><c>MaxPlies</c> below a game's natural length hits the capped-adjudication path, whose whole purpose
///   is that a capped game is NOT recorded as a genuine draw.</item>
/// </list>
/// <para>
/// The shared assertion is <c>policyLoss</c>: the loss window is NaN until a real batch runs, so "not NaN" is
/// the cheapest honest proof that the branch under test produced usable training data instead of merely not
/// throwing. Batch sizes here are deliberately below the window size for that reason (the M64.7 lesson).
/// </para>
/// </summary>
public class SelfPlayChunkVariantTests
{
    private static SelfPlayOptions Tiny() => new()
    {
        Seed = 11,
        LearningRate = 1e-3f,
        Hidden = 8,
        Search = new Mcts.Config(Simulations: 1),
        GamesPerChunk = 2,
        TempMoves = 0,
        EvalGames = 1,
        WindowCapacity = 512,
        MaxPlies = 42,
        BatchSize = 4,
    };

    /// <summary>Runs one real chunk under <paramref name="options"/> and returns the evaluated metrics.</summary>
    private static IReadOnlyList<CampaignMetric> ChunkAndEvaluate(SelfPlayOptions options)
    {
        var dir = Directory.CreateTempSubdirectory("m69-sp-variant");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4", options);

            Assert.False(c.Resume(store));
            Assert.Equal(options.GamesPerChunk, c.TrainChunk());
            return c.Evaluate().Metrics;
        }
        finally { dir.Delete(recursive: true); }
    }

    private static void AssertTrained(IReadOnlyList<CampaignMetric> metrics)
    {
        Assert.Contains(metrics, m => m.Name == "policyLoss");
        Assert.Contains(metrics, m => m.Name == "samples");
        var policyLoss = metrics.Single(m => m.Name == "policyLoss");
        Assert.False(double.IsNaN(policyLoss.Value), "policyLoss is NaN — no batch ran, so the branch under test trained nothing");
        Assert.All(metrics, m => Assert.False(double.IsNaN(m.Value), $"metric '{m.Name}' is NaN"));
        Assert.True(metrics.Single(m => m.Name == "samples").Value > 0,
            "a chunk that trained must have advanced the sample counter");
    }

    [Fact]
    public void A_chunk_with_batched_leaf_inference_trains_like_the_sequential_one()
    {
        // LeafBatch = 4 → Mcts.SearchBatched + EvaluateBatch (the batched observe/forward/split path) instead of
        // the one-leaf-at-a-time Evaluate. Same games, same window, same loss shape — a different implementation.
        AssertTrained(ChunkAndEvaluate(Tiny() with { LeafBatch = 4 }));
    }

    [Fact]
    public void A_chunk_that_samples_its_opening_moves_by_temperature_still_trains()
    {
        // TempMoves = 4: the first four plies of every game are drawn from the visit-count distribution rather
        // than argmaxed, including the numerical fallback when the cumulative walk does not terminate.
        AssertTrained(ChunkAndEvaluate(Tiny() with { TempMoves = 4 }));
    }

    [Fact]
    public void A_replay_window_smaller_than_one_chunk_evicts_and_stays_trainable()
    {
        // Two Connect-4 games produce far more than 8 samples, so AddSample must drop the oldest on every add
        // past the cap. Capacity 8 with batch 4 leaves two full batches — if eviction corrupted the window,
        // there would be no non-NaN loss to report.
        AssertTrained(ChunkAndEvaluate(Tiny() with { WindowCapacity = 8, BatchSize = 4 }));
    }

    [Fact]
    public void A_ply_capped_game_is_adjudicated_rather_than_abandoned()
    {
        // MaxPlies = 4 cuts every game long before Connect-4 can end, so the terminal result is Ongoing and the
        // capped-adjudication arm decides z. (Connect-4 exposes no material, so the honest verdict there is a
        // true 0 — the point is that the arm RUNS and yields finite targets.) Shorter games also make this the
        // cheapest test in the file, not the dearest.
        AssertTrained(ChunkAndEvaluate(Tiny() with { MaxPlies = 4, BatchSize = 4 }));
    }
}
