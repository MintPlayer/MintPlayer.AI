using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.4 — the self-play progress sidecar, driven through the cheapest real campaign in the repo.
/// <para>
/// <c>az-progress</c> exists because self-play used to persist its net and Adam state but none of its
/// <i>progress</i>: the game counter and the arena RNG were re-derived from the seed on every resume,
/// so a restarted run replayed the same games from the beginning. The existing lifecycle test never
/// asserts that the counter survives a restart, so the round-trip it was written for was untested.
/// </para>
/// <para>
/// Config is deliberately minimal: an 8-wide net, one MCTS simulation, one game per chunk. Note
/// <c>BatchSize = 16</c> — <c>TrainChunk</c> skips training entirely while the window holds fewer
/// samples than the batch, so at the shipped default of 128 this would generate games, train nothing,
/// and still look like it worked.
/// </para>
/// </summary>
public class SelfPlayProgressTests
{
    private static SelfPlayOptions Tiny(int gamesPerChunk = 2) => new()
    {
        Seed = 11,
        LearningRate = 1e-3f,
        Hidden = 8,
        Search = new Mcts.Config(Simulations: 1),
        GamesPerChunk = gamesPerChunk,
        TempMoves = 0,
        EvalGames = 1,
        WindowCapacity = 512,
        MaxPlies = 42,
        BatchSize = 16,
    };

    private static SelfPlayCampaign<Connect4State> Fresh(SelfPlayOptions? options = null)
        => new(new Connect4Game(), "connect4", options ?? Tiny());

    [Fact]
    public void A_chunk_writes_the_net_the_optimizer_and_the_progress_sidecar()
    {
        var dir = Directory.CreateTempSubdirectory("m64-sp-progress");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = Fresh();

            Assert.False(c.Resume(store));
            c.TrainChunk();
            c.Checkpoint(store);

            Assert.True(store.Exists("connect4", "az"));
            Assert.True(store.Exists("connect4", "az-adam"));
            Assert.True(store.Exists("connect4", "az-progress"));   // the one the old test never checked
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_game_counter_survives_a_restart_instead_of_replaying_from_zero()
    {
        // The whole reason az-progress exists. Without it a restarted run re-plays the same generated
        // games, and -- worse -- resets the growth stage target, because that keys off the counter.
        var dir = Directory.CreateTempSubdirectory("m64-sp-counter");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = Fresh())
            {
                first.Resume(store);
                Assert.Equal(2, first.TrainChunk());     // == GamesPerChunk
                first.Checkpoint(store);
            }

            using var second = Fresh();
            Assert.True(second.Resume(store));

            // Continues the count rather than starting over: 4, not 2.
            Assert.Equal(4, second.TrainChunk());
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_resumed_run_reports_its_metrics_without_NaN()
    {
        // policyLoss is the cheapest proof that a batch actually ran: with BatchSize above the window
        // size it would be NaN or absent, and the test above would be asserting nothing about training.
        var dir = Directory.CreateTempSubdirectory("m64-sp-metrics");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = Fresh(Tiny(gamesPerChunk: 4));

            c.Resume(store);
            c.TrainChunk();
            var metrics = c.Evaluate().Metrics;

            Assert.Contains(metrics, m => m.Name == "winRate");
            Assert.Contains(metrics, m => m.Name == "policyLoss");
            Assert.All(metrics, m => Assert.False(double.IsNaN(m.Value), $"metric '{m.Name}' is NaN"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_store_without_a_progress_sidecar_still_resumes_the_net()
    {
        // The sidecar is additive and optional by design: a store written before it existed must load
        // its net and simply start the counters at zero, not refuse to resume.
        var dir = Directory.CreateTempSubdirectory("m64-sp-nosidecar");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = Fresh())
            {
                first.Resume(store);
                first.TrainChunk();
                first.Checkpoint(store);
            }
            store.Delete("connect4", "az-progress");

            using var second = Fresh();

            Assert.True(second.Resume(store));          // the net still loads
            Assert.Equal(2, second.TrainChunk());       // counters restart, which is the documented fallback
        }
        finally { dir.Delete(recursive: true); }
    }
}
