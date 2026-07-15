using System.Security.Cryptography;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Contract test for the self-play campaign (PLAN M39.1): it plays self-play games, checkpoints the net + optimizer to
/// the expected store ids, and a fresh instance RESUMES the trained net rather than starting from random. Tiny sims /
/// game counts — this asserts the CONTRACT (play → checkpoint → resume), not strength. Marked Slow (it trains + touches
/// disk). Strength (win-rate climbing from random init) is the Lab `--game connect4` gate.
/// </summary>
public class SelfPlayCampaignTests
{
    private static SelfPlayCampaign<Connect4State> Fresh() =>
        new(new Connect4Game(), "connect4", new SelfPlayOptions
        {
            Seed = 1, LearningRate = 1e-3f, Hidden = 32, Search = new Mcts.Config(Simulations: 8),
            GamesPerChunk = 4, TempMoves = 2, EvalGames = 2, WindowCapacity = 4000, MaxPlies = 64,
        });

    [Fact]
    [Trait("Category", "Slow")]
    public void SelfPlay_Plays_Checkpoints_AndResumesTheNet()
    {
        var dir = Directory.CreateTempSubdirectory("connect4-selfplay-contract");
        try
        {
            var store = new FileModelStore(dir.FullName);

            var c1 = Fresh();
            Assert.False(c1.Resume(store));        // empty store → fresh random net
            long games1 = c1.TrainChunk();
            Assert.Equal(4, games1);               // played exactly one chunk of self-play games

            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "winRate");
            Assert.All(eval.Metrics, m => Assert.False(double.IsNaN(m.Value)));

            c1.Checkpoint(store);
            c1.Dispose();
            using (var net = store.TryOpenRead("connect4", "az")) Assert.NotNull(net);        // deployable net
            using (var adam = store.TryOpenRead("connect4", "az-adam")) Assert.NotNull(adam); // optimizer state

            // A fresh instance must load the trained net (Resume == true) and keep training without crashing.
            var c2 = Fresh();
            Assert.True(c2.Resume(store));
            long games2 = c2.TrainChunk();
            Assert.True(games2 > 0);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    /// <summary>The M39.3 robustness path: with <c>opponentRandomFrac = 1</c> every game is learner-vs-random, which
    /// exercises the separate value-credit assignment (constant learner-perspective z, not the alternating self-play
    /// z). Asserts it plays, records samples, trains, and evaluates without error.</summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void SelfPlay_against_a_random_opponent_trains_and_evaluates()
    {
        var dir = Directory.CreateTempSubdirectory("connect4-selfplay-random");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var c = new SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4", new SelfPlayOptions
            {
                Seed = 2, LearningRate = 1e-3f, Hidden = 32, Search = new Mcts.Config(Simulations: 8),
                GamesPerChunk = 6, TempMoves = 2, EvalGames = 2, WindowCapacity = 4000, MaxPlies = 64,
                TargetGames = 0, OpponentRandomFrac = 1.0,
            });

            Assert.False(c.Resume(store));
            Assert.Equal(6, c.TrainChunk());
            Assert.Contains(c.Evaluate().Metrics, m => m.Name == "winRate");
            c.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    /// <summary>The M41.2 reproducibility gate: parallel self-play generation must not change the trained result. A
    /// short Connect-4 run checkpointed with sequential generation, with parallel-at-dop-1, and with parallel-at-dop-8
    /// must produce a <b>byte-identical</b> net + optimizer checkpoint — the same guarantee the DeterministicParallel
    /// primitive is unit-tested for, verified end-to-end through the campaign (per-game RNG + ordered merge + a stable
    /// read-only net + owner-thread training). If concurrency ever leaks into the trained weights, this fails.</summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void ParallelGeneration_ProducesBitwiseIdenticalCheckpoint_AtAnyDop()
    {
        byte[] sequential = RunAndHashCheckpoint(parallel: false, maxDop: null);
        byte[] parallelDop1 = RunAndHashCheckpoint(parallel: true, maxDop: 1);
        byte[] parallelDop8 = RunAndHashCheckpoint(parallel: true, maxDop: 8);

        Assert.Equal(sequential, parallelDop1);
        Assert.Equal(sequential, parallelDop8);
    }

    // Runs a few fixed-seed chunks and returns SHA256 over the saved net + optimizer bytes.
    private static byte[] RunAndHashCheckpoint(bool parallel, int? maxDop)
    {
        var dir = Directory.CreateTempSubdirectory("connect4-selfplay-determinism");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var c = new SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4", new SelfPlayOptions
            {
                Seed = 42, LearningRate = 1e-3f, Hidden = 32, Search = new Mcts.Config(Simulations: 8),
                GamesPerChunk = 16, TempMoves = 2, EvalGames = 2, WindowCapacity = 4000, MaxPlies = 64,
                Parallel = parallel, MaxDop = maxDop,
            });
            c.Resume(store);
            for (int i = 0; i < 3; i++) c.TrainChunk();
            c.Checkpoint(store);
            c.Dispose();

            using var buffer = new MemoryStream();
            foreach (var id in new[] { "az", "az-adam" })
            {
                using var s = store.TryOpenRead("connect4", id);
                Assert.NotNull(s);
                s!.CopyTo(buffer);
            }
            return SHA256.HashData(buffer.ToArray());
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
