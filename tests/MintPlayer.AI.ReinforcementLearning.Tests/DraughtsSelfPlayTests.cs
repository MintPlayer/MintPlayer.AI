using System.Security.Cryptography;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M47.3 seam gate: the self-play contract and determinism tests instantiated with the NEW state
/// (mirroring the connect-4 <see cref="SelfPlayCampaignTests"/>) — play → checkpoint → resume on the 10×10
/// showcase variant, bitwise DOP-invariant checkpoints on the 8×8 one (determinism is variant-independent;
/// the smaller board keeps the test quick), and a <see cref="StrengthEval"/> smoke proving `--vs-minimax`
/// produces a finite number for draughts. Tiny sims/game counts — contract, not strength.
/// </summary>
public class DraughtsSelfPlayTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void SelfPlay_Plays_Checkpoints_AndResumesTheNet()
    {
        var dir = Directory.CreateTempSubdirectory("draughts-selfplay-contract");
        try
        {
            var store = new FileModelStore(dir.FullName);
            static SelfPlayCampaign<DraughtsState> Fresh() =>
                new(new DraughtsGame(DraughtsVariant.International10), "draughts", new SelfPlayOptions
                {
                    Seed = 1, LearningRate = 1e-3f, Hidden = 32, Search = new Mcts.Config(Simulations: 8),
                    GamesPerChunk = 4, TempMoves = 2, EvalGames = 2, WindowCapacity = 4000, MaxPlies = 60,
                });

            var c1 = Fresh();
            Assert.False(c1.Resume(store));
            Assert.Equal(4, c1.TrainChunk());
            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "winRate");
            Assert.All(eval.Metrics, m => Assert.False(double.IsNaN(m.Value)));
            c1.Checkpoint(store);
            c1.Dispose();
            using (var net = store.TryOpenRead("draughts", "az")) Assert.NotNull(net);
            using (var adam = store.TryOpenRead("draughts", "az-adam")) Assert.NotNull(adam);

            var c2 = Fresh();
            Assert.True(c2.Resume(store));
            Assert.True(c2.TrainChunk() > 0);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

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

    private static byte[] RunAndHashCheckpoint(bool parallel, int? maxDop)
    {
        var dir = Directory.CreateTempSubdirectory("draughts-selfplay-determinism");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var c = new SelfPlayCampaign<DraughtsState>(new DraughtsGame(DraughtsVariant.English8), "checkers8",
                new SelfPlayOptions
                {
                    Seed = 42, LearningRate = 1e-3f, Hidden = 32, Search = new Mcts.Config(Simulations: 8),
                    GamesPerChunk = 8, TempMoves = 2, EvalGames = 2, WindowCapacity = 4000, MaxPlies = 40,
                    Parallel = parallel, MaxDop = maxDop,
                });
            c.Resume(store);
            for (int i = 0; i < 2; i++) c.TrainChunk();
            c.Checkpoint(store);
            c.Dispose();

            using var buffer = new MemoryStream();
            foreach (var id in new[] { "az", "az-adam" })
            {
                using var s = store.TryOpenRead("checkers8", id);
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

    [Fact]
    public void StrengthEval_vs_minimax_produces_a_finite_result()
    {
        var game = new DraughtsGame(DraughtsVariant.English8);
        var net = new MlpNetBuilder([16]).CreateFresh(game.ObservationSize, game.PolicySize, new Xoshiro256StarStar(7));
        var r = StrengthEval.Run(game, game, net, sims: 4, depth: 1, games: 2, maxPlies: 30, openingPlies: 2, seed: 7);
        Assert.Equal(2, r.Wins + r.Draws + r.Losses);
        Assert.False(double.IsNaN(r.Score));
        Assert.False(double.IsNaN(r.AvgEndMaterial));
    }
}
