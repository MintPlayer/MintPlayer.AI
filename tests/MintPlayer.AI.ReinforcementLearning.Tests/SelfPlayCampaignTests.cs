extern alias Lab; // SelfPlayCampaign is internal to the Lab exe (InternalsVisibleTo), aliased like the other campaigns

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
    private static Lab::SelfPlayCampaign<Connect4State> Fresh() =>
        new(new Connect4Game(), "connect4", seed: 1, learningRate: 1e-3f, hidden: 32,
            selfPlayCfg: new Mcts.Config(Simulations: 8),
            gamesPerChunk: 4, tempMoves: 2, evalGames: 2, windowCapacity: 4000, maxPlies: 64);

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
            var c = new Lab::SelfPlayCampaign<Connect4State>(new Connect4Game(), "connect4", seed: 2,
                learningRate: 1e-3f, hidden: 32, selfPlayCfg: new Mcts.Config(Simulations: 8),
                gamesPerChunk: 6, tempMoves: 2, evalGames: 2, windowCapacity: 4000, maxPlies: 64,
                targetGames: 0, opponentRandomFrac: 1.0);

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
}
