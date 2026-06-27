using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class Board2048Tests
{
    private static byte[] Row(params byte[] first4)
    {
        var board = new byte[16];
        first4.CopyTo(board, 0);
        return board;
    }

    [Fact]
    public void Slide_MergesEqualPairs_OncePerMove()
    {
        // [2,2,2,2] → [4,4,_,_]: two separate pair-merges, never a chain into 8.
        var board = Row(1, 1, 1, 1);
        Assert.True(Board2048.ApplyMove(board, Board2048.ActionLeft, out int exp, out int val));
        Assert.Equal(new byte[] { 2, 2, 0, 0 }, board[..4]);
        Assert.Equal(4, exp);  // log2(4) + log2(4)
        Assert.Equal(8, val);  // 4 + 4
    }

    [Fact]
    public void Slide_MergedTileDoesNotRemerge()
    {
        // [2,2,4,_] → [4,4,_,_], NOT [8,...]: the freshly merged 4 must not merge again.
        var board = Row(1, 1, 2, 0);
        Board2048.ApplyMove(board, Board2048.ActionLeft, out _, out int val);
        Assert.Equal(new byte[] { 2, 2, 0, 0 }, board[..4]);
        Assert.Equal(4, val);
    }

    [Fact]
    public void Slide_TripleMergesLeadingPair()
    {
        // [2,2,2,_] → [4,2,_,_]: the pair nearest the slide direction merges.
        var board = Row(1, 1, 1, 0);
        Board2048.ApplyMove(board, Board2048.ActionLeft, out _, out _);
        Assert.Equal(new byte[] { 2, 1, 0, 0 }, board[..4]);
    }

    [Fact]
    public void Slide_TwoDistinctPairsBothMerge()
    {
        // [2,2,4,4] → [4,8,_,_], reward 4+8=12.
        var board = Row(1, 1, 2, 2);
        Board2048.ApplyMove(board, Board2048.ActionLeft, out int exp, out int val);
        Assert.Equal(new byte[] { 2, 3, 0, 0 }, board[..4]);
        Assert.Equal(5, exp);
        Assert.Equal(12, val);
    }

    [Fact]
    public void Slide_GapsCompress()
    {
        var board = Row(1, 0, 0, 1);
        Board2048.ApplyMove(board, Board2048.ActionLeft, out _, out int val);
        Assert.Equal(new byte[] { 2, 0, 0, 0 }, board[..4]);
        Assert.Equal(4, val);
    }

    [Fact]
    public void MoveThatChangesNothing_IsIllegal()
    {
        var board = Row(1, 0, 0, 0); // already fully left, nothing merges
        Assert.False(Board2048.ApplyMove(board, Board2048.ActionLeft, out _, out _));
        Assert.Equal(1, board[0]); // untouched
    }

    [Fact]
    public void Directions_MoveTowardTheRightEdge()
    {
        var board = Row(1, 1, 0, 0);
        Board2048.ApplyMove(board, Board2048.ActionRight, out _, out _);
        Assert.Equal(new byte[] { 0, 0, 0, 2 }, board[..4]);
    }

    [Fact]
    public void Directions_UpAndDownWorkOnColumns()
    {
        var board = new byte[16];
        board[0] = 1; board[8] = 1; // column 0, rows 0 and 2
        Board2048.ApplyMove(board, Board2048.ActionDown, out _, out _);
        Assert.Equal(2, board[12]);

        board = new byte[16];
        board[4] = 1; board[12] = 1;
        Board2048.ApplyMove(board, Board2048.ActionUp, out _, out _);
        Assert.Equal(2, board[0]);
    }

    [Fact]
    public void ValidMoves_FullCheckerboard_HasNoMoves()
    {
        // Alternating distinct exponents: nothing can slide or merge anywhere.
        var board = new byte[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                board[r * 4 + c] = (byte)((r + c) % 2 == 0 ? 1 : 2);

        Assert.All(Board2048.ValidMoves(board), valid => Assert.False(valid));
        Assert.False(Board2048.AnyMoveAvailable(board));
    }

    [Fact]
    public void Spawn_Is90Percent2_10Percent4_UniformOverEmptyCells()
    {
        var rng = new Xoshiro256StarStar(11);
        int fours = 0;
        var cellCounts = new int[16];
        const int samples = 20_000;

        for (int i = 0; i < samples; i++)
        {
            var board = new byte[16];
            board[0] = 5; // one occupied cell
            Board2048.Spawn(board, rng);
            for (int c = 1; c < 16; c++)
                if (board[c] != 0)
                {
                    cellCounts[c]++;
                    if (board[c] == 2) fours++;
                }
        }

        Assert.Equal(0.1, fours / (double)samples, 0.01);
        Assert.Equal(0, cellCounts[0]);
        for (int c = 1; c < 16; c++)
            Assert.InRange(cellCounts[c] / (double)samples, 1.0 / 15 - 0.01, 1.0 / 15 + 0.01);
    }
}

public class Env2048Tests
{
    [Fact]
    public void Reset_SpawnsExactlyTwoTiles()
    {
        var env = new Env2048();
        var (obs, _) = env.Reset(1);
        Assert.Equal(2, obs.Count(v => v > 0));
    }

    [Fact]
    public void IllegalMove_Throws()
    {
        var env = new Env2048();
        env.Reset(1);
        var mask = env.CurrentActionMask();
        int illegal = Array.IndexOf(mask, false);
        if (illegal >= 0) // initial boards usually have at least one illegal direction
            Assert.Throws<InvalidOperationException>(() => env.Step(illegal));
    }

    [Fact]
    public void RandomLegalPlay_AlwaysReachesTermination_WithConsistentMasks()
    {
        var env = new Env2048();
        var rng = new Xoshiro256StarStar(3);
        env.Reset(3);

        for (int moves = 0; ; moves++)
        {
            var mask = env.CurrentActionMask();
            int legal = mask.Count(m => m);
            Assert.True(legal > 0, "mask empty while episode not done");

            int pick = rng.NextInt(legal);
            int action = Array.FindIndex(mask, m => m && pick-- == 0);
            var step = env.Step(action);

            if (step.Done)
            {
                Assert.True(step.Terminated);
                Assert.False(Board2048.AnyMoveAvailable(env.Board));
                Assert.True(env.Score > 0);
                break;
            }
            Assert.True(moves < Env2048.MaxEpisodeMoves, "game never ended");
        }
    }

    [Fact]
    public void Episodes_AreDeterministicPerSeed()
    {
        static (int score, int maxTile) Play(ulong seed)
        {
            var env = new Env2048();
            var rng = new Xoshiro256StarStar(7);
            env.Reset(seed);
            while (true)
            {
                var mask = env.CurrentActionMask();
                int pick = rng.NextInt(mask.Count(m => m));
                var step = env.Step(Array.FindIndex(mask, m => m && pick-- == 0));
                if (step.Done) return (env.Score, env.MaxTile);
            }
        }

        Assert.Equal(Play(42), Play(42));
        Assert.NotEqual(Play(42), Play(43));
    }
}

public class MaskedDqnInfraTests
{
    [Fact]
    public void ReplayBuffer_RoundTripsNextMasks()
    {
        var buffer = new ReplayBuffer(capacity: 4, obsDim: 1, actionCount: 3);
        buffer.Add([1f], 0, 0, [2f], false, [true, false, true]);
        buffer.Add([3f], 1, 0, [4f], false); // no mask = all legal

        var rng = new Xoshiro256StarStar(1);
        var batch = buffer.Sample(32, rng);
        for (int i = 0; i < batch.Size; i++)
        {
            if (batch.Actions[i] == 0)
                Assert.Equal(new[] { true, false, true }, batch.NextMasks[(i * 3)..(i * 3 + 3)]);
            else
                Assert.Equal(new[] { true, true, true }, batch.NextMasks[(i * 3)..(i * 3 + 3)]);
        }
    }

    [Fact]
    public void GreedyQAgent_NeverPicksMaskedActions()
    {
        var rng = new Xoshiro256StarStar(2);
        var net = new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp([16, 16, 4], rng, MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu);
        var agent = new GreedyQAgent(net, 4, rng) { Epsilon = 0.5 }; // both branches exercised

        var maskRng = new Xoshiro256StarStar(3);
        for (int i = 0; i < 500; i++)
        {
            var mask = new bool[4];
            int legal = 1 + maskRng.NextInt(3);
            while (mask.Count(m => m) < legal) mask[maskRng.NextInt(4)] = true;

            var obs = new float[16];
            for (int j = 0; j < 16; j++) obs[j] = (float)maskRng.NextDouble();

            Assert.True(mask[agent.Act(obs, mask)], "exploration picked a masked action");
            Assert.True(mask[agent.Act(obs, mask, greedy: true)], "greedy picked a masked action");
        }
    }
}

public class NTuple2048Tests
{
    [Fact]
    public void Learning_ImprovesOverUntrainedBaseline()
    {
        var agent = new NTuple2048Agent();
        var evalRng = new Xoshiro256StarStar(100);
        double before = AverageScore(agent, evalRng, games: 30);

        var trainRng = new Xoshiro256StarStar(200);
        for (int g = 0; g < 2000; g++)
            agent.PlayGame(trainRng, learn: true);

        evalRng = new Xoshiro256StarStar(100);
        double after = AverageScore(agent, evalRng, games: 30);
        Assert.True(after > before * 1.5, $"before {before:F0}, after {after:F0}");
    }

    private static double AverageScore(NTuple2048Agent agent, Xoshiro256StarStar rng, int games)
    {
        double total = 0;
        for (int g = 0; g < games; g++)
            total += agent.PlayGame(rng, learn: false).Score;
        return total / games;
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Gate_Reaches2048Tile_InAtLeast10PercentOfGames()
    {
        // PRD §6 pre-registered criterion. 10k training games measured ~28% on the
        // reference run (84% at 100k); asserting ≥ 10% leaves seed margin.
        var agent = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(200);
        for (int g = 0; g < 10_000; g++)
            agent.PlayGame(trainRng, learn: true);

        var evalRng = new Xoshiro256StarStar(100);
        int hits = 0;
        for (int g = 0; g < 100; g++)
            if (agent.PlayGame(evalRng, learn: false).MaxExponent >= 11) hits++;

        Assert.True(hits >= 10, $"2048 tile reached in {hits}/100 games (need >= 10)");
    }
}

public class Expectimax2048Tests
{
    // A handful of legal mid-game boards (exponents, row-major) to exercise the search on.
    private static IEnumerable<byte[]> SampleBoards()
    {
        var rng = new Xoshiro256StarStar(7);
        for (int n = 0; n < 40; n++)
        {
            var board = new byte[16];
            Board2048.Spawn(board, rng);
            Board2048.Spawn(board, rng);
            // Walk a few random legal moves to reach varied, non-trivial positions.
            for (int step = 0; step < 10 && Board2048.AnyMoveAvailable(board); step++)
            {
                var mask = Board2048.ValidMoves(board);
                int legal = mask.Count(m => m);
                int pick = rng.NextInt(legal);
                int action = Array.FindIndex(mask, m => m && pick-- == 0);
                Board2048.ApplyMove(board, action, out _, out _);
                Board2048.Spawn(board, rng);
            }
            if (Board2048.AnyMoveAvailable(board)) yield return board;
        }
    }

    [Fact]
    public void DepthZero_ReproducesGreedyAgentChoice()
    {
        // Depth 0 is argmax [reward + V(afterstate)] with no spawn lookahead — identical to the
        // agent's 1-ply selector, so the search is a faithful generalization of it.
        var agent = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(200);
        for (int g = 0; g < 1000; g++) agent.PlayGame(trainRng, learn: true); // give V some shape

        var solver = new Expectimax2048(agent) { MaxDepth = 0 };
        Span<byte> greedyAfter = stackalloc byte[16];
        Span<byte> solverAfter = stackalloc byte[16];

        foreach (var board in SampleBoards())
        {
            int greedyAction = agent.ChooseMove(board, out int greedyReward, greedyAfter);
            int solverAction = solver.ChooseMove(board, out int solverReward, solverAfter);
            Assert.Equal(greedyAction, solverAction);
            Assert.Equal(greedyReward, solverReward);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void NeverChoosesIllegalMove(int maxDepth)
    {
        var agent = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(200);
        for (int g = 0; g < 500; g++) agent.PlayGame(trainRng, learn: true);

        var solver = new Expectimax2048(agent) { MaxDepth = maxDepth };
        Span<byte> after = stackalloc byte[16];
        Span<byte> check = stackalloc byte[16];

        foreach (var board in SampleBoards())
        {
            int action = solver.ChooseMove(board, out _, after);
            Assert.InRange(action, 0, Board2048.ActionCount - 1);
            board.CopyTo(check);
            Assert.True(Board2048.ApplyMove(check, action, out _, out _), "search picked an illegal move");
        }
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Expectimax_OutscoresGreedy_OnTheSameValueFunction()
    {
        // The whole point: looking one+ spawn(s) ahead with the SAME tables beats 1-ply greedy.
        var agent = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(200);
        for (int g = 0; g < 5000; g++) agent.PlayGame(trainRng, learn: true);

        var solver = new Expectimax2048(agent); // adaptive depth
        const int games = 40;

        double greedyAvg = 0, emaxAvg = 0;
        var greedyRng = new Xoshiro256StarStar(100);
        var emaxRng = new Xoshiro256StarStar(100);
        for (int g = 0; g < games; g++)
        {
            greedyAvg += agent.PlayGame(greedyRng, learn: false).Score;
            emaxAvg += solver.PlayGame(emaxRng).Score;
        }
        greedyAvg /= games;
        emaxAvg /= games;

        Assert.True(emaxAvg > greedyAvg, $"expectimax {emaxAvg:F0} did not beat greedy {greedyAvg:F0}");
    }
}
