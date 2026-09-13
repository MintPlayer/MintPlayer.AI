using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The net-guided A* that produces every phase-2 training sample and every "what is this net worth" number.
/// </summary>
/// <remarks>
/// The search runs BATCHED — it scores a whole round of successors in one forward pass, because the net is the
/// entire cost of the search. That is an optimisation of the evaluation ORDER, not of the evaluation, so the
/// property worth pinning is that batching changes no answer: the batched distances must equal the single-row
/// ones the rest of the code (greedy play, the calibration probe) reads.
/// </remarks>
public class BlockDudeSearchTests
{
    private static BlockDudePolicyNet Net(ulong seed = 7) => new(new Xoshiro256StarStar(seed), 32);

    [Fact]
    public void BatchedDistancesEqualTheSingleRowOnes()
    {
        var net = Net();

        // A spread of genuinely different positions, not one board repeated: a batch that happens to be uniform
        // would pass even if rows were being mixed up.
        var boards = BlockDudeLevels.All.Take(8).Select(l => BlockDudeBoard.FromGrid(l.Grid)).ToArray();

        var batched = net.Distances(boards);

        Assert.Equal(boards.Length, batched.Length);
        for (int i = 0; i < boards.Length; i++)
            Assert.Equal(net.Evaluate(boards[i]).Distance, batched[i], 3);
    }

    [Fact]
    public void AnEmptyBatchIsNotAForwardPass()
    {
        // A* generates rounds with no surviving successor (every child already reached more cheaply). Feeding a
        // zero-row tensor to the trunk would throw, so the search would die on a legal, ordinary situation.
        Assert.Empty(Net().Distances([]));
    }

    [Fact]
    public void SearchSolvesAPositionOneMoveFromTheDoor()
    {
        // An untrained net makes the heuristic noise, so this pins the search itself rather than the policy:
        // with a finite budget it must still find a solution that is one move away.
        var start = BlockDudeDemonstrations.ReverseCurriculumStarts(1, "Level 1").Single();

        var outcome = BlockDudeSearch.Solve(Net(), start, maxExpansions: 5_000, weight: 2f);

        Assert.True(outcome.Solved);
        Assert.Equal(1, outcome.Length);
    }

    [Fact]
    public void SearchReturnsAPathThatActuallyWins()
    {
        // The path is replayed through the engine, because a search that reports a solution it cannot walk would
        // feed phase 2 training samples labelled with moves nobody can play.
        var start = BlockDudeDemonstrations.ReverseCurriculumStarts(12, "Level 1").Single();

        var outcome = BlockDudeSearch.Solve(Net(), start, maxExpansions: 60_000, weight: 2f);

        Assert.True(outcome.Solved);
        var board = start;
        foreach (int move in outcome.Moves!) board = board.Apply((BlockDudeAction)move);
        Assert.True(board.Won);
    }

    [Fact]
    public void AnExhaustedBudgetFailsHonestly()
    {
        // Not "returns something": an out-of-budget search must return no path at all, so the caller counts it as
        // unsolved instead of training on a truncated one.
        var start = BlockDudeBoard.FromGrid(BlockDudeLevels.All[^1].Grid);

        Assert.False(BlockDudeSearch.Solve(Net(), start, maxExpansions: 1, weight: 2f).Solved);
    }
}
