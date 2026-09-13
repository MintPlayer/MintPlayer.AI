using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Replays the hand-played solutions to all 15 shipped levels.
/// </summary>
/// <remarks>
/// <para>This is the strongest engine conformance test in the repo, and it is strong for a reason no
/// hand-written test can match: 3,870 moves of real play across every shipped board, each of which must remain
/// legal and still arrive at the door. A rule change subtle enough to slip past the unit tests — gravity, the
/// carried block, climbing, single occupancy — will almost certainly derail at least one of these trajectories,
/// because a human solution threads a narrow path through all of those rules at once for hundreds of moves.</para>
///
/// <para>If one of these fails after an engine change, the engine changed behaviour. The recording is a fact
/// about a session that actually happened; it is not a thing to be adjusted to fit new rules.</para>
/// </remarks>
public class BlockDudeSolutionsTests
{
    [Fact]
    public void EveryShippedLevelHasARecordedSolution()
    {
        Assert.Equal(BlockDudeLevels.All.Length, BlockDudeSolutions.All.Length);

        foreach (var level in BlockDudeLevels.All)
            Assert.True(BlockDudeSolutions.For(level.Name) is not null, $"no recorded solution for '{level.Name}'");
    }

    [Fact]
    public void EveryRecordedSolutionIsLegalThroughout_AndReachesTheDoor()
    {
        foreach (var solution in BlockDudeSolutions.All)
        {
            var level = Array.Find(BlockDudeLevels.All, l => l.Name == solution.Name);
            Assert.NotNull(level);

            var board = BlockDudeBoard.FromGrid(level!.Grid);

            for (int i = 0; i < solution.Moves.Length; i++)
            {
                Assert.False(board.Won, $"'{solution.Name}' reached the door at move {i}, before its last move");

                var action = solution.Moves[i];
                Assert.True(board.IsLegal(action),
                    $"'{solution.Name}' move {i} ({action}) is illegal at ({board.PlayerX},{board.PlayerY})");

                board = board.Apply(action);
            }

            Assert.True(board.Won, $"'{solution.Name}' ran out of moves without reaching the door");
        }
    }

    [Fact]
    public void TheRecordedSolutionsSpanTheHorizonsTheTrainingDataDoesNot()
    {
        // The reason these are worth keeping as DATA and not only as a test. The curriculum's hardest rung tops
        // out around 25 optimal moves, so the value head has never seen a state hundreds of moves from the goal
        // (PRD §8.4a measured it under-estimating those by ~37 moves). These trajectories are exactly that range.
        int longest = BlockDudeSolutions.All.Max(s => s.Moves.Length);
        int total = BlockDudeSolutions.All.Sum(s => s.Moves.Length);

        Assert.True(longest > 800, $"expected a trajectory of several hundred moves, longest was {longest}");
        Assert.True(total > 3500, $"expected thousands of labelled states, got {total}");
    }
}
