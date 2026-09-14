using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Turning the human solutions into training signal. The properties here are the ones a training loop would
/// silently rely on — a wrong distance label or an unreachable start state would poison the data rather than
/// fail loudly.
/// </summary>
public class BlockDudeDemonstrationsTests
{
    [Fact]
    public void EveryDemonstratedStateCarriesTheMovesActuallyRemaining()
    {
        // The label the value head needs, and the one it has no other source for. Verified by playing the rest
        // of the path out rather than trusting the arithmetic.
        foreach (var state in BlockDudeDemonstrations.LabelledStates().Where(s => s.Level == "Level 6"))
        {
            var board = state.Board;
            var solution = BlockDudeSolutions.For(state.Level)!;
            int played = solution.Moves.Length - state.Remaining;

            for (int i = played; i < solution.Moves.Length; i++) board = board.Apply(solution.Moves[i]);

            Assert.True(board.Won, $"{state.Level}: replaying the last {state.Remaining} moves did not win");
        }
    }

    [Fact]
    public void TheLabelledStatesSpanTheRangeTheValueHeadIsBlindIn()
    {
        var remaining = BlockDudeDemonstrations.LabelledStates().Select(s => s.Remaining).ToArray();

        Assert.Equal(3870, remaining.Length);
        Assert.Equal(1, remaining.Min());
        Assert.Equal(909, remaining.Max());

        // The curriculum's hardest rung tops out around 25 optimal moves, so anything past that is range the net
        // has never been trained on. Most of this data is out there, which is the point of having it.
        Assert.True(remaining.Count(r => r > 25) > 3000,
            "expected the bulk of demonstrated states to lie beyond the curriculum's horizon");
    }

    [Fact]
    public void NoDemonstratedStateIsAlreadyWon()
    {
        // A won board has no action to imitate and a remaining distance of zero; including it would teach the
        // net to predict a move for a terminal state.
        Assert.All(BlockDudeDemonstrations.LabelledStates(), s => Assert.False(s.Board.Won));
    }

    [Fact]
    public void ReverseCurriculumStartsAreRealPositionsThatAreActuallySolvable()
    {
        // The whole mechanism: a suffix of a human path is a legitimate position on REAL shipped terrain that is
        // only `movesFromEnd` moves from the door. If that were not true the curriculum would be training on
        // states no play can reach.
        foreach (int fromEnd in new[] { 1, 5, 20 })
        {
            var starts = BlockDudeDemonstrations.ReverseCurriculumStarts(fromEnd).ToArray();
            Assert.Equal(BlockDudeSolutions.All.Length, starts.Length);

            foreach (var start in starts)
                Assert.False(start.Won, $"a start {fromEnd} moves from the end should not already be won");
        }

        // And the suffix really does finish the level: replay the last 20 moves of Level 7 from its start state.
        var solution = BlockDudeSolutions.For("Level 7")!;
        var board = BlockDudeDemonstrations.ReverseCurriculumStarts(20, "Level 7").Single();
        foreach (var action in solution.Moves.Skip(solution.Moves.Length - 20)) board = board.Apply(action);

        Assert.True(board.Won);
    }

    [Fact]
    public void AskingForMoreMovesThanALevelHasYieldsItsOpeningPosition()
    {
        // Bonus 3 is 39 moves. Once the ramp passes that, the whole level is the task — it must not drop out of
        // the curriculum, which is what skipping it would quietly do.
        var start = BlockDudeDemonstrations.ReverseCurriculumStarts(5000, "Bonus 3").Single();
        var level = Array.Find(BlockDudeLevels.All, l => l.Name == "Bonus 3")!;

        Assert.Equal(string.Join("\n", level.Grid), string.Join("\n", start.ToGrid()));
    }

    [Fact]
    public void HumanMoveCountsAreReportedForEveryLevel()
    {
        var counts = BlockDudeDemonstrations.HumanMoveCounts();

        Assert.Equal(BlockDudeLevels.All.Length, counts.Count);
        Assert.Equal(909, counts["Level 11"]);
        Assert.Equal(19, counts["Level 1"]);
    }
}
