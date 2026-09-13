using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Landmarks: letting a search aim at the human's path as well as at the door.
/// </summary>
/// <remarks>
/// The point is to rescue the searches that fail. On a long level, asking A* for a 400-move suffix inside an
/// eight-second budget mostly fails, and a failed search teaches nothing — so the levels with the most left to
/// learn produce the least data. Reaching a state the demonstration also reached is a real solution, because a
/// winning continuation from there is already known.
/// <para>
/// The danger is the other side of that: the lookup key is a 32-bit state hash, and believing a collision would
/// manufacture a training sample asserting a win that does not exist. Every hit must be confirmed by replay.
/// </para>
/// </remarks>
public class BlockDudeLandmarkTests
{
    [Fact]
    public void EveryDemonstratedStateIsALandmarkAtItsTrueRemainingDistance()
    {
        var landmarks = BlockDudeDemonstrations.PathLandmarks("Level 3");

        foreach (var state in BlockDudeDemonstrations.LabelledStates().Where(s => s.Level == "Level 3"))
        {
            Assert.True(landmarks.TryGetValue(state.Board.StateHash, out int remaining));

            // "At most" rather than "equal": a level that revisits a position keeps the SMALLER remaining, which
            // is correct — reaching it is worth the best continuation known from it.
            Assert.True(remaining <= state.Remaining);
        }
    }

    [Fact]
    public void TheDoorIsALandmarkAtZero()
    {
        // Without this the door would not be a goal of a landmark search at all, and a full solution would be
        // rejected in favour of whatever demonstrated state came before it.
        var solution = BlockDudeSolutions.For("Level 1")!;
        var board = BlockDudeBoard.FromGrid(Array.Find(BlockDudeLevels.All, l => l.Name == "Level 1")!.Grid);
        foreach (var move in solution.Moves) board = board.Apply(move);

        Assert.True(board.Won);
        Assert.Equal(0, BlockDudeDemonstrations.PathLandmarks("Level 1")[board.StateHash]);
    }

    [Fact]
    public void AConfirmedContinuationActuallyFinishesTheLevel()
    {
        var solution = BlockDudeSolutions.For("Level 5")!;
        var board = BlockDudeBoard.FromGrid(Array.Find(BlockDudeLevels.All, l => l.Name == "Level 5")!.Grid);

        const int played = 40;
        for (int i = 0; i < played; i++) board = board.Apply(solution.Moves[i]);
        int remaining = solution.Moves.Length - played;

        var tail = BlockDudeDemonstrations.ContinuationFrom(board, "Level 5", remaining);

        Assert.NotNull(tail);
        Assert.Equal(remaining, tail!.Count);
        foreach (var move in tail) board = board.Apply(move);
        Assert.True(board.Won);
    }

    [Fact]
    public void APositionThatIsNotOnThePathIsRejected()
    {
        // The collision guard, and the reason a hit is a candidate rather than a proof. A hash that happened to
        // match would splice in a tail that walks somewhere else entirely; replaying it is what catches that.
        var level = Array.Find(BlockDudeLevels.All, l => l.Name == "Level 5")!;
        var elsewhere = BlockDudeBoard.FromGrid(level.Grid).Apply(BlockDudeAction.Right);

        int wrongRemaining = BlockDudeSolutions.For("Level 5")!.Moves.Length - 40;

        Assert.Null(BlockDudeDemonstrations.ContinuationFrom(elsewhere, "Level 5", wrongRemaining));
    }

    [Fact]
    public void AnOutOfRangeRemainingIsRejectedRatherThanThrowing()
    {
        // A landmark miss surfaces as remaining = -1 in the campaign, and it must be a quiet "no", not a crash
        // that takes the training run down overnight.
        var board = BlockDudeBoard.FromGrid(Array.Find(BlockDudeLevels.All, l => l.Name == "Level 1")!.Grid);

        Assert.Null(BlockDudeDemonstrations.ContinuationFrom(board, "Level 1", -1));
        Assert.Null(BlockDudeDemonstrations.ContinuationFrom(board, "Level 1", 100_000));
    }

    [Fact]
    public void AnUnknownLevelYieldsNoLandmarksRatherThanThrowing()
    {
        Assert.Empty(BlockDudeDemonstrations.PathLandmarks("no such level"));
        Assert.Null(BlockDudeDemonstrations.ContinuationFrom(
            BlockDudeBoard.FromGrid(BlockDudeLevels.All[0].Grid), "no such level", 1));
    }

    [Fact]
    public void ALandmarkSearchEndsEitherAtTheDoorOrOnTheDemonstratedPath()
    {
        // The property the campaign relies on to label its samples. An untrained net is used deliberately: the
        // guarantee is structural, and must not depend on the heuristic being any good.
        var net = new BlockDudePolicyNet(new Xoshiro256StarStar(3), 32);
        var landmarks = BlockDudeDemonstrations.PathLandmarks("Level 3");
        var start = BlockDudeDemonstrations.ReverseCurriculumStarts(30, "Level 3").Single();

        var outcome = BlockDudeSearch.SolveToLandmark(net, start, landmarks, acceptRemaining: 15,
                                                      maxExpansions: 40_000, weight: 2f, policyWeight: 1f);

        Assert.True(outcome.Solved);

        var reached = start;
        foreach (int move in outcome.Moves!) reached = reached.Apply((BlockDudeAction)move);

        if (reached.Won) return;
        Assert.True(landmarks.TryGetValue(reached.StateHash, out int remaining));
        Assert.True(remaining <= 15, $"stopped at a landmark {remaining} moves out, past the accept threshold");
        Assert.NotNull(BlockDudeDemonstrations.ContinuationFrom(reached, "Level 3", remaining));
    }

    [Fact]
    public void NoLandmarksMeansTheSearchStillOnlyAcceptsTheDoor()
    {
        // Passing an empty table must not make everything a goal. It reduces to an ordinary search.
        var net = new BlockDudePolicyNet(new Xoshiro256StarStar(5), 32);
        var start = BlockDudeDemonstrations.ReverseCurriculumStarts(6, "Level 1").Single();

        var outcome = BlockDudeSearch.SolveToLandmark(net, start, new Dictionary<int, int>(), acceptRemaining: 99,
                                                      maxExpansions: 20_000, weight: 2f, policyWeight: 1f);

        Assert.True(outcome.Solved);
        var reached = start;
        foreach (int move in outcome.Moves!) reached = reached.Apply((BlockDudeAction)move);
        Assert.True(reached.Won);
    }
}
