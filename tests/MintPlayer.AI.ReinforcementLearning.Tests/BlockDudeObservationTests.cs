using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The observation encoding, whose constant shape is what makes a size curriculum possible (PRD §7.1a).
/// </summary>
public class BlockDudeObservationTests
{
    private static BlockDudeBoard Board(params string[] rows) => BlockDudeBoard.FromGrid(rows);

    [Fact]
    public void ObservationShapeIsIdenticalForEveryBoardSize()
    {
        // The whole point: a tiny training board and the 29x19 level 11 must produce the same shaped tensor, so
        // advancing a curriculum stage never invalidates the net or its checkpoint.
        var tiny = Board(
            "...D",
            ".PB.",
            "WWWW");

        Assert.Equal(BlockDudeBoard.ObservationSize, tiny.BuildObservation().Length);

        foreach (var level in BlockDudeLevels.All)
        {
            var board = BlockDudeBoard.FromGrid(level.Grid);
            Assert.Equal(BlockDudeBoard.ObservationSize, board.BuildObservation().Length);
        }
    }

    [Fact]
    public void ObservationSize_MatchesTheDeclaredLayout()
    {
        // 21x13 window x 4 planes + an 8x5 coarse map + 9 scalars.
        Assert.Equal((21 * 13 * 4) + (8 * 5) + 9, BlockDudeBoard.ObservationSize);
    }

    [Fact]
    public void EveryValueIsFinite_AndPlanesAreStrictlyZeroOrOne()
    {
        foreach (var level in BlockDudeLevels.All)
        {
            var observation = BlockDudeBoard.FromGrid(level.Grid).BuildObservation();
            for (int i = 0; i < observation.Length; i++)
                Assert.True(float.IsFinite(observation[i]), $"{level.Name}[{i}] is not finite");

            int planeFloats = 21 * 13 * 4;
            for (int i = 0; i < planeFloats; i++)
                Assert.True(observation[i] == 0f || observation[i] == 1f, $"{level.Name} plane[{i}] = {observation[i]}");
        }
    }

    [Fact]
    public void TheOffBoardPlaneMarksCellsBeyondTheEdge()
    {
        // A player hard against the left wall must see off-board cells to his left — without this plane the net
        // cannot distinguish open sky from solid rock past the boundary.
        var board = Board(
            "..D",
            "P..",
            "WWW");

        var observation = board.BuildObservation();
        const int windowW = 21, windowH = 13, planeSize = windowW * windowH;
        int offBoardPlane = 3 * planeSize;

        // Centre of the window is the player; one cell further left than the board allows is off-board.
        int centreRow = (windowH - 1) / 2, centreCol = (windowW - 1) / 2;
        Assert.Equal(1f, observation[offBoardPlane + centreRow * windowW + (centreCol - 1)]);
        Assert.Equal(0f, observation[offBoardPlane + centreRow * windowW + centreCol]); // the player's own cell
    }

    [Fact]
    public void TheDoorBearingPointsTowardTheDoor()
    {
        // The scalars sit at the very end; the first two are the door bearing (dx, dy), normalised.
        int bearingIndex = BlockDudeBoard.ObservationSize - 9;

        var doorToTheRight = Board(
            "....",
            "P..D",
            "WWWW");
        Assert.True(doorToTheRight.BuildObservation()[bearingIndex] > 0);

        var doorToTheLeft = Board(
            "....",
            "D..P",
            "WWWW");
        Assert.True(doorToTheLeft.BuildObservation()[bearingIndex] < 0);
    }

    [Fact]
    public void TheObservationIsDeterministic()
    {
        // Reproducible training depends on this being a pure function of the position.
        foreach (var level in BlockDudeLevels.All)
        {
            var first = BlockDudeBoard.FromGrid(level.Grid).BuildObservation();
            var second = BlockDudeBoard.FromGrid(level.Grid).BuildObservation();
            Assert.Equal(first, second);
        }
    }

    [Fact]
    public void CarryingAndFacingAreReflected()
    {
        int carryingIndex = BlockDudeBoard.ObservationSize - 7;
        int facingIndex = BlockDudeBoard.ObservationSize - 6;

        var board = Board(
            "...D",
            ".PB.",
            "WWWW");
        Assert.Equal(0f, board.BuildObservation()[carryingIndex]);
        Assert.Equal(0f, board.BuildObservation()[facingIndex]);   // spawns facing left

        var carrying = board.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Grab);
        Assert.True(carrying.Carrying);
        Assert.Equal(1f, carrying.BuildObservation()[carryingIndex]);
        Assert.Equal(1f, carrying.BuildObservation()[facingIndex]);
    }

    [Fact]
    public void WriteObservation_RejectsAMisSizedSpan()
    {
        var board = Board(
            "...D",
            ".P..",
            "WWWW");

        Assert.Throws<ArgumentException>(() => board.WriteObservation(new float[10]));

        var destination = new float[BlockDudeBoard.ObservationSize];
        board.WriteObservation(destination);
        Assert.Equal(board.BuildObservation(), destination);
    }
}
