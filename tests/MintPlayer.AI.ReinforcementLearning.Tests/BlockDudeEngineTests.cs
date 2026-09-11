using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Rule conformance for Block Dude, one test per rule recovered from the original WinForms implementation and
/// corroborated by the TI-84+CE port (PRD §4.2), plus the reach of the exact oracle.
/// </summary>
public class BlockDudeEngineTests
{
    private static BlockDudeBoard Board(params string[] rows) => BlockDudeBoard.FromGrid(rows);

    [Fact]
    public void TheShippedPack_HoldsTheElevenOriginalLevels()
    {
        Assert.Equal(11, BlockDudeLevels.All.Length);
        foreach (var board in BlockDudeLevels.LoadAll())
        {
            Assert.False(board.Won);
            Assert.InRange(board.Width, 19, 29);
            Assert.InRange(board.Height, 8, 19);
        }
    }

    [Fact]
    public void LevelElevenShipsFloatingBlocks_BecauseLoadDoesNotSettleGravity()
    {
        // The single most important content invariant: a global settle pass on load would silently rewrite this
        // level into a different puzzle.
        var board = BlockDudeLevels.Load(10);

        int unsupported = 0;
        for (int y = 0; y < board.Height - 1; y++)
            for (int x = 0; x < board.Width; x++)
                if (board.HasBlock(x, y)
                    && board.TileAt(x, y + 1) != BlockDudeTile.Wall
                    && !board.HasBlock(x, y + 1)
                    && board.TileAt(x, y + 1) != BlockDudeTile.Door)
                    unsupported++;

        Assert.Equal(14, unsupported);
    }

    [Fact]
    public void AWallCanNeverBePickedUp_OnlyABlock()
    {
        // Facing a wall with empty headroom: Grab must be refused. The TI port pins this too (`== BLOCK`).
        // The door in the top-right corner is inert here; FromGrid requires exactly one, as every real level has.
        var board = Board(
            "...D",
            ".PW.",
            "WWWW");

        var after = board.Apply(BlockDudeAction.Grab);
        Assert.False(after.Carrying);
        Assert.True(after.SameStateAs(board));
    }

    [Fact]
    public void ABlockIsPickedUpFromBeside_AndPutDownDiagonallyAhead()
    {
        var board = Board(
            "...D",
            ".PB.",
            "WWWW");

        // The player spawns facing LEFT, so turn right first, then grab.
        var facing = board.Apply(BlockDudeAction.Right);
        var holding = facing.Apply(BlockDudeAction.Grab);
        Assert.True(holding.Carrying);
        Assert.False(holding.HasBlock(2, 1));

        // Put down: the block is released diagonally up-ahead and then falls to the floor.
        var placed = holding.Apply(BlockDudeAction.Grab);
        Assert.False(placed.Carrying);
        Assert.True(placed.HasBlock(2, 1));
    }

    [Fact]
    public void ThePlayerFallsThroughADoor_ButABlockRestsOnIt()
    {
        // Player walks right onto the door cell, which does not support him, so he drops to the floor below and
        // does NOT win — the win check runs after gravity.
        var board = Board(
            "....",
            ".PD.",
            "W..W",
            "WWWW");

        var after = board.Apply(BlockDudeAction.Right);
        Assert.False(after.Won);
        Assert.Equal(2, after.PlayerY);

    }

    [Fact]
    public void ABlockFallingOntoADoorRestsOnTopOfIt()
    {
        // The asymmetry that makes doors odd: the player falls straight through a door, but a falling block
        // treats it as solid ground. Carry a block over the door and drop it.
        var board = Board(
            "......",
            ".PB...",
            "WW.DWW",
            "WWWWWW");

        var carried = board.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Grab);
        Assert.True(carried.Carrying);

        var moved = carried.Apply(BlockDudeAction.Right);
        Assert.True(moved.Carrying);
        Assert.Equal(2, moved.PlayerY);   // he dropped into the pit, still holding the block

        var dropped = moved.Apply(BlockDudeAction.Grab);
        Assert.False(dropped.Carrying);
        Assert.Equal(BlockDudeTile.Door, dropped.TileAt(3, 2));
        Assert.True(dropped.HasBlock(3, 1), "the block should come to rest on top of the door");
        Assert.False(dropped.HasBlock(3, 2), "a block must never occupy the door cell");
    }

    [Fact]
    public void WalkingIntoADoorWithSolidGroundBeneath_WinsTheLevel()
    {
        var board = Board(
            "....",
            ".PD.",
            "WWWW");

        var after = board.Apply(BlockDudeAction.Right);
        Assert.True(after.Won);
    }

    [Fact]
    public void ABlockCanNeverBePlacedIntoTheDoorCell()
    {
        // Single occupancy, with the player-in-door exception: the door must stay enterable, so a block may
        // never occupy it.
        var board = Board(
            "..D.",
            ".PB.",
            "WWWW");

        var holding = board.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Grab);
        Assert.True(holding.Carrying);

        var attempt = holding.Apply(BlockDudeAction.Grab);
        Assert.True(attempt.Carrying);            // refused
        Assert.NotEqual(BlockDudeTile.Empty, attempt.TileAt(2, 0));
        Assert.False(attempt.HasBlock(2, 0));
    }

    [Fact]
    public void ClimbingRequiresSomethingSolidToClimb_AndHeadroom()
    {
        // Nothing ahead: refused.
        var openAir = Board(
            "...D",
            ".P..",
            "WWWW");
        Assert.True(openAir.Apply(BlockDudeAction.Climb).SameStateAs(openAir));

        // A block ahead with clear headroom: the climb succeeds and lands on top of it.
        var step = Board(
            "...D",
            ".PB.",
            "WWWW");
        var climbed = step.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Climb);
        Assert.Equal(2, climbed.PlayerX);
        Assert.Equal(0, climbed.PlayerY);

        // Same shape but with a ceiling directly overhead: refused for lack of headroom.
        var ceiling = Board(
            "WWWWW",
            ".PB.D",
            "WWWWW");
        var blocked = ceiling.Apply(BlockDudeAction.Right);
        Assert.True(blocked.Apply(BlockDudeAction.Climb).SameStateAs(blocked));
    }

    [Fact]
    public void TurningToFaceANewDirectionCountsAsAMove()
    {
        // Faithful to the original: the turn itself is a step, even when the way ahead is blocked.
        var board = Board(
            "...D",
            "WPW.",
            "WWWW");

        var turned = board.Apply(BlockDudeAction.Right);
        Assert.True(turned.FacingRight);
        Assert.Equal(board.PlayerX, turned.PlayerX);
        Assert.False(turned.SameStateAs(board));
    }

    [Fact]
    public void ToGrid_RoundTripsThroughFromGrid()
    {
        foreach (var level in BlockDudeLevels.All)
        {
            var board = BlockDudeBoard.FromGrid(level.Grid);
            Assert.True(BlockDudeBoard.FromGrid(board.ToGrid()).SameStateAs(board), level.Name);
        }
    }

    [Fact]
    public void TheOracleSolvesTheEarlyLevels_AndGivesUpOnTheLateOnes()
    {
        // Documents the oracle's actual reach, which is the whole reason training data comes from generated
        // small boards rather than the shipped content (PRD §4.5).
        var reach = new List<string>();
        foreach (var (level, index) in BlockDudeLevels.All.Select((l, i) => (l, i)))
        {
            var oracle = new BlockDudeOracle(BlockDudeBoard.FromGrid(level.Grid), maxStates: 60_000);
            reach.Add(oracle.Truncated ? $"{level.Name}: truncated" : $"{level.Name}: optimal {oracle.OptimalFromStart}");
        }

        // Level 1 is small enough to label exactly, and must be genuinely solvable.
        var first = new BlockDudeOracle(BlockDudeLevels.Load(0), maxStates: 60_000);
        Assert.False(first.Truncated, string.Join(" | ", reach));
        Assert.True(first.OptimalFromStart > 0, $"Level 1 must be solvable. Reach: {string.Join(" | ", reach)}");

        // Level 11 (42 blocks on 551 cells) must be refused rather than attempted.
        var last = new BlockDudeOracle(BlockDudeLevels.Load(10), maxStates: 60_000);
        Assert.True(last.Truncated);
        Assert.Equal(-1, last.OptimalFromStart);
    }
}
