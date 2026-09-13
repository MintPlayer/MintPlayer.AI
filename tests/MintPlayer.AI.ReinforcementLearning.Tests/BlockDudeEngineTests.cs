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
    public void TheShippedPack_HoldsTheElevenOriginalLevelsAndFourBonusLevels()
    {
        Assert.Equal(15, BlockDudeLevels.All.Length);
        Assert.Equal([.. Enumerable.Range(1, 11).Select(i => $"Level {i}")],
                     BlockDudeLevels.All.Take(11).Select(l => l.Name));
        Assert.Equal([.. Enumerable.Range(1, 4).Select(i => $"Bonus {i}")],
                     BlockDudeLevels.All.Skip(11).Select(l => l.Name));

        foreach (var board in BlockDudeLevels.LoadAll())
        {
            Assert.False(board.Won);
            Assert.InRange(board.Width, 19, 29);
            Assert.InRange(board.Height, 8, 19);
        }
    }

    [Fact]
    public void ALevelMayHoldSeveralDoors_AndAnyOfThemWins()
    {
        // The bonus pack breaks the one-door-per-level habit of the originals: Bonus 2 seals its exit behind a
        // row of seven door cells, Bonus 4 offers two separate exits. FromGrid must accept both.
        var bonusDoors = BlockDudeLevels.All.Skip(11)
            .Select(l => l.Grid.Sum(r => r.Count(c => c == 'D')))
            .ToArray();

        Assert.Equal([1, 7, 1, 2], bonusDoors);

        // Any door cell is a win cell, not just the one the observation aims at.
        var twoDoors = Board(
            "DP..D",
            "WWWWW");
        Assert.True(twoDoors.Apply(BlockDudeAction.Left).Won);
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
        // The door in the top-right corner is inert here; FromGrid requires at least one.
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
    public void ACarriedBlockIsKnockedOutOfHisHands_WhenItsOwnWayForwardIsBlocked()
    {
        // The carried block travels sideways through the cell diagonally forward-and-up from where he STANDS,
        // judged before the step and before gravity (PRD §4.2). Reported on level 9: walking left off a ledge
        // with a wall diagonally ahead used to carry the block straight through that wall, because the check
        // ran against the position he LANDED on — one row lower, and empty.
        var board = Board(
            ".......D",
            "..W.....",
            "....BP..",
            "WW.WWWWW",
            "WWWWWWWW");

        // Pick the block up (he spawns facing left, so it is already the one he faces) and walk to the ledge.
        var carried = board.Apply(BlockDudeAction.Grab);
        Assert.True(carried.Carrying);

        var atLedge = carried.Apply(BlockDudeAction.Left).Apply(BlockDudeAction.Left);
        Assert.True(atLedge.Carrying);            // nothing blocked the block on the way
        Assert.Equal(3, atLedge.PlayerX);
        Assert.Equal(2, atLedge.PlayerY);

        // Now step left: the block's own destination (2,1) is wall, so it is knocked out of his hands and falls
        // in ITS OWN column, behind him, while he carries on into the hole.
        var after = atLedge.Apply(BlockDudeAction.Left);

        Assert.False(after.Carrying);
        Assert.Equal(2, after.PlayerX);
        Assert.Equal(3, after.PlayerY);           // he dropped into the hole
        Assert.True(after.HasBlock(3, 2));        // same column he left it in, resting on the floor
        Assert.False(after.HasBlock(2, 2));       // NOT dragged through the wall to above his landing cell
    }

    [Fact]
    public void WalkingIntoAWallWhileCarrying_KeepsTheBlock()
    {
        // Owner ruling 2026-09-13 (PRD §4.3). The carried-block follow-up runs only when the step ACTUALLY
        // happened: bumping a wall costs the move but never the block. Worth pinning because all three sources
        // disagreed — the recovered rule text said the follow-up runs "even when the move was blocked", and the
        // TI-84+CE port reverts the whole move instead. This is the behaviour that is intended.
        // The wall is TWO high on purpose. The rejected reading knocks the block out when the cell diagonally
        // forward-and-up from the old position is occupied — which (4,0) is — so this board separates the two
        // rules instead of merely agreeing with both.
        var board = Board(
            "....W.",
            ".PB.WD",
            "WWWWWW");

        var carried = board.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Grab);
        Assert.True(carried.Carrying);

        // Walk up to the wall: he ends at (3,1) holding the block at (3,0), facing a 2-high wall.
        var atWall = carried.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Right);
        Assert.True(atWall.Carrying);
        Assert.Equal(3, atWall.PlayerX);
        Assert.Equal(1, atWall.PlayerY);
        Assert.Equal(BlockDudeTile.Wall, atWall.TileAt(4, 0));   // the cell that would knock it out

        var bumped = atWall.Apply(BlockDudeAction.Right);

        Assert.True(bumped.Carrying, "bumping a wall must never knock the block out of his hands");
        Assert.Equal(3, bumped.PlayerX);
        Assert.Equal(1, bumped.PlayerY);
    }

    [Fact]
    public void ClimbingKeepsTheBlock_EvenWithAStoneDirectlyAboveIt()
    {
        // The mirror of the knock-out rule, and the reason it must NOT be widened to "anything solid near the
        // block": a climb moves the block diagonally up-and-forward, so the cell directly above it is never on
        // its path. A stone there is simply a low ceiling he slides out from under — he keeps the block.
        var board = Board(
            ".....D",
            ".W....",          // stone directly above the carried block
            "......",
            "BPW...",
            "WWWWWW");

        var carried = board.Apply(BlockDudeAction.Grab);
        Assert.True(carried.Carrying);
        Assert.Equal(1, carried.PlayerX);
        Assert.Equal(3, carried.PlayerY);

        var climbed = carried.Apply(BlockDudeAction.Right).Apply(BlockDudeAction.Climb);

        Assert.True(climbed.Carrying, "the stone above the block is not on the block's diagonal path");
        Assert.Equal(2, climbed.PlayerX);
        Assert.Equal(2, climbed.PlayerY);
    }

    [Fact]
    public void OnLevelTen_TheSameBlockSurvivesAClimbAndIsThenKnockedOffByAWalk()
    {
        // The two rules above meeting on REAL shipped content, one move apart, on the SAME block — which is what
        // makes the distinction concrete: a climb carries the block DIAGONALLY, a walk drags it SIDEWAYS, so the
        // very wall that a climb slips out from under is the wall that a walk knocks it against.
        //
        // The 14-move approach was found by breadth-first search over the shipped engine, so it is genuinely
        // reachable from the level's start: 1=Right, 0=Left, 2=Climb, 3=Grab.
        var board = BlockDudeLevels.Load(9);
        foreach (char step in "10311321113022")
            board = board.Apply((BlockDudeAction)(step - '0'));

        Assert.True(board.Carrying);
        Assert.Equal(19, board.PlayerX);
        Assert.Equal(15, board.PlayerY);
        Assert.Equal(BlockDudeTile.Wall, board.TileAt(19, 13));   // ceiling directly above the carried block

        // Climb up-left: the block goes diagonally to (18,13), out from under that ceiling. Still in his hands.
        var climbed = board.Apply(BlockDudeAction.Climb);
        Assert.True(climbed.Carrying, "a ceiling above the block is never on the block's diagonal path");
        Assert.Equal(18, climbed.PlayerX);
        Assert.Equal(14, climbed.PlayerY);

        // Now walk back right. The block's sideways path is that same wall at (19,13), so it is knocked out of
        // his hands and falls in its own column while he steps on and drops.
        var walked = climbed.Apply(BlockDudeAction.Right);
        Assert.False(walked.Carrying, "a walk drags the block sideways, straight into the wall");
        Assert.Equal(19, walked.PlayerX);
        Assert.Equal(15, walked.PlayerY);
        Assert.True(walked.HasBlock(18, 14));    // left in its own column, behind him
        Assert.False(walked.HasBlock(19, 14));   // NOT dragged through the wall
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
