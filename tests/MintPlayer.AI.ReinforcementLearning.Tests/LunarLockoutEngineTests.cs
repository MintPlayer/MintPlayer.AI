using MintPlayer.AI.ReinforcementLearning.Environments.LunarLockout;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Rule conformance for Lunar Lockout, one test per published rule (PRD §5.2). The rules were verified against
/// ThinkFun's own instructions, so these cases pin the *published* game, not merely the port's behaviour.
/// </summary>
public class LunarLockoutEngineTests
{
    private static LunarLockoutBoard Board(params string[] rows) => LunarLockoutBoard.FromGrid(rows);

    [Fact]
    public void EdgeIsNotABackstop_LoneRobotCannotMoveAtAll()
    {
        // A single robot has nothing to slide against. Every direction must be refused — if the edge stopped
        // robots, all four would be legal and the whole puzzle would collapse.
        var board = Board(
            ".....",
            ".....",
            "..R..",
            ".....",
            ".....");

        Assert.Empty(board.LegalActions());
    }

    [Fact]
    public void Slide_StopsAdjacentToTheBlocker()
    {
        // R at (2,1) slides right into B at (2,3) and must rest at (2,2) — the centre.
        var board = Board(
            ".....",
            ".....",
            ".R.B.",
            ".....",
            ".....");

        Assert.Equal(LunarLockoutBoard.CentreCell, board.Landing(0, LunarDirection.Right));

        var after = board.Apply(LunarLockoutBoard.EncodeAction(0, LunarDirection.Right));
        Assert.Equal(LunarLockoutBoard.CentreCell, after.TargetCell);
        Assert.True(after.IsSolved);
    }

    [Fact]
    public void Slide_IsIllegalWhenAlreadyRestingAgainstTheBlocker()
    {
        // R at (2,2) is already adjacent to B at (2,3): sliding right would move it zero cells.
        var board = Board(
            ".....",
            ".....",
            "..RB.",
            ".....",
            ".....");

        Assert.Equal(-1, board.Landing(0, LunarDirection.Right));
        Assert.False(board.IsLegal(LunarLockoutBoard.EncodeAction(0, LunarDirection.Right)));
    }

    [Fact]
    public void HelperOnCentre_BlocksButDoesNotWin()
    {
        // A helper parked on the centre is canonical: it acts as an ordinary blocker, and the position is not won.
        var board = Board(
            ".....",
            ".....",
            "R.B..",
            ".....",
            ".....");

        Assert.False(board.IsSolved);

        // R slides right and stops at (2,1), short of the occupied centre.
        var after = board.Apply(LunarLockoutBoard.EncodeAction(0, LunarDirection.Right));
        Assert.Equal(2 * LunarLockoutBoard.Size + 1, after.TargetCell);
        Assert.False(after.IsSolved);
    }

    [Fact]
    public void OnlyTheTargetRobotWinsOnCentre()
    {
        // B slides left into R and lands on the centre. That is not a win — only R counts.
        var board = Board(
            ".....",
            ".....",
            ".R..B",
            ".....",
            ".....");

        var after = board.Apply(LunarLockoutBoard.EncodeAction(1, LunarDirection.Left));
        Assert.Equal(LunarLockoutBoard.CentreCell, after.Robots[1]);
        Assert.False(after.IsSolved);
    }

    [Fact]
    public void Key_IsInvariantUnderHelperPermutation()
    {
        // Helpers only slide and block, so they are interchangeable; the canonical key must sort them, otherwise
        // the oracle explores the same position once per helper permutation.
        var a = Board(
            "B...G",
            ".....",
            "..R..",
            ".....",
            ".....");
        var b = Board(
            "G...B",
            ".....",
            "..R..",
            ".....",
            ".....");

        Assert.Equal(a.Key, b.Key);
    }

    [Fact]
    public void Key_DistinguishesTheTargetFromAHelper()
    {
        // Swapping the target with a helper is a genuinely different position and must not collide.
        var a = Board(
            "R...B",
            ".....",
            ".....",
            ".....",
            ".....");
        var b = Board(
            "B...R",
            ".....",
            ".....",
            ".....",
            ".....");

        Assert.NotEqual(a.Key, b.Key);
    }

    [Fact]
    public void Key_FitsInThirtyBits_SoTypeScriptKeepsItANumber()
    {
        // i64 keys lower to TypeScript `bigint` — correct but allocating. Every key must stay a positive i32.
        foreach (var level in LunarLockoutLevels.All)
        {
            int key = Board(level.Grid).Key;
            Assert.InRange(key, 0, (1 << 30) - 1);
        }
    }

    [Fact]
    public void FromGrid_RejectsAMalformedBoard()
    {
        Assert.Throws<ArgumentException>(() => Board(".....", ".....", "..R..", "....."));          // 4 rows
        Assert.Throws<ArgumentException>(() => Board(".....", ".....", "..R.", ".....", "....."));  // short row
        Assert.Throws<ArgumentException>(() => Board(".....", ".....", "..RR.".Replace("RR", "RR"), ".....", ".....")); // two targets
        Assert.Throws<ArgumentException>(() => Board(".....", ".....", ".....", ".....", "....."));  // no target
    }

    [Fact]
    public void ToGrid_RoundTripsThroughFromGrid()
    {
        var original = Board(
            "R...A",
            ".....",
            "..B..",
            ".....",
            ".....");

        Assert.Equal(original.Key, Board(original.ToGrid()).Key);
    }
}
