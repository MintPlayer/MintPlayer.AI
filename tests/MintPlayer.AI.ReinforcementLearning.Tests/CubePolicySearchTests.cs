using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Guards the move-canonicalization that keeps the cube solver from emitting degenerate sequences like the
/// reported <c>U'×5</c>. Action encoding (FaceletCube.QuarterTurnMoves): U=0 U'=1 D=2 D'=3 L=4 L'=5 R=6 R'=7
/// F=8 F'=9 B=10 B'=11.
/// </summary>
public class CubePolicySearchTests
{
    [Fact]
    public void Canonicalize_FivePrimeTurns_CollapseToOne()
    {
        // U'×5 ≡ U' (the reported bug). U×5 ≡ U.
        Assert.Equal(new[] { "U'" }, CubePolicySearch.Canonicalize([1, 1, 1, 1, 1]));
        Assert.Equal(new[] { "U" }, CubePolicySearch.Canonicalize([0, 0, 0, 0, 0]));
    }

    [Fact]
    public void Canonicalize_FullTurnAndInversePair_Vanish()
    {
        Assert.Empty(CubePolicySearch.Canonicalize([0, 0, 0, 0])); // U×4 = identity
        Assert.Empty(CubePolicySearch.Canonicalize([0, 1]));       // U U' cancels
    }

    [Fact]
    public void Canonicalize_TwoSameTurns_KeptAsHalfTurn()
    {
        // No U2 token in the quarter-turn move set, so a half-turn stays two quarter-turns.
        Assert.Equal(new[] { "U", "U" }, CubePolicySearch.Canonicalize([0, 0]));
    }

    [Fact]
    public void Canonicalize_CancellationReExposesNeighbour()
    {
        // U F F' U → U (F F') U → U U
        Assert.Equal(new[] { "U", "U" }, CubePolicySearch.Canonicalize([0, 8, 9, 0]));
    }

    [Fact]
    public void Canonicalize_DistinctFaces_Unchanged()
    {
        // U D U' does not reduce (the faces between block it).
        Assert.Equal(new[] { "U", "D", "U'" }, CubePolicySearch.Canonicalize([0, 2, 1]));
    }

    [Fact]
    public void Canonicalize_Output_HasNoRedundantRuns()
    {
        // A deliberately degenerate path: U'×7, then B B, then R R R.
        var result = CubePolicySearch.Canonicalize([1, 1, 1, 1, 1, 1, 1, 10, 10, 6, 6, 6]);

        for (int i = 0; i + 2 < result.Length; i++)
            Assert.False(result[i] == result[i + 1] && result[i + 1] == result[i + 2],
                $"three consecutive identical moves at {i}: {string.Join(" ", result)}");

        for (int i = 0; i + 1 < result.Length; i++)
            Assert.False(Face(result[i]) == Face(result[i + 1]) && Prime(result[i]) != Prime(result[i + 1]),
                $"a move immediately followed by its inverse at {i}: {string.Join(" ", result)}");
    }

    private static char Face(string move) => move[0];
    private static bool Prime(string move) => move.EndsWith('\'');
}
