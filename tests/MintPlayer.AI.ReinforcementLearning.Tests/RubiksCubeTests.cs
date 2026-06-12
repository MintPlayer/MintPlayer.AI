using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class FaceletCubeTests
{
    [Fact]
    public void SolvedCube_HasCanonicalKociembaString()
    {
        Assert.Equal("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
            new FaceletCube().ToKociembaString());
        Assert.True(new FaceletCube().IsSolved);
    }

    [Theory]
    [InlineData("U")]
    [InlineData("D")]
    [InlineData("L")]
    [InlineData("R")]
    [InlineData("F")]
    [InlineData("B")]
    public void FourQuarterTurns_AreIdentity(string move)
    {
        var cube = new FaceletCube();
        for (int i = 0; i < 4; i++)
            cube.Apply(move);
        Assert.True(cube.IsSolved);
    }

    [Theory]
    [InlineData("U")]
    [InlineData("R'")]
    [InlineData("F2")]
    public void MoveThenInverse_IsIdentity(string move)
    {
        var cube = new FaceletCube();
        cube.Apply(move);
        Assert.False(cube.IsSolved);
        cube.Apply(FaceletCube.InverseMove(move));
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public void HalfTurn_EqualsTwoQuarterTurns()
    {
        var half = new FaceletCube();
        half.Apply("F2");
        var quarters = new FaceletCube();
        quarters.Apply(["F", "F"]);
        Assert.Equal(half.ToKociembaString(), quarters.ToKociembaString());
    }

    // The standard "sexy move" (R U R' U') has order 6 on the cube group.
    [Fact]
    public void SexyMove_HasOrderSix()
    {
        var cube = new FaceletCube();
        for (int i = 0; i < 6; i++)
        {
            cube.Apply(["R", "U", "R'", "U'"]);
            Assert.Equal(i == 5, cube.IsSolved);
        }
    }

    // Hand-checkable single move: U cycles the top strips of the side faces F←R←B←L←F,
    // so after U the front face's top row shows the RIGHT face's color (red).
    [Fact]
    public void UMove_MovesRightTopRowToFront()
    {
        var cube = new FaceletCube();
        cube.Apply("U");
        string kociemba = cube.ToKociembaString();
        // Facelet order U(0-8) R(9-17) F(18-26): F top row = indices 18,19,20.
        Assert.Equal("RRR", kociemba.Substring(18, 3));
        // R top row shows the BACK face, B top row shows LEFT, L top row shows FRONT.
        Assert.Equal("BBB", kociemba.Substring(9, 3));
        Assert.Equal("LLL", kociemba.Substring(45, 3));
        Assert.Equal("FFF", kociemba.Substring(36, 3));
        // U face itself only rotates in place — still all U stickers.
        Assert.Equal("UUUUUUUUU", kociemba[..9]);
    }

    [Fact]
    public void ScrambleThenInverse_RestoresSolvedCube()
    {
        var rng = new Xoshiro256StarStar(123);
        var scramble = FaceletCube.ScrambleMoves(rng, 25);
        var cube = new FaceletCube();
        cube.Apply(scramble);
        Assert.False(cube.IsSolved);

        for (int i = scramble.Count - 1; i >= 0; i--)
            cube.Apply(FaceletCube.InverseMove(scramble[i]));
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public void Scramble_AvoidsConsecutiveSameFaceMoves()
    {
        var rng = new Xoshiro256StarStar(7);
        var scramble = FaceletCube.ScrambleMoves(rng, 200, quarterTurnsOnly: true);
        for (int i = 1; i < scramble.Count; i++)
            Assert.NotEqual(scramble[i][0], scramble[i - 1][0]);
        Assert.All(scramble, m => Assert.Contains(m, FaceletCube.QuarterTurnMoves));
    }

    [Fact]
    public void ColorFaces_RoundTrip()
    {
        var rng = new Xoshiro256StarStar(42);
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(rng, 20));

        var faces = cube.ToColorFaces();
        var restored = FaceletCube.FromColorFaces(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]);
        Assert.Equal(cube.ToKociembaString(), restored.ToKociembaString());
    }

    [Fact]
    public void FromColorFaces_RejectsBadInput()
    {
        var solved = new FaceletCube().ToColorFaces();
        Assert.Throws<ArgumentException>(() =>
            FaceletCube.FromColorFaces(solved[0], solved[1], solved[2], solved[3], solved[4], ["W", "W", "W"]));
        solved[2][4] = "X";
        Assert.Throws<ArgumentException>(() =>
            FaceletCube.FromColorFaces(solved[0], solved[1], solved[2], solved[3], solved[4], solved[5]));
    }
}

public class CubeValidationTests
{
    [Fact]
    public void AnyScrambledCube_IsStructurallyValid()
    {
        var rng = new Xoshiro256StarStar(99);
        for (int round = 0; round < 10; round++)
        {
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, 30));
            Assert.Null(CubeValidation.FindStructuralError(cube));
        }
    }

    [Fact]
    public void RepaintedSticker_YieldsEdgeAndCornerDiagnostics()
    {
        // Repaint one U-face corner sticker green: the UBL corner becomes impossible
        // (and the real W-O-B corner goes missing).
        var faces = new FaceletCube().ToColorFaces();
        faces[0][0] = "G";
        var cube = FaceletCube.FromColorFaces(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]);

        string? error = CubeValidation.FindStructuralError(cube);
        Assert.NotNull(error);
        Assert.Contains("Invalid corners", error);
        Assert.Contains("UBL", error);
        Assert.Contains("Missing corners", error);
    }

    [Fact]
    public void SwappedEdgeColors_YieldsDuplicateAndMissingEdges()
    {
        // Repaint the UF edge's two stickers as a second UR edge (W stays, G→R):
        // UR is now duplicated and the W-G edge is missing.
        var faces = new FaceletCube().ToColorFaces();
        faces[2][1] = "R"; // F sticker of the UF edge
        var cube = FaceletCube.FromColorFaces(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]);

        string? error = CubeValidation.FindStructuralError(cube);
        Assert.NotNull(error);
        Assert.Contains("Duplicate edges", error);
        Assert.Contains("Missing edges", error);
    }
}

public class CubeSolverTests
{
    [Fact]
    public void SolvedCube_NeedsNoMoves()
    {
        var result = CubeSolver.Solve(new FaceletCube());
        Assert.True(result.Solved);
        Assert.Empty(result.Moves);
    }

    [Fact]
    public void StructurallyInvalidCube_GetsDetailedError()
    {
        var faces = new FaceletCube().ToColorFaces();
        faces[0][0] = "G";
        var cube = FaceletCube.FromColorFaces(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]);

        var result = CubeSolver.Solve(cube);
        Assert.False(result.Solved);
        Assert.StartsWith("Invalid cube:", result.Error);
    }

    // The full two-phase gate: random 20-move scrambles must come back solved in ≤ 22
    // moves, and applying the solution to OUR cube model must yield a solved cube —
    // cross-validating the FaceletCube move tables against the independent Kociemba port.
    [Fact]
    [Trait("Category", "Slow")]
    public void TwentyRandomScrambles_SolvedWithin22Moves()
    {
        var rng = new Xoshiro256StarStar(2026_06_12);
        for (int round = 0; round < 20; round++)
        {
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, 20));

            var result = CubeSolver.Solve(cube);
            Assert.True(result.Solved, $"round {round}: {result.Error}");
            Assert.InRange(result.Moves.Length, 1, CubeSolver.MaxDepth);

            cube.Apply(result.Moves);
            Assert.True(cube.IsSolved, $"round {round}: solution did not solve the cube");
        }
    }

    // One fast non-trivial solve stays in the default bucket so the Kociemba port is
    // exercised on every test run (first solve pays the in-memory table build once).
    [Fact]
    public void SingleScramble_SolvesAndAppliesBack()
    {
        var cube = new FaceletCube();
        cube.Apply(["R", "U", "F'", "D2", "L", "B'"]);

        var result = CubeSolver.Solve(cube);
        Assert.True(result.Solved, result.Error);

        cube.Apply(result.Moves);
        Assert.True(cube.IsSolved);
    }
}
