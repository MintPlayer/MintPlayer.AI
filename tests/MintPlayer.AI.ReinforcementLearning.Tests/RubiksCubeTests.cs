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

public class RubiksCubeEnvTests
{
    [Fact]
    public void Reset_GivesOneHotObservation()
    {
        var env = new RubiksCubeEnv(maxScrambleDepth: 6);
        var (obs, _) = env.Reset(7);

        Assert.Equal(RubiksCubeEnv.ObservationSize, obs.Length);
        for (int sticker = 0; sticker < FaceletCube.FaceletCount; sticker++)
        {
            var group = obs.AsSpan(sticker * FaceletCube.FaceCount, FaceletCube.FaceCount);
            Assert.Equal(1f, group.ToArray().Sum());
            Assert.All(group.ToArray(), v => Assert.True(v is 0f or 1f));
        }
    }

    [Fact]
    public void DepthOneScramble_IsSolvedByTheInverseMove()
    {
        var env = new RubiksCubeEnv { FixedScrambleDepth = 1 };
        for (ulong seed = 0; seed < 10; seed++)
        {
            env.Reset(seed);
            Assert.Single(env.ScrambleMoves);
            string inverse = FaceletCube.InverseMove(env.ScrambleMoves[0]);
            int action = Array.IndexOf(FaceletCube.QuarterTurnMoves, inverse);
            Assert.True(action >= 0);

            var step = env.Step(action);
            Assert.True(step.Terminated);
            Assert.False(step.Truncated);
            Assert.Equal(100, step.Reward);
        }
    }

    [Fact]
    public void NonSolvingMoves_CostOne_AndEpisodeTruncatesAtMaxMoves()
    {
        var env = new RubiksCubeEnv(maxScrambleDepth: 1, maxMoves: 5) { FixedScrambleDepth = 1 };
        env.Reset(3);
        // Alternate quarter turns of two faces that are neither the scrambled face nor each
        // other's inverse (mask-legal): no prefix of f1 g1 f1 g1 … can invert a one-move
        // scramble on a third face, so the episode can only end by truncation.
        char scrambledFace = env.ScrambleMoves[0][0];
        string[] faces = [.. new[] { "U", "F", "R" }.Where(f => f[0] != scrambledFace).Take(2)];
        int first = Array.IndexOf(FaceletCube.QuarterTurnMoves, faces[0]);
        int second = Array.IndexOf(FaceletCube.QuarterTurnMoves, faces[1]);

        for (int i = 0; i < 4; i++)
        {
            var step = env.Step(i % 2 == 0 ? first : second);
            Assert.Equal(-1, step.Reward);
            Assert.False(step.Terminated);
            Assert.False(step.Truncated);
        }
        var last = env.Step(first);
        Assert.False(last.Terminated);
        Assert.True(last.Truncated);
        Assert.Throws<InvalidOperationException>(() => env.Step(first));
    }

    [Fact]
    public void ActionMask_ForbidsOnlyTheInverseOfTheLastMove()
    {
        Assert.All(RubiksCubeEnv.ActionMask(-1), legal => Assert.True(legal));

        var env = new RubiksCubeEnv { FixedScrambleDepth = 3 };
        env.Reset(5);
        int action = Array.IndexOf(FaceletCube.QuarterTurnMoves, "F");
        env.Step(action);

        var mask = env.CurrentActionMask();
        int inverse = Array.IndexOf(FaceletCube.QuarterTurnMoves, "F'");
        for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
            Assert.Equal(a != inverse, mask[a]);

        Assert.Throws<InvalidOperationException>(() => env.Step(inverse));
    }

    [Fact]
    public void ScrambleDepth_IsSampledWithinBand()
    {
        var env = new RubiksCubeEnv(maxScrambleDepth: 6);
        var depths = new HashSet<int>();
        for (ulong seed = 0; seed < 40; seed++)
        {
            env.Reset(seed);
            Assert.InRange(env.ScrambleDepth, 1, 6);
            depths.Add(env.ScrambleDepth);
        }
        Assert.True(depths.Count >= 4, "expected the curriculum to sample several depths");
    }
}

public class CubeImitationTests
{
    [Fact]
    public void ExpandToQuarterTurnActions_HandlesAllNotations()
    {
        int[] actions = CubeOracle.ExpandToQuarterTurnActions(["R", "U'", "F2"]);
        string[] moves = [.. actions.Select(a => FaceletCube.QuarterTurnMoves[a])];
        Assert.Equal(["R", "U'", "F", "F"], moves);
    }

    // The oracle's labels must actually solve: walking each labeled state's action from
    // the scrambled cube ends solved, and the distance-to-go counts down to 1.
    [Fact]
    public void LabelScramblePath_LabelsFormASolution()
    {
        var rng = new Xoshiro256StarStar(17);
        var path = CubeOracle.LabelScramblePath(rng, maxScrambleDepth: 12);
        Assert.NotNull(path);
        Assert.NotEmpty(path);

        var cube = FaceletCube.FromFacelets(path[0].Facelets);
        for (int i = 0; i < path.Count; i++)
        {
            Assert.Equal(path.Count - i, path[i].DistanceToGo);
            Assert.Equal(path[i].Facelets, cube.Facelets.ToArray());
            cube.ApplyQuarterTurn(path[i].Action);
        }
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public void CubePolicyNet_SaveLoad_RoundTripsInference()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(7), hidden: 32);
        var cube = new FaceletCube();
        cube.Apply(["R", "U", "F'"]);

        using var buffer = new MemoryStream();
        net.Save(buffer);
        buffer.Position = 0;
        var restored = CubePolicyNet.Load(buffer);

        var (logits, distance) = net.Evaluate(cube);
        var (logits2, distance2) = restored.Evaluate(cube);
        Assert.Equal(logits, logits2);
        Assert.Equal(distance, distance2);
    }

    [Fact]
    public void PolicyEvaluate_MasksTheUndoMove()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(7), hidden: 32);
        int action = Array.IndexOf(FaceletCube.QuarterTurnMoves, "R");
        var (logits, _) = net.Evaluate(new FaceletCube(), lastAction: action);
        Assert.True(float.IsNegativeInfinity(logits[RubiksCubeEnv.InverseAction(action)]));
        Assert.Equal(1, logits.Count(float.IsNegativeInfinity));
    }

    // Even an untrained net must solve depth 1-2 through the search (it enumerates the
    // neighborhood) — the contract that lookahead never makes the policy worse.
    [Fact]
    public void PolicySearch_SolvesShallowScrambles_EvenUntrained()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(3), hidden: 32);
        var cube = new FaceletCube();
        cube.Apply(["R", "U'"]);

        var result = CubePolicySearch.Solve(net, cube, maxExpansions: 5_000);
        Assert.True(result.Solved);

        cube.Apply(result.Moves);
        Assert.True(cube.IsSolved);
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
