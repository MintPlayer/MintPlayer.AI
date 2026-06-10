using RLNet.Environments.RushHour;

namespace RLNet.Tests;

public class RushHourBoardTests
{
    // Red car at (2,0)-(2,1); a vertical truck in column 2 spanning rows 0-2 blocks it.
    // Hand-solvable: the truck needs 3 down-moves (rows 3-5 is the only place a length-3
    // vertical clears row 2), then the red car needs 4 right-moves. Optimal = 7.
    private static RushHourPuzzle BlockedByTruck() => new([
        new Vehicle(2, 0, 2, Horizontal: true),
        new Vehicle(0, 2, 3, Horizontal: false),
    ]);

    [Fact]
    public void Solver_FindsHandComputedOptimal()
    {
        Assert.Equal(7, RushHourSolver.Solve(BlockedByTruck()));
    }

    [Fact]
    public void Solver_ReturnsSolutionPath_ThatActuallySolves()
    {
        var puzzle = BlockedByTruck();
        int optimal = RushHourSolver.Solve(puzzle, 1_000_000, out var solution);
        Assert.Equal(optimal, solution.Length);

        var positions = RushHourBoard.InitialPositions(puzzle);
        foreach (int action in solution)
        {
            Assert.True(RushHourBoard.ActionMask(puzzle, positions)[action], "solution contains an illegal move");
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
        }
        Assert.True(RushHourBoard.IsSolved(puzzle, positions));
    }

    // A dense 14-vehicle expert board (transcribed from a photo of the physical game):
    //   ..ABBC
    //   ..A.DC
    //   .XX.DC   <- X = red car, exit at the right of this row
    //   EFGHHH
    //   EFGIJJ
    //   EKKILL
    private static RushHourPuzzle DenseExpertBoard() => new([
        new Vehicle(2, 1, 2, Horizontal: true),  // X (red)
        new Vehicle(0, 2, 2, Horizontal: false), // A
        new Vehicle(0, 3, 2, Horizontal: true),  // B
        new Vehicle(0, 5, 3, Horizontal: false), // C — parked ON the exit cell
        new Vehicle(1, 4, 2, Horizontal: false), // D
        new Vehicle(3, 0, 3, Horizontal: false), // E
        new Vehicle(3, 1, 2, Horizontal: false), // F
        new Vehicle(3, 2, 2, Horizontal: false), // G
        new Vehicle(3, 3, 3, Horizontal: true),  // H
        new Vehicle(4, 3, 2, Horizontal: false), // I
        new Vehicle(4, 4, 2, Horizontal: true),  // J
        new Vehicle(5, 1, 2, Horizontal: true),  // K
        new Vehicle(5, 4, 2, Horizontal: true),  // L
    ]);

    [Fact]
    public void Solver_SolvesDenseExpertBoard_In56SingleCellMoves()
    {
        int optimal = RushHourSolver.Solve(DenseExpertBoard(), 2_000_000, out var solution);
        Assert.Equal(56, optimal);

        var puzzle = DenseExpertBoard();
        var positions = RushHourBoard.InitialPositions(puzzle);
        foreach (int action in solution)
        {
            Assert.True(RushHourBoard.ActionMask(puzzle, positions)[action]);
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
        }
        Assert.True(RushHourBoard.IsSolved(puzzle, positions));
    }

    // Official ThinkFun Rush Hour card #40 — the hardest card of the original deck:
    //   OAA.B.
    //   OCD.BP
    //   OCDXXP   <- X = red car, exit at the right of this row
    //   QQQE.P
    //   ..FEGG
    //   HHFII.
    // Letter order below fixes the vehicle indices used by the solution-replay test.
    private const string Level40Letters = "XOABCDPQEFGHI";

    private static RushHourPuzzle OfficialCard40() => new([
        new Vehicle(2, 3, 2, Horizontal: true),  // X (red)
        new Vehicle(0, 0, 3, Horizontal: false), // O
        new Vehicle(0, 1, 2, Horizontal: true),  // A
        new Vehicle(0, 4, 2, Horizontal: false), // B
        new Vehicle(1, 1, 2, Horizontal: false), // C
        new Vehicle(1, 2, 2, Horizontal: false), // D
        new Vehicle(1, 5, 3, Horizontal: false), // P — parked on the exit cell
        new Vehicle(3, 0, 3, Horizontal: true),  // Q
        new Vehicle(3, 3, 2, Horizontal: false), // E
        new Vehicle(4, 2, 2, Horizontal: false), // F
        new Vehicle(4, 4, 2, Horizontal: true),  // G
        new Vehicle(5, 0, 2, Horizontal: true),  // H
        new Vehicle(5, 3, 2, Horizontal: true),  // I
    ]);

    [Fact]
    public void Solver_SolvesOfficialCard40_In81SingleCellMoves()
    {
        Assert.Equal(81, RushHourSolver.Solve(OfficialCard40(), 2_000_000));
    }

    [Fact]
    public void OfficialCard40_PublishedSolution_ReplaysLegallyAndSolves()
    {
        // The card's printed solution: 51 piece-moves; the final XR3 includes 2 cells of
        // driving out through the exit, so it solves in 81 single-cell moves — exactly
        // the BFS optimum (all three expert cards' printed solutions are optimal).
        const string published =
            "PU1 IR1 ED1 QR3 FU1 HR1 OD3 AL1 DU1 CD2 " +
            "XL3 BD1 DD1 AR3 DU1 XR2 OU3 HL1 FD1 CU3 " +
            "QL3 PD1 XL1 AR1 EU4 QR2 XR1 CD3 XL1 ED1 " +
            "AL1 PU1 QR1 FU1 HR1 OD3 XL1 DD1 AL3 BU1 " +
            "EU1 DU1 XR3 OU1 HL1 FD1 IL1 GL1 QL1 PD3 " +
            "XR3";

        AssertPublishedSolutionSolves(OfficialCard40(), Level40Letters, published, expectedSingleCellMoves: 81);
    }

    /// <summary>
    /// Replays a card solution in official notation (letter + U/D/L/R + distance),
    /// asserting every single-cell slide is legal on our board. A final move may drive
    /// the red car out through the exit; counting stops once the board is solved.
    /// </summary>
    private static void AssertPublishedSolutionSolves(
        RushHourPuzzle puzzle, string letters, string published, int expectedSingleCellMoves)
    {
        var positions = RushHourBoard.InitialPositions(puzzle);
        int singleCellMoves = 0;
        bool solved = false;

        foreach (string token in published.Split(' '))
        {
            Assert.False(solved, $"solution continues with '{token}' after the puzzle was already solved");
            int vehicle = letters.IndexOf(token[0]);
            Assert.True(vehicle >= 0, $"unknown vehicle letter in '{token}'");
            int direction = token[1] is 'U' or 'L' ? 0 : 1;
            int distance = int.Parse(token[2..]);

            for (int k = 0; k < distance && !solved; k++)
            {
                Assert.True(RushHourBoard.ActionMask(puzzle, positions)[vehicle * 2 + direction],
                    $"published solution move '{token}' (cell {k + 1}/{distance}) is illegal on our board");
                positions[vehicle] += direction == 0 ? -1 : 1;
                singleCellMoves++;
                solved = RushHourBoard.IsSolved(puzzle, positions);
            }
        }

        Assert.True(solved, "published solution does not solve the puzzle");
        Assert.Equal(expectedSingleCellMoves, singleCellMoves);
    }

    // Official ThinkFun Rush Hour card #38:
    //   A..OOO
    //   ABBC..
    //   XXDC.R   <- X = red car, exit at the right of this row
    //   ..DEER
    //   ..FGGR
    //   ..FQQQ
    private const string Level38Letters = "XAOBCDREFGQ";

    private static RushHourPuzzle OfficialCard38() => new([
        new Vehicle(2, 0, 2, Horizontal: true),  // X (red)
        new Vehicle(0, 0, 2, Horizontal: false), // A
        new Vehicle(0, 3, 3, Horizontal: true),  // O
        new Vehicle(1, 1, 2, Horizontal: true),  // B
        new Vehicle(1, 3, 2, Horizontal: false), // C
        new Vehicle(2, 2, 2, Horizontal: false), // D
        new Vehicle(2, 5, 3, Horizontal: false), // R — parked on the exit cell
        new Vehicle(3, 3, 2, Horizontal: true),  // E
        new Vehicle(4, 2, 2, Horizontal: false), // F
        new Vehicle(4, 3, 2, Horizontal: true),  // G
        new Vehicle(5, 3, 3, Horizontal: true),  // Q
    ]);

    [Fact]
    public void Solver_SolvesOfficialCard38_In77SingleCellMoves()
    {
        Assert.Equal(77, RushHourSolver.Solve(OfficialCard38(), 2_000_000));
    }

    [Fact]
    public void OfficialCard38_PublishedSolution_ReplaysLegallyAndSolves()
    {
        // Like card 39, the printed solution is single-cell optimal (77; the final XR6
        // includes 2 cells of driving out through the exit).
        const string published =
            "OL1 RU2 GR1 ER1 CD2 BR2 DU1 FU1 QL3 CD1 " +
            "EL1 RD1 OR1 DU1 XR1 AD3 XL1 DD1 OL1 RU1 " +
            "ER1 CU2 QR3 FD1 DD1 BL3 CU1 DU1 AD1 EL4 " +
            "RD1 CD1 DD1 BR3 OR1 DU2 ER1 XR1 AU4 EL1 " +
            "XL1 FU2 QL1 GL4 RD2 CD1 FD1 XR6";

        AssertPublishedSolutionSolves(OfficialCard38(), Level38Letters, published, expectedSingleCellMoves: 77);
    }

    // Official ThinkFun Rush Hour card #39:
    //   ..AOOO
    //   ..AB..
    //   XXCB.R   <- X = red car, exit at the right of this row
    //   DDCEER
    //   FGHH.R
    //   FGII..
    private const string Level39Letters = "XAOBCRDEFGHI";

    private static RushHourPuzzle OfficialCard39() => new([
        new Vehicle(2, 0, 2, Horizontal: true),  // X (red)
        new Vehicle(0, 2, 2, Horizontal: false), // A
        new Vehicle(0, 3, 3, Horizontal: true),  // O
        new Vehicle(1, 3, 2, Horizontal: false), // B
        new Vehicle(2, 2, 2, Horizontal: false), // C
        new Vehicle(2, 5, 3, Horizontal: false), // R — parked on the exit cell
        new Vehicle(3, 0, 2, Horizontal: true),  // D
        new Vehicle(3, 3, 2, Horizontal: true),  // E
        new Vehicle(4, 0, 2, Horizontal: false), // F
        new Vehicle(4, 1, 2, Horizontal: false), // G
        new Vehicle(4, 2, 2, Horizontal: true),  // H
        new Vehicle(5, 2, 2, Horizontal: true),  // I
    ]);

    [Fact]
    public void Solver_SolvesOfficialCard39_In82SingleCellMoves()
    {
        Assert.Equal(82, RushHourSolver.Solve(OfficialCard39(), 2_000_000));
    }

    [Fact]
    public void OfficialCard39_PublishedSolution_ReplaysLegallyAndSolves()
    {
        // The card's printed solution. Its final "XR6" drives the red car out through the
        // exit: the board is solved (nose at the edge) after 4 of those 6 cells, so the
        // published solution is single-cell OPTIMAL (82 = the BFS minimum).
        const string published =
            "IR2 HR1 CD2 AD1 OL1 RU2 HR1 ER1 BD3 EL1 " +
            "RD1 OR1 AU1 XR3 AD1 OL1 RU1 ER1 DR2 GU4 " +
            "FU4 DL2 EL1 RD1 OR1 AU1 XL3 AD1 OL1 RU1 " +
            "ER1 BU3 EL1 RD1 OR1 AU1 CU2 HL4 CD1 AD1 " +
            "OL1 RU1 IL3 ER1 BD3 EL1 RD3 OR1 AU1 XR6";

        var puzzle = OfficialCard39();
        AssertPublishedSolutionSolves(puzzle, Level39Letters, published, expectedSingleCellMoves: 82);
    }

    [Fact]
    public void Solver_DetectsUnsolvable()
    {
        // A horizontal car on the exit row to the red car's right can never leave the row.
        var puzzle = new RushHourPuzzle([
            new Vehicle(2, 0, 2, Horizontal: true),
            new Vehicle(2, 4, 2, Horizontal: true),
        ]);
        Assert.Equal(-1, RushHourSolver.Solve(puzzle));
    }

    [Fact]
    public void Mask_BlockedRedCar_CannotMove()
    {
        var puzzle = BlockedByTruck();
        var mask = RushHourBoard.ActionMask(puzzle, RushHourBoard.InitialPositions(puzzle));

        Assert.False(mask[0]); // red left: wall
        Assert.False(mask[1]); // red right: truck at (2,2)
        Assert.False(mask[2]); // truck up: wall
        Assert.True(mask[3]);  // truck down: (3,2) is free
        for (int a = 4; a < RushHourBoard.ActionCount; a++)
            Assert.False(mask[a]); // nonexistent vehicles are masked
    }

    [Fact]
    public void Puzzle_RejectsOverlapsAndMisplacedRed()
    {
        Assert.Throws<ArgumentException>(() => new RushHourPuzzle([
            new Vehicle(2, 0, 2, true),
            new Vehicle(2, 1, 2, false), // overlaps red at (2,1)
        ]));
        Assert.Throws<ArgumentException>(() => new RushHourPuzzle([
            new Vehicle(1, 0, 2, true), // red not on exit row
        ]));
    }

    [Fact]
    public void Generator_ProducesSolvablePuzzlesInBand_Deterministically()
    {
        var puzzles = RushHourGenerator.Generate(seed: 7, count: 10, minOptimal: 4, maxOptimal: 10);
        Assert.Equal(10, puzzles.Count);
        foreach (var p in puzzles)
        {
            Assert.InRange(p.OptimalMoves, 4, 10);
            Assert.Equal(p.OptimalMoves, RushHourSolver.Solve(p));
        }

        var again = RushHourGenerator.Generate(seed: 7, count: 10, minOptimal: 4, maxOptimal: 10);
        Assert.Equal(
            puzzles.Select(p => p.Vehicles).ToArray(),
            again.Select(p => p.Vehicles).ToArray());
    }

    [Fact]
    public void Generator_VariedRedLength_ProducesBothLengths_AndDefaultStaysLengthTwo()
    {
        var varied = RushHourGenerator.Generate(seed: 99, count: 30, minOptimal: 4, maxOptimal: 10, varyRedLength: true);
        Assert.Contains(varied, p => p.Vehicles[0].Length == 2);
        Assert.Contains(varied, p => p.Vehicles[0].Length == 3);

        // Off by default — and the same seed still yields red length 2 everywhere (M6 sets unchanged).
        var classic = RushHourGenerator.Generate(seed: 99, count: 30, minOptimal: 4, maxOptimal: 10);
        Assert.All(classic, p => Assert.Equal(2, p.Vehicles[0].Length));
    }
}

public class RushHourEnvTests
{
    [Fact]
    public void SolvingEpisode_TerminatesWithBonus()
    {
        var puzzle = new RushHourPuzzle([new Vehicle(2, 0, 2, true)], optimalMoves: 4);
        var env = new RushHourEnv([puzzle]);
        env.Reset(1);

        for (int i = 0; i < 3; i++)
        {
            var mid = env.Step(1); // red right (action = vehicle 0, dir 1)
            Assert.False(mid.Done);
            Assert.Equal(-1, mid.Reward);
        }
        var final = env.Step(1);
        Assert.True(final.Terminated);
        Assert.Equal(100, final.Reward);
    }

    [Fact]
    public void Truncates_WhenMoveBudgetExhausted()
    {
        var puzzle = new RushHourPuzzle([new Vehicle(2, 0, 2, true)], optimalMoves: 4);
        var env = new RushHourEnv([puzzle], maxMoves: 3);
        env.Reset(1);
        env.Step(1);
        env.Step(0); // back and forth, never solving
        var step = env.Step(1);
        Assert.True(step.Truncated);
        Assert.False(step.Terminated);
    }

    [Fact]
    public void FixedPuzzleIndex_PinsTheEpisode()
    {
        var puzzles = RushHourGenerator.Generate(seed: 7, count: 3, minOptimal: 4, maxOptimal: 10);
        var env = new RushHourEnv(puzzles) { FixedPuzzleIndex = 2 };
        env.Reset(1);
        Assert.Same(puzzles[2], env.CurrentPuzzle);
    }

    [Fact]
    public void ShapedReward_AddsRedCarProgress()
    {
        var puzzle = new RushHourPuzzle([new Vehicle(2, 0, 2, true)], optimalMoves: 4);
        var env = new RushHourEnv([puzzle], shapedReward: true);
        env.Reset(1);
        Assert.Equal(-1 + 2, env.Step(1).Reward); // red moves right one cell
        Assert.Equal(-1 - 2, env.Step(0).Reward); // and back
    }
}
