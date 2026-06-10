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
