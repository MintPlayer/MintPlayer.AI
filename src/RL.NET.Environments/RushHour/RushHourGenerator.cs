using RLNet.Core.Random;

namespace RLNet.Environments.RushHour;

/// <summary>
/// Generates solvable puzzles with optimal length inside a difficulty band, by random
/// layout + BFS filtering. Fully deterministic from the seed, so a puzzle set is
/// reproducible data without fixture files. Horizontal vehicles never spawn on the exit
/// row (a horizontal blocker there can render the puzzle trivially unsolvable).
/// </summary>
public static class RushHourGenerator
{
    public static List<RushHourPuzzle> Generate(
        ulong seed, int count, int minOptimal, int maxOptimal,
        int minVehicles = 5, int maxVehicles = 9, int maxAttempts = 200_000)
    {
        var rng = new Xoshiro256StarStar(seed);
        var puzzles = new List<RushHourPuzzle>(count);

        for (int attempt = 0; attempt < maxAttempts && puzzles.Count < count; attempt++)
        {
            var puzzle = TryRandomLayout(rng);
            if (puzzle is null) continue;

            int optimal = RushHourSolver.Solve(puzzle);
            if (optimal >= minOptimal && optimal <= maxOptimal)
                puzzles.Add(new RushHourPuzzle(puzzle.Vehicles, optimal));
        }

        if (puzzles.Count < count)
            throw new InvalidOperationException(
                $"Only generated {puzzles.Count}/{count} puzzles in band [{minOptimal},{maxOptimal}] after {maxAttempts} attempts.");
        return puzzles;

        RushHourPuzzle? TryRandomLayout(Xoshiro256StarStar rng)
        {
            var occupied = new bool[36];
            var vehicles = new List<Vehicle>();

            // Red car: horizontal length 2 on the exit row, away from the exit.
            int redCol = rng.NextInt(3);
            vehicles.Add(new Vehicle(RushHourBoard.ExitRow, redCol, 2, Horizontal: true));
            occupied[RushHourBoard.ExitRow * 6 + redCol] = occupied[RushHourBoard.ExitRow * 6 + redCol + 1] = true;

            int targetCount = minVehicles + rng.NextInt(maxVehicles - minVehicles + 1);
            for (int tries = 0; vehicles.Count < targetCount && tries < 60; tries++)
            {
                bool horizontal = rng.NextDouble() < 0.5;
                int length = rng.NextDouble() < 0.6 ? 2 : 3;
                int row = horizontal
                    ? (rng.NextInt(5) is var r && r >= RushHourBoard.ExitRow ? r + 1 : r) // skip exit row
                    : rng.NextInt(6 - length + 1);
                int col = horizontal ? rng.NextInt(6 - length + 1) : rng.NextInt(6);

                bool free = true;
                for (int k = 0; k < length && free; k++)
                    free = !occupied[(row + (horizontal ? 0 : k)) * 6 + col + (horizontal ? k : 0)];
                if (!free) continue;

                for (int k = 0; k < length; k++)
                    occupied[(row + (horizontal ? 0 : k)) * 6 + col + (horizontal ? k : 0)] = true;
                vehicles.Add(new Vehicle(row, col, length, horizontal));
            }

            return vehicles.Count >= minVehicles ? new RushHourPuzzle([.. vehicles]) : null;
        }
    }
}
