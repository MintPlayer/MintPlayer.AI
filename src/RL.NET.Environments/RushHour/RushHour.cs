namespace RLNet.Environments.RushHour;

/// <summary>A vehicle on the 6×6 board; Row/Col is its topmost/leftmost cell.</summary>
public readonly record struct Vehicle(int Row, int Col, int Length, bool Horizontal);

/// <summary>
/// An immutable Rush Hour puzzle: vehicle 0 is the red car (always horizontal, on the
/// exit row); the goal is sliding vehicles one cell at a time until the red car's nose
/// reaches the right edge. <see cref="OptimalMoves"/> counts single-cell moves (our BFS
/// metric — note classic Rush Hour counts multi-cell slides as one move).
/// </summary>
public sealed class RushHourPuzzle
{
    public RushHourPuzzle(Vehicle[] vehicles, int optimalMoves = -1)
    {
        if (vehicles.Length == 0 || vehicles.Length > RushHourBoard.MaxVehicles)
            throw new ArgumentException($"1..{RushHourBoard.MaxVehicles} vehicles required.");
        var red = vehicles[0];
        if (!red.Horizontal || red.Row != RushHourBoard.ExitRow)
            throw new ArgumentException("Vehicle 0 must be the red car: horizontal, on the exit row.");

        Span<int> grid = stackalloc int[36];
        grid.Fill(-1);
        for (int i = 0; i < vehicles.Length; i++)
        {
            var v = vehicles[i];
            for (int k = 0; k < v.Length; k++)
            {
                int row = v.Row + (v.Horizontal ? 0 : k);
                int col = v.Col + (v.Horizontal ? k : 0);
                if (row is < 0 or >= RushHourBoard.Size || col is < 0 or >= RushHourBoard.Size)
                    throw new ArgumentException($"Vehicle {i} is out of bounds.");
                if (grid[row * RushHourBoard.Size + col] >= 0)
                    throw new ArgumentException($"Vehicles {grid[row * RushHourBoard.Size + col]} and {i} overlap.");
                grid[row * RushHourBoard.Size + col] = i;
            }
        }

        Vehicles = vehicles;
        OptimalMoves = optimalMoves;
    }

    public Vehicle[] Vehicles { get; }

    /// <summary>Optimal solution length in single-cell moves (−1 = not computed).</summary>
    public int OptimalMoves { get; }
}

/// <summary>
/// Stateless operations on (puzzle, positions) pairs, where positions[i] is vehicle i's
/// variable coordinate (Col when horizontal, Row when vertical). Action encoding:
/// vehicle·2 + direction, direction 0 = toward smaller coordinates (left/up),
/// 1 = toward larger (right/down). Fixed 32-action space, illegal entries masked.
/// </summary>
public static class RushHourBoard
{
    public const int Size = 6;
    public const int ExitRow = 2;
    public const int MaxVehicles = 16;
    public const int ActionCount = MaxVehicles * 2;

    public static int[] InitialPositions(RushHourPuzzle puzzle)
        => [.. puzzle.Vehicles.Select(v => v.Horizontal ? v.Col : v.Row)];

    /// <summary>Fills a 36-cell grid with the occupying vehicle index (−1 = empty).</summary>
    public static void FillOccupancy(RushHourPuzzle puzzle, ReadOnlySpan<int> positions, Span<int> grid)
    {
        grid.Fill(-1);
        for (int i = 0; i < puzzle.Vehicles.Length; i++)
        {
            var v = puzzle.Vehicles[i];
            for (int k = 0; k < v.Length; k++)
            {
                int row = v.Horizontal ? v.Row : positions[i] + k;
                int col = v.Horizontal ? positions[i] + k : v.Col;
                grid[row * Size + col] = i;
            }
        }
    }

    public static bool CanMove(RushHourPuzzle puzzle, ReadOnlySpan<int> positions, ReadOnlySpan<int> grid,
        int vehicle, int direction)
    {
        if (vehicle >= puzzle.Vehicles.Length) return false;
        var v = puzzle.Vehicles[vehicle];
        int pos = positions[vehicle];

        if (direction == 0)
        {
            if (pos == 0) return false;
            int row = v.Horizontal ? v.Row : pos - 1;
            int col = v.Horizontal ? pos - 1 : v.Col;
            return grid[row * Size + col] < 0;
        }
        else
        {
            if (pos + v.Length > Size - 1) return false; // target cell would be off-board
            int row = v.Horizontal ? v.Row : pos + v.Length;
            int col = v.Horizontal ? pos + v.Length : v.Col;
            return grid[row * Size + col] < 0;
        }
    }

    public static bool[] ActionMask(RushHourPuzzle puzzle, ReadOnlySpan<int> positions)
    {
        Span<int> grid = stackalloc int[36];
        FillOccupancy(puzzle, positions, grid);

        var mask = new bool[ActionCount];
        for (int i = 0; i < puzzle.Vehicles.Length; i++)
        {
            mask[i * 2] = CanMove(puzzle, positions, grid, i, 0);
            mask[i * 2 + 1] = CanMove(puzzle, positions, grid, i, 1);
        }
        return mask;
    }

    /// <summary>The red car's nose is at the right edge.</summary>
    public static bool IsSolved(RushHourPuzzle puzzle, ReadOnlySpan<int> positions)
        => positions[0] + puzzle.Vehicles[0].Length - 1 == Size - 1;

    public const int ObservationSize = 72;

    /// <summary>
    /// The canonical 72-float observation: a vehicle-identity plane ((index+1)/16,
    /// 0 = empty) and a red-car occupancy plane. Shared by the env, the oracle
    /// dataset builder and policy inference so they can never drift.
    /// </summary>
    public static void WriteObservation(RushHourPuzzle puzzle, ReadOnlySpan<int> positions, Span<float> observation)
    {
        observation.Clear();
        Span<int> grid = stackalloc int[36];
        FillOccupancy(puzzle, positions, grid);
        for (int i = 0; i < 36; i++)
        {
            if (grid[i] >= 0) observation[i] = (grid[i] + 1) / (float)MaxVehicles;
            if (grid[i] == 0) observation[36 + i] = 1f;
        }
    }

    public static string Render(RushHourPuzzle puzzle, ReadOnlySpan<int> positions)
    {
        Span<int> grid = stackalloc int[36];
        FillOccupancy(puzzle, positions, grid);

        var sb = new System.Text.StringBuilder();
        for (int row = 0; row < Size; row++)
        {
            for (int col = 0; col < Size; col++)
            {
                int v = grid[row * Size + col];
                sb.Append(v < 0 ? '·' : v == 0 ? 'R' : (char)('A' + v - 1)).Append(' ');
            }
            sb.AppendLine(row == ExitRow ? "→ exit" : string.Empty);
        }
        return sb.ToString();
    }
}
