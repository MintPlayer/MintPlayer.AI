namespace RLNet.Environments.RushHour;

/// <summary>
/// Breadth-first search over vehicle positions — the exact oracle for Rush Hour.
/// States are encoded as 4 bits per vehicle (positions are 0..5), so a puzzle of up to
/// 16 vehicles fits one ulong. Returns the optimal number of single-cell moves,
/// or −1 if unsolvable (or the state cap is hit).
/// </summary>
public static class RushHourSolver
{
    public static int Solve(RushHourPuzzle puzzle, int maxStates = 2_000_000)
        => Solve(puzzle, maxStates, out _);

    /// <param name="solution">Optimal action sequence (vehicle·2+direction), empty if unsolvable.</param>
    public static int Solve(RushHourPuzzle puzzle, int maxStates, out int[] solution)
    {
        int n = puzzle.Vehicles.Length;
        var start = RushHourBoard.InitialPositions(puzzle);

        var queue = new Queue<int[]>();
        var visited = new Dictionary<ulong, (ulong Parent, int Action)> { [Encode(start)] = (ulong.MaxValue, -1) };
        queue.Enqueue(start);

        Span<int> grid = stackalloc int[36];
        var depths = new Dictionary<ulong, int> { [Encode(start)] = 0 };

        while (queue.Count > 0 && visited.Count <= maxStates)
        {
            var positions = queue.Dequeue();
            ulong key = Encode(positions);
            int depth = depths[key];

            if (RushHourBoard.IsSolved(puzzle, positions))
            {
                solution = ReconstructPath(visited, key);
                return depth;
            }

            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            for (int vehicle = 0; vehicle < n; vehicle++)
            {
                for (int direction = 0; direction <= 1; direction++)
                {
                    if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction)) continue;

                    var next = (int[])positions.Clone();
                    next[vehicle] += direction == 0 ? -1 : 1;
                    ulong nextKey = Encode(next);
                    if (visited.ContainsKey(nextKey)) continue;

                    visited[nextKey] = (key, vehicle * 2 + direction);
                    depths[nextKey] = depth + 1;
                    queue.Enqueue(next);
                }
            }
        }

        solution = [];
        return -1;
    }

    private static ulong Encode(ReadOnlySpan<int> positions)
    {
        ulong key = 0;
        for (int i = 0; i < positions.Length; i++)
            key |= (ulong)positions[i] << (i * 4);
        return key;
    }

    private static int[] ReconstructPath(Dictionary<ulong, (ulong Parent, int Action)> visited, ulong goal)
    {
        var actions = new List<int>();
        ulong key = goal;
        while (visited[key].Action >= 0)
        {
            actions.Add(visited[key].Action);
            key = visited[key].Parent;
        }
        actions.Reverse();
        return [.. actions];
    }
}
