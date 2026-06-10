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

    /// <summary>
    /// Reorders a solution so commutable moves of the same vehicle group into one visible
    /// slide. BFS returns an arbitrary order among the equally-optimal solutions, which can
    /// split a 2-cell slide around unrelated moves (e.g. "R left 1 … 8 other moves … R left 1"
    /// when delaying the first move would allow a single "R left 2"). Length and legality
    /// are preserved — only the presentation improves.
    /// </summary>
    public static int[] CompactSolution(RushHourPuzzle puzzle, int[] actions)
    {
        var result = (int[])actions.Clone();
        bool changed = true;
        while (changed)
        {
            changed = false;
            var runs = Runs(result);
            for (int r = 0; r < runs.Count - 1 && !changed; r++)
            {
                for (int s = r + 1; s < runs.Count; s++)
                {
                    if (runs[s].Action != runs[r].Action) continue;
                    // Delay run r to just before run s, or advance run s to just after run r.
                    changed = TryMoveBlock(puzzle, result, runs[r].Start, runs[r].Length, runs[s].Start - runs[r].Length)
                           || TryMoveBlock(puzzle, result, runs[s].Start, runs[s].Length, runs[r].Start + runs[r].Length);
                    break; // only the nearest same-action run can merge with r
                }
            }
        }
        return result;
    }

    private static List<(int Start, int Length, int Action)> Runs(int[] actions)
    {
        var runs = new List<(int, int, int)>();
        for (int i = 0; i < actions.Length;)
        {
            int start = i;
            while (i < actions.Length && actions[i] == actions[start]) i++;
            runs.Add((start, i - start, actions[start]));
        }
        return runs;
    }

    /// <summary>Moves the block [start, start+length) so it begins at <paramref name="destination"/> (post-removal index).</summary>
    private static bool TryMoveBlock(RushHourPuzzle puzzle, int[] actions, int start, int length, int destination)
    {
        if (destination == start) return false;
        var candidate = new int[actions.Length];
        var block = actions.AsSpan(start, length);
        var rest = new int[actions.Length - length];
        actions.AsSpan(0, start).CopyTo(rest);
        actions.AsSpan(start + length).CopyTo(rest.AsSpan(start));

        rest.AsSpan(0, destination).CopyTo(candidate);
        block.CopyTo(candidate.AsSpan(destination));
        rest.AsSpan(destination).CopyTo(candidate.AsSpan(destination + length));

        if (SlideCount(candidate) >= SlideCount(actions) || !IsValidSolution(puzzle, candidate))
            return false;
        candidate.CopyTo(actions.AsSpan());
        return true;
    }

    /// <summary>Number of visible slides = maximal runs of the same action (classic Rush Hour counts moves this way).</summary>
    public static int SlideCount(ReadOnlySpan<int> actions)
    {
        int runs = actions.Length > 0 ? 1 : 0;
        for (int i = 1; i < actions.Length; i++)
            if (actions[i] != actions[i - 1]) runs++;
        return runs;
    }

    /// <summary>Every step legal, solved exactly at the final step and not before.</summary>
    private static bool IsValidSolution(RushHourPuzzle puzzle, int[] actions)
    {
        var positions = RushHourBoard.InitialPositions(puzzle);
        Span<int> grid = stackalloc int[36];
        for (int k = 0; k < actions.Length; k++)
        {
            int vehicle = actions[k] / 2, direction = actions[k] % 2;
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction))
                return false;
            positions[vehicle] += direction == 0 ? -1 : 1;
            if (RushHourBoard.IsSolved(puzzle, positions) != (k == actions.Length - 1))
                return false;
        }
        return actions.Length > 0;
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
