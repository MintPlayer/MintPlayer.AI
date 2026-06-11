namespace MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

/// <summary>
/// Exact supervision for imitation learning: enumerates a puzzle's ENTIRE reachable
/// state graph (forward BFS), then computes every state's true distance-to-goal
/// (multi-source backward BFS from all solved states — sliding moves are reversible,
/// so the graph is undirected) and one optimal action per state. One mid-size config
/// yields thousands of perfectly-labeled states across all depths, including depths
/// far beyond what random puzzle GENERATION can reach — deep states are common in
/// graphs even when deep START states are rare.
/// </summary>
public static class RushHourOracle
{
    /// <summary>
    /// <paramref name="OptimalActionsMask"/> has a bit set for EVERY action stepping one
    /// closer to the goal — most states have several. Training cross-entropy against a
    /// single arbitrary representative penalizes the other equally-optimal actions and
    /// flattens the policy; supervise against the full set instead.
    /// <paramref name="OptimalAction"/> is the lowest-indexed one, kept for convenience.
    /// </summary>
    public readonly record struct LabeledState(int[] Positions, int OptimalAction, uint OptimalActionsMask, int DistanceToGoal);

    /// <summary>
    /// Labels every reachable, solvable, not-yet-solved state. Returns null when the
    /// graph exceeds <paramref name="maxStates"/> or the goal is unreachable.
    /// </summary>
    public static List<LabeledState>? LabelReachableStates(RushHourPuzzle puzzle, int maxStates = 200_000)
    {
        int n = puzzle.Vehicles.Length;
        var start = RushHourBoard.InitialPositions(puzzle);
        var states = new Dictionary<ulong, int[]> { [RushHourSolver.Encode(start)] = start };
        var queue = new Queue<int[]>();
        queue.Enqueue(start);
        Span<int> grid = stackalloc int[36];

        // Forward BFS: enumerate the reachable component.
        while (queue.Count > 0)
        {
            var positions = queue.Dequeue();
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            for (int vehicle = 0; vehicle < n; vehicle++)
                for (int direction = 0; direction <= 1; direction++)
                {
                    if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction)) continue;
                    var next = (int[])positions.Clone();
                    next[vehicle] += direction == 0 ? -1 : 1;
                    if (states.TryAdd(RushHourSolver.Encode(next), next))
                    {
                        if (states.Count > maxStates) return null;
                        queue.Enqueue(next);
                    }
                }
        }

        // Multi-source backward BFS from every solved state (undirected graph).
        var distance = new Dictionary<ulong, int>();
        var frontier = new Queue<int[]>();
        foreach (var (key, positions) in states)
            if (RushHourBoard.IsSolved(puzzle, positions))
            {
                distance[key] = 0;
                frontier.Enqueue(positions);
            }
        if (distance.Count == 0) return null; // unsolvable component

        while (frontier.Count > 0)
        {
            var positions = frontier.Dequeue();
            int d = distance[RushHourSolver.Encode(positions)];
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            for (int vehicle = 0; vehicle < n; vehicle++)
                for (int direction = 0; direction <= 1; direction++)
                {
                    if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction)) continue;
                    var next = (int[])positions.Clone();
                    next[vehicle] += direction == 0 ? -1 : 1;
                    ulong nextKey = RushHourSolver.Encode(next);
                    if (!states.ContainsKey(nextKey) || distance.ContainsKey(nextKey)) continue;
                    distance[nextKey] = d + 1;
                    frontier.Enqueue(next);
                }
        }

        // Label: for each solvable non-goal state, every action stepping one closer.
        var labeled = new List<LabeledState>(distance.Count);
        foreach (var (key, d) in distance)
        {
            if (d == 0) continue;
            var positions = states[key];
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            uint mask = 0;
            for (int vehicle = 0; vehicle < n; vehicle++)
                for (int direction = 0; direction <= 1; direction++)
                {
                    if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction)) continue;
                    var next = (int[])positions.Clone();
                    next[vehicle] += direction == 0 ? -1 : 1;
                    if (distance.TryGetValue(RushHourSolver.Encode(next), out int nd) && nd == d - 1)
                        mask |= 1u << (vehicle * 2 + direction);
                }
            if (mask != 0)
                labeled.Add(new LabeledState(positions, System.Numerics.BitOperations.TrailingZeroCount(mask), mask, d));
        }
        return labeled;
    }
}
