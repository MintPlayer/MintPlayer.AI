namespace MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

/// <summary>
/// Inference strategies on top of <see cref="RushHourPolicyNet"/>:
/// a reactive greedy rollout (with cycle avoidance), and policy-guided A* — the net's
/// value head is the heuristic, so the search expands only a few hundred to a few
/// thousand states where blind BFS needs hundreds of thousands. Search is what turns
/// a good-but-imperfect learned policy into a solver for expert-depth puzzles.
/// </summary>
public static class RushHourPolicySearch
{
    /// <summary>Reactive play: best legal logit leading to an unvisited state.</summary>
    public static (bool Solved, List<int> Actions) GreedyRollout(RushHourPolicyNet net, RushHourPuzzle puzzle, int maxMoves)
    {
        var positions = RushHourBoard.InitialPositions(puzzle);
        var visited = new HashSet<ulong> { RushHourSolver.Encode(positions) };
        var actions = new List<int>();

        for (int move = 0; move < maxMoves; move++)
        {
            if (RushHourBoard.IsSolved(puzzle, positions)) return (true, actions);
            var (logits, _) = net.Evaluate(puzzle, positions);

            int best = -1, fallback = -1;
            for (int a = 0; a < RushHourBoard.ActionCount; a++)
            {
                if (float.IsNegativeInfinity(logits[a])) continue;
                if (fallback < 0 || logits[a] > logits[fallback]) fallback = a;

                positions[a / 2] += a % 2 == 0 ? -1 : 1;
                bool fresh = !visited.Contains(RushHourSolver.Encode(positions));
                positions[a / 2] -= a % 2 == 0 ? -1 : 1;
                if (fresh && (best < 0 || logits[a] > logits[best])) best = a;
            }
            if (fallback < 0) break;

            int action = best >= 0 ? best : fallback;
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
            visited.Add(RushHourSolver.Encode(positions));
            actions.Add(action);
        }
        return (RushHourBoard.IsSolved(puzzle, positions), actions);
    }

    public sealed record SearchResult(bool Solved, int[] Actions, int Expansions);

    /// <summary>A* with the learned distance-to-goal as heuristic; budgeted by node expansions.</summary>
    public static SearchResult Solve(RushHourPolicyNet net, RushHourPuzzle puzzle, int maxExpansions = 100_000)
    {
        int n = puzzle.Vehicles.Length;
        var start = RushHourBoard.InitialPositions(puzzle);
        ulong startKey = RushHourSolver.Encode(start);

        var open = new PriorityQueue<ulong, float>();
        var nodes = new Dictionary<ulong, (int[] Positions, int G, ulong Parent, int Action)>
        {
            [startKey] = (start, 0, ulong.MaxValue, -1),
        };
        var closed = new HashSet<ulong>();
        open.Enqueue(startKey, net.Evaluate(puzzle, start).Distance);

        int expansions = 0;
        Span<int> grid = stackalloc int[36];

        while (open.Count > 0 && expansions < maxExpansions)
        {
            ulong key = open.Dequeue();
            if (!closed.Add(key)) continue; // stale queue entry
            var (positions, g, _, _) = nodes[key];

            if (RushHourBoard.IsSolved(puzzle, positions))
                return new SearchResult(true, ReconstructPath(nodes, key), expansions);

            expansions++;
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            for (int vehicle = 0; vehicle < n; vehicle++)
                for (int direction = 0; direction <= 1; direction++)
                {
                    if (!RushHourBoard.CanMove(puzzle, positions, grid, vehicle, direction)) continue;
                    var next = (int[])positions.Clone();
                    next[vehicle] += direction == 0 ? -1 : 1;
                    ulong nextKey = RushHourSolver.Encode(next);
                    if (closed.Contains(nextKey)) continue;

                    int nextG = g + 1;
                    if (nodes.TryGetValue(nextKey, out var existing) && existing.G <= nextG) continue;
                    nodes[nextKey] = (next, nextG, key, vehicle * 2 + direction);
                    open.Enqueue(nextKey, nextG + net.Evaluate(puzzle, next).Distance);
                }
        }
        return new SearchResult(false, [], expansions);
    }

    private static int[] ReconstructPath(Dictionary<ulong, (int[] Positions, int G, ulong Parent, int Action)> nodes, ulong goal)
    {
        var actions = new List<int>();
        ulong key = goal;
        while (nodes[key].Action >= 0)
        {
            actions.Add(nodes[key].Action);
            key = nodes[key].Parent;
        }
        actions.Reverse();
        return [.. actions];
    }
}
