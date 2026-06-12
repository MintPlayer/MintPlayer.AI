namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Inference strategies on top of <see cref="CubePolicyNet"/>, mirroring
/// <see cref="RushHour.RushHourPolicySearch"/>: a reactive greedy rollout (no-undo mask +
/// visited-state cycle avoidance) and policy-guided A* with the value head as heuristic.
/// Solutions are capped at <see cref="MaxSolutionMoves"/> quarter-turns — Kociemba stays
/// the oracle for guaranteed short answers; this is "the AI, with lookahead".
/// </summary>
public static class CubePolicySearch
{
    /// <summary>Kociemba ≤ 22 HTM ≈ ≤ 30–40 QTM; a learned solver gets the same generous cap.</summary>
    public const int MaxSolutionMoves = 40;

    /// <summary>Reactive play: best non-undo logit leading to an unvisited state.</summary>
    public static (bool Solved, List<int> Actions) GreedyRollout(CubePolicyNet net, FaceletCube start, int maxMoves = MaxSolutionMoves)
    {
        var cube = start.Clone();
        var visited = new HashSet<string> { cube.ToKociembaString() };
        var actions = new List<int>();
        int lastAction = -1;

        for (int move = 0; move < maxMoves; move++)
        {
            if (cube.IsSolved) return (true, actions);
            var (logits, _) = net.Evaluate(cube, lastAction);

            int best = -1, fallback = -1;
            for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
            {
                if (float.IsNegativeInfinity(logits[a])) continue;
                if (fallback < 0 || logits[a] > logits[fallback]) fallback = a;

                var next = cube.Clone();
                next.ApplyQuarterTurn(a);
                if (!visited.Contains(next.ToKociembaString()) && (best < 0 || logits[a] > logits[best]))
                    best = a;
            }
            if (fallback < 0) break;

            int action = best >= 0 ? best : fallback;
            cube.ApplyQuarterTurn(action);
            visited.Add(cube.ToKociembaString());
            actions.Add(action);
            lastAction = action;
        }
        return (cube.IsSolved, actions);
    }

    public sealed record SearchResult(bool Solved, string[] Moves, int Expansions);

    /// <summary>A* with the learned distance-to-solved as heuristic; budgeted by node expansions.</summary>
    public static SearchResult Solve(CubePolicyNet net, FaceletCube start, int maxExpansions = 20_000)
    {
        if (start.IsSolved)
            return new(true, [], 0);

        var open = new PriorityQueue<Node, double>();
        var bestDepth = new Dictionary<string, int> { [start.ToKociembaString()] = 0 };
        open.Enqueue(new Node(start, null, -1, 0), net.Evaluate(start).Distance);

        int expansions = 0;
        while (open.Count > 0 && expansions < maxExpansions)
        {
            var node = open.Dequeue();
            expansions++;

            int undo = RubiksCubeEnv.InverseAction(node.Action);
            for (int action = 0; action < RubiksCubeEnv.ActionCount; action++)
            {
                if (action == undo) continue;

                var child = node.Cube.Clone();
                child.ApplyQuarterTurn(action);
                int depth = node.Depth + 1;

                if (child.IsSolved)
                    return new(true, ExtractMoves(node, action), expansions);
                if (depth >= MaxSolutionMoves) continue;

                string key = child.ToKociembaString();
                if (bestDepth.TryGetValue(key, out int seen) && seen <= depth) continue;
                bestDepth[key] = depth;

                open.Enqueue(new Node(child, node, action, depth),
                    depth + net.Evaluate(child, action).Distance);
            }
        }

        return new(false, [], expansions);
    }

    private static string[] ExtractMoves(Node parent, int finalAction)
    {
        var actions = new List<int> { finalAction };
        for (var node = parent; node is not null && node.Action >= 0; node = node.Parent)
            actions.Add(node.Action);
        actions.Reverse();
        return [.. actions.Select(a => FaceletCube.QuarterTurnMoves[a])];
    }

    private sealed record Node(FaceletCube Cube, Node? Parent, int Action, int Depth);
}
