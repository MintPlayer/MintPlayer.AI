using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Q-guided best-first search — the cube's counterpart of <c>RushHourPolicySearch</c>
/// (PRD §11 / the M11 recipe): when the greedy rollout fails, the same trained network
/// keeps choosing, now with lookahead. A* over the quarter-turn graph with
/// g = moves so far and h = the net's own estimate of moves-to-go (return ≈ 101 − moves,
/// so h ≈ 101 − maxQ; inadmissible but effective — this is "the AI with lookahead",
/// not an exact solver, and results are reported honestly as such).
/// </summary>
public static class CubeQSearch
{
    public sealed record SearchResult(bool Solved, string[] Moves, int Expansions);

    public static SearchResult Solve(GreedyQAgent agent, FaceletCube start, int maxDepth = 20, int maxExpansions = 20_000)
    {
        if (start.IsSolved)
            return new(true, [], 0);

        var open = new PriorityQueue<Node, double>();
        var bestDepth = new Dictionary<string, int> { [start.ToKociembaString()] = 0 };
        var observation = new float[RubiksCubeEnv.ObservationSize];

        open.Enqueue(new Node(start, null, -1, 0), Heuristic(agent, start, observation));

        int expansions = 0;
        while (open.Count > 0 && expansions < maxExpansions)
        {
            var node = open.Dequeue();
            expansions++;

            var mask = RubiksCubeEnv.ActionMask(node.Action);
            for (int action = 0; action < RubiksCubeEnv.ActionCount; action++)
            {
                if (!mask[action]) continue;

                var child = node.Cube.Clone();
                child.ApplyQuarterTurn(action);
                int depth = node.Depth + 1;

                if (child.IsSolved)
                    return new(true, ExtractMoves(node, action), expansions);
                if (depth >= maxDepth) continue;

                string key = child.ToKociembaString();
                if (bestDepth.TryGetValue(key, out int seen) && seen <= depth) continue;
                bestDepth[key] = depth;

                open.Enqueue(new Node(child, node, action, depth),
                    depth + Heuristic(agent, child, observation));
            }
        }

        return new(false, [], expansions);
    }

    /// <summary>Estimated moves-to-go from the Q-net (return = 101 − moves ⇒ moves ≈ 101 − maxQ).</summary>
    private static double Heuristic(GreedyQAgent agent, FaceletCube cube, float[] observation)
    {
        RubiksCubeEnv.WriteObservation(cube, observation);
        float[] q = agent.QValues(observation);
        double max = double.NegativeInfinity;
        foreach (float value in q)
            if (value > max) max = value;
        return Math.Max(0, 101 - max);
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
