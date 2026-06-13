namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// Optimal (shortest-path) planner over an <see cref="IDeterministicModel{TState}"/> by
/// breadth-first search. Because BFS expands states in order of distance, the first path it
/// reaches the goal by is provably the fewest-action solution — making it the ground-truth
/// oracle for validating learned planners/policies on tractable (shallow) instances, and a
/// usable optimal solver wherever the reachable set within <c>maxDepth</c> is small enough.
/// <para>
/// Cost is exponential in solution depth (branching ≈ ActionCount), so it is for shallow
/// problems and verification — not a substitute for a learned heuristic on deep ones.
/// </para>
/// </summary>
public static class BreadthFirstPlanner
{
    /// <summary>
    /// The fewest-action sequence from <paramref name="start"/> to a goal, or null if none
    /// exists within <paramref name="maxDepth"/> actions. Empty list ⇒ start is already a goal.
    /// </summary>
    public static IReadOnlyList<int>? FindOptimal<TState>(IDeterministicModel<TState> model, TState start, int maxDepth)
    {
        if (model.IsGoal(start)) return [];

        // Parent-linked frontier: each node remembers the action that produced it and its
        // parent, so the path is reconstructed by walking back once the goal is hit.
        var nodes = new List<(TState State, int Parent, int Action)> { (start, -1, -1) };
        var visited = new HashSet<string> { model.StateKey(start) };
        var frontier = new Queue<(int Index, int Depth)>();
        frontier.Enqueue((0, 0));

        while (frontier.Count > 0)
        {
            var (index, depth) = frontier.Dequeue();
            if (depth >= maxDepth) continue;
            var state = nodes[index].State;

            for (int action = 0; action < model.ActionCount; action++)
            {
                var next = model.Apply(state, action);
                if (model.IsGoal(next))
                    return ReconstructPath(nodes, index, action);

                if (visited.Add(model.StateKey(next)))
                {
                    nodes.Add((next, index, action));
                    frontier.Enqueue((nodes.Count - 1, depth + 1));
                }
            }
        }
        return null;
    }

    private static List<int> ReconstructPath<TState>(List<(TState State, int Parent, int Action)> nodes, int parentIndex, int finalAction)
    {
        var path = new List<int> { finalAction };
        for (int p = parentIndex; nodes[p].Action >= 0; p = nodes[p].Parent)
            path.Add(nodes[p].Action);
        path.Reverse();
        return path;
    }
}
