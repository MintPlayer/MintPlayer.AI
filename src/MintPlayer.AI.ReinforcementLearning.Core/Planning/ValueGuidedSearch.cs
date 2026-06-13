namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// Weighted A* over an <see cref="IDeterministicModel{TState}"/> guided by a learned (or exact)
/// cost-to-go. Where <see cref="GreedyValuePlanner"/> commits to the locally-best move and can get
/// stuck when the value is imperfect, this expands a frontier ordered by <c>f = g + weight·h</c>
/// (g = moves so far, h = estimated cost-to-go), so it backs out of dead ends and reaches states a
/// greedy descent never would — the inference-time ceiling-raiser for a value-iteration learner
/// (<see cref="ValueIterationTrainer{TState}"/>). Mirrors the policy-guided A* used for the
/// imitation nets.
/// <para>
/// <paramref name="weight"/> = 1 is ordinary A* (optimal when the value never over-estimates the
/// true cost); weight &gt; 1 is greedier — it expands far fewer nodes and reaches deeper, at the
/// cost of possibly non-optimal solutions. A learned value is rarely admissible, so a weight &gt; 1
/// is usually the practical choice.
/// </para>
/// </summary>
public static class ValueGuidedSearch
{
    /// <summary>
    /// Search from <paramref name="start"/> for a path to a goal, expanding at most
    /// <paramref name="maxExpansions"/> nodes. Returns the action sequence, or null if the goal was
    /// not reached within the budget. Empty list ⇒ start is already a goal.
    /// </summary>
    public static IReadOnlyList<int>? Solve<TState>(
        IDeterministicModel<TState> model, Func<TState, float> costToGo, TState start, int maxExpansions, float weight = 1f)
    {
        if (model.IsGoal(start)) return [];

        var nodes = new List<(TState State, int Parent, int Action)> { (start, -1, -1) };
        var bestG = new Dictionary<string, int> { [model.StateKey(start)] = 0 };
        var open = new PriorityQueue<int, float>(); // node index, ordered by f = g + weight·h
        open.Enqueue(0, weight * costToGo(start));

        int expansions = 0;
        while (open.Count > 0 && expansions < maxExpansions)
        {
            int index = open.Dequeue();
            expansions++;
            var state = nodes[index].State;
            int g = bestG[model.StateKey(state)];

            for (int action = 0; action < model.ActionCount; action++)
            {
                var next = model.Apply(state, action);
                if (model.IsGoal(next)) return ReconstructPath(nodes, index, action);

                string key = model.StateKey(next);
                int tentativeG = g + 1;
                if (bestG.TryGetValue(key, out int known) && known <= tentativeG) continue; // not an improvement

                bestG[key] = tentativeG;
                nodes.Add((next, index, action));
                open.Enqueue(nodes.Count - 1, tentativeG + weight * costToGo(next));
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
