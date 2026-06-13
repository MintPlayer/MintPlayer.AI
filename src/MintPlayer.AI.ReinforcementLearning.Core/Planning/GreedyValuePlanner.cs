namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// Acts greedily under a learned (or exact) cost-to-go function over an
/// <see cref="IDeterministicModel{TState}"/>: at each step take the action whose successor
/// has the lowest estimated cost-to-go, stopping at the goal. With an exact cost-to-go (e.g.
/// BFS distance) this descends optimally; with a learned value it is the inference-time policy
/// of a value-iteration learner (<see cref="ValueIterationTrainer{TState}"/>). A visited-set
/// breaks cycles so an imperfect value can't loop forever.
/// </summary>
public static class GreedyValuePlanner
{
    /// <summary>
    /// Greedily descend <paramref name="costToGo"/> from <paramref name="start"/> to a goal.
    /// Returns the action sequence taken, or null if it neither reached the goal nor had an
    /// unvisited move within <paramref name="maxSteps"/>. Empty list ⇒ start is already a goal.
    /// </summary>
    public static IReadOnlyList<int>? Solve<TState>(
        IDeterministicModel<TState> model, Func<TState, float> costToGo, TState start, int maxSteps)
    {
        if (model.IsGoal(start)) return [];

        var path = new List<int>(maxSteps);
        var current = start;
        var visited = new HashSet<string> { model.StateKey(current) };

        for (int step = 0; step < maxSteps; step++)
        {
            int bestAction = -1;
            float bestCost = float.PositiveInfinity;
            TState bestNext = default!;

            for (int action = 0; action < model.ActionCount; action++)
            {
                var next = model.Apply(current, action);
                if (model.IsGoal(next)) { path.Add(action); return path; }
                if (visited.Contains(model.StateKey(next))) continue;

                float cost = costToGo(next);
                if (cost < bestCost) { bestCost = cost; bestAction = action; bestNext = next; }
            }

            if (bestAction < 0) return null; // every move revisits a seen state — stuck
            path.Add(bestAction);
            current = bestNext;
            visited.Add(model.StateKey(current));
        }
        return null; // didn't reach the goal within the step budget
    }
}
