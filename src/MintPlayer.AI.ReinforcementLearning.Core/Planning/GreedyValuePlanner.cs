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

    /// <summary>
    /// Batched variant: evaluates all of a state's candidate successors in ONE
    /// <paramref name="batchCostToGo"/> call per step instead of ActionCount separate calls. When
    /// each evaluation is a neural-net forward, the per-call overhead dwarfs the math of a tiny
    /// single-row forward, so batching (one [≤ActionCount, …] forward per step) is far cheaper —
    /// the eval-loop hot path. Same greedy policy and result as the scalar overload.
    /// </summary>
    public static IReadOnlyList<int>? Solve<TState>(
        IDeterministicModel<TState> model, Func<IReadOnlyList<TState>, float[]> batchCostToGo, TState start, int maxSteps)
    {
        if (model.IsGoal(start)) return [];

        var path = new List<int>(maxSteps);
        var current = start;
        var visited = new HashSet<string> { model.StateKey(current) };
        var candidates = new List<TState>(model.ActionCount);
        var candidateActions = new List<int>(model.ActionCount);

        for (int step = 0; step < maxSteps; step++)
        {
            candidates.Clear();
            candidateActions.Clear();
            for (int action = 0; action < model.ActionCount; action++)
            {
                var next = model.Apply(current, action);
                if (model.IsGoal(next)) { path.Add(action); return path; } // a move reaches the goal — take it
                if (visited.Contains(model.StateKey(next))) continue;
                candidates.Add(next);
                candidateActions.Add(action);
            }
            if (candidates.Count == 0) return null; // every move revisits a seen state — stuck

            var costs = batchCostToGo(candidates);
            int best = 0;
            for (int i = 1; i < costs.Length; i++)
                if (costs[i] < costs[best]) best = i;

            path.Add(candidateActions[best]);
            current = candidates[best];
            visited.Add(model.StateKey(current));
        }
        return null; // didn't reach the goal within the step budget
    }
}
