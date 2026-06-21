using System.Diagnostics;

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
/// <c>weight</c> = 1 is ordinary A* (optimal when the value never over-estimates the true cost);
/// weight &gt; 1 is greedier — it expands far fewer nodes and reaches deeper, at the cost of possibly
/// non-optimal solutions. A learned value is rarely admissible, so a weight &gt; 1 is usually the
/// practical choice.
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
        IDeterministicModel<TState> model, Func<TState, float> costToGo, TState start, int maxExpansions,
        float weight = 1f, TimeSpan? maxTime = null)
    {
        if (model.IsGoal(start)) return [];

        var nodes = new List<(TState State, int Parent, int Action)> { (start, -1, -1) };
        var bestG = new Dictionary<string, int> { [model.StateKey(start)] = 0 };
        var open = new PriorityQueue<int, float>(); // node index, ordered by f = g + weight·h
        open.Enqueue(0, weight * costToGo(start));

        long deadlineTicks = Deadline(maxTime);
        int expansions = 0;
        while (open.Count > 0 && expansions < maxExpansions)
        {
            if (Stopwatch.GetTimestamp() >= deadlineTicks) break; // time budget exhausted
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

    /// <summary>
    /// Batch-weighted A* (BWAS, à la DeepCubeA): the same search as <see cref="Solve"/> but it expands
    /// the best <paramref name="expandBatch"/> open nodes at once and scores <b>all their successors in a
    /// single batched value call</b>. The value net is the dominant cost, and one forward over
    /// N·ActionCount states is far cheaper than N·ActionCount tiny forwards — so this is the form that
    /// makes a learned-value search usable at depth (and on the GPU, where a big batch finally pays).
    /// <para>
    /// Goal handling is <b>goal-on-pop</b>: a goal is returned only when it is dequeued as the
    /// minimum-f node, so with <paramref name="weight"/> = 1 and an admissible value the first solution
    /// returned is optimal (the Tier-1 guarantee). weight &gt; 1 expands fewer nodes and reaches deeper,
    /// at the cost of possibly non-optimal length. <paramref name="maxExpansions"/> bounds nodes expanded.
    /// </para>
    /// </summary>
    public static IReadOnlyList<int>? SolveBatched<TState>(
        IDeterministicModel<TState> model, Func<IReadOnlyList<TState>, float[]> batchCostToGo,
        TState start, int maxExpansions, float weight = 1f, int expandBatch = 100, TimeSpan? maxTime = null)
    {
        if (model.IsGoal(start)) return [];

        // The state key (computed once per node when it's first reached) is cached on the node: it's the
        // SAME string the dedup map already holds, so caching costs one reference, not an allocation — and
        // it spares the per-pop StateKey recompute (a 108-char hex string per popped node) on the hot path.
        string startKey = model.StateKey(start);
        var nodes = new List<(TState State, int Parent, int Action, int G, string Key)> { (start, -1, -1, 0, startKey) };
        var bestG = new Dictionary<string, int> { [startKey] = 0 };
        var open = new PriorityQueue<int, float>(); // node index, ordered by f = g + weight·h
        open.Enqueue(0, weight * batchCostToGo([start])[0]);

        long deadlineTicks = Deadline(maxTime);
        int expansions = 0;
        var batch = new List<int>(expandBatch);
        // pending successors awaiting one batched value call: (parent node index, action, state, g, key)
        var pendingParent = new List<int>();
        var pendingAction = new List<int>();
        var pendingState = new List<TState>();
        var pendingG = new List<int>();
        var pendingKey = new List<string>();

        while (open.Count > 0 && expansions < maxExpansions)
        {
            if (Stopwatch.GetTimestamp() >= deadlineTicks) break; // time budget exhausted — honest fail

            // Pop up to expandBatch live nodes; a goal popped here is optimal (goal-on-pop).
            batch.Clear();
            while (batch.Count < expandBatch && open.Count > 0)
            {
                int idx = open.Dequeue();
                var node = nodes[idx];
                if (node.G > bestG[node.Key]) continue; // stale: a better path superseded it (cached key, no recompute)
                if (model.IsGoal(node.State)) return ReconstructPathG(nodes, node.Parent, node.Action);
                batch.Add(idx);
            }
            if (batch.Count == 0) break;

            pendingParent.Clear(); pendingAction.Clear(); pendingState.Clear(); pendingG.Clear(); pendingKey.Clear();
            foreach (int idx in batch)
            {
                expansions++;
                var (state, _, _, g, _) = nodes[idx];
                for (int action = 0; action < model.ActionCount; action++)
                {
                    var next = model.Apply(state, action);
                    string key = model.StateKey(next);
                    int tentativeG = g + 1;
                    if (bestG.TryGetValue(key, out int known) && known <= tentativeG) continue;
                    bestG[key] = tentativeG;
                    pendingParent.Add(idx); pendingAction.Add(action); pendingState.Add(next); pendingG.Add(tentativeG); pendingKey.Add(key);
                }
            }
            if (pendingState.Count == 0) continue;

            // ONE value call for every successor generated this round (the BWAS win).
            float[] h = batchCostToGo(pendingState);
            for (int i = 0; i < pendingState.Count; i++)
            {
                int childIndex = nodes.Count;
                nodes.Add((pendingState[i], pendingParent[i], pendingAction[i], pendingG[i], pendingKey[i]));
                float hi = model.IsGoal(pendingState[i]) ? 0f : h[i]; // goal cost-to-go is exactly 0
                open.Enqueue(childIndex, pendingG[i] + weight * hi);
            }
        }
        return null;
    }

    /// <summary>Stopwatch-tick deadline for an optional wall-clock budget; <c>long.MaxValue</c> = unbounded.
    /// Ticks are monotonic and allocation-free, so the per-round check is essentially free.</summary>
    private static long Deadline(TimeSpan? maxTime)
        => maxTime is { } t ? Stopwatch.GetTimestamp() + (long)(t.TotalSeconds * Stopwatch.Frequency) : long.MaxValue;

    private static List<int> ReconstructPathG<TState>(List<(TState State, int Parent, int Action, int G, string Key)> nodes, int parentIndex, int finalAction)
    {
        var path = new List<int> { finalAction };
        for (int p = parentIndex; nodes[p].Action >= 0; p = nodes[p].Parent)
            path.Add(nodes[p].Action);
        path.Reverse();
        return path;
    }
}
