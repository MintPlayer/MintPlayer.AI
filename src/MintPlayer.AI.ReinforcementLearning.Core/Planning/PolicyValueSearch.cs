using System.Diagnostics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// Batched best-first search guided by BOTH heads of a policy/value net: a cost-to-go estimate <i>and</i> the
/// policy's prior over moves.
/// </summary>
/// <remarks>
/// <para><b>Why this exists alongside <see cref="ValueGuidedSearch"/>.</b> That search orders the frontier by
/// <c>f = g + weight·h</c>, which uses only the value head. A policy head trained by imitation or expert
/// iteration is a second, independently-trained opinion about the same state — often a far sharper one, because
/// "which move is right here" is a local question while "how far is the goal" is a global one. Ignoring it
/// throws away the better-trained half of the net.</para>
///
/// <para><b>The ordering.</b> Each node carries an accumulated <i>surprise</i>: the summed negative log
/// probability the policy assigned to the moves that reached it. A path the policy would have played has
/// surprise near zero; a path it considers absurd accumulates cost with every step. The frontier is ordered by
///
/// <code>f = g + weight·h + policyWeight·surprise</code>
///
/// which is the Levin/policy-guided-search cost with an admissible-style distance term retained. At
/// <c>policyWeight = 0</c> this is exactly <see cref="ValueGuidedSearch.SolveBatched"/>; raising it makes the
/// search follow the policy and treat deviation as expensive, so the value head only has to arbitrate <i>among
/// plausible continuations</i> rather than rank the whole state space.</para>
///
/// <para><b>It never loses completeness.</b> Surprise is finite and monotonically accumulated, so a node the
/// policy dislikes is deprioritised, never pruned. Given budget the search still reaches everything
/// uninformed search would — unlike a hard policy mask, which can make a solvable problem unsolvable.</para>
///
/// <para><b>Batched for the same reason as its value-only sibling:</b> the net forward dominates the cost, so a
/// round expands a block of open nodes and evaluates all their successors in one call. Priors are produced by
/// that same call and cached per node (ActionCount floats), because a node's priors are needed later — when it
/// is expanded — not when it is generated.</para>
/// </remarks>
public static class PolicyValueSearch
{
    /// <summary>How much cheaper a new path must be before a state is re-opened. Guards against re-expanding a
    /// state for float improvements too small to change any decision — see the note at the dedup check.</summary>
    private const float ReopenEpsilon = 1e-3f;

    /// <summary>
    /// One batched evaluation of the net over a list of states.
    /// </summary>
    /// <param name="LogPriors">Row-major <c>[states.Count × ActionCount]</c> log-probabilities over actions.
    /// They must already be normalised per row and must be finite — an illegal move should be a very negative
    /// number, not −∞, or the accumulated surprise becomes NaN and poisons the ordering.</param>
    /// <param name="CostToGo">One estimated distance-to-goal per state, in the same units as path length.</param>
    public readonly record struct Evaluation(float[] LogPriors, float[] CostToGo);

    /// <summary>
    /// Search from <paramref name="start"/> for a path to a goal.
    /// </summary>
    /// <param name="weight">Multiplier on the value head's cost-to-go. &gt; 1 is greedier and deeper.</param>
    /// <param name="policyWeight">Multiplier on accumulated surprise. 0 ignores the policy entirely (making this
    /// value-only search); around 1 makes one unit of surprise cost about one move.</param>
    /// <param name="expandBatch">Open nodes expanded per round; their successors share one evaluation.</param>
    /// <returns>The action sequence, or null if the goal was not reached within the budget. An empty list means
    /// <paramref name="start"/> is already a goal.</returns>
    public static IReadOnlyList<int>? Solve<TState>(
        IDeterministicModel<TState> model, Func<IReadOnlyList<TState>, Evaluation> evaluate,
        TState start, int maxExpansions, float weight = 1f, float policyWeight = 1f,
        int expandBatch = 64, TimeSpan? maxTime = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expandBatch);
        if (model.IsGoal(start)) return [];

        int actions = model.ActionCount;
        string startKey = model.StateKey(start);

        var nodes = new List<Node<TState>> { new(start, -1, -1, 0, 0f, startKey) };
        var priors = new List<float[]>();                       // node index → its policy log-priors
        var bestCost = new Dictionary<string, float> { [startKey] = 0f };
        var open = new PriorityQueue<int, float>();

        var startEval = evaluate([start]);
        priors.Add(Row(startEval.LogPriors, 0, actions));
        open.Enqueue(0, weight * startEval.CostToGo[0]);

        long deadlineTicks = Deadline(maxTime);
        int expansions = 0;

        var batch = new List<int>(expandBatch);
        var pending = new List<Node<TState>>();
        var pendingStates = new List<TState>();

        while (open.Count > 0 && expansions < maxExpansions)
        {
            if (Stopwatch.GetTimestamp() >= deadlineTicks) break;

            // Pop a round of live nodes. Goal-on-pop, so the first solution returned is the best one the
            // ordering knows about rather than the first one stumbled into.
            batch.Clear();
            while (batch.Count < expandBatch && open.Count > 0)
            {
                int index = open.Dequeue();
                var node = nodes[index];
                if (node.Cost > bestCost[node.Key]) continue;                 // superseded by a cheaper path
                if (model.IsGoal(node.State)) return ReconstructPath(nodes, node.Parent, node.Action);
                batch.Add(index);
            }
            if (batch.Count == 0) break;

            pending.Clear();
            pendingStates.Clear();
            foreach (int index in batch)
            {
                expansions++;
                var node = nodes[index];
                var nodePriors = priors[index];

                for (int action = 0; action < actions; action++)
                {
                    var next = model.Apply(node.State, action);
                    string key = model.StateKey(next);

                    // The dedup cost is the SAME quantity the frontier is ordered by (depth + surprise), not
                    // depth alone: two paths to one state can differ in how plausible they are, and keeping the
                    // shallower-but-implausible one would discard exactly the information the policy adds.
                    float cost = node.Cost + 1f + policyWeight * -nodePriors[action];

                    // The epsilon is not cosmetic. The value-only search dedups on integer depth, so a state can
                    // only ever be re-opened a bounded number of times; cost here is a float, so two paths of the
                    // same length that differ only in how plausible they are produce different costs, and without
                    // a threshold a state can be re-opened for improvements far too small to change any decision.
                    // That is a node-count explosion, in the one place where nodes are the whole budget.
                    if (bestCost.TryGetValue(key, out float known) && known <= cost + ReopenEpsilon) continue;

                    bestCost[key] = cost;
                    pending.Add(new(next, index, action, node.Depth + 1, cost, key));
                    pendingStates.Add(next);
                }
            }
            if (pendingStates.Count == 0) continue;

            var evaluation = evaluate(pendingStates);
            for (int i = 0; i < pending.Count; i++)
            {
                int childIndex = nodes.Count;
                nodes.Add(pending[i]);
                priors.Add(Row(evaluation.LogPriors, i, actions));
                float h = model.IsGoal(pending[i].State) ? 0f : evaluation.CostToGo[i];
                open.Enqueue(childIndex, pending[i].Cost + weight * h);
            }
        }
        return null;
    }

    /// <param name="Depth">Moves from the start — the true path length, kept separate from <paramref name="Cost"/>
    /// because the reported solution length must be a move count, not a score.</param>
    /// <param name="Cost">Depth plus accumulated policy surprise: what the frontier is ordered by.</param>
    private readonly record struct Node<TState>(TState State, int Parent, int Action, int Depth, float Cost, string Key);

    private static float[] Row(float[] flat, int row, int width)
    {
        var slice = new float[width];
        Array.Copy(flat, row * width, slice, 0, width);
        return slice;
    }

    private static List<int> ReconstructPath<TState>(List<Node<TState>> nodes, int parentIndex, int finalAction)
    {
        var path = new List<int> { finalAction };
        for (int p = parentIndex; nodes[p].Action >= 0; p = nodes[p].Parent)
            path.Add(nodes[p].Action);
        path.Reverse();
        return path;
    }

    private static long Deadline(TimeSpan? maxTime)
        => maxTime is { } t ? Stopwatch.GetTimestamp() + (long)(t.TotalSeconds * Stopwatch.Frequency) : long.MaxValue;
}
