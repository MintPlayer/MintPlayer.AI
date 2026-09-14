using System.Diagnostics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// Policy-guided beam search: keep the best <c>beamWidth</c> partial solutions by cumulative log-probability,
/// extend all of them one move, keep the best <c>beamWidth</c> again.
/// </summary>
/// <remarks>
/// <para><b>Why this exists next to the A\* searches.</b> <see cref="ValueGuidedSearch"/> and
/// <see cref="PolicyValueSearch"/> hold an open frontier that grows with the space explored, so their reach is
/// bounded by a node budget — and a node budget is a wall, not a gradient. Block Dude's Level 11 needs about 900
/// moves; no heuristic quality makes 900 moves reachable inside 200,000 nodes. Beam search spends memory as
/// <c>width × depth</c> instead of exponentially, so depth costs almost nothing and it can reach solutions that
/// are simply out of range for best-first search. That is the same reason EfficientCube uses it for the cube.</para>
///
/// <para><b>What it gives up.</b> Completeness. A solution pruned out of the beam is gone for good, so unlike
/// the A\* searches this can fail on a problem it has the budget for. The two are complements rather than
/// alternatives: best-first for short, awkward problems, beam for long ones.</para>
///
/// <para><b>It is ranked by the policy alone</b>, deliberately. Cumulative log-probability is comparable across
/// a whole beam at one depth — every candidate has taken exactly the same number of moves — which is precisely
/// the comparison a value head trained by regression is measured to be bad at over long horizons.</para>
///
/// <para><b>One batched evaluation per depth step</b>, since the net dominates the cost.</para>
/// </remarks>
public static class PolicyBeamSearch
{
    /// <summary>
    /// Search from <paramref name="start"/> for a path to a goal.
    /// </summary>
    /// <param name="logPriors">Batched: row-major <c>[states.Count × ActionCount]</c> log-probabilities. Must be
    /// finite — an illegal move should be a large negative number, not −∞, which would make every cumulative
    /// score NaN and destroy the ranking.</param>
    /// <param name="beamWidth">Partial solutions carried forward per step. The whole cost/quality dial.</param>
    /// <param name="maxDepth">Give up after this many moves. Must exceed the expected solution length.</param>
    /// <returns>The action sequence, or null if no goal was reached. Empty ⇒ <paramref name="start"/> is a goal.</returns>
    public static IReadOnlyList<int>? Solve<TState>(
        IDeterministicModel<TState> model, Func<IReadOnlyList<TState>, float[]> logPriors,
        TState start, int beamWidth = 256, int maxDepth = 1_000, TimeSpan? maxTime = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beamWidth);
        if (model.IsGoal(start)) return [];

        int actions = model.ActionCount;

        // Parent pointers rather than a path array per node: at depth 900 with a wide beam, copying a growing
        // path for every candidate at every step is the dominant cost and the dominant allocation.
        var nodes = new List<(int Parent, int Action)> { (-1, -1) };
        var beam = new List<Entry<TState>> { new(start, 0, 0.0) };

        // Visited across ALL depths, not just within a step. A beam of near-duplicates that keep walking back
        // into positions already tried is exactly the looping failure a greedy policy shows here, and per-step
        // dedup alone does not prevent it.
        var visited = new HashSet<string> { model.StateKey(start) };

        long deadlineTicks = Deadline(maxTime);
        var states = new List<TState>();
        var candidates = new List<Entry<TState>>();

        for (int depth = 0; depth < maxDepth && beam.Count > 0; depth++)
        {
            if (Stopwatch.GetTimestamp() >= deadlineTicks) break;

            states.Clear();
            foreach (var entry in beam) states.Add(entry.State);
            float[] priors = logPriors(states);

            candidates.Clear();
            for (int i = 0; i < beam.Count; i++)
            {
                for (int action = 0; action < actions; action++)
                {
                    var next = model.Apply(beam[i].State, action);
                    if (model.IsGoal(next))
                    {
                        nodes.Add((beam[i].Node, action));
                        return ReconstructPath(nodes, nodes.Count - 1);
                    }

                    string key = model.StateKey(next);
                    if (!visited.Add(key)) continue;

                    nodes.Add((beam[i].Node, action));
                    candidates.Add(new(next, nodes.Count - 1, beam[i].Score + priors[i * actions + action]));
                }
            }
            if (candidates.Count == 0) break;

            // Highest cumulative log-probability wins. Every candidate is at the same depth, so the scores are
            // directly comparable with no length normalisation needed.
            candidates.Sort(static (a, b) => b.Score.CompareTo(a.Score));
            beam.Clear();
            for (int i = 0; i < candidates.Count && i < beamWidth; i++) beam.Add(candidates[i]);
        }
        return null;
    }

    private readonly record struct Entry<TState>(TState State, int Node, double Score);

    private static List<int> ReconstructPath(List<(int Parent, int Action)> nodes, int index)
    {
        var path = new List<int>();
        for (int i = index; nodes[i].Action >= 0; i = nodes[i].Parent) path.Add(nodes[i].Action);
        path.Reverse();
        return path;
    }

    private static long Deadline(TimeSpan? maxTime)
        => maxTime is { } t ? Stopwatch.GetTimestamp() + (long)(t.TotalSeconds * Stopwatch.Frequency) : long.MaxValue;
}
