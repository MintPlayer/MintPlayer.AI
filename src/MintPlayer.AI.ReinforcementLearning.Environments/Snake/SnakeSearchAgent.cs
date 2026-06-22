using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>Tunable weights for <see cref="SnakeSearchAgent"/>. The defaults make eating dominate, treat a
/// self-trap as nearly as bad as death, and use the net value only as a tiebreak for "which safe way to go".</summary>
public sealed record SnakeSearchOptions
{
    /// <summary>Plies to look ahead. Higher catches traps that form further out, at ~3× cost per extra ply.</summary>
    public int MaxDepth { get; init; } = 6;

    /// <summary>Live (non-terminal) nodes carried to the next ply, best-scoring first. Caps the per-ply node count
    /// so deep horizons stay cheap; raise for stronger (slower) play.</summary>
    public int BeamWidth { get; init; } = 32;

    /// <summary>Reward per food eaten along a line — the dominant term; eating safely beats any non-eating line.</summary>
    public float FoodWeight { get; init; } = 10_000f;

    /// <summary>Penalty for a leaf whose reachable space can no longer hold the body (a guaranteed future trap).
    /// Smaller than the food reward so the snake still eats when it can do so without boxing itself in.</summary>
    public float TrapPenalty { get; init; } = 50_000f;

    /// <summary>Weight on the trained net's position value (max Q at the leaf) — the learned "which way is good".</summary>
    public float NetWeight { get; init; } = 500f;

    /// <summary>Weight on reachable free space (tiebreak: prefer keeping more room).</summary>
    public float SpaceWeight { get; init; } = 5f;

    /// <summary>Pull toward food when no line eats within the horizon (tiebreak by L1 head→food distance).</summary>
    public float FoodDistWeight { get; init; } = 1f;
}

/// <summary>
/// Receding-horizon, net-guided look-ahead for Snake (PLAN M27 — the EfficientCube idea of searching on top of a
/// learned value function, applied to Snake). Snake's dynamics are deterministic between food (the food RNG lives in
/// the env's saved state), so the agent can simulate every legal action sequence to a fixed horizon <i>exactly</i> on
/// a private clone of the env. Each simulated leaf is scored by the trained <see cref="DuelingQNet"/>'s value PLUS an
/// exact flood-fill survivability term — the net says "which way is good", the search guarantees "don't walk into a
/// box you can't get out of", which is the failure mode that caps a reactive learned policy. The agent plays the first
/// move of the best line and re-searches next tick.
/// <para>
/// The agent reads the live env's state directly (via <see cref="SnakeEnv.SaveState"/>), so it is driven by a
/// parameterless <see cref="Act"/> — the observation/mask passed to the usual selector contract are ignored.
/// </para>
/// </summary>
public sealed class SnakeSearchAgent
{
    private readonly SnakeEnv _live;
    private readonly SnakeEnv _sim;           // private clone for branching; safeMask off → reversal-only legality
    private readonly GreedyQAgent _value;     // net leaf evaluator
    private readonly SnakeSearchOptions _opt;

    public SnakeSearchAgent(SnakeEnv live, IValueNet net, SnakeSearchOptions? options = null)
    {
        _live = live;
        _sim = new SnakeEnv(live.Size, safeMask: false);
        _value = new GreedyQAgent(net, SnakeEnv.ActionCount);
        _opt = options ?? new SnakeSearchOptions();
    }

    /// <summary>Plays the first move of the highest-scoring line found within the horizon.</summary>
    public int Act()
    {
        var root = _live.SaveState();
        int rootFood = _live.FoodEaten;

        // The current beam: live (non-terminal) states to expand further, paired with the first move that led there.
        var beam = new List<(byte[] State, int First)> { (root, -1) };
        double bestScore = double.NegativeInfinity;
        int bestFirst = FallbackAction(root);

        for (int depth = 0; depth < _opt.MaxDepth && beam.Count > 0; depth++)
        {
            var next = new List<(byte[] State, int First, double Score)>();
            foreach (var (state, first) in beam)
            {
                _sim.RestoreState(state);
                var mask = _sim.CurrentActionMask(); // reversal-only (this sim has safeMask off)
                for (int a = 0; a < SnakeEnv.ActionCount; a++)
                {
                    if (!mask[a]) continue;
                    _sim.RestoreState(state);
                    var step = _sim.Step(a);
                    int childFirst = first < 0 ? a : first;
                    double score = LeafScore(step, rootFood, depth + 1);

                    if (score > bestScore) { bestScore = score; bestFirst = childFirst; }

                    // A still-alive, non-final node can be expanded another ply.
                    if (!step.Done && depth + 1 < _opt.MaxDepth)
                        next.Add((_sim.SaveState(), childFirst, score));
                }
            }

            // Prune to the best BeamWidth survivors before going deeper.
            if (next.Count > _opt.BeamWidth)
            {
                next.Sort((x, y) => y.Score.CompareTo(x.Score));
                next.RemoveRange(_opt.BeamWidth, next.Count - _opt.BeamWidth);
            }
            beam = next.ConvertAll(n => (n.State, n.First));
        }

        return bestFirst;
    }

    /// <summary>
    /// Scores the simulated position <paramref name="step"/> just reached. A board-full win dominates everything; a
    /// death is dominated by everything but ranked so a later death beats a sooner one (buys time for the tail to
    /// clear); a survived position is rewarded for food eaten, penalized hard if it can no longer fit its body, and
    /// nudged by net value, free space, and proximity to food.
    /// </summary>
    private double LeafScore(StepResult<float[]> step, int rootFood, int depth)
    {
        bool boardFull = step.Terminated && _sim.Length == _sim.Cells;
        if (boardFull) return 1e9;
        if (step.Terminated) return -1e6 + depth * 1_000; // death — prefer to delay the unavoidable

        int foodGained = _sim.FoodEaten - rootFood;
        int free = _sim.FreeSpaceAhead();
        double score = foodGained * _opt.FoodWeight;
        if (free < _sim.Length) score -= _opt.TrapPenalty;
        score += free * _opt.SpaceWeight;

        var obs = step.Observation;
        // The forward pass dominates the per-node cost, so skip it entirely when the net term is disabled
        // (empirically the net contributes almost nothing here — the flood-fill survival term carries the search).
        if (_opt.NetWeight != 0f)
        {
            var q = _value.QValues(obs);
            float bestQ = float.NegativeInfinity;
            for (int a = 0; a < q.Length; a++) bestQ = MathF.Max(bestQ, q[a]);
            score += bestQ * _opt.NetWeight;
        }

        // L1 head→food distance is the last three obs slots' source; recover it from the food-distance feature
        // (index RayFeatures+4+2), normalized by 2·Size. Pull toward food only as a final tiebreak.
        float foodDist = obs[SnakeEnv.RayFeatures + 4 + 2] * (2f * _sim.Size);
        score -= foodDist * _opt.FoodDistWeight;
        return score;
    }

    private int FallbackAction(byte[] state)
    {
        _sim.RestoreState(state);
        var mask = _sim.CurrentActionMask();
        for (int a = 0; a < SnakeEnv.ActionCount; a++)
            if (mask[a]) return a;
        return 0;
    }
}
