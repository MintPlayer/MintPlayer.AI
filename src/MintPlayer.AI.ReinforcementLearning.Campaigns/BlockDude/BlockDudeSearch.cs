using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Net-guided weighted A* over Block Dude, for measuring what a trained policy is worth WITH lookahead in front
/// of it — as opposed to <see cref="BlockDudeGreedy"/>, which deliberately measures the policy alone.
/// </summary>
/// <remarks>
/// <para><b>Why this is worth having even though the gate stays greedy.</b> The gate's stance — a policy that
/// only wins with search has not learned the game — is right for deciding when a curriculum rung is passed. It
/// is the wrong instrument for asking "can this net play the shipped levels at all", because a deterministic
/// argmax policy is guaranteed to be trapped by the first state it revisits: the greedy bench ends EVERY shipped
/// level in a loop within 2-15 steps. That measures the absence of a tie-break, not the absence of knowledge.</para>
///
/// <para><b>The heuristic is free.</b> The net's value head is already regressed onto distance-to-goal in optimal
/// moves (Huber, scaled by <see cref="BlockDudePolicyNet.DistanceScale"/>), which is exactly the cost-to-go
/// <see cref="ValueGuidedSearch"/> wants. Nothing new is trained to enable this.</para>
///
/// <para><b>A learned heuristic is not admissible</b>, so weight 1 buys optimality it cannot actually guarantee
/// while expanding far more nodes. A weight above 1 is the practical setting — greedier, deeper, and the
/// solutions it finds are not claimed to be optimal.</para>
///
/// <para>Irreversibility is handled by the search itself rather than specially: a block set down wrongly yields
/// successors whose value is poor and whose subtree contains no goal, so A* abandons that branch instead of
/// being stranded in it the way a greedy descent is.</para>
/// </remarks>
public static class BlockDudeSearch
{
    /// <summary>What a search attempt produced. <paramref name="Moves"/> is null when the goal was not reached
    /// inside the expansion budget.</summary>
    public readonly record struct Outcome(IReadOnlyList<int>? Moves, int Expansions)
    {
        public bool Solved => Moves is not null;
        public int Length => Moves?.Count ?? 0;
    }

    /// <summary>Searches for a solution to <paramref name="start"/> under <paramref name="net"/>.</summary>
    /// <param name="maxExpansions">Node budget. Block Dude states are large, so this is the memory lever.</param>
    /// <param name="weight">f = g + weight·h. Above 1 trades optimality for depth.</param>
    /// <param name="maxTime">Wall-clock ceiling, so one hopeless level cannot stall a whole benchmark.</param>
    public static Outcome Solve(BlockDudePolicyNet net, BlockDudeBoard start,
                                int maxExpansions = 200_000, float weight = 2f, TimeSpan? maxTime = null)
    {
        var model = new Model();
        var moves = ValueGuidedSearch.Solve(
            model, board => net.Evaluate(board).Distance, start, maxExpansions, weight, maxTime);
        return new(moves, maxExpansions);
    }

    /// <summary>The forward model. Block Dude is already deterministic and goal-directed, so this is a pure
    /// adapter — no rules live here.</summary>
    private sealed class Model : IDeterministicModel<BlockDudeBoard>
    {
        public int ActionCount => BlockDudeBoard.ActionCount;

        public bool IsGoal(BlockDudeBoard state) => state.Won;

        public BlockDudeBoard Apply(BlockDudeBoard state, int action) => state.Apply((BlockDudeAction)action);

        // The engine's own 32-bit state hash, widened to a string because that is the seam's key type. Collisions
        // would make the search treat two different positions as one; at these depths that is rare, and the
        // failure mode is a missed solution rather than an invalid one.
        public string StateKey(BlockDudeBoard state) => state.StateHash.ToString();
    }
}
