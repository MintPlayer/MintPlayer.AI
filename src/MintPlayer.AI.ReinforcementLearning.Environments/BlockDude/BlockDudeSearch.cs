using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

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
/// <para><b>Batched, not per-node.</b> The net forward is essentially the whole cost of this search, and a
/// 1181-wide input through a 512×512 trunk is a matrix multiply that is wildly under-fed by a single row. So it
/// runs <see cref="ValueGuidedSearch.SolveBatched"/>: expand a block of open nodes, then score ALL their
/// successors in one pass. Same search, same ordering; the budget simply buys several times more of it, which
/// matters twice over here — once at the bench and once inside expert iteration, where every training sample is
/// produced by a search like this one.</para>
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
    /// <param name="expandBatch">How many open nodes to expand per round. Their successors are scored in ONE
    /// forward pass, which is the entire point — see the remarks on batching above.</param>
    public static Outcome Solve(BlockDudePolicyNet net, BlockDudeBoard start,
                                int maxExpansions = 200_000, float weight = 2f, TimeSpan? maxTime = null,
                                int expandBatch = DefaultExpandBatch)
        => Solve(net.Distances, start, maxExpansions, weight, maxTime, expandBatch);

    /// <summary>
    /// The same search against an arbitrary batched cost-to-go — the seam that lets a bench substitute a
    /// different heuristic for the net's.
    /// </summary>
    /// <remarks>
    /// Its reason for existing is <see cref="ZeroHeuristic"/>. "The net solves 7/15 with search" is not
    /// interpretable on its own: uninformed breadth-first search solves some of these levels too, and without
    /// that control number there is no way to tell whether the value head is guiding the search or merely
    /// riding along. Comparing the two is what says whether effort belongs in the heuristic or elsewhere.
    /// </remarks>
    public static Outcome Solve(Func<IReadOnlyList<BlockDudeBoard>, float[]> costToGo, BlockDudeBoard start,
                                int maxExpansions = 200_000, float weight = 2f, TimeSpan? maxTime = null,
                                int expandBatch = DefaultExpandBatch)
    {
        var model = new Model();
        var moves = ValueGuidedSearch.SolveBatched(
            model, costToGo, start, maxExpansions, weight, expandBatch, maxTime);
        return new(moves, maxExpansions);
    }

    /// <summary>h = 0 everywhere: weighted A* degenerates to uniform-cost (breadth-first) search. The control
    /// for "is the learned heuristic worth anything".</summary>
    public static float[] ZeroHeuristic(IReadOnlyList<BlockDudeBoard> boards) => new float[boards.Count];

    /// <summary>
    /// A flat prior over all actions — the control for <see cref="SolveByBeam"/>, as <see cref="ZeroHeuristic"/>
    /// is for the A* tiers.
    /// </summary>
    /// <remarks>
    /// Needed because beam search and A* are different search SHAPES, so the h = 0 control cannot be used to
    /// attribute a beam result: comparing beam-with-a-policy against best-first-without-a-heuristic would credit
    /// the policy for the change of shape. With uniform priors every candidate at a depth scores identically, so
    /// the beam keeps whichever states it happens to enumerate first — an uninformed width-limited sweep, which
    /// is exactly the "what does the net add" baseline.
    /// </remarks>
    public static float[] UniformPriors(IReadOnlyList<BlockDudeBoard> boards)
    {
        var priors = new float[boards.Count * BlockDudeBoard.ActionCount];
        Array.Fill(priors, -MathF.Log(BlockDudeBoard.ActionCount));
        return priors;
    }

    /// <summary>Beam search against an arbitrary batched prior — the seam <see cref="UniformPriors"/> needs.</summary>
    public static Outcome SolveByBeam(Func<IReadOnlyList<BlockDudeBoard>, float[]> logPriors, BlockDudeBoard start,
                                      int beamWidth = 256, int maxDepth = 1_200, TimeSpan? maxTime = null)
        => new(PolicyBeamSearch.Solve(new Model(), logPriors, start, beamWidth, maxDepth, maxTime), beamWidth);

    /// <summary>
    /// Search using BOTH heads — the value head as cost-to-go and the policy head as a prior over moves.
    /// </summary>
    /// <remarks>
    /// <para>The value-only <see cref="Solve(BlockDudePolicyNet, BlockDudeBoard, int, float, TimeSpan?, int)"/>
    /// discards the better-trained half of the net. The policy reaches ~89% agreement with demonstrated moves
    /// and solves shipped levels outright with no lookahead at all; the value head is measured compressed at
    /// long horizons (PRD §8.4a). Asking the accurate head "which move" and the inaccurate one only "roughly how
    /// far" plays to what each actually knows.</para>
    ///
    /// <para>The policy is a preference, never a filter: deviating from it costs, so the search still reaches
    /// anything an uninformed search would, given budget. That matters in a game where the winning line is
    /// frequently the move the policy ranks second.</para>
    /// </remarks>
    /// <param name="policyWeight">How many moves' worth of cost one nat of policy surprise is worth. 0 reduces
    /// this exactly to the value-only search.</param>
    public static Outcome SolveWithPolicy(BlockDudePolicyNet net, BlockDudeBoard start,
                                          int maxExpansions = 200_000, float weight = 2f, float policyWeight = 1f,
                                          TimeSpan? maxTime = null, int expandBatch = DefaultExpandBatch)
    {
        var model = new Model();
        var moves = PolicyValueSearch.Solve(
            model,
            boards => { var (priors, distances) = net.EvaluateBatch(boards); return new(priors, distances); },
            start, maxExpansions, weight, policyWeight, expandBatch, maxTime);
        return new(moves, maxExpansions);
    }

    /// <summary>
    /// Policy beam search — the tier for the LONG levels, where a node budget is a wall rather than a dial.
    /// </summary>
    /// <remarks>
    /// <para>Seven shipped levels are solved by nothing: not greedy, not uninformed search, not either head,
    /// not both. They are the large ones — Level 11 is 29×19 and 909 human moves — and no heuristic quality
    /// makes a 900-move solution reachable inside a 200,000-node A\* frontier. Beam search costs
    /// <c>width × depth</c> instead, so depth is nearly free and these levels are at least *in range*.</para>
    ///
    /// <para>It is ranked by the policy head alone. That is not an oversight: every candidate in the beam is at
    /// the same depth, so cumulative log-probabilities compare directly, whereas the long-horizon comparison is
    /// exactly what the value head is measured to be bad at.</para>
    ///
    /// <para>It trades away completeness — a solution pruned from the beam is gone — so this complements the A\*
    /// tiers rather than replacing them.</para>
    /// </remarks>
    public static Outcome SolveByBeam(BlockDudePolicyNet net, BlockDudeBoard start,
                                      int beamWidth = 256, int maxDepth = 1_200, TimeSpan? maxTime = null)
    {
        var model = new Model();
        var moves = PolicyBeamSearch.Solve(
            model, boards => net.EvaluateBatch(boards).LogPriors, start, beamWidth, maxDepth, maxTime);
        return new(moves, beamWidth);
    }

    /// <summary>Open nodes expanded per round. 4 actions each, so the net sees up to 256 positions per forward —
    /// enough for the matrix multiply to amortise, small enough that the frontier stays close to best-first.</summary>
    public const int DefaultExpandBatch = 64;

    /// <summary>
    /// Searches for the door OR for a demonstrated position at most <paramref name="acceptRemaining"/> moves
    /// from it, using both heads.
    /// </summary>
    /// <remarks>
    /// <para>The failure mode this addresses: on a long level, asking A* for a 400-move suffix inside an 8-second
    /// budget almost always fails, and a failed search produces no training data at all — so the levels with the
    /// most left to learn produce the least signal. Aiming at the human's path as well as the door converts most
    /// of those failures into real data, because reaching a state the demonstration also reached means a winning
    /// continuation from there is already known.</para>
    ///
    /// <para>The door is itself a landmark with remaining 0, so a full solution is still found when one exists,
    /// and the caller distinguishes the two outcomes by asking whether the final board is won. Only a genuine
    /// win should move a level's frontier — otherwise the curriculum would advance on the strength of the
    /// human's work rather than the net's.</para>
    /// </remarks>
    /// <param name="landmarks">State hash → moves remaining, from <c>BlockDudeDemonstrations.PathLandmarks</c>.</param>
    /// <param name="acceptRemaining">How close to the door a landmark must be to count. Set well below the
    /// start's own remaining distance, or the search stops at the first demonstrated state it stumbles onto and
    /// learns nothing.</param>
    public static Outcome SolveToLandmark(BlockDudePolicyNet net, BlockDudeBoard start,
                                          IReadOnlyDictionary<int, int> landmarks, int acceptRemaining,
                                          int maxExpansions = 200_000, float weight = 2f, float policyWeight = 1f,
                                          TimeSpan? maxTime = null, int expandBatch = DefaultExpandBatch)
    {
        var model = new Model(landmarks, acceptRemaining);
        var moves = PolicyValueSearch.Solve(
            model,
            boards => { var (priors, distances) = net.EvaluateBatch(boards); return new(priors, distances); },
            start, maxExpansions, weight, policyWeight, expandBatch, maxTime);
        return new(moves, maxExpansions);
    }

    /// <summary>The forward model. Block Dude is already deterministic and goal-directed, so this is a pure
    /// adapter — no rules live here.</summary>
    private sealed class Model(IReadOnlyDictionary<int, int>? landmarks = null, int acceptRemaining = -1)
        : IDeterministicModel<BlockDudeBoard>
    {
        public int ActionCount => BlockDudeBoard.ActionCount;

        public bool IsGoal(BlockDudeBoard state)
            => state.Won
            || (landmarks is not null
                && landmarks.TryGetValue(state.StateHash, out int remaining)
                && remaining <= acceptRemaining);

        public BlockDudeBoard Apply(BlockDudeBoard state, int action) => state.Apply((BlockDudeAction)action);

        // The engine's own 32-bit state hash, widened to a string because that is the seam's key type. Collisions
        // would make the search treat two different positions as one; at these depths that is rare, and the
        // failure mode is a missed solution rather than an invalid one.
        public string StateKey(BlockDudeBoard state) => state.StateHash.ToString();
    }
}
