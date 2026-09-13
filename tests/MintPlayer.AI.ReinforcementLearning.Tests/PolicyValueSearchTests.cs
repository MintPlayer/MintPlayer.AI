using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Policy-guided best-first search: the value head says how far, the policy head says which way.
/// </summary>
/// <remarks>
/// Exercised on a toy model rather than a real game so each property is isolated from whether some net happens
/// to be good. The properties that matter are the ones a wrong implementation would break silently: the policy
/// must bias the order, it must NEVER prune, and at policy weight 0 the whole thing must reduce to the
/// value-only search it generalises.
/// </remarks>
public class PolicyValueSearchTests
{
    /// <summary>
    /// A line of integers 0..Length. Action 0 steps forward, action 1 steps back. The goal is the far end.
    /// </summary>
    private sealed class Line(int length) : IDeterministicModel<int>
    {
        public int ActionCount => 2;
        public bool IsGoal(int state) => state == length;
        public int Apply(int state, int action) => Math.Clamp(action == 0 ? state + 1 : state - 1, 0, length);
        public string StateKey(int state) => state.ToString();
    }

    /// <summary>Log-priors that favour <paramref name="favoured"/> by a wide margin, and a given cost-to-go.</summary>
    private static Func<IReadOnlyList<int>, PolicyValueSearch.Evaluation> Eval(
        int favoured, Func<int, float> costToGo, float margin = 5f)
        => states =>
        {
            var priors = new float[states.Count * 2];
            var h = new float[states.Count];
            for (int i = 0; i < states.Count; i++)
            {
                // log-softmax of (margin, 0) placed on the favoured action.
                float logSum = MathF.Log(MathF.Exp(margin) + 1f);
                priors[i * 2 + favoured] = margin - logSum;
                priors[i * 2 + (1 - favoured)] = -logSum;
                h[i] = costToGo(states[i]);
            }
            return new(priors, h);
        };

    [Fact]
    public void AGoalStartNeedsNoMoves()
    {
        var path = PolicyValueSearch.Solve(new Line(0), Eval(0, _ => 0f), 0, maxExpansions: 10);
        Assert.Equal([], path);
    }

    [Fact]
    public void FindsTheShortestPathWhenPolicyAndValueAgree()
    {
        var model = new Line(10);
        var path = PolicyValueSearch.Solve(model, Eval(0, s => 10 - s), 0, maxExpansions: 1_000, weight: 1f);

        Assert.NotNull(path);
        Assert.Equal(10, path!.Count);
        Assert.All(path, a => Assert.Equal(0, a));
    }

    [Fact]
    public void TheGoalIsStillFoundWhenThePolicyPointsTheWrongWay()
    {
        // The headline safety property. The policy here is confidently, uniformly WRONG (it always prefers
        // stepping backwards) and the value head is useless (flat). A search that treated the prior as a filter
        // would never arrive; this one must, because surprise is a cost and not a wall.
        var model = new Line(8);
        var path = PolicyValueSearch.Solve(model, Eval(1, _ => 0f), 0, maxExpansions: 10_000,
                                           weight: 1f, policyWeight: 1f);

        Assert.NotNull(path);
        Assert.Equal(8, path!.Count);
    }

    /// <summary>
    /// A complete binary tree of the given depth; the goal is the single leaf reached by taking action 0 every
    /// time. State is <c>1</c> followed by the actions taken, so every path is a distinct state and nothing
    /// dedups — unlike a line, where a misdirected search is rescued by running out of anywhere else to go.
    /// </summary>
    private sealed class Tree(int depth) : IDeterministicModel<int>
    {
        public int ActionCount => 2;
        public bool IsGoal(int state) => state == 1 << depth;               // depth moves, all of them action 0
        public int Apply(int state, int action)
            => 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)state) > depth ? state : state << 1 | action;
        public string StateKey(int state) => state.ToString();
    }

    [Fact]
    public void AHelpfulPolicyReachesTheGoalInFarFewerExpansions()
    {
        // The reason the feature exists. The value head is flat, so the prior alone has to carry the search
        // through a tree with 4,096 leaves. The budget sits between "walk straight to it" and "look around":
        // a right prior needs ~12 expansions, a wrong one has to unpack an exponential subtree first.
        var model = new Tree(12);

        var guided = PolicyValueSearch.Solve(model, Eval(0, _ => 0f), 1, maxExpansions: 60,
                                             weight: 1f, policyWeight: 1f, expandBatch: 1);
        var misguided = PolicyValueSearch.Solve(model, Eval(1, _ => 0f), 1, maxExpansions: 60,
                                                weight: 1f, policyWeight: 1f, expandBatch: 1);

        Assert.NotNull(guided);
        Assert.Null(misguided);
    }

    [Fact]
    public void TheGoalIsStillFoundInATreeThePolicyPointsAwayFrom()
    {
        // Same hostile prior as above, budget large enough to exhaust the tree: deprioritised is not pruned.
        var model = new Tree(8);

        var path = PolicyValueSearch.Solve(model, Eval(1, _ => 0f), 1, maxExpansions: 100_000,
                                           weight: 1f, policyWeight: 1f);

        Assert.NotNull(path);
        Assert.Equal(8, path!.Count);
        Assert.All(path, a => Assert.Equal(0, a));
    }

    [Fact]
    public void PolicyWeightZeroIgnoresThePolicyEntirely()
    {
        // At policyWeight 0 this must be value-only search, so a maximally hostile prior changes nothing.
        var model = new Line(6);

        var withHostilePrior = PolicyValueSearch.Solve(model, Eval(1, s => 6 - s, margin: 50f), 0,
                                                       maxExpansions: 1_000, weight: 1f, policyWeight: 0f);

        Assert.NotNull(withHostilePrior);
        Assert.Equal(6, withHostilePrior!.Count);
    }

    [Fact]
    public void AnExhaustedBudgetReturnsNothing()
    {
        Assert.Null(PolicyValueSearch.Solve(new Line(50), Eval(0, _ => 0f), 0, maxExpansions: 2, expandBatch: 1));
    }

    [Fact]
    public void AZeroTimeBudgetStopsImmediatelyRatherThanRunningToCompletion()
    {
        Assert.Null(PolicyValueSearch.Solve(new Line(50), Eval(0, _ => 0f), 0, maxExpansions: 1_000_000,
                                            weight: 1f, policyWeight: 1f, expandBatch: 8,
                                            maxTime: TimeSpan.Zero));
    }

    [Fact]
    public void TheReturnedPathIsWalkable()
    {
        // A path is only worth anything if replaying it lands on the goal — the search reports move indices, and
        // an off-by-one in the parent chain would produce a plausible-looking list that goes somewhere else.
        var model = new Line(12);
        var path = PolicyValueSearch.Solve(model, Eval(0, s => 12 - s), 0, maxExpansions: 5_000,
                                           weight: 2f, policyWeight: 1f);

        Assert.NotNull(path);
        int state = 0;
        foreach (int action in path!) state = model.Apply(state, action);
        Assert.True(model.IsGoal(state));
    }
}
