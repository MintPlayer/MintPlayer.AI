using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Policy beam search: the tier for problems whose solutions are too DEEP for a node-budgeted frontier.
/// </summary>
/// <remarks>
/// Its whole reason for existing is that it costs width × depth rather than holding an exponential frontier, so
/// the property worth pinning hardest is exactly that — it must find a solution hundreds of moves long inside a
/// budget that no best-first search could. And its known weakness (it is incomplete: a pruned solution is gone)
/// has to be demonstrated rather than left as a footnote, since anyone reading "search" will otherwise assume
/// the guarantees of the A* siblings.
/// </remarks>
public class PolicyBeamSearchTests
{
    /// <summary>A line of integers; action 0 advances, action 1 steps back. Goal is the far end.</summary>
    private sealed class Line(int length) : IDeterministicModel<int>
    {
        public int ActionCount => 2;
        public bool IsGoal(int state) => state == length;
        public int Apply(int state, int action) => Math.Clamp(action == 0 ? state + 1 : state - 1, 0, length);
        public string StateKey(int state) => state.ToString();
    }

    /// <summary>A complete binary tree; the goal leaf is reached by taking action 0 every time.</summary>
    private sealed class Tree(int depth) : IDeterministicModel<int>
    {
        public int ActionCount => 2;
        public bool IsGoal(int state) => state == 1 << depth;
        public int Apply(int state, int action)
            => 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)state) > depth ? state : state << 1 | action;
        public string StateKey(int state) => state.ToString();
    }

    /// <summary>Log-priors favouring <paramref name="favoured"/>, normalised over two actions.</summary>
    private static Func<IReadOnlyList<int>, float[]> Priors(int favoured, float margin = 3f)
        => states =>
        {
            var priors = new float[states.Count * 2];
            float logSum = MathF.Log(MathF.Exp(margin) + 1f);
            for (int i = 0; i < states.Count; i++)
            {
                priors[i * 2 + favoured] = margin - logSum;
                priors[i * 2 + (1 - favoured)] = -logSum;
            }
            return priors;
        };

    [Fact]
    public void AGoalStartNeedsNoMoves()
        => Assert.Equal([], PolicyBeamSearch.Solve(new Line(0), Priors(0), 0, beamWidth: 4, maxDepth: 4));

    [Fact]
    public void ReachesASolutionHundredsOfMovesDeep()
    {
        // The point of the whole class. A 900-move solution is simply out of range for a node-budgeted
        // frontier; here it costs one narrow beam per step.
        var path = PolicyBeamSearch.Solve(new Line(900), Priors(0), 0, beamWidth: 8, maxDepth: 1_000);

        Assert.NotNull(path);
        Assert.Equal(900, path!.Count);
    }

    [Fact]
    public void TheReturnedPathIsWalkable()
    {
        // Parent-pointer reconstruction over hundreds of steps is where an off-by-one hides, and it would
        // produce a plausible-looking list that goes somewhere else.
        var model = new Line(300);
        var path = PolicyBeamSearch.Solve(model, Priors(0), 0, beamWidth: 8, maxDepth: 400);

        Assert.NotNull(path);
        int state = 0;
        foreach (int action in path!) state = model.Apply(state, action);
        Assert.True(model.IsGoal(state));
    }

    [Fact]
    public void AWiderBeamSurvivesAPolicyThatPointsTheWrongWay()
    {
        // Width is the quality dial: the same hostile prior that defeats a beam of 1 is absorbed by a wider one.
        var model = new Tree(10);

        Assert.Null(PolicyBeamSearch.Solve(model, Priors(1, margin: 8f), 1, beamWidth: 1, maxDepth: 12));
        Assert.NotNull(PolicyBeamSearch.Solve(model, Priors(1, margin: 8f), 1, beamWidth: 4_000, maxDepth: 12));
    }

    [Fact]
    public void ItIsIncompleteAndThatIsTheTradeItMakes()
    {
        // Documented rather than hidden. Unlike the A* tiers, a solution pruned out of the beam is gone for
        // good, so this can fail on a problem it has ample depth budget for. Anyone reading "search" would
        // otherwise assume the siblings' guarantees.
        var narrow = PolicyBeamSearch.Solve(new Tree(14), Priors(1, margin: 10f), 1, beamWidth: 1, maxDepth: 20);

        Assert.Null(narrow);
    }

    [Fact]
    public void ADepthLimitIsRespected()
        => Assert.Null(PolicyBeamSearch.Solve(new Line(50), Priors(0), 0, beamWidth: 8, maxDepth: 10));

    [Fact]
    public void AZeroTimeBudgetStopsImmediately()
        => Assert.Null(PolicyBeamSearch.Solve(new Line(900), Priors(0), 0, beamWidth: 8, maxDepth: 1_000,
                                              maxTime: TimeSpan.Zero));

    [Fact]
    public void ADeadEndedBeamGivesUpRatherThanSpinning()
    {
        // Every successor is already visited, so the candidate set empties. Without the explicit break this
        // would idle to maxDepth doing nothing — cheap here, minutes of a training run on a real board.
        var path = PolicyBeamSearch.Solve(new Line(3), Priors(1), 3, beamWidth: 8, maxDepth: 1_000_000);

        Assert.Equal([], path);   // start IS the goal on a 3-line starting at 3
    }
}
