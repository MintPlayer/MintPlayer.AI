using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class RushHourImitationTests
{
    // Hand-verified: red at (2,0), vertical truck col 2 rows 0-2; optimal = 7
    // (truck down ×3 = action 3, then red right ×4 = action 1).
    private static RushHourPuzzle BlockedByTruck() => new([
        new Vehicle(2, 0, 2, Horizontal: true),
        new Vehicle(0, 2, 3, Horizontal: false),
    ]);

    [Fact]
    public void Oracle_LabelsEveryReachableState_WithExactDistances()
    {
        var puzzle = BlockedByTruck();
        var labeled = RushHourOracle.LabelReachableStates(puzzle);

        Assert.NotNull(labeled);
        var start = labeled.Single(s => s.Positions.SequenceEqual(RushHourBoard.InitialPositions(puzzle)));
        Assert.Equal(7, start.DistanceToGoal);          // matches the BFS solver
        Assert.Equal(3, start.OptimalAction);           // truck down is the only optimal first move

        // Every label is self-consistent: applying the optimal action lands on a state
        // labeled exactly one move closer (or on the goal).
        var byKey = labeled.ToDictionary(s => RushHourSolver.Encode(s.Positions));
        foreach (var state in labeled)
        {
            var next = (int[])state.Positions.Clone();
            next[state.OptimalAction / 2] += state.OptimalAction % 2 == 0 ? -1 : 1;
            int nextDistance = RushHourBoard.IsSolved(puzzle, next)
                ? 0
                : byKey[RushHourSolver.Encode(next)].DistanceToGoal;
            Assert.Equal(state.DistanceToGoal - 1, nextDistance);
        }
    }

    [Fact]
    public void Oracle_ReturnsNull_ForUnsolvableConfigs()
    {
        var unsolvable = new RushHourPuzzle([
            new Vehicle(2, 0, 2, Horizontal: true),
            new Vehicle(2, 4, 2, Horizontal: true), // horizontal blocker on the exit row
        ]);
        Assert.Null(RushHourOracle.LabelReachableStates(unsolvable));
    }

    [Fact]
    public void PolicyNet_CheckpointRoundTrip_IsBitwiseIdentical()
    {
        var original = new RushHourPolicyNet(new Xoshiro256StarStar(5), hidden: 64);
        using var stream = new MemoryStream();
        original.Save(stream);
        stream.Position = 0;
        var restored = RushHourPolicyNet.Load(stream);

        var puzzle = BlockedByTruck();
        var positions = RushHourBoard.InitialPositions(puzzle);
        var (logitsA, distanceA) = original.Evaluate(puzzle, positions);
        var (logitsB, distanceB) = restored.Evaluate(puzzle, positions);
        Assert.Equal(logitsA, logitsB);
        Assert.Equal(distanceA, distanceB);
    }

    [Fact]
    public void Search_SolvesWithinBudget_EvenWithAnUntrainedHeuristic()
    {
        // With a useless heuristic A* degrades toward uniform-cost search — it must
        // still find a valid solution on a small graph, just with more expansions.
        var net = new RushHourPolicyNet(new Xoshiro256StarStar(11), hidden: 32);
        var result = RushHourPolicySearch.Solve(net, BlockedByTruck(), maxExpansions: 10_000);

        Assert.True(result.Solved);
        var positions = RushHourBoard.InitialPositions(BlockedByTruck());
        foreach (int action in result.Actions)
        {
            Assert.True(RushHourBoard.ActionMask(BlockedByTruck(), positions)[action]);
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
        }
        Assert.True(RushHourBoard.IsSolved(BlockedByTruck(), positions));
        Assert.True(result.Actions.Length >= 7, "shorter than the proven optimum — impossible");
    }
}
