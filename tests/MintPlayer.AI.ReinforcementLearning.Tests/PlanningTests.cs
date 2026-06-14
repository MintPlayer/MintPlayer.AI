using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The teacher-free optimal planner (foundation for the value-iteration trainer and its
/// ground-truth oracle). BFS over the cube model must return a fewest-action solution that
/// actually solves shallow scrambles, with no help from Kociemba.
/// </summary>
public class PlanningTests
{
    [Fact]
    public void Bfs_SolvesShallowScrambles_Optimally()
    {
        var model = new CubeModel();
        for (int depth = 1; depth <= 5; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(1000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            if (cube.IsSolved) continue; // rare cancellation to identity

            var solution = BreadthFirstPlanner.FindOptimal(model, cube, maxDepth: depth);

            Assert.NotNull(solution);
            Assert.True(solution!.Count <= depth, $"depth {depth}: solution {solution.Count} > scramble {depth}");

            var replay = cube.Clone();
            foreach (int action in solution) replay.ApplyQuarterTurn(action);
            Assert.True(replay.IsSolved, $"depth {depth}: BFS solution did not solve the cube");
        }
    }

    [Fact]
    public void Bfs_ReturnsEmpty_ForAnAlreadySolvedCube()
    {
        var solution = BreadthFirstPlanner.FindOptimal(new CubeModel(), new FaceletCube(), maxDepth: 5);
        Assert.NotNull(solution);
        Assert.Empty(solution!);
    }

    [Fact]
    public void Bfs_ReturnsNull_WhenNoSolutionWithinDepth()
    {
        // A depth-5 scramble cannot be solved in 1 move, so a too-shallow search must give up.
        var rng = new Xoshiro256StarStar(42);
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(rng, 5, quarterTurnsOnly: true));

        Assert.Null(BreadthFirstPlanner.FindOptimal(new CubeModel(), cube, maxDepth: 1));
    }

    [Fact]
    public void ValueGuidedSearch_WithExactValue_FindsOptimal()
    {
        // With an exact cost-to-go and weight 1, weighted A* is optimal — same length as BFS.
        var model = new CubeModel();
        var cache = new Dictionary<string, float>();
        float ExactCost(FaceletCube state)
        {
            string key = model.StateKey(state);
            if (cache.TryGetValue(key, out float cached)) return cached;
            float cost = BreadthFirstPlanner.FindOptimal(model, state, maxDepth: 5)?.Count ?? 1000f;
            cache[key] = cost;
            return cost;
        }

        for (int depth = 1; depth <= 4; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(3000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            if (cube.IsSolved) continue;

            int optimal = BreadthFirstPlanner.FindOptimal(model, cube, maxDepth: 5)!.Count;
            var solution = ValueGuidedSearch.Solve(model, ExactCost, cube, maxExpansions: 5_000, weight: 1f);

            Assert.NotNull(solution);
            Assert.Equal(optimal, solution!.Count);
            var replay = cube.Clone();
            foreach (int action in solution) replay.ApplyQuarterTurn(action);
            Assert.True(replay.IsSolved);
        }
    }

    [Fact]
    public void ValueGuidedSearch_ReturnsNull_WhenExpansionBudgetTooSmall()
    {
        // Zero heuristic (uninformed) + a 1-expansion budget can't reach a depth-4 goal.
        var rng = new Xoshiro256StarStar(99);
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(rng, 4, quarterTurnsOnly: true));

        Assert.Null(ValueGuidedSearch.Solve(new CubeModel(), _ => 0f, cube, maxExpansions: 1));
    }
}
