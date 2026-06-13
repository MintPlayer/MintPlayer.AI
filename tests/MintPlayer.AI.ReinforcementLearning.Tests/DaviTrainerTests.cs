using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The deep approximate value-iteration (DAVI) trainer — the SDK's teacher-free,
/// self-improvement path (distinct from the exact tabular Solvers.ValueIteration). Two checks:
/// the greedy policy/lookahead is correct given an exact value (deterministic, fast), and the
/// full DAVI loop actually learns to solve shallow cubes with no Kociemba (slow, gated) —
/// validated against the BFS optimum.
/// </summary>
public class DaviTrainerTests
{
    private static float[] Featurize(FaceletCube cube)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);
        return obs;
    }

    [Fact]
    public void GreedyPlanner_WithExactDistance_DescendsOptimally()
    {
        // An exact cost-to-go (BFS distance, cached) must make the greedy policy solve in the
        // optimal number of moves — this isolates the lookahead/argmin/solver logic from learning.
        var model = new CubeModel();
        var cache = new Dictionary<string, float>();
        float ExactCost(FaceletCube state)
        {
            string key = model.StateKey(state);
            if (cache.TryGetValue(key, out float cached)) return cached;
            var optimal = BreadthFirstPlanner.FindOptimal(model, state, maxDepth: 5);
            float cost = optimal?.Count ?? 1000f;
            cache[key] = cost;
            return cost;
        }

        for (int depth = 1; depth <= 3; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(2000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            if (cube.IsSolved) continue;

            int optimalLength = BreadthFirstPlanner.FindOptimal(model, cube, maxDepth: 5)!.Count;
            var solution = GreedyValuePlanner.Solve(model, ExactCost, cube, maxSteps: depth + 3);

            Assert.NotNull(solution);
            Assert.Equal(optimalLength, solution!.Count); // exact value ⇒ optimal descent
            var replay = cube.Clone();
            foreach (int action in solution) replay.ApplyQuarterTurn(action);
            Assert.True(replay.IsSolved);
        }
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Davi_LearnsToSolveShallowCubes_TeacherFree()
    {
        var model = new CubeModel();
        var sampleRng = new Xoshiro256StarStar(123);
        const int maxScramble = 3;

        var net = new Mlp([RubiksCubeEnv.ObservationSize, 256, 256, 1], new Xoshiro256StarStar(7), Activation.Relu);
        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, new ValueIterationOptions
        {
            BatchSize = 128,
            LearningRate = 1e-3f,
            DistanceScale = 1f,  // predict raw cost-to-go: well-separated targets (1,2,3) → robust argmin
            TargetUpdateInterval = 100,
        });

        FaceletCube Sample()
        {
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(sampleRng, 1 + sampleRng.NextInt(maxScramble), quarterTurnsOnly: true));
            return cube;
        }

        trainer.Train(Sample, iterations: 6000);

        int solved = 0, total = 0;
        for (int depth = 1; depth <= maxScramble; depth++)
            for (int episode = 0; episode < 10; episode++)
            {
                var evalRng = new Xoshiro256StarStar((ulong)(50_000 + 100 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                total++;
                if (cube.IsSolved || trainer.Solve(cube, maxSteps: depth + 6) is not null) solved++;
            }

        // Teacher-free: the net learned cost-to-go purely from the model and the goal.
        Assert.True(solved >= (int)(0.8 * total), $"DAVI solved only {solved}/{total} shallow cubes");
    }
}
