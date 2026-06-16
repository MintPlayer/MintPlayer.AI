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
    public void BatchedSearch_WithExactValue_IsOptimal()
    {
        // BWAS (goal-on-pop) with weight=1 and an exact, admissible value (BFS distance) must return
        // OPTIMAL-length solutions — the Tier-1 "provably QTM-optimal" guarantee, isolated from learning.
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
        float[] BatchExact(IReadOnlyList<FaceletCube> states)
        {
            var r = new float[states.Count];
            for (int i = 0; i < states.Count; i++) r[i] = ExactCost(states[i]);
            return r;
        }

        for (int depth = 1; depth <= 3; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(8000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            if (cube.IsSolved) continue;

            int optimalLength = BreadthFirstPlanner.FindOptimal(model, cube, maxDepth: 5)!.Count;
            var solution = ValueGuidedSearch.SolveBatched(model, BatchExact, cube, maxExpansions: 2000, weight: 1f, expandBatch: 16);

            Assert.NotNull(solution);
            Assert.Equal(optimalLength, solution!.Count); // weight=1 + admissible value ⇒ optimal
            var replay = cube.Clone();
            foreach (int action in solution) replay.ApplyQuarterTurn(action);
            Assert.True(replay.IsSolved);
        }
    }

    [Fact]
    public void BatchedSearch_SolvesWhatNonBatchedSolves()
    {
        // The batched search must reach the goal on the same shallow cubes the non-batched search does
        // (same model, same value) — batching changes throughput, not reachability.
        var model = new CubeModel();
        var net = new Mlp([RubiksCubeEnv.ObservationSize, 64, 1], new Xoshiro256StarStar(13), Activation.Relu);
        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, new Adam(net.Parameters(), 1e-3f),
            new ValueIterationOptions { DistanceScale = 1f });

        for (int depth = 1; depth <= 4; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(9000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));

            var nonBatched = trainer.SolveWithSearch(cube, maxExpansions: 20_000, weight: 2f);
            var batched = trainer.SolveWithSearchBatched(cube, maxExpansions: 20_000, weight: 2f, expandBatch: 32);

            Assert.Equal(nonBatched is null, batched is null);
            if (batched is not null)
            {
                var replay = cube.Clone();
                foreach (int action in batched) replay.ApplyQuarterTurn(action);
                Assert.True(replay.IsSolved);
            }
        }
    }

    [Fact]
    public void BatchedGreedySolve_MatchesPerSuccessorSolve()
    {
        // trainer.Solve now batches each step's successors into one forward; it must produce the
        // exact same path as evaluating successors one at a time (independent rows ⇒ same values).
        var model = new CubeModel();
        var net = new Mlp([RubiksCubeEnv.ObservationSize, 64, 1], new Xoshiro256StarStar(5), Activation.Relu);
        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, new Adam(net.Parameters(), 1e-3f),
            new ValueIterationOptions { DistanceScale = 1f });

        for (int depth = 1; depth <= 4; depth++)
        {
            var rng = new Xoshiro256StarStar((ulong)(4000 + depth));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));

            var batched = trainer.Solve(cube, maxSteps: depth + 4);
            var perSuccessor = GreedyValuePlanner.Solve(model, trainer.Value, cube, maxSteps: depth + 4);

            Assert.Equal(perSuccessor is null, batched is null);
            if (perSuccessor is not null) Assert.Equal(perSuccessor, batched);
        }
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void BatchedGreedySolve_IsFasterThanPerSuccessor()
    {
        // Quantify the eval optimization on the campaign's net shape (1024×3).
        var model = new CubeModel();
        var net = new Mlp([RubiksCubeEnv.ObservationSize, 1024, 1024, 1024, 1], new Xoshiro256StarStar(9), Activation.Relu);
        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, new Adam(net.Parameters(), 1e-3f),
            new ValueIterationOptions { DistanceScale = 1f });

        var cubes = new List<FaceletCube>();
        for (int i = 0; i < 40; i++)
        {
            var c = new FaceletCube();
            c.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar((ulong)(6000 + i)), 6, quarterTurnsOnly: true));
            cubes.Add(c);
        }

        foreach (var c in cubes) { trainer.Solve(c, 14); GreedyValuePlanner.Solve(model, trainer.Value, c, 14); } // warm up

        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (var c in cubes) GreedyValuePlanner.Solve(model, trainer.Value, c, maxSteps: 14);
        sw.Stop();
        double perSuccessorMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        foreach (var c in cubes) trainer.Solve(c, maxSteps: 14);
        sw.Stop();
        double batchedMs = sw.Elapsed.TotalMilliseconds;

        System.Console.WriteLine($"[eval perf] per-successor {perSuccessorMs:F0} ms, batched {batchedMs:F0} ms, speedup {perSuccessorMs / batchedMs:F2}×");
        Assert.True(batchedMs < perSuccessorMs, $"batched ({batchedMs:F0} ms) should beat per-successor ({perSuccessorMs:F0} ms)");
    }

    [Fact]
    public void BatchedSearch_TimeDeadline_StopsEarly()
    {
        // A wall-clock budget bounds the search regardless of the expansion ceiling: a zero time budget trips
        // the per-round deadline check before any expansion (honest null), while the same search with no time
        // limit solves the cube — so the EFFECTIVE budget is the deadline, with maxExpansions only a ceiling.
        var model = new CubeModel();
        float[] ZeroHeuristic(IReadOnlyList<FaceletCube> states) => new float[states.Count]; // f = g ⇒ plain BFS
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(4242), 3, quarterTurnsOnly: true));

        var stopped = ValueGuidedSearch.SolveBatched(model, ZeroHeuristic, cube, maxExpansions: 1_000_000, weight: 2f, expandBatch: 32, maxTime: TimeSpan.Zero);
        Assert.Null(stopped); // deadline tripped before expanding

        var solved = ValueGuidedSearch.SolveBatched(model, ZeroHeuristic, cube, maxExpansions: 1_000_000, weight: 2f, expandBatch: 32, maxTime: null);
        Assert.NotNull(solved); // no time limit ⇒ the same search reaches the goal
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Davi_LearnsToSolveShallowCubes_TeacherFree()
    {
        var model = new CubeModel();
        var sampleRng = new Xoshiro256StarStar(123);
        const int maxScramble = 3;

        var net = new Mlp([RubiksCubeEnv.ObservationSize, 256, 256, 1], new Xoshiro256StarStar(7), Activation.Relu);
        var options = new ValueIterationOptions
        {
            BatchSize = 128,
            LearningRate = 1e-3f,
            DistanceScale = 1f,  // predict raw cost-to-go: well-separated targets (1,2,3) → robust argmin
            TargetUpdateInterval = 100,
        };
        var trainer = new ValueIterationTrainer<FaceletCube>(model, Featurize, net, new Adam(net.Parameters(), options.LearningRate), options);

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
