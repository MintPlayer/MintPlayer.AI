using MintPlayer.AI.ReinforcementLearning.Core.Agents;
using MintPlayer.AI.ReinforcementLearning.Core.Agents.Tabular;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Solvers;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

// Usage: RLDemo.Console [grid|lake|cartpole|ppo|2048|2048dqn|rushhour|cube ...] [seed] [--load] [--save] [--data <dir>]
//        no env args = run everything except 2048dqn (DQN needs a long budget there).
//        --load: skip training when the model store has a checkpoint; --save: checkpoint after training.
string[] knownSections = ["grid", "lake", "cartpole", "ppo", "2048", "2048dqn", "rushhour", "cube"];
var selected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
ulong masterSeed = 42;
bool loadModels = false, saveModels = false;
string dataDir = "data";
for (int argIndex = 0; argIndex < args.Length; argIndex++)
{
    var arg = args[argIndex];
    if (knownSections.Contains(arg, StringComparer.OrdinalIgnoreCase)) selected.Add(arg);
    else if (arg.Equals("--load", StringComparison.OrdinalIgnoreCase)) loadModels = true;
    else if (arg.Equals("--save", StringComparison.OrdinalIgnoreCase)) saveModels = true;
    else if (arg.Equals("--data", StringComparison.OrdinalIgnoreCase) && argIndex + 1 < args.Length) dataDir = args[++argIndex];
    else if (ulong.TryParse(arg, out var s)) masterSeed = s;
    else Console.WriteLine($"(ignoring unknown argument '{arg}')");
}
bool ShouldRun(string name) => selected.Count == 0
    ? name != "2048dqn"
    : selected.Contains(name);
bool animate = !Console.IsOutputRedirected;
var store = new FileModelStore(dataDir);

Console.WriteLine("MintPlayer.AI.ReinforcementLearning demo");
Console.WriteLine($"master seed: {masterSeed}   (usage: RLDemo.Console [{string.Join('|', knownSections)} ...] [seed] [--load] [--save] [--data <dir>])");
if (loadModels || saveModels)
    Console.WriteLine($"model store: {store.RootDirectory}  (load: {loadModels}, save: {saveModels})");
Console.WriteLine();

Mlp? TryLoadMlp(string envId, string algoId)
{
    if (!loadModels) return null;
    using var stream = store.TryOpenRead(envId, algoId);
    if (stream is null) return null;
    Console.WriteLine($"loaded '{envId}.{algoId}' from {store.PathOf(envId, algoId)} — skipping training");
    return MlpCheckpoint.Load(stream);
}

void SaveMlp(string envId, string algoId, Mlp network)
{
    if (!saveModels) return;
    store.Save(envId, algoId, s => MlpCheckpoint.Save(network, s));
    Console.WriteLine($"saved '{envId}.{algoId}' to {store.PathOf(envId, algoId)}");
}

if (ShouldRun("grid"))
{
    Console.WriteLine("=== GridWorld 4x4 — tabular Q-learning (deterministic, step -0.04, goal +1) ===");
    var seeds = new SeedSequence(masterSeed);
    var env = new GridWorldEnv();
    var agent = new QLearningAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var result = TabularTrainer.Train(env, agent, new TabularTrainingOptions
    {
        Episodes = 3000,
        Epsilon = new LinearSchedule(1.0, 0.01, 2000),
        Alpha = new LinearSchedule(0.5, 0.1, 2000),
    }, seeds.Derive(RngStreams.Environment));
    sw.Stop();

    var oracle = ValueIteration.Solve(env, gamma: 0.99);
    int optimalStates = Enumerable.Range(0, env.StateCount)
        .Count(state => env.IsTerminal(state) || oracle.IsOptimalAction(state, agent.GreedyAction(state)));

    Console.WriteLine($"trained 3000 episodes ({result.TotalSteps:N0} steps) in {sw.ElapsedMilliseconds} ms");
    Console.WriteLine($"greedy policy optimal (vs value iteration) in {optimalStates}/{env.StateCount} states");
    Console.WriteLine();
    PrintPolicyAndValues(env, agent, oracle);
    if (animate) AnimateGridEpisode(env, agent, seeds.Derive(RngStreams.Evaluation), "GridWorld — greedy playback");
    Console.WriteLine();
}

if (ShouldRun("lake"))
{
    Console.WriteLine("=== FrozenLake 4x4 — tabular Q-learning (slippery 1/3-1/3-1/3, Gymnasium-comparable) ===");
    var seeds = new SeedSequence(masterSeed);
    var env = new FrozenLakeEnv();
    var agent = new QLearningAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var result = TabularTrainer.Train(env, agent, new TabularTrainingOptions
    {
        Episodes = 100_000,
        Epsilon = new LinearSchedule(1.0, 0.01, 80_000),
        Alpha = new LinearSchedule(0.25, 0.01, 80_000),
        ProgressInterval = 20_000,
        OnProgress = p => Console.WriteLine(
            $"  episode {p.Episode,6}/{p.TotalEpisodes}  avg return (last 100): {p.AvgReturn100:F2}  epsilon: {p.Epsilon:F3}"),
    }, seeds.Derive(RngStreams.Environment));
    sw.Stop();
    Console.WriteLine($"trained 100,000 episodes ({result.TotalSteps:N0} steps) in {sw.ElapsedMilliseconds} ms");

    var eval = Evaluator.Evaluate(env, agent, episodes: 1000, seeds.Derive(RngStreams.Evaluation));
    Console.WriteLine($"greedy success rate: {eval.SuccessRate():P1} over 1000 episodes (solved threshold: 70%)");
    Console.WriteLine();
    PrintPolicyAndValues(env, agent, ValueIteration.Solve(env, gamma: 0.99));
    if (animate)
        for (int i = 1; i <= 2; i++)
            AnimateGridEpisode(env, agent, seeds.Derive(RngStreams.Evaluation + i), $"FrozenLake — greedy playback {i}/2");
    Console.WriteLine();
}

if (ShouldRun("cartpole"))
{
    Console.WriteLine("=== CartPole-v1 — Double DQN from scratch (solved: mean return >= 475/500) ===");
    var seeds = new SeedSequence(masterSeed);
    var env = new CartPoleEnv();

    GreedyQAgent agent;
    if (TryLoadMlp("cartpole", "dqn") is { } network)
    {
        agent = new GreedyQAgent(network, 2);
    }
    else
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = DqnTrainer.Train(env, new DqnOptions
        {
            MaxSteps = 150_000,
            SolveThreshold = 475,
            OnProgress = p => Console.WriteLine(
                $"  step {p.Step,7}/{p.MaxSteps}  eval mean return: {p.EvalMeanReturn,6:F1}  epsilon: {p.Epsilon:F3}  loss: {p.LastLoss:F4}"),
        }, seeds);
        sw.Stop();
        Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalSeconds:F1} s");
        agent = result.Agent;
        SaveMlp("cartpole", "dqn", (Mlp)result.Network);
    }

    var eval = Evaluator.Evaluate(env, agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
    Console.WriteLine($"final greedy eval: {eval.MeanReturn:F1} mean return over 100 episodes " +
                      $"({(eval.MeanReturn >= 475 ? "SOLVED" : "not solved")})");
    Console.WriteLine();
    if (animate) AnimateCartPole(env, agent, seeds.Derive(RngStreams.Evaluation + 1));
    Console.WriteLine();
}

if (ShouldRun("ppo"))
{
    Console.WriteLine("=== CartPole-v1 — PPO from scratch (8 vectorized envs, GAE, clipped surrogate) ===");
    var seeds = new SeedSequence(masterSeed);
    var evalEnv = new CartPoleEnv();

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var result = PpoTrainer.Train(_ => new CartPoleEnv(), evalEnv, new PpoOptions
    {
        TotalSteps = 400_000,
        SolveThreshold = 475,
        ParallelEnvs = true,
        OnProgress = p =>
        {
            if (p.EnvSteps % 20_480 == 0)
                Console.WriteLine(
                    $"  step {p.EnvSteps,7}/{p.TotalSteps}  avg return (last 100 eps): {p.AvgReturn100,6:F1}  " +
                    $"kl: {p.ApproxKl:F4}  clip: {p.ClipFraction:P0}  expl.var: {p.ExplainedVariance:F2}  lr: {p.LearningRate:E1}");
        },
    }, seeds);
    sw.Stop();

    var eval = Evaluator.Evaluate(evalEnv, result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
    Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalSeconds:F1} s");
    Console.WriteLine($"final greedy eval: {eval.MeanReturn:F1} mean return over 100 episodes " +
                      $"({(eval.MeanReturn >= 475 ? "SOLVED" : "not solved")})");
    Console.WriteLine();
    if (animate) AnimateCartPole(evalEnv, result.Agent, seeds.Derive(RngStreams.Evaluation + 1));
    Console.WriteLine();
}

if (ShouldRun("2048"))
{
    Console.WriteLine("=== 2048 — afterstate TD(0) with n-tuple network (17×4-tuples, ~4.5 MB of weights) ===");
    Console.WriteLine("    solved criterion (PRD): reach the 2048 tile in >= 10% of 100 greedy games");

    NTuple2048Agent agent;
    var loadedStream = loadModels ? store.TryOpenRead("2048", "ntuple") : null;
    if (loadedStream is not null)
    {
        using (loadedStream) agent = NTuple2048Agent.Load(loadedStream);
        Console.WriteLine($"loaded '2048.ntuple' from {store.PathOf("2048", "ntuple")} — skipping training");
    }
    else
    {
        agent = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(new SeedSequence(masterSeed).Derive(RngStreams.Policy));

        const int totalGames = 100_000;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int game = 1; game <= totalGames; game++)
        {
            agent.PlayGame(trainRng, learn: true);
            if (game % 10_000 == 0)
            {
                var (rate2048, rate1024, avgScore, bestTile) = Eval2048(agent, new SeedSequence(masterSeed).Derive(RngStreams.Evaluation), games: 100);
                Console.WriteLine($"  game {game,7}/{totalGames}  greedy eval: avg score {avgScore,6:F0}  " +
                                  $"1024-rate {rate1024:P0}  2048-rate {rate2048:P0}  best tile {bestTile}");
            }
        }
        sw.Stop();
        Console.WriteLine($"trained {totalGames:N0} self-play games in {sw.Elapsed.TotalSeconds:F0} s");
        if (saveModels)
        {
            store.Save("2048", "ntuple", s => agent.Save(s));
            Console.WriteLine($"saved '2048.ntuple' to {store.PathOf("2048", "ntuple")}");
        }
    }

    var final = Eval2048(agent, new SeedSequence(masterSeed).Derive(RngStreams.Evaluation), games: 100);
    Console.WriteLine($"final: 2048-rate {final.Rate2048:P0} over 100 games " +
                      $"({(final.Rate2048 >= 0.10 ? "SOLVED" : "not solved")}, target >= 10%)");
    Console.WriteLine();

    // High-score lens: 2048 is a chase-the-largest-tile game, so compare the shipped 1-ply
    // greedy policy against test-time expectimax over the SAME value tables (both reuse `agent`).
    CompareExpectimax2048(agent, new SeedSequence(masterSeed).Derive(RngStreams.Evaluation), games: 100);
    Console.WriteLine();

    if (animate) Animate2048(agent, new SeedSequence(masterSeed).Derive(RngStreams.Evaluation + 1));
    Console.WriteLine();
}

if (ShouldRun("2048dqn"))
{
    Console.WriteLine("=== 2048 — generic masked Double DQN (demonstrates IActionMaskProvider) ===");
    var seeds = new SeedSequence(masterSeed);
    var env = new Env2048();

    GreedyQAgent agent;
    if (TryLoadMlp("2048", "dqn") is { } network)
    {
        agent = new GreedyQAgent(network, 4);
    }
    else
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = DqnTrainer.Train(env, new DqnOptions
        {
            Hidden = [256, 256],
            MaxSteps = 300_000,
            BufferCapacity = 100_000,
            Epsilon = new LinearSchedule(1.0, 0.05, 100_000),
            EvalEvery = 25_000,
            EvalEpisodes = 10,
            OnProgress = p => Console.WriteLine(
                $"  step {p.Step,7}/{p.MaxSteps}  eval mean return: {p.EvalMeanReturn,7:F1}  epsilon: {p.Epsilon:F3}  loss: {p.LastLoss:F4}"),
        }, seeds);
        sw.Stop();
        Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalMinutes:F1} min");
        agent = result.Agent;
        SaveMlp("2048", "dqn", (Mlp)result.Network);
    }

    // Report in game terms: play 50 greedy games, count tiles reached.
    int games2048 = 0, games1024 = 0, games512 = 0;
    double totalScore = 0;
    for (int g = 0; g < 50; g++)
    {
        env.Reset(seeds.Derive(RngStreams.Evaluation) + (ulong)g);
        while (true)
        {
            var step = env.Step(agent.Act(env.CurrentObservation(), env.CurrentActionMask(), greedy: true));
            if (step.Done) break;
        }
        totalScore += env.Score;
        if (env.MaxTile >= 2048) games2048++;
        if (env.MaxTile >= 1024) games1024++;
        if (env.MaxTile >= 512) games512++;
    }
    Console.WriteLine($"50 greedy games: avg score {totalScore / 50:F0}, 512-rate {games512 / 50.0:P0}, " +
                      $"1024-rate {games1024 / 50.0:P0}, 2048-rate {games2048 / 50.0:P0}");
    Console.WriteLine();
}

if (ShouldRun("rushhour"))
{
    Console.WriteLine("=== Rush Hour — masked Double DQN on a 30-puzzle easy set (optimal 4-10 moves) ===");
    Console.WriteLine("    gate (PRD): >= 90% of the set solved within 2x the BFS-optimal move count");
    // The quick M6 showcase set. (The web playground trains its own model on a much
    // larger generated set so it can handle arbitrary user-drawn boards.)
    var puzzles = RushHourGenerator.Generate(seed: 99, count: 30, minOptimal: 4, maxOptimal: 10, varyRedLength: true);
    Console.WriteLine($"generated {puzzles.Count} puzzles, avg optimal {puzzles.Average(p => p.OptimalMoves):F1} moves. Example:");
    Console.WriteLine(RushHourBoard.Render(puzzles[0], RushHourBoard.InitialPositions(puzzles[0])));

    var seeds = new SeedSequence(masterSeed);
    var env = new RushHourEnv(puzzles, maxMoves: 60);

    GreedyQAgent agent;
    if (TryLoadMlp("rushhour", "dqn") is { } network)
    {
        agent = new GreedyQAgent(network, RushHourBoard.ActionCount);
    }
    else
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = DqnTrainer.Train(env, new DqnOptions
        {
            Hidden = [128, 128],
            Gamma = 0.98,
            LearningRate = 5e-4f,
            MaxSteps = 200_000,
            BufferCapacity = 100_000,
            Epsilon = new LinearSchedule(1.0, 0.05, 60_000),
            EvalEvery = 10_000,
            EvalEpisodes = 20,
            SolveThreshold = 88, // mean return ~ solving random puzzles within ~2x optimal
            OnProgress = p => Console.WriteLine(
                $"  step {p.Step,7}/{p.MaxSteps}  eval mean return: {p.EvalMeanReturn,6:F1}  epsilon: {p.Epsilon:F3}  loss: {p.LastLoss:F4}"),
        }, seeds);
        sw.Stop();
        Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalMinutes:F1} min");
        agent = result.Agent;
        SaveMlp("rushhour", "dqn", (Mlp)result.Network);
    }

    var (solvedInBudget, solvedAtAll) = EvaluateRushHourGate(env, agent, puzzles);
    Console.WriteLine($"gate: {solvedInBudget}/{puzzles.Count} puzzles solved within 2x optimal " +
                      $"({solvedInBudget / (double)puzzles.Count:P0}, target >= 90%); {solvedAtAll}/{puzzles.Count} solved within 60 moves");
    Console.WriteLine();
    if (animate) AnimateRushHour(env, agent, puzzleIndex: 0);
    Console.WriteLine();
}

if (ShouldRun("cube"))
{
    Console.WriteLine("=== Rubik's Cube — Double DQN on shallow scrambles (quarter-turn depths 1-6) ===");
    Console.WriteLine("    gate (PRD §11): >= 90% of 100 eval scrambles (depths 1-6) solved within 20 moves");

    var seeds = new SeedSequence(masterSeed);
    var env = new RubiksCubeEnv(maxScrambleDepth: 6, maxMoves: 20);

    GreedyQAgent agent;
    if (TryLoadMlp("cube", "dqn") is { } cubeNetwork)
    {
        agent = new GreedyQAgent(cubeNetwork, RubiksCubeEnv.ActionCount);
    }
    else
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        // Same recipe as the web playground's CubeModelService.
        var result = DqnTrainer.Train(env, new DqnOptions
        {
            Hidden = [256, 256],
            Gamma = 0.99,
            LearningRate = 5e-4f,
            MaxSteps = 600_000,
            BufferCapacity = 200_000,
            Epsilon = new LinearSchedule(1.0, 0.05, 200_000),
            EvalEvery = 10_000,
            EvalEpisodes = 100,
            SolveThreshold = 88, // ~90% solved on the 1-6 band (return = 101 - moves)
            OnProgress = p => Console.WriteLine(
                $"  step {p.Step,7}/{p.MaxSteps}  eval mean return: {p.EvalMeanReturn,6:F1}  epsilon: {p.Epsilon:F3}  loss: {p.LastLoss:F4}"),
        }, seeds);
        sw.Stop();
        Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalMinutes:F1} min");
        agent = result.Agent;
        SaveMlp("cube", "dqn", (Mlp)result.Network);
    }

    // Greedy rollouts first; failures retried with the Q-guided lookahead, mirroring the
    // solve API (aiMode greedy/search — the Rush Hour M11 pattern, honest either way).
    int totalSolved = 0, totalEpisodes = 0, totalGreedy = 0;
    for (int depth = 1; depth <= 6; depth++)
    {
        env.FixedScrambleDepth = depth;
        int greedySolved = 0, searchSolved = 0;
        const int episodes = 100;
        for (int episode = 0; episode < episodes; episode++)
        {
            env.Reset((ulong)(1000 * depth + episode));
            var obs = env.CurrentObservation();
            bool solved = false;
            while (true)
            {
                var step = env.Step(agent.Act(obs, env.CurrentActionMask(), greedy: true));
                obs = step.Observation;
                if (step.Terminated) { solved = true; greedySolved++; break; }
                if (step.Truncated) break;
            }
            if (!solved)
            {
                var start = new FaceletCube();
                start.Apply(env.ScrambleMoves);
                if (CubeQSearch.Solve(agent, start).Solved) searchSolved++;
            }
        }
        env.FixedScrambleDepth = null;
        totalGreedy += greedySolved;
        totalSolved += greedySolved + searchSolved;
        totalEpisodes += episodes;
        Console.WriteLine($"  depth {depth}: {greedySolved}/{episodes} greedy, +{searchSolved} with lookahead = {greedySolved + searchSolved}/{episodes} within 20 moves");
    }
    Console.WriteLine($"gate: {totalSolved}/{totalEpisodes} solved ({totalSolved / (double)totalEpisodes:P1}, target >= 90%); greedy alone {totalGreedy / (double)totalEpisodes:P1}");
    Console.WriteLine();
}

Console.WriteLine("done.");
return;

static (int SolvedInBudget, int SolvedAtAll) EvaluateRushHourGate(RushHourEnv env, GreedyQAgent agent, IReadOnlyList<RushHourPuzzle> puzzles)
{
    int solvedInBudget = 0, solvedAtAll = 0;
    for (int i = 0; i < puzzles.Count; i++)
    {
        env.FixedPuzzleIndex = i;
        env.Reset(1);
        var obs = env.CurrentObservation();
        while (true)
        {
            var step = env.Step(agent.Act(obs, env.CurrentActionMask(), greedy: true));
            obs = step.Observation;
            if (step.Terminated)
            {
                solvedAtAll++;
                if (env.MovesUsed <= 2 * puzzles[i].OptimalMoves) solvedInBudget++;
                break;
            }
            if (step.Truncated) break;
        }
    }
    env.FixedPuzzleIndex = null;
    return (solvedInBudget, solvedAtAll);
}

static void AnimateRushHour(RushHourEnv env, GreedyQAgent agent, int puzzleIndex)
{
    Console.WriteLine($"--- Rush Hour — greedy playback, puzzle {puzzleIndex} (optimal {env.CurrentPuzzle.OptimalMoves} moves) ---");
    env.FixedPuzzleIndex = puzzleIndex;
    env.Reset(1);
    var obs = env.CurrentObservation();
    int frameTop = Console.CursorTop;

    while (true)
    {
        Console.SetCursorPosition(0, frameTop);
        Console.Write(env.RenderString());
        Thread.Sleep(250);

        var step = env.Step(agent.Act(obs, env.CurrentActionMask(), greedy: true));
        obs = step.Observation;

        if (step.Done)
        {
            Console.SetCursorPosition(0, frameTop);
            Console.Write(env.RenderString());
            Console.WriteLine(step.Terminated
                ? $"solved in {env.MovesUsed} moves (optimal {env.CurrentPuzzle.OptimalMoves})!"
                : "move budget exhausted without solving.");
            break;
        }
    }
    env.FixedPuzzleIndex = null;
}

static (double Rate2048, double Rate1024, double AvgScore, int BestTile) Eval2048(
    NTuple2048Agent agent, ulong seed, int games)
{
    var rng = new Xoshiro256StarStar(seed);
    int hits2048 = 0, hits1024 = 0, bestExp = 0;
    double totalScore = 0;
    for (int g = 0; g < games; g++)
    {
        var (score, maxExp) = agent.PlayGame(rng, learn: false);
        totalScore += score;
        if (maxExp >= 11) hits2048++;
        if (maxExp >= 10) hits1024++;
        bestExp = Math.Max(bestExp, maxExp);
    }
    return (hits2048 / (double)games, hits1024 / (double)games, totalScore / games, 1 << bestExp);
}

static void CompareExpectimax2048(NTuple2048Agent agent, ulong seed, int games)
{
    Console.WriteLine($"  high-score A/B over {games} games (greedy 1-ply vs adaptive expectimax, same value tables):");
    var greedy = RunPolicy2048(rng => agent.PlayGame(rng, learn: false), seed, games);
    var emax = RunPolicy2048(new Expectimax2048(agent).PlayGame, seed, games);
    Console.WriteLine($"    greedy    : avg {greedy.AvgScore,7:F0}  best {greedy.BestTile,6}  {greedy.Histogram}  ({greedy.MsPerGame:F1} ms/game)");
    Console.WriteLine($"    expectimax: avg {emax.AvgScore,7:F0}  best {emax.BestTile,6}  {emax.Histogram}  ({emax.MsPerGame:F1} ms/game)");
}

static (double AvgScore, int BestTile, string Histogram, double MsPerGame) RunPolicy2048(
    Func<Xoshiro256StarStar, (int Score, int MaxExponent)> play, ulong seed, int games)
{
    var scores = new int[games];
    var maxExps = new int[games];
    var perGameMs = new double[games];

    // Games are independent and only READ the value tables, so fan them out across cores. Each
    // game is timed on its own thread, so the reported ms/game stays a real single-playout cost
    // (what matters for serving) even though wall-clock is divided by the core count.
    Parallel.For(0, games, g =>
    {
        var rng = new Xoshiro256StarStar(seed + (ulong)g);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        (scores[g], maxExps[g]) = play(rng);
        sw.Stop();
        perGameMs[g] = sw.Elapsed.TotalMilliseconds;
    });

    var counts = new Dictionary<int, int>();
    foreach (var e in maxExps) counts[e] = counts.GetValueOrDefault(e) + 1;
    string histogram = string.Join(" ", counts.Keys.OrderByDescending(e => e).Select(e => $"{1 << e}×{counts[e]}"));
    return (scores.Average(), 1 << maxExps.Max(), histogram, perGameMs.Average());
}

static void Animate2048(NTuple2048Agent agent, ulong seed)
{
    Console.WriteLine("--- 2048 — greedy playback (every 4th move) ---");
    var env = new Env2048();
    env.Reset(seed);
    int frameTop = Console.CursorTop;
    Span<byte> scratch = stackalloc byte[16];

    for (int move = 0; ; move++)
    {
        if (move % 4 == 0)
        {
            Console.SetCursorPosition(0, frameTop);
            Console.Write(env.RenderString());
            Thread.Sleep(50);
        }

        int action = agent.ChooseMove(env.Board, out _, scratch);
        var step = env.Step(action);
        if (step.Done)
        {
            Console.SetCursorPosition(0, frameTop);
            Console.Write(env.RenderString());
            Console.WriteLine($"game over after {move + 1} moves — final score {env.Score}, best tile {env.MaxTile}");
            break;
        }
    }
}

static void PrintPolicyAndValues(GridEnvironmentBase env, TabularAgent agent, ValueIterationResult oracle)
{
    Console.WriteLine("learned policy:                value iteration values:");
    for (int row = 0; row < env.Rows; row++)
    {
        var policy = new System.Text.StringBuilder("  ");
        var values = new System.Text.StringBuilder("  ");
        for (int col = 0; col < env.Cols; col++)
        {
            int state = row * env.Cols + col;
            char cell = env.CellAt(state);
            policy.Append(env.IsTerminal(state) ? cell : "<v>^"[agent.GreedyAction(state)]).Append(' ');
            values.Append(env.IsTerminal(state) ? $"  {cell}   " : $"{oracle.Values[state],6:F2} ");
        }
        Console.WriteLine($"{policy}               {values}");
    }
    Console.WriteLine();
}

static void AnimateGridEpisode(GridEnvironmentBase env, TabularAgent agent, ulong seed, string title)
{
    Console.WriteLine($"--- {title} ---");
    var (state, _) = env.Reset(seed);
    double episodeReturn = 0;
    int steps = 0;
    int frameTop = Console.CursorTop;

    while (true)
    {
        Console.SetCursorPosition(0, frameTop);
        Console.Write(env.RenderString());
        Console.WriteLine($"step {steps,3}  return {episodeReturn,6:F2}   ");
        Thread.Sleep(120);

        var step = env.Step(agent.Act(state, greedy: true));
        episodeReturn += step.Reward;
        steps++;
        state = step.Observation;

        if (step.Done)
        {
            Console.SetCursorPosition(0, frameTop);
            Console.Write(env.RenderString());
            Console.WriteLine($"step {steps,3}  return {episodeReturn,6:F2}  -> {(step.Terminated && episodeReturn > 0 ? "GOAL!" : step.Terminated ? "fell/ended" : "time limit")}");
            break;
        }
    }
    Console.WriteLine();
}

static void AnimateCartPole(CartPoleEnv env, IAgent<float[], int> agent, ulong seed)
{
    const int maxFrames = 250;
    Console.WriteLine($"--- CartPole — greedy playback (showing up to {maxFrames} of 500 steps) ---");
    var (obs, _) = env.Reset(seed);
    int steps = 0;
    int frameTop = Console.CursorTop;

    while (true)
    {
        Console.SetCursorPosition(0, frameTop);
        Console.Write(env.RenderString());
        Console.WriteLine($"step {steps,3}/500   x={obs[0],6:F2}  theta={obs[2] * 180 / Math.PI,6:F1}°   ");
        Thread.Sleep(20);

        var step = env.Step(agent.Act(obs, greedy: true));
        obs = step.Observation;
        steps++;

        if (step.Done || steps >= maxFrames)
        {
            Console.SetCursorPosition(0, frameTop);
            Console.Write(env.RenderString());
            Console.WriteLine(step.Truncated
                ? $"step {steps,3}/500   survived the full episode — pole balanced!"
                : step.Terminated
                    ? $"step {steps,3}/500   pole fell."
                    : $"step {steps,3}/500   still balancing after {maxFrames} shown steps — calling it a win.");
            break;
        }
    }
}
