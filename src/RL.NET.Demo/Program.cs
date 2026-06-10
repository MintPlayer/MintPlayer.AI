using RLNet.Core.Agents;
using RLNet.Core.Agents.Tabular;
using RLNet.Core.Environments;
using RLNet.Core.Random;
using RLNet.Core.Schedules;
using RLNet.Core.Solvers;
using RLNet.Core.Training;
using RLNet.Environments;

// Usage: RL.NET.Demo [grid|lake|cartpole ...] [seed]
//        no env args = run everything.
var selected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
ulong masterSeed = 42;
foreach (var arg in args)
{
    if (ulong.TryParse(arg, out var s)) masterSeed = s;
    else selected.Add(arg);
}
bool ShouldRun(string name) => selected.Count == 0 || selected.Contains(name);
bool animate = !Console.IsOutputRedirected;

Console.WriteLine("RL.NET demo");
Console.WriteLine($"master seed: {masterSeed}   (usage: RL.NET.Demo [grid|lake|cartpole|ppo ...] [seed])");
Console.WriteLine();

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

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var result = DqnTrainer.Train(env, new DqnOptions
    {
        MaxSteps = 150_000,
        SolveThreshold = 475,
        OnProgress = p => Console.WriteLine(
            $"  step {p.Step,7}/{p.MaxSteps}  eval mean return: {p.EvalMeanReturn,6:F1}  epsilon: {p.Epsilon:F3}  loss: {p.LastLoss:F4}"),
    }, seeds);
    sw.Stop();

    var eval = Evaluator.Evaluate(env, result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
    Console.WriteLine($"trained {result.StepsTrained:N0} env steps in {sw.Elapsed.TotalSeconds:F1} s");
    Console.WriteLine($"final greedy eval: {eval.MeanReturn:F1} mean return over 100 episodes " +
                      $"({(eval.MeanReturn >= 475 ? "SOLVED" : "not solved")})");
    Console.WriteLine();
    if (animate) AnimateCartPole(env, result.Agent, seeds.Derive(RngStreams.Evaluation + 1));
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

Console.WriteLine("done.");
return;

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
