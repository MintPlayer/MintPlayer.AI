using RLNet.Core.Agents.Tabular;
using RLNet.Core.Random;
using RLNet.Core.Schedules;
using RLNet.Core.Solvers;
using RLNet.Core.Training;
using RLNet.Environments;

ulong masterSeed = args.Length > 0 && ulong.TryParse(args[0], out var s) ? s : 42UL;
bool animate = !Console.IsOutputRedirected;

Console.WriteLine("RL.NET demo — tabular Q-learning (milestone M1)");
Console.WriteLine($"master seed: {masterSeed}");
Console.WriteLine();

// ---------------------------------------------------------------------------
// Part 1 — GridWorld (deterministic): sanity-check against the exact solution
// ---------------------------------------------------------------------------
Console.WriteLine("=== GridWorld 4x4 (deterministic, step -0.04, goal +1) ===");
{
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
    Console.WriteLine("learned policy:                value iteration values:");
    PrintPolicyAndValues(env, agent, oracle);

    if (animate) AnimateGreedyEpisode(env, agent, seeds.Derive(RngStreams.Evaluation), "GridWorld — greedy playback");
}

// ---------------------------------------------------------------------------
// Part 2 — FrozenLake (slippery): learning under stochastic dynamics
// ---------------------------------------------------------------------------
Console.WriteLine();
Console.WriteLine("=== FrozenLake 4x4 (slippery 1/3-1/3-1/3, Gymnasium-comparable) ===");
{
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
    var oracle = ValueIteration.Solve(env, gamma: 0.99);
    Console.WriteLine($"greedy success rate: {eval.SuccessRate():P1} over 1000 episodes (solved threshold: 70%)");
    Console.WriteLine();
    Console.WriteLine("learned policy:                value iteration values:");
    PrintPolicyAndValues(env, agent, oracle);

    if (animate)
        for (int i = 1; i <= 3; i++)
            AnimateGreedyEpisode(env, agent, seeds.Derive(RngStreams.Evaluation + i), $"FrozenLake — greedy playback {i}/3");
}

Console.WriteLine();
Console.WriteLine("done. (S start, F frozen/free, H hole, G goal, @ agent, <v>^ greedy action)");
return;

static void PrintPolicyAndValues(GridEnvironmentBase env, TabularAgent agent, ValueIterationResult oracle)
{
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

static void AnimateGreedyEpisode(GridEnvironmentBase env, TabularAgent agent, ulong seed, string title)
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
