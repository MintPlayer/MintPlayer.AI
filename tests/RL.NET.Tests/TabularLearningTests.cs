using RLNet.Core.Agents.Tabular;
using RLNet.Core.Random;
using RLNet.Core.Schedules;
using RLNet.Core.Solvers;
using RLNet.Core.Training;
using RLNet.Environments;

namespace RLNet.Tests;

public class TabularLearningTests
{
    private static QLearningAgent TrainGridWorld(ulong masterSeed)
    {
        var seeds = new SeedSequence(masterSeed);
        var env = new GridWorldEnv();
        var agent = new QLearningAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };
        TabularTrainer.Train(env, agent, new TabularTrainingOptions
        {
            Episodes = 3000,
            Epsilon = new LinearSchedule(1.0, 0.01, 2000),
            Alpha = new LinearSchedule(0.5, 0.1, 2000),
        }, seeds.Derive(RngStreams.Environment));
        return agent;
    }

    [Fact]
    public void QLearning_GridWorld_GreedyPolicyIsOptimal_EveryState()
    {
        var env = new GridWorldEnv();
        var oracle = ValueIteration.Solve(env, gamma: 0.99);
        var agent = TrainGridWorld(masterSeed: 42);

        for (int s = 0; s < env.StateCount; s++)
        {
            if (env.IsTerminal(s)) continue;
            Assert.True(
                oracle.IsOptimalAction(s, agent.GreedyAction(s)),
                $"Greedy action in state {s} is not optimal per value iteration.");
        }
    }

    [Fact]
    public void QLearning_GridWorld_QValuesApproachOracle()
    {
        var env = new GridWorldEnv();
        var oracle = ValueIteration.Solve(env, gamma: 0.99);
        var agent = TrainGridWorld(masterSeed: 42);

        // Along the greedy trajectory Q-values should be near the exact optimum.
        Assert.Equal(oracle.Values[14], agent.Q[14, agent.GreedyAction(14)], 1e-2);
    }

    [Fact]
    public void Sarsa_GridWorld_GreedyPolicyIsOptimal()
    {
        var seeds = new SeedSequence(42);
        var env = new GridWorldEnv();
        var oracle = ValueIteration.Solve(env, gamma: 0.99);
        var agent = new SarsaAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };
        TabularTrainer.Train(env, agent, new TabularTrainingOptions
        {
            Episodes = 5000,
            Epsilon = new LinearSchedule(1.0, 0.01, 4000),
            Alpha = new LinearSchedule(0.5, 0.05, 4000),
        }, seeds.Derive(RngStreams.Environment));

        for (int s = 0; s < env.StateCount; s++)
        {
            if (env.IsTerminal(s)) continue;
            Assert.True(
                oracle.IsOptimalAction(s, agent.GreedyAction(s)),
                $"SARSA greedy action in state {s} is not optimal per value iteration.");
        }
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void QLearning_FrozenLake_MedianSuccessRate_MeetsGymnasiumThreshold()
    {
        // Gate from PLAN.md M1: success ≥ 0.70 over eval episodes, median of 3 seeds.
        var rates = new List<double>();
        foreach (ulong masterSeed in new ulong[] { 42, 43, 44 })
        {
            var seeds = new SeedSequence(masterSeed);
            var env = new FrozenLakeEnv();
            var agent = new QLearningAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };
            TabularTrainer.Train(env, agent, new TabularTrainingOptions
            {
                Episodes = 100_000,
                Epsilon = new LinearSchedule(1.0, 0.01, 80_000),
                Alpha = new LinearSchedule(0.25, 0.01, 80_000),
            }, seeds.Derive(RngStreams.Environment));

            var eval = Evaluator.Evaluate(env, agent, episodes: 1000, seeds.Derive(RngStreams.Evaluation));
            rates.Add(eval.SuccessRate());
        }

        rates.Sort();
        double median = rates[1];
        Assert.True(median >= 0.70, $"Median success rate {median:P1} (all: {string.Join(", ", rates.Select(r => r.ToString("P1")))})");
    }

    // Determinism is checked on the stochastic env with a short run: on deterministic
    // GridWorld a fully-converged Q-table is seed-independent (it reaches the exact
    // fixed point), so only a partially-trained stochastic run can distinguish seeds.
    private static QLearningAgent TrainFrozenLakeShort(ulong masterSeed)
    {
        var seeds = new SeedSequence(masterSeed);
        var env = new FrozenLakeEnv();
        var agent = new QLearningAgent(env.StateCount, env.ActionCount, seeds.CreateRng(RngStreams.Policy)) { Gamma = 0.99 };
        TabularTrainer.Train(env, agent, new TabularTrainingOptions
        {
            Episodes = 2000,
            Epsilon = new LinearSchedule(1.0, 0.1, 1500),
            Alpha = new LinearSchedule(0.25, 0.05, 1500),
        }, seeds.Derive(RngStreams.Environment));
        return agent;
    }

    [Fact]
    public void Training_IsBitwiseDeterministic_GivenSameMasterSeed()
    {
        var first = TrainFrozenLakeShort(masterSeed: 7);
        var second = TrainFrozenLakeShort(masterSeed: 7);

        Assert.Equal(first.Q.Cast<double>(), second.Q.Cast<double>());
    }

    [Fact]
    public void Training_Differs_GivenDifferentMasterSeed()
    {
        var first = TrainFrozenLakeShort(masterSeed: 7);
        var second = TrainFrozenLakeShort(masterSeed: 8);

        Assert.NotEqual(first.Q.Cast<double>(), second.Q.Cast<double>());
    }
}
