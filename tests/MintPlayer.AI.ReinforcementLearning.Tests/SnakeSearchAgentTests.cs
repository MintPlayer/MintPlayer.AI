using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class SnakeSearchAgentTests
{
    private static DuelingQNet UntrainedNet() =>
        new(SnakeEnv.ObservationSize, [32, 32], SnakeEnv.ActionCount, new Xoshiro256StarStar(0));

    // Plays a full episode driven by the search agent; returns food eaten. Every move Step accepts proves legality
    // (Step throws on the reversal), so a completed episode is itself the "only legal moves" assertion.
    private static int PlaySearch(SnakeEnv env, ulong seed, SnakeSearchOptions opts)
    {
        var agent = new SnakeSearchAgent(env, UntrainedNet(), opts);
        env.Reset(seed);
        while (true)
        {
            var step = env.Step(agent.Act());
            if (step.Done) break;
        }
        return env.FoodEaten;
    }

    [Fact]
    public void Search_BeatsReactiveGreedy_WithTheSameUntrainedNet()
    {
        // The net is random, so reactive greedy ≈ random play. Pure look-ahead survival (net weight 0) should eat
        // far more on the same net — the whole point of search: it doesn't depend on a good value function to avoid
        // walking into walls/itself. A small grid keeps the (long, healthy) search episode quick.
        var net = UntrainedNet();
        var greedyEnv = new SnakeEnv(7);
        var greedy = new GreedyQAgent(net, SnakeEnv.ActionCount);
        var (obs, _) = greedyEnv.Reset(1);
        while (true)
        {
            var step = greedyEnv.Step(greedy.Act(obs, greedyEnv.CurrentActionMask(), greedy: true));
            obs = step.Observation;
            if (step.Done) break;
        }

        int searchFood = PlaySearch(new SnakeEnv(7), 1, new SnakeSearchOptions { MaxDepth = 6, NetWeight = 0f });

        Assert.True(searchFood > greedyEnv.FoodEaten,
            $"search ate {searchFood}, reactive greedy ate {greedyEnv.FoodEaten} — search should dominate a random net");
        Assert.True(searchFood > 5, $"search only ate {searchFood} — survival look-ahead should sustain a long game");
    }

    [Fact]
    public void Search_IsDeterministic_ForAGivenSeed()
    {
        var opts = new SnakeSearchOptions { MaxDepth = 6 };
        int a = PlaySearch(new SnakeEnv(7), 7, opts);
        int b = PlaySearch(new SnakeEnv(7), 7, opts);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Search_NeverReturnsTheReversal()
    {
        // After moving right a few times, Left is the reversal; the planner must never pick it (Step would throw).
        var env = new SnakeEnv();
        var agent = new SnakeSearchAgent(env, UntrainedNet(), new SnakeSearchOptions { MaxDepth = 6, NetWeight = 0f });
        env.Reset(3);
        for (int i = 0; i < 50; i++)
        {
            int action = agent.Act();
            Assert.True(env.CurrentActionMask()[action], $"planner chose masked action {action}");
            if (env.Step(action).Done) break;
        }
    }
}
