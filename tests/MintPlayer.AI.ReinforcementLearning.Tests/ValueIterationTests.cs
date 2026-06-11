using MintPlayer.AI.ReinforcementLearning.Core.Solvers;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class ValueIterationTests
{
    [Fact]
    public void GridWorld_ValuesIncreaseTowardGoal()
    {
        var env = new GridWorldEnv();
        var result = ValueIteration.Solve(env, gamma: 0.99);

        // State 14 is one step from the goal; state 0 is six steps away.
        Assert.True(result.Values[14] > result.Values[0]);
        // One step from goal: reward +1, no discounting of further value.
        Assert.Equal(1.0, result.Values[14], precision: 9);
    }

    [Fact]
    public void GridWorld_OptimalPolicy_MovesTowardGoal()
    {
        var env = new GridWorldEnv();
        var result = ValueIteration.Solve(env, gamma: 0.99);

        // From the start, both right and down lie on a shortest path; up and left never do.
        Assert.True(result.IsOptimalAction(0, GridEnvironmentBase.ActionRight));
        Assert.True(result.IsOptimalAction(0, GridEnvironmentBase.ActionDown));
        Assert.False(result.IsOptimalAction(0, GridEnvironmentBase.ActionUp));
        Assert.False(result.IsOptimalAction(0, GridEnvironmentBase.ActionLeft));
    }

    [Fact]
    public void FrozenLake_OptimalPolicy_Solves70PercentOfEpisodes()
    {
        var env = new FrozenLakeEnv();
        var oracle = ValueIteration.Solve(env, gamma: 0.999);

        var agent = new OraclePolicy(oracle);
        var eval = MintPlayer.AI.ReinforcementLearning.Core.Training.Evaluator.Evaluate(env, agent, episodes: 2000, seed: 7);

        // Gymnasium's registered reward_threshold for FrozenLake-v1 is 0.70.
        Assert.True(eval.SuccessRate() >= 0.70, $"VI-optimal policy success rate was {eval.SuccessRate():P1}");
    }

    private sealed class OraclePolicy(ValueIterationResult oracle) : MintPlayer.AI.ReinforcementLearning.Core.Agents.IAgent<int, int>
    {
        public int Act(int observation, bool greedy = false)
        {
            int best = 0;
            for (int a = 1; a < oracle.Q.GetLength(1); a++)
                if (oracle.Q[observation, a] > oracle.Q[observation, best]) best = a;
            return best;
        }
    }
}
