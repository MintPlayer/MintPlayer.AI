using RLNet.Core.Environments;
using RLNet.Environments;

namespace RLNet.Tests;

public class GridWorldTests
{
    [Fact]
    public void Reset_StartsTopLeft()
    {
        var env = new GridWorldEnv();
        var (state, _) = env.Reset(1);
        Assert.Equal(0, state);
    }

    [Fact]
    public void StepRight_MovesOneColumn_WithStepPenalty()
    {
        var env = new GridWorldEnv();
        env.Reset(1);
        var step = env.Step(GridEnvironmentBase.ActionRight);
        Assert.Equal(1, step.Observation);
        Assert.Equal(GridWorldEnv.StepReward, step.Reward);
        Assert.False(step.Terminated);
        Assert.False(step.Truncated);
    }

    [Fact]
    public void BumpingWall_StaysInPlace_StillCostsStep()
    {
        var env = new GridWorldEnv();
        env.Reset(1);
        var step = env.Step(GridEnvironmentBase.ActionUp);
        Assert.Equal(0, step.Observation);
        Assert.Equal(GridWorldEnv.StepReward, step.Reward);
    }

    [Fact]
    public void EnteringGoal_TerminatesWithPlusOne()
    {
        var env = new GridWorldEnv();
        env.Reset(1);
        // Walk right 3, down 3 → state 15 (goal).
        for (int i = 0; i < 3; i++) env.Step(GridEnvironmentBase.ActionRight);
        for (int i = 0; i < 2; i++) env.Step(GridEnvironmentBase.ActionDown);
        var final = env.Step(GridEnvironmentBase.ActionDown);

        Assert.Equal(15, final.Observation);
        Assert.Equal(GridWorldEnv.GoalReward, final.Reward);
        Assert.True(final.Terminated);
        Assert.False(final.Truncated);
    }

    [Fact]
    public void SteppingAfterDone_Throws()
    {
        var env = new GridWorldEnv();
        env.Reset(1);
        for (int i = 0; i < 3; i++) env.Step(GridEnvironmentBase.ActionRight);
        for (int i = 0; i < 3; i++) env.Step(GridEnvironmentBase.ActionDown);
        Assert.Throws<InvalidOperationException>(() => env.Step(GridEnvironmentBase.ActionDown));
    }

    [Fact]
    public void NeverReachingGoal_TruncatesAt100Steps_NotTerminated()
    {
        var env = new GridWorldEnv();
        env.Reset(1);
        StepResult<int> step = default;
        for (int i = 0; i < 100; i++)
            step = env.Step(GridEnvironmentBase.ActionUp); // bump the wall forever

        Assert.True(step.Truncated);
        Assert.False(step.Terminated);
    }
}

public class FrozenLakeTests
{
    [Fact]
    public void HoleTerminatesWithZeroReward()
    {
        var env = new FrozenLakeEnv(slippery: false);
        env.Reset(1);
        env.Step(GridEnvironmentBase.ActionDown);     // 0 -> 4
        var step = env.Step(GridEnvironmentBase.ActionRight); // 4 -> 5 (H)

        Assert.Equal(5, step.Observation);
        Assert.Equal(0.0, step.Reward);
        Assert.True(step.Terminated);
    }

    [Fact]
    public void GoalGivesPlusOne()
    {
        var env = new FrozenLakeEnv(slippery: false);
        env.Reset(1);
        // Safe path on the standard map: down, down, right, right, down, right.
        env.Step(GridEnvironmentBase.ActionDown);
        env.Step(GridEnvironmentBase.ActionDown);
        env.Step(GridEnvironmentBase.ActionRight);
        env.Step(GridEnvironmentBase.ActionRight);
        env.Step(GridEnvironmentBase.ActionDown);
        var final = env.Step(GridEnvironmentBase.ActionRight);

        Assert.Equal(15, final.Observation);
        Assert.Equal(1.0, final.Reward);
        Assert.True(final.Terminated);
    }

    [Fact]
    public void SlipperyModel_HasThreeOutcomes_OneThirdEach()
    {
        var env = new FrozenLakeEnv();
        var transitions = env.Model(state: 9, action: GridEnvironmentBase.ActionRight).ToList();

        Assert.Equal(3, transitions.Count);
        Assert.All(transitions, t => Assert.Equal(1.0 / 3.0, t.Probability, precision: 12));
        // Intended right (9→10) plus perpendiculars down (9→13) and up (9→5).
        Assert.Equal(new[] { 5, 10, 13 }, transitions.Select(t => t.NextState).Order().ToArray());
    }

    [Fact]
    public void SlipperyDynamics_MatchModelDistribution()
    {
        var env = new FrozenLakeEnv();
        const int samples = 30_000;

        // From state 0 stepping right: intended 1 (right), perpendiculars 4 (down) and 0 (up = bump).
        var counts = new Dictionary<int, int> { [4] = 0, [1] = 0, [0] = 0 };
        env.Reset(123);
        for (int i = 0; i < samples; i++)
        {
            env.Reset();
            var step = env.Step(GridEnvironmentBase.ActionRight);
            counts[step.Observation]++;
        }

        foreach (var (_, count) in counts)
            Assert.InRange(count / (double)samples, 1.0 / 3.0 - 0.02, 1.0 / 3.0 + 0.02);
    }

    [Fact]
    public void TimeLimit_TruncatesAt100Steps()
    {
        var env = new FrozenLakeEnv(slippery: false);
        env.Reset(1);
        StepResult<int> step = default;
        for (int i = 0; i < 100; i++)
            step = env.Step(GridEnvironmentBase.ActionUp); // bump the top wall forever

        Assert.True(step.Truncated);
        Assert.False(step.Terminated);
    }

    [Fact]
    public void ModelProbabilities_SumToOne_ForAllStateActions()
    {
        var env = new FrozenLakeEnv();
        for (int s = 0; s < env.StateCount; s++)
        {
            if (env.IsTerminal(s)) continue;
            for (int a = 0; a < env.ActionCount; a++)
                Assert.Equal(1.0, env.Model(s, a).Sum(t => t.Probability), precision: 12);
        }
    }
}
