using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class MountainCarEnvTests
{
    private const int Left = 0, None = 1, Right = 2;

    [Fact]
    public void Reset_DrawsStartStateInBand_WithZeroVelocity()
    {
        var env = new MountainCarEnv();
        var (obs, _) = env.Reset(1);
        Assert.Equal(2, obs.Length);
        Assert.InRange(env.Position, -0.6, -0.4);
        Assert.Equal(0.0, env.Velocity);
        // Observation is normalised to ~[-1,1].
        Assert.InRange(obs[0], -1f, 1f);
        Assert.Equal(0f, obs[1]); // velocity 0 → normalised 0
    }

    [Fact]
    public void Step_AppliesTheGymnasiumDynamics()
    {
        var env = new MountainCarEnv();
        env.Reset(1);
        env.SetState(0.0, 0.0);
        env.Step(Right); // v += (2-1)*0.001 + cos(0)*(-0.0025) = -0.0015 ; pos += v
        Assert.Equal(-0.0015, env.Velocity, 6);
        Assert.Equal(-0.0015, env.Position, 6);
    }

    [Fact]
    public void ReachingGoal_Terminates()
    {
        var env = new MountainCarEnv();
        env.Reset(1);
        env.SetState(0.49, MountainCarEnv.MaxSpeed); // one push past the flag
        var step = env.Step(Right);
        Assert.True(step.Terminated);
        Assert.False(step.Truncated);
        Assert.True(env.Position >= MountainCarEnv.GoalPosition);
    }

    [Fact]
    public void StepCap_TruncatesNotTerminates()
    {
        var env = new MountainCarEnv(maxEpisodeSteps: 5);
        env.Reset(1); // starts near -0.5; won't reach the goal in 5 idle steps
        StepResult<float[]> last = default;
        for (int i = 0; i < 5; i++) last = env.Step(None);
        Assert.True(last.Truncated);
        Assert.False(last.Terminated);
    }

    [Fact]
    public void Reward_IsMinusOnePerStep()
    {
        var env = new MountainCarEnv();
        env.Reset(1);
        Assert.Equal(-1.0, env.Step(Left).Reward);
        Assert.Equal(-1.0, env.Step(None).Reward);
        Assert.Equal(-1.0, env.Step(Right).Reward);
    }

    [Fact]
    public void SaveRestoreState_ResumesIdentically()
    {
        var a = new MountainCarEnv();
        a.Reset(5);
        for (int i = 0; i < 10; i++) a.Step(Right);
        var snapshot = a.SaveState();
        var ra = a.Step(Left);

        var b = new MountainCarEnv();
        b.Reset(999);
        b.RestoreState(snapshot);
        var rb = b.Step(Left);

        Assert.Equal(ra.Observation, rb.Observation);
        Assert.Equal(ra.Reward, rb.Reward);
        Assert.Equal(ra.Terminated, rb.Terminated);
    }
}
