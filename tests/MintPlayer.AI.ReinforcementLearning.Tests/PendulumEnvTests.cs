using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class PendulumEnvTests
{
    [Fact]
    public void Reset_DrawsStartStateInBand_AndUnitCircleObs()
    {
        var env = new PendulumEnv();
        var (obs, _) = env.Reset(1);

        Assert.Equal(3, obs.Length);
        Assert.Equal(1.0, obs[0] * obs[0] + obs[1] * obs[1], 4); // cos²+sin² == 1
        Assert.InRange(obs[2], -1f, 1f);                          // θ̇ drawn from [-1,1]
        Assert.InRange(env.Theta, -Math.PI, Math.PI);
    }

    [Fact]
    public void ActionSpace_IsBoxTorque_PlusMinusTwo()
    {
        var box = Assert.IsType<BoxSpace>(new PendulumEnv().ActionSpace);
        Assert.Equal(1, box.Dimensions);
        Assert.Equal(-2f, box.Low[0]);
        Assert.Equal(2f, box.High[0]);
    }

    [Fact]
    public void Step_AppliesGymnasiumDynamics()
    {
        var env = new PendulumEnv();
        env.Reset(1);
        env.SetState(Math.PI / 2, 0.0); // rod horizontal, at rest

        var step = env.Step([0f]); // no torque

        // newθ̇ = θ̇ + (3g/2l·sinθ + 3/ml²·τ)·dt = 0 + (15·1 + 0)·0.05 = 0.75
        double expectedThetaDot = (3.0 * 10.0 / 2.0 * 1.0) * 0.05;
        Assert.Equal(expectedThetaDot, env.AngularVelocity, 6);
        Assert.Equal(Math.PI / 2 + expectedThetaDot * 0.05, env.Theta, 6);
        Assert.Equal((float)expectedThetaDot, step.Observation[2], 5);

        // cost from the PRE-update angle (π/2): θ² + 0.1·θ̇² + 0.001·τ² = (π/2)²
        Assert.Equal(-(Math.PI / 2) * (Math.PI / 2), step.Reward, 6);
    }

    [Fact]
    public void Step_ClampsTorqueToBounds_NoThrow()
    {
        var over = new PendulumEnv();
        over.Reset(1); over.SetState(0.0, 0.0);
        var a = over.Step([5f]); // way past +2

        var clamped = new PendulumEnv();
        clamped.Reset(1); clamped.SetState(0.0, 0.0);
        var b = clamped.Step([2f]);

        Assert.Equal(b.Observation[2], a.Observation[2], 6); // identical: 5 was clamped to 2
        Assert.Equal(b.Reward, a.Reward, 6);
    }

    [Fact]
    public void Reward_IsNonPositive_AndZeroAtUprightRest()
    {
        var env = new PendulumEnv();
        env.Reset(1); env.SetState(0.0, 0.0); // upright, at rest
        var step = env.Step([0f]);
        Assert.Equal(0.0, step.Reward, 9); // the maximum attainable reward
    }

    [Fact]
    public void StepCap_TruncatesNeverTerminates()
    {
        var env = new PendulumEnv();
        env.Reset(1);
        StepResult<float[]> last = default;
        for (int i = 0; i < PendulumEnv.DefaultMaxEpisodeSteps; i++)
            last = env.Step([0f]);

        Assert.True(last.Truncated);
        Assert.False(last.Terminated); // Pendulum has no terminal state
        Assert.True(last.Done);
    }

    [Fact]
    public void SameSeed_ReproducesTrajectory()
    {
        var a = new PendulumEnv();
        var b = new PendulumEnv();
        var (oa, _) = a.Reset(42);
        var (ob, _) = b.Reset(42);
        Assert.Equal(oa, ob);

        for (int i = 0; i < 20; i++)
        {
            var sa = a.Step([0.5f]);
            var sb = b.Step([0.5f]);
            Assert.Equal(sa.Observation, sb.Observation);
            Assert.Equal(sa.Reward, sb.Reward);
        }
    }

    [Fact]
    public void SaveRestoreState_ResumesIdentically()
    {
        var a = new PendulumEnv();
        a.Reset(5);
        for (int i = 0; i < 5; i++) a.Step([0.3f]);
        var snapshot = a.SaveState();

        var ra = a.Step([-0.7f]);

        var b = new PendulumEnv();
        b.Reset(999);
        b.RestoreState(snapshot);
        var rb = b.Step([-0.7f]);

        Assert.Equal(ra.Observation, rb.Observation);
        Assert.Equal(ra.Reward, rb.Reward);
        Assert.Equal(ra.Truncated, rb.Truncated);
    }
}
