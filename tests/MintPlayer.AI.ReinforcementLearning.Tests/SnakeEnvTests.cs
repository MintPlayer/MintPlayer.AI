using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class SnakeEnvTests
{
    private const int Up = 0, Down = 1, Left = 2, Right = 3;
    private static int Head(SnakeEnv e) => (e.Size / 2) * e.Size + (e.Size / 2);

    [Fact]
    public void Reset_StartsLength3_HeadCentred_RayObservation()
    {
        var env = new SnakeEnv();
        var (obs, _) = env.Reset(1);

        Assert.Equal(SnakeEnv.ObservationSize, obs.Length); // rays + flood + food + tail + heading + length
        Assert.Equal(3, env.Length);
        Assert.Equal(Head(env), env.Body.First());

        // Every ray's wall channel (the 3rd of each triple) is a positive 1/distance — the head always sees walls.
        for (int d = 0; d < SnakeEnv.RayDirections; d++)
            Assert.InRange(obs[d * SnakeEnv.RayChannels + 2], 1f / env.Size, 1f);

        // Heading one-hot (after rays + flood(4) + food(3) + tail(3)) = exactly one bit, Right (index 3).
        int headingStart = SnakeEnv.RayFeatures + 4 + 3 + 3;
        var heading = obs[headingStart..(headingStart + 4)];
        Assert.Equal(1, heading.Count(v => v == 1f));
        Assert.Equal(1f, heading[3]); // Right

        // Food Δ (signed, normalized) sits right after the flood-fill block and matches the food's position.
        int foodStart = SnakeEnv.RayFeatures + 4;
        int hr = Head(env) / env.Size, hc = Head(env) % env.Size;
        int fr = env.Food / env.Size, fc = env.Food % env.Size;
        Assert.Equal((fc - hc) / (float)env.Size, obs[foodStart + 0]);
        Assert.Equal((fr - hr) / (float)env.Size, obs[foodStart + 1]);
    }

    [Fact]
    public void ConfigurableGridSize_KeepsObservationInvariant()
    {
        var small = new SnakeEnv(6);
        var (obs, _) = small.Reset(1);
        Assert.Equal(6, small.Size);
        Assert.Equal(36, small.Cells);
        Assert.Equal(SnakeEnv.ObservationSize, obs.Length); // egocentric patch + scalars → fixed size across grids
        Assert.Equal(3, small.Length);
        Assert.Equal(Head(small), small.Body.First()); // (3,3) = 21
    }

    [Fact]
    public void Mask_ForbidsOnlyTheReversal()
    {
        var env = new SnakeEnv();
        env.Reset(1); // head facing right → neck is to the left → Left is the reversal
        var mask = env.CurrentActionMask();
        Assert.True(mask[Up]);
        Assert.True(mask[Down]);
        Assert.False(mask[Left]);
        Assert.True(mask[Right]);
    }

    [Fact]
    public void Step_ReversalAction_Throws()
    {
        var env = new SnakeEnv();
        env.Reset(1);
        Assert.Throws<ArgumentException>(() => env.Step(Left));
    }

    [Fact]
    public void EatingFood_GrowsSnake_AndRewardsPlusOne()
    {
        for (ulong seed = 1; seed < 500; seed++)
        {
            var env = new SnakeEnv();
            env.Reset(seed);
            int head = Head(env), food = env.Food;
            int action = food switch
            {
                _ when food == head - env.Size => Up,
                _ when food == head + env.Size => Down,
                _ when food == head + 1 => Right,
                _ => -1,
            };
            if (action < 0) continue;

            var step = env.Step(action);
            Assert.Equal(SnakeEnv.FoodReward, step.Reward);
            Assert.False(step.Terminated);
            Assert.Equal(4, env.Length);
            Assert.Equal(1, env.FoodEaten);
            return;
        }
        Assert.Fail("no seed in range placed food adjacent to the start head");
    }

    [Fact]
    public void WalkingIntoWall_TerminatesWithDeathReward()
    {
        var env = new SnakeEnv();
        env.Reset(1);
        for (int i = 0; i < env.Size / 2; i++) // reach the top row
        {
            var s = env.Step(Up);
            Assert.False(s.Terminated, $"unexpected early death at up #{i}");
        }
        var death = env.Step(Up); // step off the top edge
        Assert.True(death.Terminated);
        Assert.False(death.Truncated);
        Assert.Equal(SnakeEnv.DeathReward, death.Reward);
    }

    [Fact]
    public void SameSeed_ProducesSameObservation()
    {
        var a = new SnakeEnv();
        var b = new SnakeEnv();
        var (oa, _) = a.Reset(42);
        var (ob, _) = b.Reset(42);
        Assert.Equal(a.Food, b.Food);
        Assert.Equal(oa, ob);
    }

    [Fact]
    public void SaveRestoreState_ResumesIdentically()
    {
        var a = new SnakeEnv();
        a.Reset(5);
        for (int i = 0; i < 5; i++) a.Step(Right);
        var snapshot = a.SaveState();

        var ra = a.Step(Up);

        var b = new SnakeEnv();
        b.Reset(999);
        b.RestoreState(snapshot);
        var rb = b.Step(Up);

        Assert.Equal(ra.Observation, rb.Observation);
        Assert.Equal(ra.Reward, rb.Reward);
        Assert.Equal(ra.Terminated, rb.Terminated);
        Assert.Equal(a.Food, b.Food);
        Assert.Equal(a.Length, b.Length);
    }
}
