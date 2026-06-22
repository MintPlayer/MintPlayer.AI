using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class SnakeEnvTests
{
    private const int Up = 0, Down = 1, Left = 2, Right = 3;
    private static int Head(SnakeEnv e) => (e.Size / 2) * e.Size + (e.Size / 2);

    [Fact]
    public void Reset_StartsLength3_HeadCentred_EgocentricObservation()
    {
        var env = new SnakeEnv();
        var (obs, _) = env.Reset(1);

        Assert.Equal(SnakeEnv.ObservationSize, obs.Length); // patch (2 planes) + scalars
        Assert.Equal(3, env.Length);
        Assert.Equal(Head(env), env.Body.First());

        // The patch centre (the head's own cell) is never an obstacle.
        int side = SnakeEnv.PatchSide, centre = (side / 2) * side + (side / 2);
        Assert.Equal(0f, obs[centre]);

        // Heading one-hot lives in the scalar block: exactly one bit set, and it's Right.
        var heading = obs[(SnakeEnv.PatchSize + 3)..(SnakeEnv.PatchSize + 7)];
        Assert.Equal(1, heading.Count(v => v == 1f));
        Assert.Equal(1f, obs[SnakeEnv.PatchSize + 6]); // Right

        // Food Δ (signed, normalized) matches the food's position relative to the head.
        int hr = Head(env) / env.Size, hc = Head(env) % env.Size;
        int fr = env.Food / env.Size, fc = env.Food % env.Size;
        Assert.Equal((fc - hc) / (float)env.Size, obs[SnakeEnv.PatchSize + 0]);
        Assert.Equal((fr - hr) / (float)env.Size, obs[SnakeEnv.PatchSize + 1]);
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
