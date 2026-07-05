using System.Reflection;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class SnakeEnvTests
{
    private const int Up = 0, Down = 1, Left = 2, Right = 3;
    private static int Head(SnakeEnv e) => (e.Size / 2) * e.Size + (e.Size / 2);

    // Build a SnakeEnv state buffer (RestoreState format) with an arbitrary body, so a tail-follow can be set
    // up deterministically without depending on random food placement. rng state is unused (no food spawn).
    private static byte[] State(int[] bodyHeadFirst, int food, int heading)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(0UL); w.Write(0UL); w.Write(0UL); w.Write(1UL); // rng (unused — we never eat)
        w.Write(bodyHeadFirst.Length);
        foreach (int c in bodyHeadFirst) w.Write(c);
        w.Write(food);
        w.Write(heading);
        w.Write(0);     // foodEaten
        w.Write(0);     // elapsedSteps
        w.Write(0);     // stepsSinceFood
        w.Write(false); // done
        w.Flush();
        return ms.ToArray();
    }

    // Read the single-source core's occupancy (List<bool> over cells) as a set of occupied cell indices. SnakeEnv
    // is now a facade over the generated PgSnakeEnv; the occupancy lives in the core, not a _occupied HashSet.
    private static HashSet<int> Occupied(SnakeEnv e)
    {
        var core = (global::PgSnakeEnv)typeof(SnakeEnv).GetField("_core", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(e)!;
        var set = new HashSet<int>();
        for (int i = 0; i < core.occupied.Count; i++)
            if (core.occupied[i]) set.Add(i);
        return set;
    }

    [Fact]
    public void TailFollowMove_KeepsOccupiedTrackingInSyncWithBody()
    {
        // A length-4 snake coiled into the 2×2 square {(6,5),(6,6),(7,5),(7,6)}: head (7,5) facing Left,
        // tail (6,5). Moving Up steps the head onto (6,5) — the exact cell the tail vacates this non-eating
        // tick: the canonical "tail-follow" the AI does constantly to survive.
        var env = new SnakeEnv();
        int c65 = 6 * 12 + 5, c66 = 6 * 12 + 6, c75 = 7 * 12 + 5, c76 = 7 * 12 + 6;
        env.RestoreState(State([c75, c76, c66, c65], food: 0, heading: Left));

        var step = env.Step(Up); // head → (6,5), the vacating tail cell
        Assert.False(step.Terminated);
        Assert.Equal(4, env.Length);

        // Collision detection reads _occupied (SnakeEnv.cs:138). Every body cell MUST be in it, or a later move
        // onto an untracked cell passes through the snake's own body without dying — the reported AI-mode bug.
        Assert.Equal(env.Body.ToHashSet(), Occupied(env));
    }

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
