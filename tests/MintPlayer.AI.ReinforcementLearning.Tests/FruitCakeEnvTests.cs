using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class FruitCakeEnvTests
{
    private static readonly int[] Actions = [3, 7, 3, 7, 5, 9, 2, 11, 6, 0, 13, 1];

    [Fact]
    public void Reset_ObservationAndSpaces_MatchContract()
    {
        var env = new FruitCakeEnv();
        var (obs, _) = env.Reset(1);

        Assert.Equal(FruitCakeEnv.ObservationSize, obs.Length);
        Assert.True(env.ObservationSpace.Contains(obs)); // all features normalized to [0, 1]
        Assert.True(env.ActionSpace.Contains(FruitCakeEnv.ColumnCount - 1));
        Assert.False(env.ActionSpace.Contains(FruitCakeEnv.ColumnCount));
        Assert.InRange(env.CurrentTier, 1, FruitCatalog.MaxDroppableTier);
        Assert.InRange(env.NextTier, 1, FruitCatalog.MaxDroppableTier);

        var step = env.Step(7);
        Assert.True(double.IsFinite(step.Reward));
        Assert.Equal(FruitCakeEnv.ObservationSize, step.Observation.Length);
    }

    [Fact]
    public void SameSeed_SameActions_ProducesIdenticalTrajectory()
    {
        var a = new FruitCakeEnv();
        var b = new FruitCakeEnv();
        a.Reset(12345);
        b.Reset(12345);

        for (int i = 0; i < 48; i++)
        {
            int act = Actions[i % Actions.Length];
            var sa = a.Step(act);
            var sb = b.Step(act);
            Assert.Equal(sa.Observation, sb.Observation);
            Assert.Equal(sa.Reward, sb.Reward);
            Assert.Equal(sa.Terminated, sb.Terminated);
            Assert.Equal(sa.Truncated, sb.Truncated);
            if (sa.Done) break;
        }
        Assert.Equal(a.Score, b.Score);
    }

    [Fact]
    public void SaveRestoreState_ResumesIdentically()
    {
        var a = new FruitCakeEnv();
        a.Reset(5);
        for (int i = 0; i < 8; i++) a.Step(Actions[i % Actions.Length]);
        Assert.False(a.Drops >= 500); // sanity: not done yet
        var snapshot = a.SaveState();

        var ra = a.Step(7);

        var b = new FruitCakeEnv();
        b.Reset(999);
        b.RestoreState(snapshot);
        var rb = b.Step(7);

        Assert.Equal(ra.Observation, rb.Observation);
        Assert.Equal(ra.Reward, rb.Reward);
        Assert.Equal(ra.Terminated, rb.Terminated);
        Assert.Equal(a.Score, b.Score);
    }

    [Fact]
    public void StepAfterDone_Throws()
    {
        var env = new FruitCakeEnv();
        env.Reset(7);
        StepResult<float[]> last = default;
        // Piling everything into the corner column overflows the danger line well before the truncation cap.
        for (int i = 0; i < 500; i++)
        {
            last = env.Step(0);
            if (last.Done) break;
        }
        Assert.True(last.Terminated, "expected a danger-line/eject game-over, not truncation");
        Assert.False(last.Truncated);
        Assert.Throws<InvalidOperationException>(() => env.Step(0));
    }

    [Fact]
    public void BuildObservation_TierGrid_EncodesOccupiedCellsAndLeavesEmptiesZero()
    {
        var world = new FruitCakeWorld(enableRotation: false);
        const int tier = 7;                 // r = 84px
        const float x = 300f, y = 700f;
        world.SpawnFruit(tier, x, y);

        var obs = FruitCakeEnv.BuildObservation(world, current: 1, next: 2);
        int gridBase = FruitCakeEnv.ColumnCount * 5 + 5 + 5 + 3; // after A-D + one-hots + globals; grid precedes big-fruit
        float binW = FruitCakeWorld.Width / FruitCakeEnv.ColumnCount;
        float binH = FruitCakeWorld.Height / FruitCakeEnv.GridRows;

        // The cell over the fruit's center carries its tier (÷11)…
        int col = (int)(x / binW), row = (int)(y / binH);
        Assert.Equal(tier / 11f, obs[gridBase + row * FruitCakeEnv.ColumnCount + col], 1e-5f);
        // …while a far top-left cell the fruit can't reach stays empty.
        Assert.Equal(0f, obs[gridBase + 0 * FruitCakeEnv.ColumnCount + 0], 1e-5f);
    }

    [Fact]
    public void BuildObservation_EncodesTwoBiggestFruitPositions()
    {
        var world = new FruitCakeWorld(enableRotation: false);
        world.SpawnFruit(3, 496f, 765f);   // filler — lowest tier, must be ignored
        world.SpawnFruit(10, 124f, 680f);  // biggest
        world.SpawnFruit(8, 310f, 425f);   // second biggest

        var obs = FruitCakeEnv.BuildObservation(world, current: 1, next: 2);
        int b = FruitCakeEnv.ObservationSize - 6;

        Assert.Equal(124f / FruitCakeWorld.Width, obs[b + 0], 1e-5f);
        Assert.Equal(680f / FruitCakeWorld.Height, obs[b + 1], 1e-5f);
        Assert.Equal(10f / 11f, obs[b + 2], 1e-5f);
        Assert.Equal(310f / FruitCakeWorld.Width, obs[b + 3], 1e-5f);
        Assert.Equal(425f / FruitCakeWorld.Height, obs[b + 4], 1e-5f);
        Assert.Equal(8f / 11f, obs[b + 5], 1e-5f);
    }

    [Fact]
    public void BuildObservation_EmptyBoard_UsesNeutralBigFruitSentinel()
    {
        var obs = FruitCakeEnv.BuildObservation(new FruitCakeWorld(enableRotation: false), current: 1, next: 2);
        int b = FruitCakeEnv.ObservationSize - 6;
        Assert.Equal<float[]>([0.5f, 1f, 0f, 0.5f, 1f, 0f], obs[b..(b + 6)]); // floor-centre, tier 0 for both slots
    }

    [Fact]
    public void BuildObservation_BigFruitTie_BrokenByLowerThenLeftmost()
    {
        var world = new FruitCakeWorld(enableRotation: false);
        world.SpawnFruit(9, 200f, 300f);   // same tier, higher up
        world.SpawnFruit(9, 400f, 700f);   // same tier, lower on the board → ranks first

        var obs = FruitCakeEnv.BuildObservation(world, 1, 2);
        int b = FruitCakeEnv.ObservationSize - 6;
        Assert.Equal(400f / FruitCakeWorld.Width, obs[b + 0], 1e-5f); // big1 = the lower fruit
        Assert.Equal(200f / FruitCakeWorld.Width, obs[b + 3], 1e-5f); // big2 = the higher fruit
    }

    [Fact]
    public void World_SameTierContact_Merges_AndScores()
    {
        var world = new FruitCakeWorld(enableRotation: false);
        // Two cherries (tier 1, r=24) overlapping in the same column.
        world.SpawnFruit(1, 300f, 786f);
        world.SpawnFruit(1, 300f, 826f);

        int points = world.Step(1f / 60f);

        Assert.Equal(FruitCatalog.ByTier(1).MergePoints, points); // a tier-1 merge scores 1
        Assert.Equal(1, world.Count);                              // two fruit became one
        Assert.Equal(2, world.Bodies[0].Tier);                    // …of the next tier
    }

    [Fact]
    public void World_DropSettlesQuickly_NotAFixedHorizon()
    {
        var world = new FruitCakeWorld(enableRotation: false);
        world.SpawnFruit(1, 300f, 100f); // a single fruit falling onto an empty floor

        int steps = 0;
        while (steps < 600)
        {
            int gained = world.Step(1f / 60f);
            steps++;
            if (steps >= 8 && gained == 0 && world.MaxSpeed() < 30f) break; // let it accelerate before testing rest
        }

        Assert.True(steps is > 8 and < 250, $"a lone drop should fall then early-settle quickly; took {steps} sub-steps");
    }
}
