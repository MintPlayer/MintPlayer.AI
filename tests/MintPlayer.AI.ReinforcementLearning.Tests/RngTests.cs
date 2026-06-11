using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class RngTests
{
    [Fact]
    public void SameSeed_ProducesIdenticalSequence()
    {
        var a = new Xoshiro256StarStar(123);
        var b = new Xoshiro256StarStar(123);
        for (int i = 0; i < 1000; i++)
            Assert.Equal(a.NextUInt64(), b.NextUInt64());
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentSequences()
    {
        var a = new Xoshiro256StarStar(1);
        var b = new Xoshiro256StarStar(2);
        Assert.NotEqual(a.NextUInt64(), b.NextUInt64());
    }

    [Fact]
    public void NextDouble_StaysInUnitInterval()
    {
        var rng = new Xoshiro256StarStar(7);
        for (int i = 0; i < 10_000; i++)
        {
            double d = rng.NextDouble();
            Assert.InRange(d, 0.0, 0.9999999999999999);
        }
    }

    [Fact]
    public void NextInt_StaysInBounds_AndHitsAllValues()
    {
        var rng = new Xoshiro256StarStar(7);
        var seen = new bool[4];
        for (int i = 0; i < 1000; i++)
        {
            int v = rng.NextInt(4);
            Assert.InRange(v, 0, 3);
            seen[v] = true;
        }
        Assert.All(seen, Assert.True);
    }

    [Fact]
    public void SeedSequence_StreamsAreIndependentButDeterministic()
    {
        var seq1 = new SeedSequence(42);
        var seq2 = new SeedSequence(42);

        Assert.Equal(seq1.Derive(0), seq2.Derive(0));
        Assert.NotEqual(seq1.Derive(0), seq1.Derive(1));
        Assert.NotEqual(new SeedSequence(42).Derive(0), new SeedSequence(43).Derive(0));
    }

    [Fact]
    public void StateRoundTrip_ResumesSequence()
    {
        var rng = new Xoshiro256StarStar(99);
        rng.NextUInt64();
        var state = rng.GetState();
        ulong expected = rng.NextUInt64();

        var resumed = new Xoshiro256StarStar(0);
        resumed.SetState(state.S0, state.S1, state.S2, state.S3);
        Assert.Equal(expected, resumed.NextUInt64());
    }
}
