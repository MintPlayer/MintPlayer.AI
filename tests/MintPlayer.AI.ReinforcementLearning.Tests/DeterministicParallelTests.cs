using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class DeterministicParallelTests
{
    // A non-trivial per-item payload: draw a handful of values from the item's RNG so the result depends on
    // BOTH the index (via makeItem) and the derived RNG stream — the thing that must stay order-independent.
    private static double[] MakeItem(int i, Xoshiro256StarStar rng)
    {
        var v = new double[8];
        v[0] = i;
        for (int k = 1; k < v.Length; k++) v[k] = rng.NextDouble();
        return v;
    }

    private static double[][] Run(bool parallel, int? maxDop, int count = 200, long baseIndex = 0, ulong seed = 1234)
        => DeterministicParallel.Generate(count, new SeedSequence(seed), RngStreams.Policy, baseIndex, MakeItem, parallel, maxDop);

    [Fact]
    public void ParallelOutput_IsBitwiseIdenticalToSequential()
    {
        var sequential = Run(parallel: false, maxDop: null);
        var parallel = Run(parallel: true, maxDop: null);

        Assert.Equal(sequential.Length, parallel.Length);
        for (int i = 0; i < sequential.Length; i++)
            Assert.Equal(sequential[i], parallel[i]); // element-wise double equality — bitwise for these values
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void Output_IsInvariantToDegreeOfParallelism(int dop)
    {
        var baseline = Run(parallel: false, maxDop: null);
        var atDop = Run(parallel: true, maxDop: dop);

        for (int i = 0; i < baseline.Length; i++)
            Assert.Equal(baseline[i], atDop[i]);
    }

    [Fact]
    public void Results_AreInAscendingIndexOrder()
    {
        // makeItem stamps v[0] = i, so index order is observable regardless of completion order.
        var results = Run(parallel: true, maxDop: 8);
        for (int i = 0; i < results.Length; i++)
            Assert.Equal(i, results[i][0]);
    }

    [Fact]
    public void EachIndex_DerivesADistinctRngStream()
    {
        var results = Run(parallel: false, maxDop: null, count: 64);
        // Compare the RNG-derived tail (skip v[0]=index) — no two items should share a stream.
        for (int i = 0; i < results.Length; i++)
            for (int j = i + 1; j < results.Length; j++)
                Assert.NotEqual(results[i][1..], results[j][1..]);
    }

    [Fact]
    public void BaseIndex_ShiftsTheRngStreams()
    {
        // Item at local 0 with baseIndex 100 must equal item at local 100 with baseIndex 0 (same global index).
        var fromZero = Run(parallel: false, maxDop: null, count: 200, baseIndex: 0);
        var shifted = Run(parallel: false, maxDop: null, count: 1, baseIndex: 100);
        Assert.Equal(fromZero[100][1..], shifted[0][1..]);
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentOutput()
    {
        var a = Run(parallel: false, maxDop: null, seed: 1);
        var b = Run(parallel: false, maxDop: null, seed: 2);
        Assert.NotEqual(a[0][1..], b[0][1..]);
    }

    [Fact]
    public void RawSeedOverload_ReproducesTheCubeHandRolledWorkerSeeding()
    {
        // The cube campaigns seeded generator w as Xoshiro(roundBase + φ·(w+1)). Migrating them onto the primitive
        // with (baseSeed: roundBase, baseIndex: 1) must derive byte-identical per-worker RNG streams — otherwise the
        // refactor would silently change cube training output. This locks that equivalence.
        const ulong roundBase = 123456789UL;
        const ulong golden = 0x9E3779B97F4A7C15UL;
        const int workers = 6;

        var viaPrimitive = DeterministicParallel.Generate(workers, roundBase, baseIndex: 1,
            (_, rng) => new[] { rng.NextUInt64(), rng.NextUInt64(), rng.NextUInt64() }, parallel: false);

        for (int w = 0; w < workers; w++)
        {
            var expected = new Xoshiro256StarStar(unchecked(roundBase + golden * (ulong)(w + 1)));
            for (int i = 0; i < 3; i++) Assert.Equal(expected.NextUInt64(), viaPrimitive[w][i]);
        }
    }

    [Fact]
    public void ZeroCount_ReturnsEmpty()
        => Assert.Empty(Run(parallel: true, maxDop: null, count: 0));

    [Fact]
    public void NegativeCount_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Run(parallel: false, maxDop: null, count: -1));
}
