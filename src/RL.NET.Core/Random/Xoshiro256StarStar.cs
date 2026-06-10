namespace RLNet.Core.Random;

/// <summary>
/// xoshiro256** PRNG with a fixed, version-stable implementation.
/// <see cref="System.Random"/> is deliberately avoided: its algorithm is not
/// contractually stable across .NET versions, which would break seeded reproducibility.
/// </summary>
public sealed class Xoshiro256StarStar
{
    private ulong _s0, _s1, _s2, _s3;

    public Xoshiro256StarStar(ulong seed)
    {
        // Seed the state via SplitMix64, per the xoshiro authors' recommendation.
        _s0 = SplitMix64.Next(ref seed);
        _s1 = SplitMix64.Next(ref seed);
        _s2 = SplitMix64.Next(ref seed);
        _s3 = SplitMix64.Next(ref seed);
    }

    public ulong NextUInt64()
    {
        ulong result = ulong.RotateLeft(_s1 * 5, 7) * 9;
        ulong t = _s1 << 17;

        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = ulong.RotateLeft(_s3, 45);

        return result;
    }

    /// <summary>Uniform double in [0, 1).</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>Uniform double in [min, max).</summary>
    public double NextDouble(double min, double max) => min + NextDouble() * (max - min);

    /// <summary>Uniform float in [0, 1).</summary>
    public float NextFloat() => (NextUInt64() >> 40) * (1.0f / (1U << 24));

    /// <summary>Uniform int in [0, maxExclusive).</summary>
    public int NextInt(int maxExclusive)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxExclusive);
        // Modulo bias is negligible for the small ranges used in RL (actions, states, buffer indices).
        return (int)(NextUInt64() % (ulong)maxExclusive);
    }

    public (ulong S0, ulong S1, ulong S2, ulong S3) GetState() => (_s0, _s1, _s2, _s3);

    public void SetState(ulong s0, ulong s1, ulong s2, ulong s3) => (_s0, _s1, _s2, _s3) = (s0, s1, s2, s3);
}

internal static class SplitMix64
{
    public static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
