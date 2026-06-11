using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>Describes the set of valid observations or actions of an environment.</summary>
public abstract class Space<T>
{
    public abstract T Sample(Xoshiro256StarStar rng);
    public abstract bool Contains(T value);
}

/// <summary>{0, 1, ..., N-1}</summary>
public sealed class DiscreteSpace : Space<int>
{
    public DiscreteSpace(int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        N = n;
    }

    public int N { get; }

    public override int Sample(Xoshiro256StarStar rng) => rng.NextInt(N);

    public override bool Contains(int value) => value >= 0 && value < N;
}

/// <summary>Axis-aligned box in R^n with per-dimension bounds.</summary>
public sealed class BoxSpace : Space<float[]>
{
    public BoxSpace(float[] low, float[] high)
    {
        if (low.Length != high.Length)
            throw new ArgumentException("low and high must have the same length.");
        Low = low;
        High = high;
    }

    public BoxSpace(float low, float high, int dimensions)
        : this(
            Enumerable.Repeat(low, dimensions).ToArray(),
            Enumerable.Repeat(high, dimensions).ToArray())
    {
    }

    public float[] Low { get; }
    public float[] High { get; }
    public int Dimensions => Low.Length;

    public override float[] Sample(Xoshiro256StarStar rng)
    {
        var result = new float[Dimensions];
        for (int i = 0; i < result.Length; i++)
            result[i] = (float)rng.NextDouble(Low[i], High[i]);
        return result;
    }

    public override bool Contains(float[] value)
    {
        if (value.Length != Dimensions) return false;
        for (int i = 0; i < value.Length; i++)
            if (value[i] < Low[i] || value[i] > High[i]) return false;
        return true;
    }
}
