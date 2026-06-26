using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class NoisyLinearTests
{
    private static Tensor Batch() => new([0.5f, -1f, 0.25f, 2f, -0.3f, 0.1f, 1.5f, -0.7f], 2, 4); // [B=2, in=4]

    [Fact]
    public void Backward_GivesGradientsToBothMeanAndSigma()
    {
        // NoisyNets' core requirement: gradients must reach μ AND σ (so exploration is learned),
        // but never the sampled noise (which is a constant). σ getting a gradient proves the latter
        // path — σ only ever appears multiplied by the constant ε.
        var rng = new Xoshiro256StarStar(1);
        var layer = new NoisyLinear(4, 3, rng) { NoiseEnabled = true };
        layer.ResampleNoise(rng);

        layer.Forward(Batch()).Sum().Backward();

        Assert.Contains(layer.MeanWeight.Grad!, g => g != 0f);
        Assert.Contains(layer.SigmaWeight.Grad!, g => g != 0f);
        Assert.Contains(layer.MeanBias.Grad!, g => g != 0f);
        Assert.Contains(layer.SigmaBias.Grad!, g => g != 0f);
    }

    [Fact]
    public void SamplingForwards_DifferAfterResample()
    {
        var rng = new Xoshiro256StarStar(2);
        var layer = new NoisyLinear(4, 3, rng) { NoiseEnabled = true };
        var x = Batch();

        layer.ResampleNoise(rng);
        var first = (float[])layer.Forward(x).Data.Clone();
        layer.ResampleNoise(rng);
        var second = layer.Forward(x).Data;

        Assert.False(first.AsSpan().SequenceEqual(second), "fresh noise should change the output");
    }

    [Fact]
    public void EvalMode_IsDeterministic_AndEqualsMeanOnly()
    {
        var rng = new Xoshiro256StarStar(3);
        var layer = new NoisyLinear(4, 3, rng); // NoiseEnabled defaults to false
        var x = Batch();

        var before = (float[])layer.Forward(x).Data.Clone();
        layer.ResampleNoise(rng); // must be irrelevant while noise is off
        var after = layer.Forward(x).Data;
        Assert.Equal(before, after);

        // ...and the deterministic output is exactly the mean-weight linear map.
        var meanOnly = x.MatMul(layer.MeanWeight).AddBias(layer.MeanBias).Data;
        Assert.Equal(meanOnly, after);
    }

    [Fact]
    public void Parameters_AreTheFourLearnableTensors_ExcludingNoise()
    {
        var rng = new Xoshiro256StarStar(4);
        var layer = new NoisyLinear(4, 3, rng);
        var ps = layer.Parameters().ToList();

        Assert.Equal(4, ps.Count);
        Assert.All(ps, p => Assert.True(p.RequiresGrad));
        Assert.Equal([layer.MeanWeight, layer.SigmaWeight, layer.MeanBias, layer.SigmaBias], ps);
    }
}
