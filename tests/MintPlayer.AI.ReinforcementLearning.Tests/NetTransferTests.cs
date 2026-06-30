using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Covers <see cref="IValueNet.GrowInput"/> / <see cref="NetTransfer"/>: growing the input dimension must be
/// function-preserving (the new features start at zero weight, so the net's output on the original features is
/// unchanged for any value of the new ones), every other parameter must transfer exactly, and the new in-weights
/// must be zero. Mirrors the function-preservation style of <c>ResidualMlpTests.WidenTo_*</c>.
/// </summary>
public class NetTransferTests
{
    /// <summary>Column-concatenate two row-major [B, *] batches into [B, aCols+bCols].</summary>
    private static Tensor Concat(Tensor a, Tensor b)
    {
        int rows = a.Rows, ac = a.Cols, bc = b.Cols, nc = ac + bc;
        var data = new float[rows * nc];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < ac; c++) data[r * nc + c] = a.Data[r * ac + c];
            for (int c = 0; c < bc; c++) data[r * nc + ac + c] = b.Data[r * bc + c];
        }
        return new Tensor(data, rows, nc);
    }

    private static float MaxAbsDiff(float[] x, float[] y)
    {
        Assert.Equal(x.Length, y.Length);
        float m = 0f;
        for (int i = 0; i < x.Length; i++) m = MathF.Max(m, MathF.Abs(x[i] - y[i]));
        return m;
    }

    [Fact]
    public void GrowInput_Mlp_PreservesFunctionOnOriginalFeatures()
    {
        var rng = new Xoshiro256StarStar(4);
        var net = new Mlp([4, 16, 8, 2], rng, Activation.Relu);
        var grown = net.GrowInput(7);
        Assert.Equal(7, grown.InputSize);

        var old = Tensor.RandomNormal(rng, 0f, 1f, 5, 4);
        var extra = Tensor.RandomNormal(rng, 0f, 5f, 5, 3); // arbitrary new-feature values
        float diff = MaxAbsDiff(net.Forward(old).Data, grown.Forward(Concat(old, extra)).Data);
        Assert.True(diff < 1e-5f, $"Mlp grown output diverged from original by {diff}");
    }

    [Fact]
    public void GrowInput_DuelingQNet_PreservesFunctionOnOriginalFeatures()
    {
        var rng = new Xoshiro256StarStar(7);
        var net = new DuelingQNet(5, [16, 16], 3, rng);
        var grown = net.GrowInput(9);
        Assert.Equal(9, grown.InputSize);

        var old = Tensor.RandomNormal(rng, 0f, 1f, 6, 5);
        var extra = Tensor.RandomNormal(rng, 0f, 3f, 6, 4);
        float diff = MaxAbsDiff(net.Forward(old).Data, grown.Forward(Concat(old, extra)).Data);
        Assert.True(diff < 1e-5f, $"DuelingQNet grown output diverged by {diff}");
    }

    [Fact]
    public void GrowInput_NoisyDuelingQNet_PreservesFunction_WithNoiseOff()
    {
        var rng = new Xoshiro256StarStar(13);
        var net = new DuelingQNet(5, [16], 4, rng, noisy: true); // noise off by default → deterministic means
        var grown = net.GrowInput(8);
        Assert.True(((DuelingQNet)grown).Noisy);

        var old = Tensor.RandomNormal(rng, 0f, 1f, 4, 5);
        var extra = Tensor.RandomNormal(rng, 0f, 2f, 4, 3);
        float diff = MaxAbsDiff(net.Forward(old).Data, grown.Forward(Concat(old, extra)).Data);
        Assert.True(diff < 1e-5f, $"Noisy DuelingQNet grown output diverged by {diff}");
    }

    [Fact]
    public void GrowInput_ResidualMlp_PreservesFunctionOnOriginalFeatures()
    {
        var rng = new Xoshiro256StarStar(21);
        var net = new ResidualMlp(6, 8, 2, rng);
        var grown = net.GrowInput(10);
        Assert.Equal(10, grown.InputSize);

        var old = Tensor.RandomNormal(rng, 0f, 1f, 5, 6);
        var extra = Tensor.RandomNormal(rng, 0f, 4f, 5, 4);
        float diff = MaxAbsDiff(net.Forward(old).Data, grown.Forward(Concat(old, extra)).Data);
        Assert.True(diff < 1e-5f, $"ResidualMlp grown output diverged by {diff}");
    }

    [Fact]
    public void GrowInput_ZeroesNewInputWeights_AndPreservesOldRows()
    {
        var rng = new Xoshiro256StarStar(30);
        var net = new Mlp([3, 8, 1], rng, Activation.Relu);
        var grown = (Mlp)net.GrowInput(5);

        var oldW = net.Layers[0].Weight;     // [3, 8]
        var newW = grown.Layers[0].Weight;   // [5, 8]
        int cols = oldW.Cols;

        // Old input rows copied verbatim.
        for (int i = 0; i < oldW.Data.Length; i++)
            Assert.Equal(oldW.Data[i], newW.Data[i]);
        // New input rows are exactly zero.
        for (int i = oldW.Data.Length; i < newW.Data.Length; i++)
            Assert.Equal(0f, newW.Data[i]);
        Assert.Equal(5 * cols, newW.Data.Length);
    }

    [Fact]
    public void GrowInput_CopiesAllNonInputParametersExactly()
    {
        var rng = new Xoshiro256StarStar(31);
        var net = new DuelingQNet(4, [12], 3, rng);
        var grown = net.GrowInput(6);

        using var a = net.Parameters().GetEnumerator();
        using var b = grown.Parameters().GetEnumerator();
        a.MoveNext(); b.MoveNext(); // skip the input weight (shape changed)
        while (a.MoveNext() && b.MoveNext())
            Assert.Equal(a.Current.Data, b.Current.Data);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(3)]
    public void GrowInput_NotLarger_Throws(int newInputSize)
    {
        var net = new Mlp([4, 8, 1], new Xoshiro256StarStar(1), Activation.Relu);
        Assert.Throws<ArgumentException>(() => net.GrowInput(newInputSize));
    }
}
