using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Finite-difference gradient checks (central differences) for every autograd op.
/// Float32 FD is noisy, so tolerances are relative-or-absolute 1e-2 with h = 1e-3,
/// and inputs avoid kink points (ReLU/clamp boundaries).
/// </summary>
public class GradCheckTests
{
    private const float H = 1e-3f;
    private const float Tolerance = 1e-2f;

    /// <summary>Checks d(loss)/d(input) for every element of every input tensor.</summary>
    private static void CheckGradients(Func<Tensor[], Tensor> lossFn, params Tensor[] inputs)
    {
        var loss = lossFn(inputs);
        loss.Backward();

        foreach (var input in inputs)
        {
            Assert.NotNull(input.Grad);
            for (int i = 0; i < input.Length; i++)
            {
                float original = input.Data[i];
                input.Data[i] = original + H;
                float lossPlus = lossFn(inputs).Data[0];
                input.Data[i] = original - H;
                float lossMinus = lossFn(inputs).Data[0];
                input.Data[i] = original;

                float numeric = (lossPlus - lossMinus) / (2 * H);
                float analytic = input.Grad![i];
                float scale = Math.Max(1f, Math.Max(Math.Abs(numeric), Math.Abs(analytic)));
                Assert.True(
                    Math.Abs(numeric - analytic) / scale < Tolerance,
                    $"Gradient mismatch at element {i}: numeric {numeric}, analytic {analytic}");
            }
        }
    }

    private static Tensor Param(float[] data, params int[] shape) => new(data, shape) { RequiresGrad = true };

    // A tensor filled with deterministic values in [-0.8, 0.8] — enough spread to exercise a conv, away from
    // ReLU/clamp kinks (conv itself is kink-free, but this keeps composite checks stable).
    private static Tensor Rand(Xoshiro256StarStar rng, params int[] shape)
    {
        int len = 1;
        foreach (int d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)(rng.NextDouble() * 1.6 - 0.8);
        return new Tensor(data, shape) { RequiresGrad = true };
    }

    [Fact]
    public void MatMul_Gradients()
        => CheckGradients(
            t => t[0].MatMul(t[1]).Sum(),
            Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3),
            Param([0.2f, -0.7f, 1.3f, 0.6f, -0.9f, 0.4f], 3, 2));

    [Fact]
    public void Add_And_Sub_Gradients()
        => CheckGradients(
            t => t[0].Add(t[1]).Mul(t[0].Sub(t[1])).Sum(),
            Param([0.5f, -1.2f, 0.3f, 0.8f], 2, 2),
            Param([0.2f, -0.7f, 1.3f, 0.6f], 2, 2));

    [Fact]
    public void AddBias_Gradients()
        => CheckGradients(
            t => t[0].AddBias(t[1]).Square().Sum(),
            Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3),
            Param([0.3f, -0.5f, 0.9f], 3));

    [Fact]
    public void Mul_WithSharedTensor_Gradients()
        => CheckGradients(t => t[0].Mul(t[0]).Sum(), Param([0.5f, -1.2f, 0.3f], 3));

    [Fact]
    public void Relu_Gradients()
        => CheckGradients(t => t[0].Relu().Sum(), Param([0.5f, -1.2f, 0.3f, -0.8f], 4));

    [Fact]
    public void Tanh_Gradients()
        => CheckGradients(t => t[0].Tanh().Sum(), Param([0.5f, -1.2f, 0.3f, 1.7f], 4));

    [Fact]
    public void Exp_And_Log_Gradients()
        => CheckGradients(t => t[0].Exp().Add(t[0].Log()).Sum(), Param([0.5f, 1.2f, 0.3f], 3));

    [Fact]
    public void Clamp_Gradients()
        => CheckGradients(t => t[0].Clamp(-1f, 1f).Square().Sum(), Param([0.5f, -1.8f, 0.3f, 1.9f], 4));

    [Fact]
    public void Min_Gradients()
        => CheckGradients(
            t => t[0].Min(t[1]).Sum(),
            Param([0.5f, -1.2f, 0.9f], 3),
            Param([0.2f, 0.7f, 1.3f], 3));

    [Fact]
    public void Gather_Gradients()
        => CheckGradients(
            t => t[0].Gather([2, 0]).Square().Sum(),
            Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3));

    [Fact]
    public void Mean_And_SumRows_Gradients()
        => CheckGradients(
            t => t[0].SumRows().Square().Mean(),
            Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3));

    [Fact]
    public void LogSoftmax_Gradients()
        => CheckGradients(
            t => t[0].LogSoftmax().Gather([1, 2]).Sum(),
            Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3));

    [Fact]
    public void HuberLoss_Gradients_BothBranches()
        // diffs 0.3 (quadratic) and 2.0 (linear) exercise both branches of the loss.
        => CheckGradients(
            t => t[0].HuberLoss(new Tensor([0.2f, -1.5f], 2)),
            Param([0.5f, 0.5f], 2));

    [Fact]
    public void MseLoss_Gradients()
        => CheckGradients(
            t => t[0].MseLoss(new Tensor([0.2f, -0.7f, 1.3f], 3)),
            Param([0.5f, -1.2f, 0.3f], 3));

    [Fact]
    public void Conv2D_Gradients_SamePadding()
    {
        // 3×3 "SAME" conv (pad 1, stride 1): the residual-tower workhorse. Checks dInput, dWeight, dBias.
        var rng = new Xoshiro256StarStar(11);
        const int inC = 2, inH = 4, inW = 4, outC = 3, k = 3;
        CheckGradients(
            t => t[0].Conv2D(t[1], t[2], inC, inH, inW, outC, k, k, stride: 1, pad: 1).Square().Sum(),
            Rand(rng, 2, inC * inH * inW), Rand(rng, inC * k * k, outC), Rand(rng, outC));
    }

    [Fact]
    public void Conv2D_Gradients_StridedValid()
    {
        // stride 2, no padding — exercises the striding + boundary (dropped) columns of im2col/col2im.
        var rng = new Xoshiro256StarStar(23);
        const int inC = 2, inH = 5, inW = 5, outC = 2, k = 3;
        CheckGradients(
            t => t[0].Conv2D(t[1], t[2], inC, inH, inW, outC, k, k, stride: 2, pad: 0).Square().Sum(),
            Rand(rng, 1, inC * inH * inW), Rand(rng, inC * k * k, outC), Rand(rng, outC));
    }

    [Fact]
    public void Conv2D_Gradients_OneByOne()
    {
        // 1×1 conv — the shape the AlphaZero policy/value heads use to reduce channels.
        var rng = new Xoshiro256StarStar(31);
        const int inC = 3, inH = 3, inW = 3, outC = 2, k = 1;
        CheckGradients(
            t => t[0].Conv2D(t[1], t[2], inC, inH, inW, outC, k, k, stride: 1, pad: 0).Square().Sum(),
            Rand(rng, 2, inC * inH * inW), Rand(rng, inC * k * k, outC), Rand(rng, outC));
    }

    [Fact]
    public void FullMlp_CompositeLoss_Gradients()
    {
        // End-to-end check through Linear→Tanh→Linear→LogSoftmax→Gather, the REINFORCE shape.
        var rng = new Xoshiro256StarStar(5);
        var net = new Mlp([3, 8, 2], rng, Activation.Tanh);
        var input = new Tensor([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3);
        var advantages = new Tensor([1.5f, -0.5f], 2);

        CheckGradients(
            _ => new Categorical(net.Forward(input)).LogProb([0, 1]).Mul(advantages).Mean().MulScalar(-1f),
            net.Parameters().ToArray());
    }

    [Fact]
    public void Entropy_Gradients()
    {
        var logits = Param([0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f], 2, 3);
        CheckGradients(t => new Categorical(t[0]).Entropy().Mean(), logits);
    }

    [Fact]
    public void NoGrad_RecordsNothing()
    {
        var a = Param([1f, 2f], 2);
        Tensor result;
        using (GradMode.NoGrad())
            result = a.Square().Sum();

        // Backward on a no-grad scalar is legal but propagates nothing.
        result.Backward();
        Assert.Null(a.Grad);
    }

    [Fact]
    public void Detach_CutsTheGraph()
    {
        var a = Param([1f, 2f], 2);
        var detached = a.Square().Detach();
        var loss = detached.Sum();
        loss.Backward();
        Assert.Null(a.Grad);
    }
}
