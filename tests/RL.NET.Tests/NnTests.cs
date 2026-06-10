using RLNet.Core.Nn;
using RLNet.Core.Numerics;
using RLNet.Core.Random;

namespace RLNet.Tests;

public class NnTests
{
    [Fact]
    public void Adam_MinimizesSimpleQuadratic()
    {
        // Minimize (x - 3)^2; Adam should walk x from 0 to ~3.
        var x = new Tensor([0f], 1) { RequiresGrad = true };
        var target = new Tensor([3f], 1);
        var adam = new Adam([x], learningRate: 0.1f);

        for (int i = 0; i < 500; i++)
        {
            adam.ZeroGrad();
            x.MseLoss(target).Backward();
            adam.Step();
        }

        Assert.Equal(3f, x.Data[0], 1e-2f);
    }

    [Fact]
    public void Mlp_LearnsXor()
    {
        var rng = new Xoshiro256StarStar(3);
        var net = new Mlp([2, 8, 1], rng, Activation.Tanh);
        var adam = new Adam(net.Parameters(), 0.05f);
        var inputs = new Tensor([0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f], 4, 2);
        var targets = new Tensor([0f, 1f, 1f, 0f], 4, 1);

        float loss = float.MaxValue;
        for (int i = 0; i < 1000; i++)
        {
            adam.ZeroGrad();
            var lossTensor = net.Forward(inputs).MseLoss(targets);
            lossTensor.Backward();
            adam.Step();
            loss = lossTensor.Data[0];
        }

        Assert.True(loss < 0.01f, $"XOR loss after 1000 steps was {loss}");
    }

    [Fact]
    public void ClipGradNorm_ScalesLargeGradients()
    {
        var x = new Tensor([0f, 0f], 2) { RequiresGrad = true };
        var adam = new Adam([x]);
        x.MseLoss(new Tensor([300f, 400f], 2)).Backward();

        float preClipNorm = adam.ClipGradNorm(1f);

        // Gradient of MSE is 2(x−t)/N = (−300, −400) → norm 500.
        Assert.Equal(500f, preClipNorm, 0.1f);
        double postNorm = Math.Sqrt(x.Grad![0] * x.Grad[0] + x.Grad[1] * x.Grad[1]);
        Assert.Equal(1.0, postNorm, 1e-3);
    }

    [Fact]
    public void Mlp_CopyFrom_MakesOutputsIdentical()
    {
        var rng = new Xoshiro256StarStar(4);
        var online = new Mlp([4, 16, 2], rng, Activation.Relu);
        var target = new Mlp([4, 16, 2], rng, Activation.Relu);
        var input = Tensor.RandomNormal(rng, 0f, 1f, 5, 4);

        Assert.NotEqual(online.Forward(input).Data, target.Forward(input).Data);
        target.CopyFrom(online);
        Assert.Equal(online.Forward(input).Data, target.Forward(input).Data);
    }

    [Fact]
    public void Categorical_SamplesMatchDistribution()
    {
        // Logits ln(1), ln(2), ln(7) → probabilities 0.1, 0.2, 0.7.
        var logits = new Tensor([MathF.Log(1f), MathF.Log(2f), MathF.Log(7f)], 1, 3);
        var dist = new Categorical(logits);
        var rng = new Xoshiro256StarStar(6);

        var counts = new int[3];
        const int samples = 30_000;
        for (int i = 0; i < samples; i++)
            counts[dist.Sample(rng)[0]]++;

        Assert.Equal(0.1, counts[0] / (double)samples, 0.02);
        Assert.Equal(0.2, counts[1] / (double)samples, 0.02);
        Assert.Equal(0.7, counts[2] / (double)samples, 0.02);
        Assert.Equal(2, dist.Mode()[0]);
    }

    [Fact]
    public void Categorical_UniformEntropy_IsLogN()
    {
        var dist = new Categorical(Tensor.Zeros(1, 4));
        Assert.Equal(MathF.Log(4f), dist.Entropy().Data[0], 1e-5f);
    }

    [Fact]
    public void PolicyGradient_IncreasesLogProbOfRewardedAction()
    {
        // The M2 sanity check for REINFORCE: after one ascent step on an action with
        // positive advantage, the policy must assign that action higher probability.
        var rng = new Xoshiro256StarStar(8);
        var net = new Mlp([3, 8, 2], rng, Activation.Tanh);
        var adam = new Adam(net.Parameters(), 0.01f);
        var obs = new Tensor([0.5f, -0.2f, 0.9f], 1, 3);

        float Before() => new Categorical(net.Forward(obs)).LogProb([1]).Data[0];

        float logProbBefore = Before();
        adam.ZeroGrad();
        var loss = new Categorical(net.Forward(obs)).LogProb([1])
            .Mul(new Tensor([2.0f], 1))   // positive advantage
            .Mean().MulScalar(-1f);       // gradient ascent via negated loss
        loss.Backward();
        adam.Step();

        Assert.True(Before() > logProbBefore, "log-prob of the rewarded action did not increase");
    }

    [Fact]
    public void Training_IsDeterministic_GivenSameSeed()
    {
        static float[] Run()
        {
            var rng = new Xoshiro256StarStar(11);
            var net = new Mlp([2, 8, 1], rng, Activation.Tanh);
            var adam = new Adam(net.Parameters(), 0.05f);
            var inputs = new Tensor([0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f], 4, 2);
            var targets = new Tensor([0f, 1f, 1f, 0f], 4, 1);
            for (int i = 0; i < 100; i++)
            {
                adam.ZeroGrad();
                net.Forward(inputs).MseLoss(targets).Backward();
                adam.Step();
            }
            return net.Forward(inputs).Data;
        }

        Assert.Equal(Run(), Run());
    }
}
