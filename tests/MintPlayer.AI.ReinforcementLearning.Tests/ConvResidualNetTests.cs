using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Contract for the M42.2 convolutional residual policy/value net: correct head shapes, an exact checkpoint
/// round-trip, and that it actually learns (a fixed batch's AlphaZero loss falls under Adam). Small dims — this
/// asserts the net WORKS end-to-end through the Conv2D op, not chess strength (that's the Lab `--game chess --arch
/// conv` gate, M42.3).
/// </summary>
public class ConvResidualNetTests
{
    private const int Planes = 3, H = 4, W = 4, Actions = 10, Filters = 8, Blocks = 2, Batch = 4;

    private static ConvResidualPolicyValueNet Net(ulong seed = 7)
        => new(Planes, H, W, Actions, Filters, Blocks, new Xoshiro256StarStar(seed));

    private static Tensor Obs(ulong seed = 1)
        => Tensor.RandomNormal(new Xoshiro256StarStar(seed), 0f, 0.5f, Batch, Planes * H * W);

    [Fact]
    public void Forward_ProducesPolicyAndValueHeadShapes()
    {
        var (logits, value) = Net().Forward(Obs());
        Assert.Equal([Batch, Actions], logits.Shape);
        Assert.Equal([Batch, 1], value.Shape);
        Assert.All(logits.Data, v => Assert.False(float.IsNaN(v)));
        Assert.All(value.Data, v => Assert.False(float.IsNaN(v)));
    }

    [Fact]
    public void SaveLoad_RoundTripsExactly()
    {
        const string kind = "selfplay-pv-conv";
        var net = Net();
        var obs = Obs(2);
        var (logitsBefore, valueBefore) = net.Forward(obs);

        using var ms = new MemoryStream();
        net.Save(ms, kind);
        ms.Position = 0;
        var reloaded = ConvResidualPolicyValueNet.Load(ms, kind, Actions);

        var (logitsAfter, valueAfter) = reloaded.Forward(obs);
        Assert.Equal(logitsBefore.Data, logitsAfter.Data);   // bitwise — same weights, same math
        Assert.Equal(valueBefore.Data, valueAfter.Data);
    }

    [Fact]
    public void Training_OnAFixedBatch_ReducesTheLoss()
    {
        var net = Net(3);
        var adam = new Adam(net.Parameters(), 3e-3f);
        var obs = Obs(4);

        // A fixed target: a peaked policy (argmax over a random logit vector) + a target value.
        var rng = new Xoshiro256StarStar(5);
        var pi = new float[Batch * Actions];
        var z = new float[Batch];
        for (int b = 0; b < Batch; b++)
        {
            int hot = rng.NextInt(Actions);
            pi[b * Actions + hot] = 1f;
            z[b] = b % 2 == 0 ? 0.7f : -0.7f;
        }
        var piT = new Tensor(pi, Batch, Actions);
        var zT = new Tensor(z, Batch);

        float Step()
        {
            var (logits, value) = net.Forward(obs);
            var ce = logits.LogSoftmax().Mul(piT).Sum().MulScalar(-1f / Batch);
            var vl = value.Reshape(Batch).Tanh().MseLoss(zT);
            var loss = ce.Add(vl);
            adam.ZeroGrad();
            loss.Backward();
            adam.ClipGradNorm(5f);
            adam.Step();
            return loss.Data[0];
        }

        float first = Step();
        float last = first;
        for (int i = 0; i < 60; i++) last = Step();

        Assert.True(last < first * 0.6f, $"loss did not fall enough: {first:F4} → {last:F4}");
    }
}
