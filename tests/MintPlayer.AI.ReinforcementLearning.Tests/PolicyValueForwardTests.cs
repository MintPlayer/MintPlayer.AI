using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M43.1: the inference seam <see cref="IPolicyValueForward"/>. The default <see cref="AutogradPolicyValueForward"/>
/// must be bitwise-identical to calling the net's own <c>Forward</c> — so routing self-play inference through the seam
/// changes nothing on the CPU path, and the GPU-resident impl (M43.2) has an exact reference to match against.
/// </summary>
public class PolicyValueForwardTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public void Autograd_forward_matches_net_forward_bitwise(int rows)
    {
        const int obsSize = 18 * 8 * 8, actions = 4672;
        var net = new ConvResidualPolicyValueNet(planes: 18, boardH: 8, boardW: 8, actions: actions,
            filters: 8, blocks: 2, new Xoshiro256StarStar(1));
        var rng = new Xoshiro256StarStar(2);
        var obs = new float[rows * obsSize];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)(rng.NextDouble() * 2 - 1);

        var (logitsT, valueT) = net.Forward(new Tensor(obs, rows, obsSize));
        var (logits, value) = new AutogradPolicyValueForward(net, obsSize).Forward(obs, rows);

        Assert.Equal(rows * actions, logits.Length);
        Assert.Equal(rows, value.Length);
        for (int i = 0; i < logits.Length; i++) Assert.Equal(logitsT.Data[i], logits[i]);
        for (int i = 0; i < value.Length; i++) Assert.Equal(valueT.Data[i], value[i]);
    }
}
