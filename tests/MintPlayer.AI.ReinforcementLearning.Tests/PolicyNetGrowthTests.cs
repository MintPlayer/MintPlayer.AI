using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class PolicyNetGrowthTests
{
    private static Tensor RandomObs(int size, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[size];
        for (int i = 0; i < size; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor(data, 1, size);
    }

    [Fact]
    public void PolicyValueNet_WidenTo_IsFunctionPreserving()
    {
        var net = new PolicyValueNet(20, [12, 16], 5, new Xoshiro256StarStar(1));
        var x = RandomObs(20, 42);
        var (l0, v0) = net.Forward(x);
        var logits0 = (float[])l0.Data.Clone(); float val0 = v0.Data[0];

        var wider = net.WidenTo([24, 40], new Xoshiro256StarStar(2)); // Net2WiderNet
        Assert.Equal([24, 40], wider.Trunk);
        var (l1, v1) = wider.Forward(x);
        for (int a = 0; a < logits0.Length; a++) Assert.Equal(logits0[a], l1.Data[a], 3);
        Assert.Equal(val0, v1.Data[0], 3);
    }

    [Fact]
    public void PolicyValueNet_Deepen_IsFunctionPreserving()
    {
        var net = new PolicyValueNet(20, [16, 16], 5, new Xoshiro256StarStar(3));
        var x = RandomObs(20, 7);
        var (l0, v0) = net.Forward(x);
        var logits0 = (float[])l0.Data.Clone(); float val0 = v0.Data[0];

        var deeper = net.Deepen(new Xoshiro256StarStar(4)); // Net2DeeperNet (identity-init layer)
        Assert.Equal([16, 16, 16], deeper.Trunk);
        var (l1, v1) = deeper.Forward(x);
        for (int a = 0; a < logits0.Length; a++) Assert.Equal(logits0[a], l1.Data[a], 3);
        Assert.Equal(val0, v1.Data[0], 3);
    }

    [Fact]
    public void CubePolicyNet_LoadsLegacyV1Checkpoint()
    {
        // A shipped cube.policy.ckpt is v1: header + one hidden width + the four layers' floats (trunk1, trunk2,
        // policy head, value head), which is exactly Parameters() order. Hand-write that layout and confirm the
        // refactored (variable-depth) loader still reads it and reproduces the forward pass.
        var reference = new CubePolicyNet(new Xoshiro256StarStar(9), hidden: 8);
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            CheckpointFormat.WriteHeader(w, CubePolicyNet.CheckpointKind, 1);
            w.Write(8); // single hidden width (v1)
            foreach (var p in reference.Parameters()) CheckpointFormat.WriteFloats(w, p.Data);
        }
        ms.Position = 0;

        var loaded = CubePolicyNet.Load(ms);
        Assert.Equal([8, 8], loaded.Trunk);
        var x = RandomObs(RubiksCubeEnv.ObservationSize, 123);
        var (lr, vr) = reference.Forward(x);
        var (ll, vl) = loaded.Forward(x);
        Assert.Equal(lr.Data, ll.Data);
        Assert.Equal(vr.Data[0], vl.Data[0]);
    }

    [Fact]
    public void CubePolicyNet_GrownCheckpoint_RoundTrips()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(5), hidden: 8)
            .WidenTo([16, 16], new Xoshiro256StarStar(6))
            .Deepen(new Xoshiro256StarStar(7)); // → [16,16,16]
        using var ms = new MemoryStream();
        net.Save(ms);
        ms.Position = 0;
        var loaded = CubePolicyNet.Load(ms);

        Assert.Equal([16, 16, 16], loaded.Trunk);
        var x = RandomObs(RubiksCubeEnv.ObservationSize, 321);
        Assert.Equal(net.Forward(x).Logits.Data, loaded.Forward(x).Logits.Data);
    }
}
