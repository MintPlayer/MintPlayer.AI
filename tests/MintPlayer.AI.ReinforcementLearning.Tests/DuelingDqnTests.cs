using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class DuelingDqnTests
{
    private static void AssertParamsEqual(IValueNet a, IValueNet b)
    {
        var pa = a.Parameters().ToArray();
        var pb = b.Parameters().ToArray();
        Assert.Equal(pa.Length, pb.Length);
        for (int i = 0; i < pa.Length; i++) Assert.Equal(pa[i].Data, pb[i].Data);
    }

    [Fact]
    public void DuelingQNet_Forward_ProducesQPerAction()
    {
        var net = new DuelingQNet(4, [16, 16], 3, new Xoshiro256StarStar(5));
        var q = net.Forward(new Tensor([0.1f, -0.2f, 0.3f, 0.05f, 0.0f, 0.4f, -0.1f, 0.2f], 2, 4));
        Assert.Equal(2, q.Rows);
        Assert.Equal(3, q.Cols);
        Assert.All(q.Data, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void DuelingQNet_AddingConstantToAdvantageStream_DoesNotChangeQ()
    {
        // The mean-subtraction makes Q invariant to a constant advantage shift — the identifiability
        // fix that defines dueling. We test it via the equivalent net property: two nets that differ
        // only by the advantage-head bias (a constant per action shifts cancel after mean-centering only
        // if uniform) — here we assert the structural invariant that Q's per-row mean equals V, i.e.
        // shifting every advantage logit by the same constant c leaves Q unchanged.
        var net = new DuelingQNet(4, [8], 4, new Xoshiro256StarStar(1));
        var x = new Tensor([0.2f, 0.1f, -0.3f, 0.4f], 1, 4);
        var q1 = (float[])net.Forward(x).Data.Clone();

        // Add a constant to every advantage-head bias → uniform advantage shift → Q must be unchanged.
        var advBias = net.Parameters().Last(); // advantage head bias is the final parameter
        for (int i = 0; i < advBias.Data.Length; i++) advBias.Data[i] += 2.5f;
        var q2 = net.Forward(x).Data;

        for (int a = 0; a < q1.Length; a++) Assert.Equal(q1[a], q2[a], 4);
    }

    [Fact]
    public void DuelingQNetCheckpoint_RoundTrip_PreservesForward()
    {
        var net = new DuelingQNet(6, [16, 12], 5, new Xoshiro256StarStar(7));
        using var ms = new MemoryStream();
        DuelingQNetCheckpoint.Save(net, ms);
        ms.Position = 0;
        var loaded = DuelingQNetCheckpoint.Load(ms);

        AssertParamsEqual(net, loaded);
        var x = new Tensor([0.1f, 0.2f, -0.1f, 0.3f, 0.0f, -0.2f], 1, 6);
        Assert.Equal(net.Forward(x).Data, loaded.Forward(x).Data);
    }

    [Fact]
    public void QNetCheckpoint_RoundTrips_BothNetworkTypes()
    {
        IValueNet mlp = new Mlp([4, 8, 3], new Xoshiro256StarStar(2), Activation.Relu);
        IValueNet dueling = new DuelingQNet(4, [8], 3, new Xoshiro256StarStar(3));

        foreach (var net in new[] { mlp, dueling })
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true)) QNetCheckpoint.Write(net, w);
            ms.Position = 0;
            using var r = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true);
            var loaded = QNetCheckpoint.Read(r);
            Assert.Equal(net.GetType(), loaded.GetType());
            AssertParamsEqual(net, loaded);
        }
    }

    private static DqnOptions Options(int steps) => new()
    {
        Dueling = true,
        Hidden = [32, 32],
        BufferCapacity = 5_000,
        WarmupSteps = 200,
        BatchSize = 32,
        TargetSyncEvery = 200,
        EvalEvery = int.MaxValue, // no eval interruptions → clean determinism
        MaxSteps = steps,
    };

    [Fact]
    public void Dqn_Dueling_ResumesBitwiseIdentically()
    {
        const ulong masterSeed = 123;
        var resultA = DqnTrainer.Train(new CartPoleEnv(), Options(4_000), new SeedSequence(masterSeed));

        var resultB1 = DqnTrainer.Train(new CartPoleEnv(), Options(2_000), new SeedSequence(masterSeed));
        using var stream = new MemoryStream();
        resultB1.State.Save(stream);
        stream.Position = 0;
        var restored = DqnTrainingState.Load(stream);
        Assert.IsType<DuelingQNet>(restored.Online); // the type tag survived the round-trip
        var resultB2 = DqnTrainer.Train(new CartPoleEnv(), Options(4_000), new SeedSequence(masterSeed), resume: restored);

        AssertParamsEqual(resultA.Network, resultB2.Network);
        AssertParamsEqual(resultA.State.Target, resultB2.State.Target);
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Dqn_Dueling_SolvesCartPole_MedianOf3Seeds()
    {
        // RL is statistical — single-seed pass/fail lies (PRD §3 testing discipline), so gate on the
        // median of 3, matching the PPO gate. Dueling DQN reaches ~440 per seed on this budget.
        var finals = new List<double>();
        foreach (ulong seed in new ulong[] { 1, 2, 3 })
        {
            var seeds = new SeedSequence(seed);
            var result = DqnTrainer.Train(new CartPoleEnv(), new DqnOptions
            {
                Dueling = true,
                MaxSteps = 120_000,
                SolveThreshold = 475,
            }, seeds);
            var eval = Evaluator.Evaluate(new CartPoleEnv(), result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
            finals.Add(eval.MeanReturn);
        }

        finals.Sort();
        Assert.True(finals[1] >= 400,
            $"median dueling-DQN eval {finals[1]:F1} (all: {string.Join(", ", finals.Select(f => f.ToString("F0")))})");
    }
}
