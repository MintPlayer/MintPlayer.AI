using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class SacTests
{
    private static void AssertParamsEqual(IModule a, IModule b)
    {
        var pa = a.Parameters().ToArray();
        var pb = b.Parameters().ToArray();
        Assert.Equal(pa.Length, pb.Length);
        for (int i = 0; i < pa.Length; i++) Assert.Equal(pa[i].Data, pb[i].Data);
    }

    [Fact]
    public void ConcatCols_RoutesGradientToEachHalf()
    {
        var a = new Tensor([1f, 2f, 3f, 4f], 2, 2) { RequiresGrad = true };
        var b = new Tensor([5f, 6f], 2, 1) { RequiresGrad = true };

        var cat = a.ConcatCols(b);
        Assert.Equal(3, cat.Cols);
        Assert.Equal([1f, 2f, 5f, 3f, 4f, 6f], cat.Data); // rows interleaved correctly

        cat.Sum().Backward();
        Assert.Equal([1f, 1f, 1f, 1f], a.Grad!);
        Assert.Equal([1f, 1f], b.Grad!);
    }

    [Fact]
    public void SliceCols_ScattersGradientBackToSourceColumns()
    {
        var x = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], 2, 4) { RequiresGrad = true };
        var mid = x.SliceCols(1, 2);
        Assert.Equal([2f, 3f, 6f, 7f], mid.Data);

        mid.Sum().Backward();
        Assert.Equal([0f, 1f, 1f, 0f, 0f, 1f, 1f, 0f], x.Grad!); // only the sliced columns receive grad
    }

    [Fact]
    public void Normal_RSample_IsBoundedAndLogProbFinite()
    {
        var mean = new Tensor([0.0f, 0.0f], 1, 2);
        var logStd = new Tensor([-0.5f, -0.5f], 1, 2);
        var dist = new Normal(mean, logStd);

        var (action, logProb) = dist.RSample(new Xoshiro256StarStar(3));
        Assert.Equal([1, 2], action.Shape);
        Assert.All(action.Data, v => Assert.InRange(v, -1f, 1f)); // tanh-squashed
        Assert.Single(logProb.Data);
        Assert.True(float.IsFinite(logProb.Data[0]));

        Assert.Equal(MathF.Tanh(0f), dist.Mode().Data[0], 6); // greedy = tanh(mean)
    }

    [Fact]
    public void Normal_FromNetOutput_SplitsAndClampsLogStd()
    {
        // Output [mean(2) | logStd(2)] with a huge log-σ that must clamp to LogStdMax before exp.
        var netOut = new Tensor([0.1f, -0.2f, 50f, 50f], 1, 4);
        var dist = Normal.FromNetOutput(netOut, 2);
        var (_, logProb) = dist.RSample(new Xoshiro256StarStar(1));
        Assert.True(float.IsFinite(logProb.Data[0])); // would be NaN/Inf if log-σ weren't clamped
    }

    private static SacOptions ResumeOptions(int steps) => new()
    {
        Hidden = [32, 32],
        BufferCapacity = 5_000,
        WarmupSteps = 200,
        BatchSize = 32,
        EvalEvery = int.MaxValue, // no eval interruptions → clean determinism
        MaxSteps = steps,
    };

    [Fact]
    public void Sac_ResumesBitwiseIdentically()
    {
        const ulong masterSeed = 123;
        var resultA = SacTrainer.Train(new PendulumEnv(), ResumeOptions(3_000), new SeedSequence(masterSeed));

        var resultB1 = SacTrainer.Train(new PendulumEnv(), ResumeOptions(1_500), new SeedSequence(masterSeed));
        using var stream = new MemoryStream();
        resultB1.State.Save(stream);
        stream.Position = 0;
        var restored = SacTrainingState.Load(stream);
        var resultB2 = SacTrainer.Train(new PendulumEnv(), ResumeOptions(3_000), new SeedSequence(masterSeed), resume: restored);

        AssertParamsEqual(resultA.Actor, resultB2.Actor);
        AssertParamsEqual(resultA.Critic1, resultB2.Critic1);
        AssertParamsEqual(resultA.State.Target1, resultB2.State.Target1);
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Sac_SolvesPendulum_MedianOf3Seeds()
    {
        // RL is statistical — single-seed pass/fail lies (PRD §3), so gate on the median of 3. A competent
        // SAC reaches a mean return around −150 on Pendulum; random ≈ −1200. Gate generously at −250.
        var finals = new List<double>();
        foreach (ulong seed in new ulong[] { 1, 2, 3 })
        {
            var seeds = new SeedSequence(seed);
            var result = SacTrainer.Train(new PendulumEnv(), new SacOptions
            {
                Hidden = [128, 128],
                MaxSteps = 25_000,
                WarmupSteps = 1_000,
                EvalEvery = 5_000,
                SolveThreshold = -150,
            }, seeds);
            var eval = Evaluator.Evaluate(new PendulumEnv(), result.Agent, episodes: 50, seeds.Derive(RngStreams.Evaluation));
            finals.Add(eval.MeanReturn);
        }

        finals.Sort();
        Assert.True(finals[1] >= -250,
            $"median SAC eval {finals[1]:F1} (all: {string.Join(", ", finals.Select(f => f.ToString("F0")))})");
    }
}
