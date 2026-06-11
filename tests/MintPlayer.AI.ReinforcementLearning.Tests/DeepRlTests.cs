using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class ReplayBufferTests
{
    [Fact]
    public void Wraparound_OverwritesOldestEntries()
    {
        var buffer = new ReplayBuffer(capacity: 3, obsDim: 1, actionCount: 5);
        for (int i = 0; i < 5; i++)
            buffer.Add([i], i, i, [i + 0.5f], false);

        Assert.Equal(3, buffer.Count);
        // Entries 0 and 1 were overwritten by 3 and 4; sampling can only yield 2, 3, 4.
        var rng = new Xoshiro256StarStar(1);
        for (int i = 0; i < 50; i++)
        {
            var batch = buffer.Sample(4, rng);
            Assert.All(batch.Actions, a => Assert.InRange(a, 2, 4));
        }
    }

    [Fact]
    public void StoresTerminatedFlag_NotTruncated()
    {
        // The trainer must pass terminated only. This guards the call-site convention:
        // a truncated-but-not-terminated transition is stored with terminated=false,
        // so its TD target still bootstraps.
        var buffer = new ReplayBuffer(capacity: 2, obsDim: 1, actionCount: 2);
        buffer.Add([1f], 0, 1.0, [2f], terminated: false); // truncated transition
        buffer.Add([3f], 1, 1.0, [4f], terminated: true);  // true terminal

        var rng = new Xoshiro256StarStar(1);
        var batch = buffer.Sample(64, rng);
        for (int i = 0; i < batch.Size; i++)
            Assert.Equal(batch.Actions[i] == 1, batch.Terminated[i]);
    }
}

public class DqnUnitTests
{
    [Fact]
    public void OverfitOneTransition_QConvergesToReward()
    {
        // Single terminal transition (s, a, r=1, terminated): Q(s,a) must converge to
        // exactly r. The classic 'can the optimizer learn at all' sanity check.
        var rng = new Xoshiro256StarStar(2);
        var net = new Mlp([4, 32, 2], rng, Activation.Relu);
        var adam = new Adam(net.Parameters(), 1e-2f);
        var obs = new Tensor([0.1f, -0.2f, 0.3f, 0.05f], 1, 4);
        var target = new Tensor([1f], 1);

        float loss = float.MaxValue;
        for (int i = 0; i < 500; i++)
        {
            adam.ZeroGrad();
            var lossTensor = net.Forward(obs).Gather([1]).HuberLoss(target);
            lossTensor.Backward();
            adam.Step();
            loss = lossTensor.Data[0];
        }

        Assert.True(loss < 1e-5f, $"loss after 500 steps: {loss}");
        Assert.Equal(1f, net.Forward(obs).Gather([1]).Data[0], 1e-2f);
    }

    [Fact]
    public void TruncatedTransitions_StillBootstrap_ProbeEnvConvergesToGeometricValue()
    {
        // Probe environment (research-recommended): constant reward 1, never terminates,
        // truncates every 5 steps. Correct truncation handling lets Q converge toward
        // r/(1−γ) = 10; the classic done-flag bug (storing done = terminated||truncated)
        // caps Q at the 5-step partial sum ≈ 4.1. Asserting Q > 8 separates the two.
        var seeds = new SeedSequence(9);
        var result = DqnTrainer.Train(new ConstantRewardTruncatingEnv(), new DqnOptions
        {
            Hidden = [32],
            Gamma = 0.9,
            LearningRate = 1e-2f,
            MaxSteps = 6000,
            WarmupSteps = 200,
            BatchSize = 32,
            TargetSyncEvery = 200,
            Epsilon = new LinearSchedule(1.0, 0.1, 1000),
            EvalEvery = int.MaxValue, // probe env has no meaningful greedy eval
        }, seeds);

        using (GradMode.NoGrad())
        {
            var q = result.Network.Forward(new Tensor([0f], 1, 1));
            float maxQ = Math.Max(q.Data[0], q.Data[1]);
            Assert.True(maxQ > 8f, $"Q after training: {maxQ:F2} (expected ≈ 10; ≈ 4 indicates the truncation done-flag bug)");
        }
    }

    private sealed class ConstantRewardTruncatingEnv : MintPlayer.AI.ReinforcementLearning.Core.Environments.IEnvironment<float[], int>
    {
        private int _steps;

        public MintPlayer.AI.ReinforcementLearning.Core.Environments.Space<float[]> ObservationSpace { get; } = new MintPlayer.AI.ReinforcementLearning.Core.Environments.BoxSpace(-1f, 1f, 1);
        public MintPlayer.AI.ReinforcementLearning.Core.Environments.Space<int> ActionSpace { get; } = new MintPlayer.AI.ReinforcementLearning.Core.Environments.DiscreteSpace(2);

        public (float[] Observation, MintPlayer.AI.ReinforcementLearning.Core.Environments.EnvInfo Info) Reset(ulong? seed = null)
        {
            _steps = 0;
            return ([0f], MintPlayer.AI.ReinforcementLearning.Core.Environments.EnvInfo.Empty);
        }

        public MintPlayer.AI.ReinforcementLearning.Core.Environments.StepResult<float[]> Step(int action)
            => new([0f], 1.0, Terminated: false, Truncated: ++_steps >= 5, MintPlayer.AI.ReinforcementLearning.Core.Environments.EnvInfo.Empty);

        public string RenderString() => "·";
    }
}

public class CartPoleSolveTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void Reinforce_SolvesCartPole_Above400_MedianOf3Seeds()
    {
        var finals = new List<double>();
        foreach (ulong seed in new ulong[] { 1, 2, 3 })
        {
            var seeds = new SeedSequence(seed);
            var env = new CartPoleEnv();
            var result = ReinforceTrainer.Train(env, new ReinforceOptions
            {
                MaxEpisodes = 3000,
                SolveThreshold = 450,
            }, seeds);

            var eval = Evaluator.Evaluate(env, result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
            finals.Add(eval.MeanReturn);
        }

        finals.Sort();
        Assert.True(finals[1] >= 400, $"median eval return {finals[1]:F1} (all: {string.Join(", ", finals.Select(f => f.ToString("F0")))})");
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Dqn_SolvesCartPole_Above475_MedianOf3Seeds()
    {
        var finals = new List<double>();
        foreach (ulong seed in new ulong[] { 1, 2, 3 })
        {
            var seeds = new SeedSequence(seed);
            var env = new CartPoleEnv();
            var result = DqnTrainer.Train(env, new DqnOptions
            {
                MaxSteps = 150_000,
                SolveThreshold = 475,
            }, seeds);

            var eval = Evaluator.Evaluate(env, result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
            finals.Add(eval.MeanReturn);
        }

        finals.Sort();
        Assert.True(finals[1] >= 475, $"median eval return {finals[1]:F1} (all: {string.Join(", ", finals.Select(f => f.ToString("F0")))})");
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Dqn_IsDeterministic_GivenSameMasterSeed()
    {
        static float[] Run()
        {
            var seeds = new SeedSequence(5);
            var result = DqnTrainer.Train(new CartPoleEnv(), new DqnOptions
            {
                MaxSteps = 3000,
                WarmupSteps = 500,
                Epsilon = new LinearSchedule(1.0, 0.05, 2000),
            }, seeds);
            return result.Network.Forward(new Tensor([0.01f, 0.02f, -0.01f, 0.03f], 1, 4)).Data;
        }

        Assert.Equal(Run(), Run());
    }
}
