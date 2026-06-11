using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class PpoTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void Ppo_SolvesCartPole_Above475_MedianOf3Seeds()
    {
        var finals = new List<double>();
        foreach (ulong seed in new ulong[] { 1, 2, 3 })
        {
            var seeds = new SeedSequence(seed);
            var evalEnv = new CartPoleEnv();
            var result = PpoTrainer.Train(_ => new CartPoleEnv(), evalEnv, new PpoOptions
            {
                TotalSteps = 400_000,
                SolveThreshold = 475,
            }, seeds);

            var eval = Evaluator.Evaluate(evalEnv, result.Agent, episodes: 100, seeds.Derive(RngStreams.Evaluation));
            finals.Add(eval.MeanReturn);
        }

        finals.Sort();
        Assert.True(finals[1] >= 475, $"median eval return {finals[1]:F1} (all: {string.Join(", ", finals.Select(f => f.ToString("F0")))})");
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Ppo_ParallelEnvs_MatchSequential_AtMetricLevel()
    {
        // Env stepping is bitwise-identical between modes (VectorEnvTests proves it),
        // and the learner consumes rollouts in a fixed order, so short trainings from
        // the same master seed should produce identical actor outputs.
        static float[] Run(bool parallel)
        {
            var seeds = new SeedSequence(13);
            var result = PpoTrainer.Train(_ => new CartPoleEnv(), new CartPoleEnv(), new PpoOptions
            {
                TotalSteps = 8 * 128 * 3, // 3 iterations
                ParallelEnvs = parallel,
                EvalEveryIterations = int.MaxValue,
            }, seeds);
            return result.Actor.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor([0.01f, 0.02f, -0.01f, 0.03f], 1, 4)).Data;
        }

        Assert.Equal(Run(parallel: false), Run(parallel: true));
    }
}
