using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// RANKING PRD M51.2: the γ=0 dense all-action regression (<see cref="DqnOptions.DenseTargets"/>) must fix
/// exactly the failure chosen-action-only regression cannot see — a policy that never samples the best
/// action still learns its Q, so the argmax ranking comes out right.
/// </summary>
public class DqnDenseTargetsTests
{
    /// <summary>Single constant state, 3 actions with fixed rewards, episodes truncate on a move budget —
    /// the smallest environment where per-action ranking is the whole problem.</summary>
    private sealed class ThreeArmEnv : IEnvironment<float[], int>
    {
        public static readonly float[] ArmRewards = [0.1f, 1.0f, 0.4f];
        private int _moves;

        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 4);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(3);

        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
        {
            _moves = 0;
            return (Obs(), EnvInfo.Empty);
        }

        public StepResult<float[]> Step(int action)
        {
            _moves++;
            return new StepResult<float[]>(Obs(), ArmRewards[action], false, _moves >= 8, EnvInfo.Empty);
        }

        public string RenderString() => $"moves={_moves}";

        private static float[] Obs() => [1f, 0.5f, 0.25f, 0.125f];
    }

    [Fact]
    public void DenseTargets_TeachRankingOfNeverSampledActions()
    {
        // Pure-greedy behavior (ε ≡ 0) from a random init: without dense targets the policy locks onto
        // whatever arm the init favors and the other arms' Q-values are never updated. The dense targets
        // must rank all three arms correctly anyway.
        var options = new DqnOptions
        {
            Hidden = [16],
            Gamma = 0,
            LearningRate = 1e-2f,
            BufferCapacity = 512,
            BatchSize = 16,
            WarmupSteps = 32,
            Epsilon = new LinearSchedule(0, 0, 1),
            MaxSteps = 600,
            EvalEvery = 10_000, // never — keep the run pure data collection + training
            DenseTargets = _ => ThreeArmEnv.ArmRewards,
        };

        var result = DqnTrainer.Train(new ThreeArmEnv(), options, new SeedSequence(7));

        var q = result.Agent.QValues([1f, 0.5f, 0.25f, 0.125f]);
        Assert.True(q[1] > q[2] && q[2] > q[0],
            $"dense targets must rank Q(1) > Q(2) > Q(0); got [{q[0]:F3}, {q[1]:F3}, {q[2]:F3}]");
    }

    [Fact]
    public void DenseTargets_NaN_LeavesActionsUnsupervised()
    {
        // Only arm 1 gets a dense target; the never-sampled, never-supervised arms must not converge to it.
        var options = new DqnOptions
        {
            Hidden = [16],
            Gamma = 0,
            LearningRate = 1e-2f,
            BufferCapacity = 512,
            BatchSize = 16,
            WarmupSteps = 32,
            Epsilon = new LinearSchedule(0, 0, 1),
            MaxSteps = 600,
            EvalEvery = 10_000,
            DenseTargets = _ => [float.NaN, 1.0f, float.NaN],
        };

        var result = DqnTrainer.Train(new ThreeArmEnv(), options, new SeedSequence(7));

        var q = result.Agent.QValues([1f, 0.5f, 0.25f, 0.125f]);
        Assert.True(Math.Abs(q[1] - 1.0f) < 0.15f, $"supervised arm must approach 1.0; got {q[1]:F3}");
    }

    [Fact]
    public void DenseTargets_RequireGammaZero()
    {
        var options = new DqnOptions { Gamma = 0.99, DenseTargets = _ => ThreeArmEnv.ArmRewards };
        Assert.Throws<ArgumentException>(() => DqnTrainer.Train(new ThreeArmEnv(), options, new SeedSequence(1)));
    }
}
