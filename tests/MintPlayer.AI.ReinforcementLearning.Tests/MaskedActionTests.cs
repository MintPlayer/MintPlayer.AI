using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// A 4-step env where the most rewarding action (3, +10) is ALWAYS illegal (masked). A correct masked
/// learner can never take it, so the best achievable is the legal action 0 (+1/step). If masking leaked,
/// PPO would chase the +10 and the return would blow past the legal maximum — and the violation counter
/// (incremented whenever the masked action is actually stepped) would be non-zero.
/// </summary>
internal sealed class TemptingMaskEnv(int[] violations) : IEnvironment<float[], int>, IActionMaskProvider
{
    public const int Actions = 4, EpisodeLen = 4;
    private int _step;

    public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 2);
    public Space<int> ActionSpace { get; } = new DiscreteSpace(Actions);

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null) { _step = 0; return (Obs(), EnvInfo.Empty); }

    public bool[] CurrentActionMask() => [true, true, true, false]; // action 3 is illegal

    public StepResult<float[]> Step(int action)
    {
        if (action == 3) violations[0]++; // must never happen under correct masking
        double reward = action switch { 0 => 1.0, 3 => 10.0, _ => 0.0 };
        _step++;
        return new StepResult<float[]>(Obs(), reward, _step >= EpisodeLen, false, EnvInfo.Empty);
    }

    private float[] Obs() => [1f, _step / (float)EpisodeLen];
    public string RenderString() => $"step {_step}";
}

public class MaskedActionTests
{
    [Fact]
    public void Ppo_RespectsActionMask_NeverTakesIllegal_AndLearnsBestLegal()
    {
        var violations = new int[1];
        var seeds = new SeedSequence(7);
        var evalEnv = new TemptingMaskEnv(violations);
        var result = PpoTrainer.Train(_ => new TemptingMaskEnv(violations), evalEnv, new PpoOptions
        {
            TotalSteps = 40_000,
            NumEnvs = 8,
            RolloutSteps = 64,
            EvalEveryIterations = int.MaxValue,
        }, seeds);

        var eval = Evaluator.Evaluate(evalEnv, (o, m) => result.Agent.Act(o, m, greedy: true),
            episodes: 50, seeds.Derive(RngStreams.Evaluation));

        // Contract: the illegal action was never stepped, in training rollouts OR eval.
        Assert.Equal(0, violations[0]);
        // Never exceeded the legal maximum (would only be possible by collecting the masked +10).
        Assert.True(eval.MeanReturn <= TemptingMaskEnv.EpisodeLen + 0.01,
            $"return {eval.MeanReturn:F2} exceeds the legal max {TemptingMaskEnv.EpisodeLen} — mask bypassed?");
        // Still learned the best legal action (0 → +1/step).
        Assert.True(eval.MeanReturn >= 3.0, $"return {eval.MeanReturn:F2} — did not learn the best legal action");
    }

    [Fact]
    public void PolicyAgent_MaskedAct_NeverReturnsIllegalAction()
    {
        var net = new Mlp([2, 8, 4], new Xoshiro256StarStar(1), Activation.Tanh);
        var agent = new PolicyAgent(net, new Xoshiro256StarStar(2));
        var mask = new[] { false, true, false, false }; // only action 1 is legal

        for (int i = 0; i < 200; i++)
        {
            Assert.Equal(1, agent.Act([1f, 0.5f], mask, greedy: false)); // sampled
            Assert.Equal(1, agent.Act([1f, 0.5f], mask, greedy: true));  // argmax
        }
    }
}
