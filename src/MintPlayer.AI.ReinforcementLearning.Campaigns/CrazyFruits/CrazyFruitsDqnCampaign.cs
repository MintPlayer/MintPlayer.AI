using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Crazy Fruits DQN campaign (`--game crazyfruits`, PLAN M49) on the shared <see cref="DqnScoreCampaign"/>
/// spine — score-maximizing: eval is the mean episodic score over fixed-seed greedy episodes. Trains the
/// masked Double+Dueling <see cref="DuelingQNet"/> (masking is load-bearing on match-3: unmasked DQN scores
/// below random in the literature — PRD §3.7). Both envs are constructor dependencies (M46.2). All
/// resume/train/keep-best/checkpoint/growth/telemetry plumbing lives in the base; this type supplies only
/// the hyperparameters and the score eval.
/// </summary>
public sealed partial class CrazyFruitsDqnCampaign : DqnScoreCampaign
{
    [Inject] private readonly CrazyFruitsEnv evalEnv;

    public override string Environment => "crazyfruits";
    protected override string StepNoun => "moves";
    protected override string GateLabel => "mean score";
    protected override string DisplayName => "Crazy Fruits DQN";
    protected override string FreshStartDetail => $" (8×8, 6 fruits, {evalEnv.MoveBudget}-move episodes)";
    protected override int ObservationSize => CrazyFruitsEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => CrazyFruitsEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => CrazyFruitsEnv.ActionLabels;

    /// <summary>The Snake recipe re-pointed (masked Double+Dueling DQN); MaxSteps is managed per chunk.</summary>
    protected override DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = Options.Hidden,
        Gamma = Options.Gamma,
        LearningRate = Options.LearningRate,
        BufferCapacity = 100_000,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        Epsilon = new LinearSchedule(Options.EpsilonStart, 0.05, 30_000),
        EvalEpisodes = 20,
    };

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (score, meanReturn) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new("score", score, "F1"),
            new("return", meanReturn, "F2"),
        };
        return (score, metrics, $"mean score {score:F1} | return {meanReturn:F2}");
    }

    /// <summary>Mean episode score + return over fixed-seed greedy masked episodes (score is the gate metric).</summary>
    private (double Score, double Return) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, CrazyFruitsEnv.ActionCount);
        double totalScore = 0, totalReturn = 0;
        for (int e = 0; e < Options.EvalEpisodes; e++)
        {
            var (obs, _) = evalEnv.Reset((ulong)(5_000 + e));
            double epReturn = 0;
            while (true)
            {
                int action = agent.Act(obs, evalEnv.CurrentActionMask(), greedy: true);
                var step = evalEnv.Step(action);
                epReturn += step.Reward;
                obs = step.Observation;
                if (step.Done) break;
            }
            totalScore += evalEnv.Score;
            totalReturn += epReturn;
        }
        return (totalScore / Options.EvalEpisodes, totalReturn / Options.EvalEpisodes);
    }
}
