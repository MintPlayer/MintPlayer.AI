using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Snake DQN campaign (`--game snake`, PLAN M22) on the shared <see cref="DqnScoreCampaign"/> spine — the
/// score-maximizing paradigm: an *infinite-goal* game whose eval is the mean episodic score (food), not a solve
/// rate. Trains the masked Double+Dueling <see cref="DuelingQNet"/> on a small injected grid (typically 6×6 —
/// fast, dense food; the size-invariant observation transfers) and evaluates mean food on the injected eval grid
/// (typically the deployed 12×12). Both envs are constructor dependencies (PLAN M46.2): the caller owns grid
/// size / step penalty / safe mask. All resume/train/keep-best/checkpoint/growth/telemetry plumbing lives in the
/// base; this type supplies only the M22 hyperparameters and the food eval. The constructor is source-generated:
/// the [Inject] eval env plus the base's forwarded dependencies (train env, options, logger).
/// </summary>
public sealed partial class SnakeDqnCampaign : DqnScoreCampaign
{
    [Inject] private readonly SnakeEnv evalEnv;

    /// <summary>The base's train env, downcast for its grid size — this campaign is always handed a
    /// <see cref="SnakeEnv"/> (the generated ctor's param is the base's <see cref="IEnvironment{TObs,TAct}"/>,
    /// which can't be narrowed per subclass).</summary>
    private SnakeEnv TrainSnakeEnv => (SnakeEnv)TrainEnv;

    public override string Environment => "snake";
    protected override string StepNoun => "steps";
    protected override string GateLabel => $"food@{evalEnv.Size}";
    protected override string DisplayName => "Snake DQN";
    protected override string FreshStartDetail => $" (train {TrainSnakeEnv.Size}×{TrainSnakeEnv.Size}, eval {evalEnv.Size}×{evalEnv.Size})";
    protected override int ObservationSize => SnakeEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => SnakeEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => SnakeEnv.ActionLabels;

    /// <summary>The proven M22 config (masked Double+Dueling DQN); MaxSteps is managed per chunk by the runner.</summary>
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
        // Fresh runs explore from ε=1.0; a warm-start continuation passes a low ε (e.g. 0.2) so it refines the
        // already-competent net instead of randomizing it away.
        Epsilon = new LinearSchedule(Options.EpsilonStart, 0.05, 30_000),
        EvalEpisodes = 20,
    };

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (food, meanReturn) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new($"food{evalEnv.Size}", food, "F2"),
            new("return", meanReturn, "F2"),
        };
        return (food, metrics, $"food@{evalEnv.Size} {food:F2} | return {meanReturn:F2}");
    }

    /// <summary>Mean food + return over fixed-seed greedy episodes on the DEPLOYED grid (food is the gate metric).</summary>
    private (double Food, double Return) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, SnakeEnv.ActionCount);
        double totalFood = 0, totalReturn = 0;
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
            totalFood += evalEnv.FoodEaten;
            totalReturn += epReturn;
        }
        return (totalFood / Options.EvalEpisodes, totalReturn / Options.EvalEpisodes);
    }
}
