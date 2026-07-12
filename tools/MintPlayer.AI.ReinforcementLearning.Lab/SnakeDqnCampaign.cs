using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Snake DQN campaign (`--game snake`, PLAN M22) on the shared <see cref="DqnScoreCampaign"/> spine — the
/// score-maximizing paradigm: an *infinite-goal* game whose eval is the mean episodic score (food), not a solve
/// rate. Trains the masked Double+Dueling <see cref="DuelingQNet"/> on a small 6×6 grid (fast, dense food; the
/// size-invariant observation transfers to the demo grid) and evaluates mean food on the deployed 12×12 grid. All
/// resume/train/keep-best/checkpoint/growth/telemetry plumbing lives in the base; this type supplies only the env,
/// the M22 hyperparameters, and the 12×12 food eval.
/// </summary>
internal sealed class SnakeDqnCampaign(ulong seed, int trainGrid, int evalGrid, int chunkSteps, long targetSteps, int evalEpisodes, float learningRate, float epsilonStart, int[] hidden, double gamma, float stepPenalty, bool safeMask, bool grow = false, int growEvery = 5000)
    : DqnScoreCampaign(seed, chunkSteps, targetSteps, learningRate, grow, growEvery)
{
    private readonly SnakeEnv _env = new(trainGrid, stepPenalty, safeMask);
    private readonly SnakeEnv _evalEnv = new(evalGrid, stepPenalty, safeMask);

    public override string Environment => "snake";
    protected override IEnvironment<float[], int> TrainEnv => _env;
    protected override string StepNoun => "steps";
    protected override string GateLabel => $"food@{evalGrid}";
    protected override string DisplayName => "Snake DQN";
    protected override string FreshStartDetail => $" (train {trainGrid}×{trainGrid}, eval {evalGrid}×{evalGrid})";
    protected override int ObservationSize => SnakeEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => SnakeEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => SnakeEnv.ActionLabels;

    /// <summary>The proven M22 config (masked Double+Dueling DQN); MaxSteps is managed per chunk by the runner.</summary>
    protected override DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = hidden,
        Gamma = gamma,
        LearningRate = learningRate,
        BufferCapacity = 100_000,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        // Fresh runs explore from ε=1.0; a warm-start continuation passes a low ε (e.g. 0.2) so it refines the
        // already-competent net instead of randomizing it away.
        Epsilon = new LinearSchedule(epsilonStart, 0.05, 30_000),
        EvalEpisodes = 20,
    };

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (food, meanReturn) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new($"food{evalGrid}", food, "F2"),
            new("return", meanReturn, "F2"),
        };
        return (food, metrics, $"food@{evalGrid} {food:F2} | return {meanReturn:F2}");
    }

    /// <summary>Mean food + return over fixed-seed greedy episodes on the DEPLOYED grid (food is the gate metric).</summary>
    private (double Food, double Return) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, SnakeEnv.ActionCount);
        double totalFood = 0, totalReturn = 0;
        for (int e = 0; e < evalEpisodes; e++)
        {
            var (obs, _) = _evalEnv.Reset((ulong)(5_000 + e));
            double epReturn = 0;
            while (true)
            {
                int action = agent.Act(obs, _evalEnv.CurrentActionMask(), greedy: true);
                var step = _evalEnv.Step(action);
                epReturn += step.Reward;
                obs = step.Observation;
                if (step.Done) break;
            }
            totalFood += _evalEnv.FoodEaten;
            totalReturn += epReturn;
        }
        return (totalFood / evalEpisodes, totalReturn / evalEpisodes);
    }
}
