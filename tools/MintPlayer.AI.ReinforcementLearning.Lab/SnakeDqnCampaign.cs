using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Snake DQN campaign (`--game snake`, PLAN M22) — the score-maximizing paradigm: an *infinite-goal* game whose
/// eval is the mean episodic score (food), not a solve rate. Trains the masked Double+Dueling
/// <see cref="DuelingQNet"/> on a small 6×6 grid (fast, dense food; the size-invariant observation transfers to
/// the demo grid) and evaluates mean food on the deployed 12×12 grid. All the resume/warm-start/keep-best/
/// checkpoint plumbing lives in <see cref="DqnCampaignBase"/>; this type supplies only the envs, the DQN config,
/// and the food-based eval.
/// </summary>
internal sealed class SnakeDqnCampaign(ulong seed, int trainGrid, int evalGrid, int chunkSteps, long targetSteps, int evalEpisodes, float learningRate, float epsilonStart, int[] hidden, double gamma, float stepPenalty, bool safeMask)
    : DqnCampaignBase(seed, chunkSteps, targetSteps)
{
    private readonly SnakeEnv _env = new(trainGrid, stepPenalty, safeMask);
    private readonly SnakeEnv _evalEnv = new(evalGrid, stepPenalty, safeMask);

    public override string Environment => "snake";
    protected override IEnvironment<float[], int> TrainEnv => _env;
    protected override string StepNoun => "steps";
    protected override string GateNoun => $"food@{evalGrid}";

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
        var (food, meanReturn) = RunEval(net);
        var metrics = new CampaignMetric[]
        {
            new($"food{evalGrid}", food, "F2"),
            new("return", meanReturn, "F2"),
        };
        return (food, metrics, $"food@{evalGrid} {food:F2} | return {meanReturn:F2}");
    }

    /// <summary>Mean food + return over fixed-seed greedy episodes on the DEPLOYED grid (food is the gate metric).</summary>
    private (double Food, double Return) RunEval(IValueNet net)
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
