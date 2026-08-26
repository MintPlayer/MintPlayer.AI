using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Tetris DQN campaign (`--game tetris`, PLAN M54) on the shared <see cref="DqnScoreCampaign"/> spine —
/// score-maximizing where the score currency is LINES: eval is the mean lines cleared over fixed-seed
/// greedy 500-piece episodes (uniform pieces, no garbage — the benchmark-honest protocol A; the
/// discriminative garbage-survival protocol B runs in the Lab's `--baselines`). Trains the masked
/// Double+Dueling <see cref="DuelingQNet"/> over afterstate macro-actions with γ=0.995 (TETRIS_PRD.md
/// §3.7 — long-horizon survival; the γ=0 dense-target recipe structurally does not transfer here).
/// ε-greedy by default; <see cref="TetrisDqnOptions.Noisy"/> is the pre-registered escalation lever.
/// </summary>
public sealed partial class TetrisDqnCampaign : DqnScoreCampaign
{
    [Inject] private readonly TetrisEnv evalEnv;

    private TetrisDqnOptions Typed => Options as TetrisDqnOptions ?? new TetrisDqnOptions();

    public override string Environment => "tetris";
    protected override string StepNoun => "placements";
    protected override string GateLabel => "mean lines";
    protected override string DisplayName => "Tetris DQN";
    protected override string FreshStartDetail => $" (10×20, afterstate placements, {evalEnv.PieceBudget}-piece episodes)";
    protected override int ObservationSize => TetrisEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => TetrisEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => TetrisEnv.ActionLabels;

    /// <summary>The Crazy Fruits recipe re-pointed at long-horizon survival (PRD §3.7 locks).</summary>
    protected override DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = Options.Hidden,
        Gamma = Options.Gamma,
        NStep = Typed.NStep,
        NoisyNets = Typed.Noisy,
        LearningRate = Options.LearningRate,
        BufferCapacity = 100_000,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        Epsilon = new LinearSchedule(Options.EpsilonStart, 0.05, 30_000),
        EvalEpisodes = Options.EvalEpisodes,
    };

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (lines, pieces, topOuts) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new("lines", lines, "F1"),
            new("pieces", pieces, "F1"),
            new("topouts", topOuts, "F0"),
        };
        return (lines, metrics, $"mean lines {lines:F1} | mean pieces {pieces:F1} | top-outs {topOuts}/{Options.EvalEpisodes}");
    }

    /// <summary>Mean lines + pieces survived over fixed-seed greedy masked episodes (lines is the gate metric).</summary>
    private (double Lines, double Pieces, int TopOuts) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, TetrisEnv.ActionCount);
        double totalLines = 0, totalPieces = 0;
        int topOuts = 0;
        for (int e = 0; e < Options.EvalEpisodes; e++)
        {
            var (obs, _) = evalEnv.Reset((ulong)(5_000 + e));
            while (true)
            {
                int action = agent.Act(obs, evalEnv.CurrentActionMask(), greedy: true);
                var step = evalEnv.Step(action);
                obs = step.Observation;
                if (step.Done)
                {
                    if (step.Terminated) topOuts++;
                    break;
                }
            }
            totalLines += evalEnv.Lines;
            totalPieces += evalEnv.PiecesPlaced;
        }
        return (totalLines / Options.EvalEpisodes, totalPieces / Options.EvalEpisodes, topOuts);
    }
}
