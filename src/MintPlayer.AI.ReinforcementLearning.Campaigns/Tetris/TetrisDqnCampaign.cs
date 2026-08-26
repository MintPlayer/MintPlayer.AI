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
    protected override string GateLabel => "mean score";
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
        var (score, lines, tetrises, pieces, topOuts) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new("score", score, "F0"),
            new("lines", lines, "F1"),
            new("tetrises", tetrises, "F2"),
            new("pieces", pieces, "F1"),
            new("topouts", topOuts, "F0"),
        };
        return (score, metrics,
            $"mean score {score:F0} | lines {lines:F1} | tetrises {tetrises:F2} | pieces {pieces:F1} | top-outs {topOuts}/{Options.EvalEpisodes}");
    }

    /// <summary>Mean NES score (the gate metric — the owner's objective), lines, tetrises and pieces
    /// survived over fixed-seed greedy masked episodes. Top-outs are the stack-and-camp watchdog.</summary>
    private (double Score, double Lines, double Tetrises, double Pieces, int TopOuts) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, TetrisEnv.ActionCount);
        double totalScore = 0, totalLines = 0, totalTetrises = 0, totalPieces = 0;
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
            totalScore += evalEnv.Score;
            totalLines += evalEnv.Lines;
            totalTetrises += evalEnv.Tetrises;
            totalPieces += evalEnv.PiecesPlaced;
        }
        int n = Options.EvalEpisodes;
        return (totalScore / n, totalLines / n, totalTetrises / n, totalPieces / n, topOuts);
    }
}
