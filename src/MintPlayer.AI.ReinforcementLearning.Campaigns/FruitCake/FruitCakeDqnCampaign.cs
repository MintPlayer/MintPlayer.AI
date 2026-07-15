using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// FruitCake DQN campaign (`--game fruitcake`, PRD FRUITCAKE_AI §4.4/§7-A2) on the shared
/// <see cref="DqnScoreCampaign"/> spine — the score-maximizing paradigm (an open-ended game whose eval is the mean
/// episodic score). Trains a Double+Dueling <see cref="DuelingQNet"/> on the headless <see cref="FruitCakeEnv"/>
/// (one step = one drop, simulated to rest in pure compute; rotation off). Both envs are constructor dependencies
/// (PLAN M46.2): the caller owns the training env's reward shaping (its <c>ShapingGamma</c> should match the
/// learner's γ for policy-invariance), and the eval env stays a plain game so keep-best/A/B always judge real
/// merge points, never the shaped signal. This type supplies the hyperparameters, the mean-score eval, and the
/// plain→noisy / grow-input warm-net adaptation; the base owns resume/train/keep-best/checkpoint/growth/telemetry.
/// </summary>
public sealed class FruitCakeDqnCampaign(FruitCakeEnv trainEnv, FruitCakeEnv evalEnv, FruitCakeDqnOptions options)
    : DqnScoreCampaign(trainEnv, options)
{
    public override string Environment => "fruitcake";
    // Noisy training is a SEPARATE line with its own resume state: it must never resume a PLAIN state as a plain net
    // (which would then train with ε=0 and NO exploration). The deployable NetId is shared and keep-best gated.
    protected override string StateId => options.Noisy ? "dqn-noisy-state" : "dqn-state";
    protected override string StepNoun => "drops";
    protected override string GateLabel => "mean score";
    protected override string DisplayName => "FruitCake DQN";
    protected override int ObservationSize => FruitCakeEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => FruitCakeEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => FruitCakeEnv.ActionLabels;

    /// <summary>Double+Dueling DQN; MaxSteps is managed per chunk by the runner (drops, not physics sub-steps).</summary>
    protected override DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        NoisyNets = options.Noisy, // learned exploration replaces ε-greedy; the trainer forces ε=0 and resamples noise

        Hidden = Options.Hidden,
        Gamma = Options.Gamma,
        NStep = options.NStep, // n-step returns: propagate the sparse high-tier reward backward faster (PRD §4.C C4)
        LearningRate = Options.LearningRate,
        BufferCapacity = 100_000,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        // Fresh runs explore from ε=1.0; a warm-start continuation passes a low ε (e.g. 0.2) so it refines the
        // already-competent net instead of randomizing it away.
        Epsilon = new LinearSchedule(Options.EpsilonStart, 0.05, 30_000),
        EvalEpisodes = 10,
    };

    protected override IValueNet AdaptWarmNet(DuelingQNet loaded)
    {
        // Promote-plain→noisy: the shipped net is plain, but a noisy run needs a noisy net. Copy its trained weights
        // into the means + add fresh σ — behaviorally identical with noise off, so the run continues the ~current
        // policy and merely adds learnable exploration.
        IValueNet warm = options.Noisy && !loaded.Noisy ? loaded.ToNoisy(Seeds.CreateRng(RngStreams.Init)) : loaded;
        // If the observation has gained features since this net was trained (e.g. the big-fruit-position inputs),
        // grow its input to fit — function-preserving (new inputs zero-init), so the baseline eval and warm-start
        // continue the trained net rather than retraining from scratch.
        if (warm.InputSize != FruitCakeEnv.ObservationSize)
        {
            Log($"growing the loaded net's input {warm.InputSize} → {FruitCakeEnv.ObservationSize} (observation gained features; weights preserved)");
            warm = warm.GrowInput(FruitCakeEnv.ObservationSize);
        }
        return warm;
    }

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (score, maxTier, meanReturn) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new("score", score, "F2"),
            new("maxTier", maxTier, "F2"),
            new("return", meanReturn, "F2"),
        };
        return (score, metrics, $"score {score:F2} | maxTier {maxTier:F2} | return {meanReturn:F2}");
    }

    /// <summary>Mean episodic score, mean max-tier reached, and mean return over fixed-seed greedy episodes.</summary>
    private (double Score, double MaxTier, double Return) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);
        double totalScore = 0, totalMaxTier = 0, totalReturn = 0;
        for (int e = 0; e < Options.EvalEpisodes; e++)
        {
            var (obs, _) = evalEnv.Reset((ulong)(5_000 + e));
            double epReturn = 0;
            int epMaxTier = 0;
            while (true)
            {
                int action = agent.Act(obs, greedy: true);
                var step = evalEnv.Step(action);
                epReturn += step.Reward;
                obs = step.Observation;
                foreach (var b in evalEnv.World.Bodies)
                    if (b.Tier > epMaxTier) epMaxTier = b.Tier;
                if (step.Done) break;
            }
            totalScore += evalEnv.Score;
            totalMaxTier += epMaxTier;
            totalReturn += epReturn;
        }
        return (totalScore / Options.EvalEpisodes, totalMaxTier / Options.EvalEpisodes, totalReturn / Options.EvalEpisodes);
    }
}
