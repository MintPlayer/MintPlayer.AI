using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// FruitCake DQN campaign (`--game fruitcake`, PRD FRUITCAKE_AI §4.4/§7-A2) as an <see cref="ITrainingCampaign"/>
/// on <see cref="CampaignRunner"/> — the score-maximizing paradigm (an open-ended game whose eval is the mean
/// episodic score, not a solve rate). Trains a Double+Dueling <see cref="DuelingQNet"/> on the headless
/// <see cref="FruitCakeEnv"/> (one step = one drop, simulated to rest in pure compute; rotation off). Resumable
/// bitwise-identically via <see cref="DqnTrainingState"/>; each chunk raises the absolute
/// <see cref="DqnOptions.MaxSteps"/> and continues. Persists the deployable net under `fruitcake`/`dqn` (the id
/// the web's <c>FruitCakeModelService</c> will load, A3) plus the full resume state under `fruitcake`/`dqn-state`.
/// </summary>
internal sealed class FruitCakeDqnCampaign(ulong seed, int chunkSteps, long targetSteps, int evalEpisodes, float learningRate, float epsilonStart, int[] hidden, double gamma)
    : ITrainingCampaign
{
    private const string NetId = "dqn";         // deployable DuelingQNet — the id the web loads
    private const string StateId = "dqn-state"; // full DqnTrainingState for lossless resume

    private readonly FruitCakeEnv _env = new();
    private readonly FruitCakeEnv _evalEnv = new();
    private readonly SeedSequence _seeds = new(seed);

    private DqnTrainingState? _state;
    private IValueNet? _warmNet; // deployable net to warm-start from when there's no full resume state
    // Save-best: DQN eval is noisy, so the deployable net is only overwritten when the eval mean-score IMPROVES on
    // the best seen (seeded from the starting net, so a noisy dip never regresses a good shipped model).
    private double _bestScore = double.NegativeInfinity;
    private double _lastEvalScore = double.NegativeInfinity;

    public string Environment => "fruitcake";

    /// <summary>Double+Dueling DQN; MaxSteps is managed per chunk by the runner (drops, not physics sub-steps).</summary>
    private DqnOptions BaseOptions => new()
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
        EvalEpisodes = 10,
    };

    public bool Resume(IModelStore store)
    {
        bool resumed = false;
        using (var s = store.TryOpenRead(Environment, StateId))
        {
            if (s is not null)
            {
                _state = DqnTrainingState.Load(s);
                Log($"resumed FruitCake DQN at {_state.StepsCompleted:N0} drops (last eval return {_state.LastEval:F2})");
                resumed = true;
            }
        }
        // No full resume state, but the deployable net may exist (e.g. the shipped checkpoint). Warm-start from it:
        // a fresh optimizer/replay buffer continues the trained net rather than discarding its weights.
        if (!resumed)
        {
            using var net = store.TryOpenRead(Environment, NetId);
            if (net is not null)
            {
                _warmNet = DuelingQNetCheckpoint.Load(net);
                Log($"warm-starting from the deployable FruitCake net '{NetId}' (fresh optimizer + replay buffer)");
                resumed = true;
            }
        }

        // Seed save-best from the starting net's score so training only ever ships a net that BEATS it.
        var startNet = _state?.Online ?? _warmNet;
        if (startNet is not null)
        {
            _bestScore = EvalNet(startNet).Score;
            Log($"baseline mean score: {_bestScore:F2} (will only re-save the deployable net when an eval beats this)");
        }
        else
        {
            Log("starting fresh FruitCake DQN training");
        }
        return resumed;
    }

    public long TrainChunk()
    {
        int from = _state?.StepsCompleted ?? 0;
        int to = from + chunkSteps;
        if (targetSteps > 0) to = (int)Math.Min(to, targetSteps);
        // EvalEvery == the chunk size so the trainer's own eval fires at most once per chunk — the campaign's
        // authoritative eval is the mean score in Evaluate(). MaxSteps is ABSOLUTE: resuming raises the ceiling.
        var options = BaseOptions with { MaxSteps = to, EvalEvery = Math.Max(1, chunkSteps) };
        var result = DqnTrainer.Train(_env, options, _seeds, resume: _state, warmStart: _state is null ? _warmNet : null);
        _state = result.State;
        return _state.StepsCompleted;
    }

    /// <summary>Score-maximizing: no hard goal. Stops on the runner's time budget, or an optional absolute drop cap.</summary>
    public bool IsComplete => targetSteps > 0 && (_state?.StepsCompleted ?? 0) >= targetSteps;

    public CampaignEval Evaluate()
    {
        int steps = _state?.StepsCompleted ?? 0;
        var net = _state?.Online ?? _warmNet;
        if (net is null)
            return new CampaignEval([new("drops", 0, "0")], "no model yet (train first)");

        var (score, maxTier, meanReturn) = EvalNet(net);
        _lastEvalScore = score; // Checkpoint() ships the net only when this beats the best seen

        float loss = _state?.LastLoss ?? 0f;
        var metrics = new List<CampaignMetric>
        {
            new("drops", steps, "0"),
            new("score", score, "F2"),
            new("maxTier", maxTier, "F2"),
            new("return", meanReturn, "F2"),
            new("loss", loss, "F4"),
        };
        return new CampaignEval(metrics,
            $"drops {steps:N0} | score {score:F2} | maxTier {maxTier:F2} | return {meanReturn:F2} | loss {loss:F4}");
    }

    public void Checkpoint(IModelStore store)
    {
        if (_state is null) return;
        // The resume state always tracks the latest net (so a continuation picks up where it left off).
        store.Save(Environment, StateId, s => _state.Save(s));
        // The DEPLOYABLE net is save-best: only overwrite it when this eval beat the best seen (DQN eval is noisy).
        if (_lastEvalScore > _bestScore)
        {
            _bestScore = _lastEvalScore;
            store.Save(Environment, NetId, s => DuelingQNetCheckpoint.Save((DuelingQNet)_state.Online, s));
            Log($"new best mean score {_bestScore:F2} → saved deployable net '{NetId}'");
        }
    }

    public void Dispose() { }

    /// <summary>Mean episodic score, mean max-tier reached, and mean return over fixed-seed greedy episodes.</summary>
    private (double Score, double MaxTier, double Return) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);
        double totalScore = 0, totalMaxTier = 0, totalReturn = 0;
        for (int e = 0; e < evalEpisodes; e++)
        {
            var (obs, _) = _evalEnv.Reset((ulong)(5_000 + e));
            double epReturn = 0;
            int epMaxTier = 0;
            while (true)
            {
                int action = agent.Act(obs, greedy: true);
                var step = _evalEnv.Step(action);
                epReturn += step.Reward;
                obs = step.Observation;
                foreach (var b in _evalEnv.World.Bodies)
                    if (b.Tier > epMaxTier) epMaxTier = b.Tier;
                if (step.Done) break;
            }
            totalScore += _evalEnv.Score;
            totalMaxTier += epMaxTier;
            totalReturn += epReturn;
        }
        return (totalScore / evalEpisodes, totalMaxTier / evalEpisodes, totalReturn / evalEpisodes);
    }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
