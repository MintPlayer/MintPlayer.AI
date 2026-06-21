using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Snake DQN campaign (`--game snake`, PLAN M22) as an <see cref="ITrainingCampaign"/> on
/// <see cref="CampaignRunner"/> (PLAN M25) — the score-maximizing paradigm: an *infinite-goal* game whose eval is
/// the mean episodic score (food), not a solve rate. Trains the masked Double+Dueling <see cref="DuelingQNet"/> on a
/// small 6×6 grid (fast, dense food; the size-invariant observation transfers to the demo grid) and evaluates mean
/// food on the deployed 12×12 grid. Resumable bitwise-identically via <see cref="DqnTrainingState"/>: each chunk
/// raises the absolute <see cref="DqnOptions.MaxSteps"/> and continues from the prior state. Persists the deployable
/// net under `snake`/`dqn` (the id the web's <c>SnakeModelService</c> loads) plus the full resume state under
/// `snake`/`dqn-state`.
/// </summary>
internal sealed class SnakeDqnCampaign(ulong seed, int trainGrid, int evalGrid, int chunkSteps, long targetSteps, int evalEpisodes, float learningRate, float epsilonStart)
    : ITrainingCampaign
{
    private const string NetId = "dqn";          // deployable DuelingQNet — the id the web loads
    private const string StateId = "dqn-state";  // full DqnTrainingState for lossless resume

    private readonly SnakeEnv _env = new(trainGrid);
    private readonly SnakeEnv _evalEnv = new(evalGrid);
    private readonly SeedSequence _seeds = new(seed);

    private DqnTrainingState? _state;
    private IValueNet? _warmNet; // the deployable net to warm-start from when there's no full resume state

    public string Environment => "snake";

    /// <summary>The proven M22 config (masked Double+Dueling DQN); MaxSteps is managed per chunk by the runner.</summary>
    private DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = [128, 128],
        Gamma = 0.99,
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

    public bool Resume(IModelStore store)
    {
        using (var s = store.TryOpenRead(Environment, StateId))
        {
            if (s is not null)
            {
                _state = DqnTrainingState.Load(s);
                Log($"resumed Snake DQN at {_state.StepsCompleted:N0} steps (last eval return {_state.LastEval:F2})");
                return true;
            }
        }
        // No full resume state, but the deployable net may be present (e.g. the shipped checkpoint, or one produced
        // by the old web one-shot trainer). Warm-start from it: a fresh optimizer/replay buffer continues the
        // trained net rather than throwing away its weights. Lets us further-train (and eval) the shipped model.
        using (var net = store.TryOpenRead(Environment, NetId))
        {
            if (net is not null)
            {
                _warmNet = DuelingQNetCheckpoint.Load(net);
                Log($"warm-starting from the deployable Snake net '{NetId}' (fresh optimizer + replay buffer)");
                return true;
            }
        }
        Log($"starting fresh Snake DQN training (train {trainGrid}×{trainGrid}, eval {evalGrid}×{evalGrid})");
        return false;
    }

    public long TrainChunk()
    {
        int from = _state?.StepsCompleted ?? 0;
        int to = from + chunkSteps;
        if (targetSteps > 0) to = (int)Math.Min(to, targetSteps);
        // EvalEvery == the chunk size so the trainer's own (6×6) eval fires at most once per chunk — the campaign's
        // authoritative eval is the 12×12 food in Evaluate(). MaxSteps is ABSOLUTE: resuming raises the ceiling.
        var options = BaseOptions with { MaxSteps = to, EvalEvery = Math.Max(1, chunkSteps) };
        // First chunk: resume the full state if present, else warm-start from the deployable net (if any).
        var result = DqnTrainer.Train(_env, options, _seeds, resume: _state, warmStart: _state is null ? _warmNet : null);
        _state = result.State;
        return _state.StepsCompleted;
    }

    /// <summary>Score-maximizing: no hard goal. Stops on the runner's time budget, or an optional absolute step cap.</summary>
    public bool IsComplete => targetSteps > 0 && (_state?.StepsCompleted ?? 0) >= targetSteps;

    public CampaignEval Evaluate()
    {
        int steps = _state?.StepsCompleted ?? 0;
        // Eval the trained net if we've trained this run, else the warm-start net (so eval-only works on the
        // shipped checkpoint too); only "no model at all" yields the placeholder.
        var net = _state?.Online ?? _warmNet;
        if (net is null)
            return new CampaignEval([new("steps", 0, "0")], "no model yet (train first)");

        // Mean food + return over fixed-seed greedy episodes on the DEPLOYED grid — food is the gate metric
        // (M22: train 6×6 → ~22 food on the 12×12 demo grid), return is the optimization target.
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
        double food = totalFood / evalEpisodes;
        double meanReturn = totalReturn / evalEpisodes;

        float loss = _state?.LastLoss ?? 0f;
        var metrics = new List<CampaignMetric>
        {
            new("steps", steps, "0"),
            new($"food{evalGrid}", food, "F2"),
            new("return", meanReturn, "F2"),
            new("loss", loss, "F4"),
        };
        return new CampaignEval(metrics,
            $"steps {steps:N0} | food@{evalGrid} {food:F2} | return {meanReturn:F2} | loss {loss:F4}");
    }

    public void Checkpoint(IModelStore store)
    {
        if (_state is null) return;
        store.Save(Environment, NetId, s => DuelingQNetCheckpoint.Save((DuelingQNet)_state.Online, s));
        store.Save(Environment, StateId, s => _state.Save(s));
    }

    public void Dispose() { }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
