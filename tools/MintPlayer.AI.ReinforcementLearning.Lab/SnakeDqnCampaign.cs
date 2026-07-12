using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
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
internal sealed class SnakeDqnCampaign(ulong seed, int trainGrid, int evalGrid, int chunkSteps, long targetSteps, int evalEpisodes, float learningRate, float epsilonStart, int[] hidden, double gamma, float stepPenalty, bool safeMask, bool grow = false, int growEvery = 5000)
    : ITrainingCampaign, INetworkTelemetrySource
{
    // Progressive-growth demo (shared with FruitCake via DqnGrowth): grow the net wider+deeper on the growEvery cadence.
    private readonly Xoshiro256StarStar _growRng = new(seed ^ 0x9E3779B97F4A7C15UL);
    private const string NetId = "dqn";          // deployable DuelingQNet — the id the web loads
    private const string StateId = "dqn-state";  // full DqnTrainingState for lossless resume

    private readonly SnakeEnv _env = new(trainGrid, stepPenalty, safeMask);
    private readonly SnakeEnv _evalEnv = new(evalGrid, stepPenalty, safeMask);
    private readonly SeedSequence _seeds = new(seed);

    private DqnTrainingState? _state;
    private IValueNet? _warmNet; // the deployable net to warm-start from when there's no full resume state
    // Save-best: DQN eval is noisy, so the deployable net is only overwritten when the 12×12 eval IMPROVES on the
    // best seen (seeded from the starting net, so a noisy dip never regresses a good shipped model). The resume
    // state always tracks the latest net.
    private double _bestFood = double.NegativeInfinity;
    private double _lastEvalFood = double.NegativeInfinity;

    public string Environment => "snake";

    /// <summary>The proven M22 config (masked Double+Dueling DQN); MaxSteps is managed per chunk by the runner.</summary>
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
        EvalEpisodes = 20,
    };

    public bool Resume(IModelStore store)
    {
        bool resumed = false;
        using (var s = store.TryOpenRead(Environment, StateId))
        {
            if (s is not null)
            {
                _state = DqnTrainingState.Load(s);
                Log($"resumed Snake DQN at {_state.StepsCompleted:N0} steps (last eval return {_state.LastEval:F2})");
                resumed = true;
            }
        }
        // No full resume state, but the deployable net may be present (e.g. the shipped checkpoint, or one produced
        // by the old web one-shot trainer). Warm-start from it: a fresh optimizer/replay buffer continues the
        // trained net rather than throwing away its weights. Lets us further-train (and eval) the shipped model.
        if (!resumed)
        {
            using var net = store.TryOpenRead(Environment, NetId);
            if (net is not null)
            {
                _warmNet = DuelingQNetCheckpoint.Load(net);
                Log($"warm-starting from the deployable Snake net '{NetId}' (fresh optimizer + replay buffer)");
                resumed = true;
            }
        }

        // Seed save-best from the starting net's score so training only ever ships a net that BEATS it.
        var startNet = _state?.Online ?? _warmNet;
        if (startNet is not null)
        {
            _bestFood = EvalNet(startNet).Food;
            Log($"baseline food@{evalGrid}: {_bestFood:F2} (will only re-save the deployable net when an eval beats this)");
        }
        else
        {
            Log($"starting fresh Snake DQN training (train {trainGrid}×{trainGrid}, eval {evalGrid}×{evalGrid})");
        }
        return resumed;
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
        _state = DqnGrowth.Maybe(_state, grow, growEvery, learningRate, _growRng, Log);
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

        var (food, meanReturn) = EvalNet(net);
        _lastEvalFood = food; // Checkpoint() ships the net only when this beats the best seen

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
        // The resume state always tracks the latest net (so a continuation picks up where it left off).
        store.Save(Environment, StateId, s => _state.Save(s));
        // The DEPLOYABLE net is save-best: only overwrite it when this eval beat the best seen — DQN eval is
        // noisy and the last net is often not the best (here the 270k net scored 23.3 but 300k fell to 19.5).
        if (_lastEvalFood > _bestFood)
        {
            _bestFood = _lastEvalFood;
            store.Save(Environment, NetId, s => DuelingQNetCheckpoint.Save((DuelingQNet)_state.Online, s));
            Log($"new best food@{evalGrid} {_bestFood:F2} → saved deployable net '{NetId}'");
        }
    }

    public void Dispose() { }

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

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "dueling-q";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => (_state?.Online ?? _warmNet) is { } net ? [.. net.Parameters()] : null;
    NetworkMetrics INetworkTelemetrySource.Sample()
        => new(_state?.StepsCompleted ?? 0, targetSteps, _state?.LastLoss ?? double.NaN, _lastEvalFood, double.NaN);

    // Environment-aware neuron labels + live values: inputs are named observation features (177-dim egocentric
    // vision + scalars), outputs are the 4 move directions with their Q-values, hidden neurons show activations.
    IReadOnlyList<string>? INetworkTelemetrySource.InputLabels => SnakeEnv.ObservationLabels;
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => SnakeEnv.ActionLabels;
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo()
    {
        var obs = _state?.CurrentObs;
        if ((_state?.Online ?? _warmNet) is not { } net || obs is null || obs.Length != SnakeEnv.ObservationSize) return null;
        try
        {
            var input = (float[])obs.Clone();
            return (input, net.Forward(new Tensor(input, 1, input.Length)).Data.AsSpan().ToArray());
        }
        catch { return null; }
    }
    float[][]? INetworkTelemetrySource.SampleActivations()
    {
        var obs = _state?.CurrentObs;
        if ((_state?.Online ?? _warmNet) is not DuelingQNet dqn || obs is null || obs.Length != SnakeEnv.ObservationSize) return null;
        try { return dqn.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length)); }
        catch { return null; }
    }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
