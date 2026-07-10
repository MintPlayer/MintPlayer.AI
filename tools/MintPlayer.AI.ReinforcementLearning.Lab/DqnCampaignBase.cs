using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Shared spine for the score-maximizing DQN campaigns (Snake, FruitCake). Both train a Double+Dueling
/// <see cref="DuelingQNet"/> resumably via <see cref="DqnTrainingState"/>, ship a keep-best deployable net under
/// <c>&lt;env&gt;/dqn</c> (the id the web loads) plus the full resume state under <c>&lt;env&gt;/&lt;StateId&gt;</c>,
/// and follow the identical resume → per-chunk-train → keep-best-checkpoint lifecycle. Subclasses supply only the
/// genuinely game-specific pieces: the environment, the DQN hyperparameters, the eval rollout + metrics, and
/// (optionally) how a loaded net is adapted before warm-starting.
/// <para>
/// Keep-best: DQN eval is noisy, so the deployable net is overwritten only when the gate metric IMPROVES on the
/// best seen (seeded in <see cref="Resume"/> from the starting net's score, so a noisy dip never regresses a good
/// shipped model). The resume state always tracks the latest net, so a continuation picks up where it left off.
/// </para>
/// </summary>
internal abstract class DqnCampaignBase(ulong seed, int chunkSteps, long targetSteps) : ITrainingCampaign
{
    protected const string NetId = "dqn"; // deployable DuelingQNet — the id the web loads

    protected SeedSequence Seeds { get; } = new(seed);
    protected DqnTrainingState? State { get; private set; }
    private IValueNet? _warmNet; // deployable net to warm-start from when there's no full resume state
    private double _bestGate = double.NegativeInfinity;
    private double _lastGate = double.NegativeInfinity;

    /// <summary>The model-store environment id (e.g. "snake", "fruitcake").</summary>
    public abstract string Environment { get; }

    /// <summary>Model-store id for the full resume state; a separate line (e.g. noisy) overrides it.</summary>
    protected virtual string StateId => "dqn-state";

    /// <summary>The env the trainer learns on (may carry reward shaping and/or action masking).</summary>
    protected abstract IEnvironment<float[], int> TrainEnv { get; }

    /// <summary>DQN hyperparameters; <see cref="DqnOptions.MaxSteps"/>/<see cref="DqnOptions.EvalEvery"/> are set per chunk by the base.</summary>
    protected abstract DqnOptions BaseOptions { get; }

    /// <summary>Label for the step counter in metrics/logs ("steps", "drops", …).</summary>
    protected abstract string StepNoun { get; }

    /// <summary>Human name of the keep-best gate metric for logs (e.g. "food@12", "mean score").</summary>
    protected abstract string GateNoun { get; }

    /// <summary>
    /// Evaluate a net on the deployed setting. Returns the keep-best GATE metric, the display metrics (the base
    /// prepends the step count and appends loss), and a summary fragment (the base wraps it with step count + loss).
    /// </summary>
    protected abstract (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net);

    /// <summary>Adapt a loaded deployable net before warm-starting. Default identity; FruitCake promotes plain→noisy and grows the input.</summary>
    protected virtual IValueNet AdaptWarmNet(DuelingQNet loaded) => loaded;

    public bool Resume(IModelStore store)
    {
        bool resumed = false;
        using (var s = store.TryOpenRead(Environment, StateId))
        {
            if (s is not null)
            {
                State = DqnTrainingState.Load(s);
                Log($"resumed {Environment} DQN at {State.StepsCompleted:N0} {StepNoun} (last eval return {State.LastEval:F2})");
                resumed = true;
            }
        }
        // No full resume state, but the deployable net may be present (e.g. the shipped checkpoint). Warm-start from
        // it: a fresh optimizer/replay buffer continues the trained net rather than discarding its weights.
        if (!resumed)
        {
            using var net = store.TryOpenRead(Environment, NetId);
            if (net is not null)
            {
                _warmNet = AdaptWarmNet(DuelingQNetCheckpoint.Load(net));
                Log($"warm-starting from the deployable {Environment} net '{NetId}' (fresh optimizer + replay buffer)");
                resumed = true;
            }
        }

        // Seed keep-best from the starting net's score so training only ever ships a net that BEATS it.
        var startNet = State?.Online ?? _warmNet;
        if (startNet is not null)
        {
            _bestGate = EvaluateNet(startNet).Gate;
            Log($"baseline {GateNoun}: {_bestGate:F2} (will only re-save the deployable net when an eval beats this)");
        }
        else
        {
            Log($"starting fresh {Environment} DQN training");
        }
        return resumed;
    }

    public long TrainChunk()
    {
        int from = State?.StepsCompleted ?? 0;
        int to = from + chunkSteps;
        if (targetSteps > 0) to = (int)Math.Min(to, targetSteps);
        // EvalEvery == the chunk size so the trainer's own eval fires at most once per chunk — the campaign's
        // authoritative eval is Evaluate(). MaxSteps is ABSOLUTE: resuming raises the ceiling.
        var options = BaseOptions with { MaxSteps = to, EvalEvery = Math.Max(1, chunkSteps) };
        // First chunk: resume the full state if present, else warm-start from the deployable net (if any).
        var result = DqnTrainer.Train(TrainEnv, options, Seeds, resume: State, warmStart: State is null ? _warmNet : null);
        State = result.State;
        return State.StepsCompleted;
    }

    /// <summary>Score-maximizing: no hard goal. Stops on the runner's time budget, or an optional absolute step cap.</summary>
    public bool IsComplete => targetSteps > 0 && (State?.StepsCompleted ?? 0) >= targetSteps;

    public CampaignEval Evaluate()
    {
        int steps = State?.StepsCompleted ?? 0;
        // Eval the trained net if we've trained this run, else the warm-start net (so eval-only works on the
        // shipped checkpoint too); only "no model at all" yields the placeholder.
        var net = State?.Online ?? _warmNet;
        if (net is null)
            return new CampaignEval([new(StepNoun, 0, "0")], "no model yet (train first)");

        var (gate, metrics, summary) = EvaluateNet(net);
        _lastGate = gate; // Checkpoint() ships the net only when this beats the best seen

        float loss = State?.LastLoss ?? 0f;
        var full = new List<CampaignMetric> { new(StepNoun, steps, "0") };
        full.AddRange(metrics);
        full.Add(new("loss", loss, "F4"));
        return new CampaignEval(full, $"{StepNoun} {steps:N0} | {summary} | loss {loss:F4}");
    }

    public void Checkpoint(IModelStore store)
    {
        if (State is null) return;
        // The resume state always tracks the latest net (so a continuation picks up where it left off).
        store.Save(Environment, StateId, s => State.Save(s));
        // The DEPLOYABLE net is keep-best: only overwrite it when this eval beat the best seen (DQN eval is noisy).
        if (_lastGate > _bestGate)
        {
            _bestGate = _lastGate;
            store.Save(Environment, NetId, s => DuelingQNetCheckpoint.Save((DuelingQNet)State.Online, s));
            Log($"new best {GateNoun} {_bestGate:F2} → saved deployable net '{NetId}'");
        }
    }

    public virtual void Dispose() { }

    protected static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
