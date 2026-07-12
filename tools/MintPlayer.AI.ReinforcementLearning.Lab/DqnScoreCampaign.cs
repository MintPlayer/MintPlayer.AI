using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// The shared spine of a <b>score-maximizing</b> DQN campaign (Snake, FruitCake): an open-ended game whose eval is
/// a mean episodic score, not a solve rate. It owns everything these campaigns do identically — resume-or-warm-start
/// from the model store, per-chunk training against a raised absolute <see cref="DqnOptions.MaxSteps"/>, the
/// keep-best gate on the deployable net (DQN eval is noisy, so the shipped net is only overwritten when an eval
/// beats the best seen), the full-state checkpoint, progressive net growth (<see cref="DqnGrowth"/>, PLAN M37), and
/// the live-telemetry seam (<see cref="INetworkTelemetrySource"/>, PLAN M36) a viewer samples while it trains.
/// <para>
/// A subclass supplies only what genuinely differs: its env + DQN hyperparameters (<see cref="BaseOptions"/>), how
/// to evaluate the net (<see cref="EvaluateNet"/>), the neuron labels, and — for a net whose warm-start needs
/// adaptation (plain→noisy, grow-input) — <see cref="AdaptWarmNet"/>. Resume/train/checkpoint stay byte-identical
/// to the hand-rolled campaigns, so a run is bitwise-reproducible across the refactor.
/// </para>
/// </summary>
internal abstract class DqnScoreCampaign(ulong seed, int chunkSteps, long targetSteps, float learningRate, bool grow, int growEvery)
    : ITrainingCampaign, INetworkTelemetrySource
{
    protected const string NetId = "dqn";  // deployable DuelingQNet — the id the web loads

    // Progressive-growth demo (DqnGrowth): grow the net wider+deeper on the growEvery cadence. Seeded independently
    // of the training streams so growth is deterministic and never perturbs the trainer's RNG.
    private readonly Xoshiro256StarStar _growRng = new(seed ^ 0x9E3779B97F4A7C15UL);
    protected readonly SeedSequence Seeds = new(seed);

    protected DqnTrainingState? State;
    private IValueNet? _warmNet; // deployable net to warm-start from when there's no full resume state
    // Save-best: the deployable net is only overwritten when the gate metric IMPROVES on the best seen (seeded from
    // the starting net, so a noisy dip never regresses a good shipped model). The resume state tracks the latest net.
    private double _bestGate = double.NegativeInfinity;
    private double _lastGate = double.NegativeInfinity;

    // ── Game-specific surface ──────────────────────────────────────────────────────────────────────────────────
    public abstract string Environment { get; }
    /// <summary>Model-store id for the full resume state (default "dqn-state"; a separate line, e.g. noisy, overrides).</summary>
    protected virtual string StateId => "dqn-state";
    protected abstract IEnvironment<float[], int> TrainEnv { get; }
    protected abstract DqnOptions BaseOptions { get; }
    /// <summary>The progress unit shown in logs/metrics ("steps" / "drops").</summary>
    protected abstract string StepNoun { get; }
    /// <summary>The gate metric's label for logs ("food@12" / "mean score").</summary>
    protected abstract string GateLabel { get; }
    /// <summary>Human display name for the campaign ("Snake DQN" / "FruitCake DQN").</summary>
    protected abstract string DisplayName { get; }
    /// <summary>Optional suffix on the fresh-start log (e.g. " (train 6×6, eval 12×12)"); empty by default.</summary>
    protected virtual string FreshStartDetail => "";
    /// <summary>The net's observation width, used to guard the telemetry forward pass.</summary>
    protected abstract int ObservationSize { get; }
    protected abstract IReadOnlyList<string>? InputLabels { get; }
    protected abstract IReadOnlyList<string>? OutputLabels { get; }
    /// <summary>Adapt a freshly-loaded deployable net before warm-starting; identity by default (FruitCake promotes
    /// plain→noisy and grows the input to fit an enriched observation).</summary>
    protected virtual IValueNet AdaptWarmNet(DuelingQNet loaded) => loaded;
    /// <summary>Run the greedy eval and return the keep-best <c>Gate</c> value plus the game-specific middle metrics
    /// and a summary body; the base frames them with the step count and loss.</summary>
    protected abstract (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net);

    /// <summary>The net being trained/served: the live online net, else the warm-start net (for eval-only runs).</summary>
    protected IValueNet? CurrentNet => State?.Online ?? _warmNet;

    /// <summary>The learning rate, exposed so a subclass's <see cref="BaseOptions"/> reuses the value it already
    /// hands the base (which owns it for optimizer rebuilds on growth) rather than capturing the ctor param twice.</summary>
    protected float LearningRate => learningRate;

    public bool Resume(IModelStore store)
    {
        bool resumed = false;
        using (var s = store.TryOpenRead(Environment, StateId))
        {
            if (s is not null)
            {
                State = DqnTrainingState.Load(s);
                Log($"resumed {DisplayName} at {State.StepsCompleted:N0} {StepNoun} (last eval return {State.LastEval:F2})");
                resumed = true;
            }
        }
        // No full resume state, but the deployable net may be present (the shipped checkpoint). Warm-start from it: a
        // fresh optimizer/replay buffer continues the trained net rather than discarding its weights.
        if (!resumed)
        {
            using var net = store.TryOpenRead(Environment, NetId);
            if (net is not null)
            {
                _warmNet = AdaptWarmNet(DuelingQNetCheckpoint.Load(net));
                Log($"warm-starting from the deployable {DisplayName} net '{NetId}' (fresh optimizer + replay buffer)");
                resumed = true;
            }
        }

        // Seed save-best from the starting net's score so training only ever ships a net that BEATS it.
        var startNet = CurrentNet;
        if (startNet is not null)
        {
            _bestGate = EvaluateNet(startNet).Gate;
            Log($"baseline {GateLabel}: {_bestGate:F2} (will only re-save the deployable net when an eval beats this)");
        }
        else
        {
            Log($"starting fresh {DisplayName} training{FreshStartDetail}");
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
        State = DqnGrowth.Maybe(State, grow, growEvery, learningRate, _growRng, Log);
        return State.StepsCompleted;
    }

    /// <summary>Score-maximizing: no hard goal. Stops on the runner's time budget, or an optional absolute step cap.</summary>
    public bool IsComplete => targetSteps > 0 && (State?.StepsCompleted ?? 0) >= targetSteps;

    public CampaignEval Evaluate()
    {
        int steps = State?.StepsCompleted ?? 0;
        // Eval the trained net if we've trained this run, else the warm-start net (so eval-only works on the shipped
        // checkpoint too); only "no model at all" yields the placeholder.
        var net = CurrentNet;
        if (net is null)
            return new CampaignEval([new(StepNoun, 0, "0")], "no model yet (train first)");

        var (gate, metrics, summary) = EvaluateNet(net);
        _lastGate = gate; // Checkpoint() ships the net only when this beats the best seen
        float loss = State?.LastLoss ?? 0f;

        var full = new List<CampaignMetric>(metrics.Count + 2) { new(StepNoun, steps, "0") };
        full.AddRange(metrics);
        full.Add(new("loss", loss, "F4"));
        return new CampaignEval(full, $"{StepNoun} {steps:N0} | {summary} | loss {loss:F4}");
    }

    public void Checkpoint(IModelStore store)
    {
        if (State is null) return;
        // The resume state always tracks the latest net (so a continuation picks up where it left off).
        store.Save(Environment, StateId, s => State.Save(s));
        // The DEPLOYABLE net is save-best: only overwrite it when this eval beat the best seen (DQN eval is noisy).
        if (_lastGate > _bestGate)
        {
            _bestGate = _lastGate;
            store.Save(Environment, NetId, s => DuelingQNetCheckpoint.Save((DuelingQNet)State.Online, s));
            Log($"new best {GateLabel} {_bestGate:F2} → saved deployable net '{NetId}'");
        }
    }

    public virtual void Dispose() { }

    // ── Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ──────
    string INetworkTelemetrySource.NetKind => "dueling-q";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => CurrentNet is { } net ? [.. net.Parameters()] : null;
    NetworkMetrics INetworkTelemetrySource.Sample()
        => new(State?.StepsCompleted ?? 0, targetSteps, State?.LastLoss ?? double.NaN, _lastGate, double.NaN);
    IReadOnlyList<string>? INetworkTelemetrySource.InputLabels => InputLabels;
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => OutputLabels;
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo()
    {
        var obs = State?.CurrentObs;
        if (CurrentNet is not { } net || obs is null || obs.Length != ObservationSize) return null;
        try
        {
            // Forward the most-recent observation for the net's current per-action Q-values. Read-only (no Backward),
            // so a concurrent training step is unaffected; a fresh array so the viewer never sees it mutated.
            var input = (float[])obs.Clone();
            return (input, net.Forward(new Tensor(input, 1, input.Length)).Data.AsSpan().ToArray());
        }
        catch { return null; }
    }
    float[][]? INetworkTelemetrySource.SampleActivations()
    {
        var obs = State?.CurrentObs;
        if (CurrentNet is not DuelingQNet dqn || obs is null || obs.Length != ObservationSize) return null;
        try { return dqn.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length)); }
        catch { return null; }
    }

    protected static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
