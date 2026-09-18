using System.Text;
using Microsoft.Extensions.Logging;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Block Dude imitation campaign: a size curriculum over generated boards, each labelled exactly by
/// <see cref="BlockDudeOracle"/>, trained supervised into <see cref="BlockDudePolicyNet"/> (soft cross-entropy
/// over ALL optimal actions + Huber on distance-to-goal).
/// </summary>
/// <remarks>
/// <para><b>Reproducible from a blank slate.</b> Every input to round <i>k</i> is either derived from the seed or
/// persisted in <see cref="BlockDudeTrainingState"/>: the stage, the counters, the shuffle/growth RNG states, and
/// the generator's attempt counter. Board generation derives its RNG from the attempt index rather than carrying
/// a stream, so rejections cannot shift it and worker count cannot change it.</para>
///
/// <para><b>The curriculum gate is evaluated inside <see cref="TrainChunk"/> on a SAMPLE cadence</b>, never in
/// <see cref="Evaluate"/>: the runner fires evaluation on the wall clock, so a gate there would make the
/// trajectory depend on machine speed.</para>
///
/// <para>Rejection reasons are counted, not just accepted boards. A quietly-falling accept rate biases the label
/// set toward shallow puzzles, and at the top rungs the generator legitimately rejects most candidates.</para>
/// </remarks>
public sealed class BlockDudeImitationCampaign : ITrainingCampaign, INetworkTelemetrySource
{
    // M64.7: was a const. `TrainChunk` RETURNS EARLY DOING NOTHING while fewer than this many samples
    // have been collected, so a hard-coded value made a small-batch test silently train nothing while
    // still looking like it passed. Shipped value 256 stays the default, so training is unchanged.
    private int BatchSize => _options.BatchSize;

    private readonly BlockDudeImitationOptions _options;
    private readonly ILogger? _logger;
    private readonly BlockDudeIds.NetIds _ids;
    private readonly ulong _fingerprint;

    private BlockDudePolicyNet _net = null!;
    private Adam _adam = null!;
    private BlockDudeTrainingState _state = null!;
    private List<BlockDudeBoard> _gateBoards = [];
    private int _gateBoardsStage = -1;

    // Held from Resume so a new best can be written the INSTANT the gate improves. Gating runs on a sample
    // cadence inside training while Checkpoint runs on the runner's wall clock — roughly 1.3M samples apart at
    // the observed throughput — so deferring the save to Checkpoint would persist a net that is not the one the
    // gate measured, which defeats the point of keeping a best at all.
    private IModelStore? _store;

    private double _liveLoss = double.NaN, _liveAcc = double.NaN;
    private double _windowCe, _windowHuber, _windowAcc;
    private int _windowBatches;
    private readonly Dictionary<BlockDudeGenerationOutcome, int> _outcomes = [];

    public BlockDudeImitationCampaign(BlockDudeImitationOptions options, ILogger? logger = null)
    {
        _options = options;
        _logger = logger;
        _ids = BlockDudeIds.ForPhase(options.Phase);
        _fingerprint = BlockDudeTrainingState.Fingerprint(
            options.Seed, BlockDudeBoard.ObservationSize, BlockDudeBoard.ActionCount, options.LearningRate, options.Phase);
    }

    public string Environment => BlockDudeIds.Environment;

    public bool IsComplete => _options.TargetSamples > 0 && _state.TotalSamples >= _options.TargetSamples;

    public bool Resume(IModelStore store)
    {
        _store = store;

        if (_options.Fresh)
        {
            // Blank slate. Deleting all four ids together matters: a net without its progress sidecar resumes
            // as an unlabelled warm start at stage 0, which is a different run than the operator asked for — and
            // a surviving best net would let a blank-slate run ship weights it never trained.
            foreach (string id in new[] { _ids.Policy, _ids.PolicyAdam, _ids.State, _ids.PolicyBest })
                if (store.Delete(BlockDudeIds.Environment, id)) Log($"--fresh: deleted {BlockDudeIds.Environment}.{id}");
        }

        bool resumed = false;
        using (var stream = store.TryOpenRead(BlockDudeIds.Environment, _ids.Policy))
        {
            if (stream is not null)
            {
                try
                {
                    _net = BlockDudePolicyNet.Load(stream);
                    resumed = true;
                    Log($"resumed the policy net (trunk [{string.Join(",", _net.Trunk)}])");
                }
                catch (InvalidDataException ex)
                {
                    // The exact-length guard in PolicyValueNet. Previously this half-loaded in silence.
                    Log($"STALE net checkpoint ignored: {ex.Message}");
                }
            }
        }

        if (!resumed)
        {
            var initRng = new Xoshiro256StarStar(_options.Seed ^ 0xDEADBEEF);
            // Rung 0 of the Block Dude ladder IS the default trunk, so a growing run and a plain one start from
            // the SAME architecture and diverge only when the gate saturates. (The shared DqnGrowth ladder starts
            // at [16] and tops out below this game's default — growing on it would shrink the net.)
            _net = new BlockDudePolicyNet(initRng, BlockDudeGrowth.Ladder.TrunkFor(0));
            Log(_options.Grow
                ? $"initialized a fresh GROWING policy net (rung 0, trunk [{string.Join(",", BlockDudeGrowth.Ladder.TrunkFor(0))}]) " +
                  $"— grows one rung after {_options.GrowPatience} gates without a +{_options.GrowMinImprovement:P0} best"
                : "initialized a fresh policy net");
        }

        _adam = AdamState.LoadOrInit(store, BlockDudeIds.Environment, _ids.PolicyAdam, _net.Parameters(), _options.LearningRate, Log);

        _state = BlockDudeTrainingState.TryLoad(
                     store, BlockDudeIds.Environment, _ids.State, _fingerprint,
                     BlockDudeBoard.ObservationSize, BlockDudeBoard.ActionCount, Log)
                 ?? NewState();

        if (_state.TotalSamples > 0)
            Log($"resumed curriculum: stage {_state.Stage}, {_state.TotalSamples:N0} samples " +
                $"({_state.AcceptedBoards:N0} boards accepted of {_state.BoardAttempts:N0} drawn)");
        else if (resumed)
            Log("net resumed WITHOUT a progress sidecar — treating it as a warm start and beginning the curriculum at stage 0");

        return resumed;
    }

    private BlockDudeTrainingState NewState()
    {
        var rates = new double[BlockDudeCurriculum.Stages.Length];
        Array.Fill(rates, -1);
        return new BlockDudeTrainingState
        {
            ObservationSize = BlockDudeBoard.ObservationSize,
            ActionCount = BlockDudeBoard.ActionCount,
            RunFingerprint = _fingerprint,
            Stage = _options.PinStage ?? 0,
            GateRates = rates,
            ShuffleRng = new Xoshiro256StarStar(_options.Seed ^ 0x5A3F1C09UL),
            GrowRng = new Xoshiro256StarStar(_options.Seed ^ 0x6C0FFEEUL),
        };
    }

    public long TrainChunk()
    {
        int stage = CurrentStage;
        var rung = BlockDudeCurriculum.Stages[stage];
        var samples = new List<Sample>();

        // Keep drawing until there is at least one full batch. A small board yields far fewer labelled states
        // than SamplesPerBoard asks for — at the first rung the median is under a hundred — so a fixed
        // boards-per-round quietly produced rounds that trained NOTHING and returned as if they had worked.
        int maxBoards = Math.Max(_options.BoardsPerRound * 8, _options.BoardsPerRound + 8);
        for (int i = 0; i < _options.BoardsPerRound || samples.Count < BatchSize; i++)
        {
            if (i >= maxBoards) break;
            var board = DrawBoard(rung.Spec);
            if (board is null) break;           // generator exhausted its attempt budget
            CollectSamples(board, samples);
        }
        if (samples.Count < BatchSize) return _state.TotalSamples;

        Shuffle(samples);
        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = TrainStep(samples, offset, BatchSize);
            _liveLoss = ce + huber;
            _liveAcc = acc;
            _windowCe += ce;
            _windowHuber += huber;
            _windowAcc += acc;
            _windowBatches++;

            _state.TotalSamples += BatchSize;
            _state.StageSamples += BatchSize;
        }

        // Growth is driven by the gate, inside MaybeGateAndAdvance — there is nothing to do on a sample cadence.
        MaybeGateAndAdvance();
        return _state.TotalSamples;
    }

    private int CurrentStage => Math.Clamp(_options.PinStage ?? _state.Stage, 0, _options.MaxStage);

    /// <summary>Draws one accepted board, deriving each candidate's RNG from the attempt counter so the stream is
    /// a pure function of how many candidates have been drawn — never of how many were accepted.</summary>
    private BlockDudeBoard? DrawBoard(BlockDudeStageSpec spec)
    {
        for (int attempt = 0; attempt < _options.MaxGenerationAttempts; attempt++)
        {
            var rng = new Xoshiro256StarStar(_options.Seed ^ ((ulong)_state.BoardAttempts * 0x9E3779B97F4A7C15UL));
            _state.BoardAttempts++;

            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            _outcomes[outcome] = _outcomes.TryGetValue(outcome, out int n) ? n + 1 : 1;

            if (outcome == BlockDudeGenerationOutcome.Truncated) _state.TruncatedBoards++;
            if (board is not null)
            {
                _state.AcceptedBoards++;
                return board;
            }
            _state.RejectedBoards++;
        }
        return null;
    }

    private void CollectSamples(BlockDudeBoard board, List<Sample> into)
    {
        var oracle = new BlockDudeOracle(board, BlockDudeCurriculum.Stages[CurrentStage].Spec.OracleMaxStates);
        if (oracle.Truncated) return;

        var labelled = new List<(BlockDudeBoard Board, int Distance, int Mask)>();
        foreach (var (state, distance, mask) in oracle.LabelledStates())
            if (distance > 0 && mask != 0) labelled.Add((state, distance, mask));
        if (labelled.Count == 0) return;

        int wanted = Math.Min(_options.SamplesPerBoard, labelled.Count);
        for (int i = 0; i < wanted; i++)
        {
            var pick = labelled[_state.ShuffleRng.NextInt(labelled.Count)];
            var observation = new float[BlockDudeBoard.ObservationSize];
            pick.Board.WriteObservation(observation);

            // A large FINITE penalty, not -inf. An illegal action's log-probability would be -inf, and its
            // supervision weight is 0, so the cross-entropy term becomes 0 * -inf = NaN — which silently poisons
            // the reported loss for the rest of the run. exp(-1e9) is 0 in float, so the masking is just as
            // absolute, and the logged CE stays a real number that divergence can actually be spotted in.
            const float illegal = -1e9f;
            var maskOffsets = new float[BlockDudeBoard.ActionCount];
            for (int a = 0; a < BlockDudeBoard.ActionCount; a++)
                maskOffsets[a] = pick.Board.IsLegal((BlockDudeAction)a) ? 0f : illegal;

            into.Add(new Sample(observation, maskOffsets, (uint)pick.Mask, pick.Distance));
        }
    }

    private void Shuffle(List<Sample> samples)
    {
        for (int i = samples.Count - 1; i > 0; i--)
        {
            int j = _state.ShuffleRng.NextInt(i + 1);
            (samples[i], samples[j]) = (samples[j], samples[i]);
        }
    }

    /// <summary>Refreshes the stage gate on a SAMPLE cadence and applies the curriculum's pure advance rule.</summary>
    private void MaybeGateAndAdvance()
    {
        if (_options.PinStage is not null) return;
        if (_state.TotalSamples - _state.SamplesAtLastGate < _options.GateEverySamples) return;

        _state.SamplesAtLastGate = _state.TotalSamples;
        int stage = CurrentStage;
        double rate = GateSolveRate(stage);
        _state.GateRates[stage] = rate;

        // Flag a new best for the next Checkpoint. Ordered by (stage, gate): reaching a harder rung always wins,
        // and within a rung the higher solve rate wins. Measured 2026-09-11, the gate swings ~10 points between
        // consecutive evals on a 64-board hold-out (61% -> 50% -> 52% with loss falling throughout), so without
        // this the shipped net is close to a random draw from the last few evals.
        if (stage > _state.BestStage || (stage == _state.BestStage && rate > _state.BestGate))
        {
            _state.BestStage = stage;
            _state.BestGate = rate;
            _state.BestSamples = _state.TotalSamples;
            SaveBestNet();
        }

        MaybeGrow(rate);

        int next = BlockDudeCurriculum.Advance(stage, _state.StageSamples, rate, out bool forced);
        next = Math.Min(next, _options.MaxStage);
        if (next == stage) return;

        Log(forced
            ? $"stage {stage} -> {next} FORCED at {_state.StageSamples:N0} samples (gate {rate:P0} never reached " +
              $"{BlockDudeCurriculum.Stages[stage].PromoteSolveRate:P0})"
            : $"stage {stage} -> {next} (gate {rate:P0})");
        _state.Stage = next;
        _state.StageSamples = 0;

        // A harder rung gates on harder boards, so the solve rate legitimately DROPS on promotion. To a detector
        // watching for "no new highs" that is indistinguishable from saturation, and it would grow the net for the
        // one reason that is not a capacity problem. The window starts again on the new rung's own scale.
        ResetPlateau();
    }

    /// <summary>
    /// Adds one rung of capacity when the gate has stopped producing new highs — the net is fitting harder without
    /// solving more, which is what saturation looks like from outside.
    /// </summary>
    /// <remarks>
    /// Deliberately driven by the GATE rather than the loss. Falling loss with a flat gate is the exact signature
    /// this is for: the net is still learning to reproduce the oracle's chosen action and still failing to solve
    /// boards, so more capacity is the plausible lever. A loss-driven trigger cannot see that difference — it would
    /// read the falling loss as healthy progress and never fire.
    /// </remarks>
    private void MaybeGrow(double rate)
    {
        var result = SaturationGrowth.Observe(
            _net, Progress, rate, _state.TotalSamples, BlockDudeGrowth.Ladder,
            new GrowthSettings(_options.Grow, _options.GrowPatience, _options.GrowMinImprovement),
            _state.GrowRng, Log);

        Progress = result.Progress;
        if (!result.Grew) return;

        _net = result.Net;
        _adam = new Adam(_net.Parameters(), _options.LearningRate);   // moments are keyed to the parameter set
    }

    /// <summary>The generic growth cursor, projected onto this campaign's own persisted state.</summary>
    private GrowthProgress Progress
    {
        get => new(_state.GrowthRung, new GrowthPlateau.State(_state.PlateauBest, _state.PlateauEvals),
                   _state.SamplesAtLastGrowth);
        set
        {
            _state.GrowthRung = value.Rung;
            _state.PlateauBest = value.Plateau.Best;
            _state.PlateauEvals = value.Plateau.EvalsSinceBest;
            _state.SamplesAtLastGrowth = value.LastGrowthAt;
        }
    }

    /// <summary>Starts the saturation window again. Every caller documents its own reason.</summary>
    private void ResetPlateau() => Progress = SaturationGrowth.ResetWindow(Progress);

    /// <summary>Greedy solve rate over the rung's fixed hold-out. Greedy, not search: it measures the policy
    /// itself, and the irreversibility of the game means a policy that needs search to avoid dead ends has not
    /// learned the rung.</summary>
    private double GateSolveRate(int stage)
    {
        if (_gateBoardsStage != stage)
        {
            _gateBoards = BlockDudeCurriculum.GateBoardsFor(stage);
            _gateBoardsStage = stage;
        }
        if (_gateBoards.Count == 0) return 0;

        int solved = 0;
        foreach (var board in _gateBoards)
            if (GreedySolves(board)) solved++;
        return solved / (double)_gateBoards.Count;
    }

    private bool GreedySolves(BlockDudeBoard start)
        => BlockDudeGreedy.Run(_net, start, BlockDudeCurriculum.Stages[CurrentStage].Spec.MaxOptimal * 3).Solved;

    public CampaignEval Evaluate()
    {
        // READ-ONLY. The runner calls this on a wall clock, so anything mutated here would make the training
        // trajectory depend on machine speed.
        int stage = CurrentStage;
        double ce = _windowBatches > 0 ? _windowCe / _windowBatches : double.NaN;
        double huber = _windowBatches > 0 ? _windowHuber / _windowBatches : double.NaN;
        double acc = _windowBatches > 0 ? _windowAcc / _windowBatches : double.NaN;
        _windowCe = _windowHuber = _windowAcc = 0;
        _windowBatches = 0;

        long drawn = Math.Max(1, _state.BoardAttempts);
        double acceptRate = _state.AcceptedBoards / (double)drawn;

        var metrics = new List<CampaignMetric>
        {
            new("stage", stage, "0"),
            new("samples", _state.TotalSamples, "0"),
            new("stage_samples", _state.StageSamples, "0"),
            new("ce", ce, "F4"),
            new("huber", huber, "F4"),
            new("acc", acc, "F3"),
            new("gate", _state.GateRates[stage], "F3"),
            new("accept_rate", acceptRate, "F3"),
            new("boards", _state.AcceptedBoards, "0"),
            new("truncated", _state.TruncatedBoards, "0"),
        };

        var report = new StringBuilder();
        report.Append($"stage {stage} | {_state.TotalSamples:N0} samples | ce {ce:F3} acc {acc:P0} | ");
        report.Append($"gate {(_state.GateRates[stage] < 0 ? "-" : _state.GateRates[stage].ToString("P0"))} | ");
        report.Append($"boards {_state.AcceptedBoards:N0}/{_state.BoardAttempts:N0} ({acceptRate:P0})");

        string worst = string.Join(" ", _outcomes
            .Where(kv => kv.Key != BlockDudeGenerationOutcome.Accepted)
            .OrderByDescending(kv => kv.Value)
            .Take(3)
            .Select(kv => $"{kv.Key}={kv.Value}"));
        if (worst.Length > 0) report.Append($" | rejected: {worst}");

        return new CampaignEval(metrics, report.ToString());
    }

    public void Checkpoint(IModelStore store)
    {
        // The RESUME net is always the latest weights, so a continuation stays consistent with the Adam moments
        // written beside it.
        store.Save(BlockDudeIds.Environment, _ids.Policy, s => _net.Save(s));
        AdamState.Save(store, BlockDudeIds.Environment, _ids.PolicyAdam, _adam);
        store.Save(BlockDudeIds.Environment, _ids.State, s => _state.Save(s));
    }

    /// <summary>Writes the deployable net, at the exact moment its gate was measured.</summary>
    private void SaveBestNet()
    {
        if (_store is null) return;
        _store.Save(BlockDudeIds.Environment, _ids.PolicyBest, s => _net.Save(s));
        Log($"new best: stage {_state.BestStage} gate {_state.BestGate:P0} at {_state.BestSamples:N0} samples " +
            $"-> saved deployable net '{_ids.PolicyBest}'");
    }

    public void Dispose() { }

    private (double Ce, double Huber, double Acc) TrainStep(List<Sample> samples, int offset, int batch)
    {
        const int actions = BlockDudeBoard.ActionCount;
        var obs = new float[batch * BlockDudeBoard.ObservationSize];
        var maskOffsets = new float[batch * actions];
        var weights = new float[batch * actions];
        var targets = new float[batch];

        for (int i = 0; i < batch; i++)
        {
            var s = samples[offset + i];
            s.Obs.CopyTo(obs.AsSpan(i * BlockDudeBoard.ObservationSize));
            s.MaskOffsets.CopyTo(maskOffsets.AsSpan(i * actions));

            // Soft target over ALL optimal actions: a single arbitrary label penalises the other equally-good
            // moves and flattens the policy.
            float w = 1f / System.Numerics.BitOperations.PopCount(s.LabelMask);
            for (uint bits = s.LabelMask; bits != 0; bits &= bits - 1)
                weights[i * actions + System.Numerics.BitOperations.TrailingZeroCount(bits)] = w;
            targets[i] = s.Distance / BlockDudePolicyNet.DistanceScale;
        }

        var (logits, value) = _net.Forward(new Tensor(obs, batch, BlockDudeBoard.ObservationSize));
        var logProbs = logits.Add(new Tensor(maskOffsets, batch, actions)).LogSoftmax();
        var ce = logProbs.Mul(new Tensor(weights, batch, actions)).Sum().MulScalar(-1f / batch);
        var huber = value.Reshape(batch).HuberLoss(new Tensor(targets, batch));
        var loss = ce.Add(huber);

        _adam.ZeroGrad();
        loss.Backward();
        _adam.ClipGradNorm(5f);
        _adam.Step();

        int correct = 0;
        for (int i = 0; i < batch; i++)
        {
            int argmax = 0;
            for (int a = 1; a < actions; a++)
                if (logProbs.Data[i * actions + a] > logProbs.Data[i * actions + argmax]) argmax = a;
            if ((samples[offset + i].LabelMask >> argmax & 1) != 0) correct++;   // any optimal action counts
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    private void Log(string message) => _logger?.LogInformation("[blockdude] {Message}", message);

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "blockdude-policy";

    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];

    /// <summary>
    /// Eval is the CURRICULUM GATE — the solve rate over this rung's held-out boards — not the batch accuracy
    /// beside it in the log. They diverge exactly where it matters: a net can predict the oracle's chosen action
    /// nine times in ten and still solve barely half the boards, which is the shape of the stage-3 plateau. The
    /// gate is the number that decides promotion, so it is the one worth watching evolve. NaN until the first
    /// gate of the run fires, which the viewer renders as "—".
    /// </summary>
    NetworkMetrics INetworkTelemetrySource.Sample()
    {
        long step = ReferenceEquals(_state, null) ? 0 : _state.TotalSamples;
        double gate = double.NaN;
        if (!ReferenceEquals(_state, null) && _state.Stage >= 0 && _state.Stage < _state.GateRates.Length)
            gate = _state.GateRates[_state.Stage];
        return new(step, _options.TargetSamples, _liveLoss, gate, double.NaN);
    }

    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => ["Left", "Right", "Climb", "Grab"];

    // An imitation campaign has no running environment, so the viewer forwards ONE fixed position every frame —
    // the start of shipped level 1 — and you watch this net's move preferences and hidden activations for that
    // board evolve. Read-only forward on the CPU (imitation has no GPU path); it never touches training state.
    private float[]? _probeObs;
    private float[] ProbeObs()
    {
        if (_probeObs is null)
        {
            var obs = new float[BlockDudeBoard.ObservationSize];
            BlockDudeLevels.Load(0).WriteObservation(obs);
            _probeObs = obs;
        }
        return _probeObs;
    }

    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo()
    {
        if (ReferenceEquals(_net, null)) return null;
        try
        {
            var obs = ProbeObs();
            var (logits, _) = _net.Forward(new Tensor((float[])obs.Clone(), 1, obs.Length));
            return ((float[])obs.Clone(), [.. logits.Data]);
        }
        catch { return null; }
    }

    float[][]? INetworkTelemetrySource.SampleActivations()
    {
        if (ReferenceEquals(_net, null)) return null;
        try
        {
            var obs = ProbeObs();
            return _net.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length));
        }
        catch { return null; }
    }

    private sealed record Sample(float[] Obs, float[] MaskOffsets, uint LabelMask, float Distance);
}
