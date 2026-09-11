using System.Text;
using Microsoft.Extensions.Logging;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
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
public sealed class BlockDudeImitationCampaign : ITrainingCampaign
{
    private const int BatchSize = 256;

    private readonly BlockDudeImitationOptions _options;
    private readonly ILogger? _logger;
    private readonly BlockDudeIds.NetIds _ids;
    private readonly ulong _fingerprint;

    private BlockDudePolicyNet _net = null!;
    private Adam _adam = null!;
    private BlockDudeTrainingState _state = null!;
    private List<BlockDudeBoard> _gateBoards = [];
    private int _gateBoardsStage = -1;

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
        if (_options.Fresh)
        {
            // Blank slate. Deleting all three ids together matters: a net without its progress sidecar resumes
            // as an unlabelled warm start at stage 0, which is a different run than the operator asked for.
            foreach (string id in new[] { _ids.Policy, _ids.PolicyAdam, _ids.State })
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
            _net = _options.Grow ? new BlockDudePolicyNet(initRng, DqnGrowth.Start) : new BlockDudePolicyNet(initRng);
            Log(_options.Grow
                ? $"initialized a fresh GROWING policy net (start trunk [{string.Join(",", DqnGrowth.Start)}])"
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

        // Growth keys off the persisted sample counter. That is exactly the counter the other imitation
        // campaigns failed to persist, which made a restarted run re-grow its trunk from the first stage.
        var grown = PolicyGrowth.Maybe(
            _net, _state.TotalSamples, _options.Grow, _options.GrowEvery, _options.LearningRate, _state.GrowRng, Log);
        if (grown is not null)
        {
            _net = grown.Value.Net;
            _adam = grown.Value.Adam;
        }

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

        int next = BlockDudeCurriculum.Advance(stage, _state.StageSamples, rate, out bool forced);
        next = Math.Min(next, _options.MaxStage);
        if (next == stage) return;

        Log(forced
            ? $"stage {stage} -> {next} FORCED at {_state.StageSamples:N0} samples (gate {rate:P0} never reached " +
              $"{BlockDudeCurriculum.Stages[stage].PromoteSolveRate:P0})"
            : $"stage {stage} -> {next} (gate {rate:P0})");
        _state.Stage = next;
        _state.StageSamples = 0;
    }

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
    {
        var current = start;
        var seen = new HashSet<int>();
        int budget = BlockDudeCurriculum.Stages[CurrentStage].Spec.MaxOptimal * 3;

        for (int step = 0; step < budget; step++)
        {
            if (current.Won) return true;
            var action = _net.Greedy(current);
            if (action is null) return false;

            var next = current.Apply(action.Value);
            if (next.SameStateAs(current)) return false;          // refused move: stuck
            if (!seen.Add(next.StateHash) && seen.Count > budget) return false;
            current = next;
        }
        return current.Won;
    }

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
        store.Save(BlockDudeIds.Environment, _ids.Policy, s => _net.Save(s));
        AdamState.Save(store, BlockDudeIds.Environment, _ids.PolicyAdam, _adam);
        store.Save(BlockDudeIds.Environment, _ids.State, s => _state.Save(s));
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

    private sealed record Sample(float[] Obs, float[] MaskOffsets, uint LabelMask, float Distance);
}
