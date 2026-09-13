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
/// Phase 2: expert iteration on the SHIPPED levels, using the human demonstrations as a reverse curriculum.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> Phase 1 imitates an exact oracle on generated boards, and measurement showed
/// that ceiling is structural rather than a matter of more samples (PRD §8.4a): the generator cannot express
/// 53% of shipped topologies, the oracle cannot label boards that size at all, and tripling capacity moved the
/// gate not at all. Every source of phase-1 data is out-of-distribution by construction.</para>
///
/// <para><b>The mechanism.</b> A suffix of a human solution is a real position on real shipped terrain that is
/// only N moves from the door. So the net is never asked to solve more of a level than it has just shown it can:
/// search finishes the suffix, the solution becomes training data, and the frontier moves outward. Measured
/// before building this — Level 11, 909 moves and untouchable from its opening, is solved 35 moves from the end
/// (PRD §6a). Fifteen demonstrations become thousands of tasks on terrain no generator produces.</para>
///
/// <para><b>Per-level frontiers, not one global one.</b> Level 3 was already solvable 76 moves out while Level 5
/// stalled at 10. A single shared frontier would be held to the hardest level and waste the easy ones.</para>
///
/// <para><b>Two sources in every batch.</b> Searched solutions cluster wherever the frontier currently is, so on
/// their own the far-distance labels would fade from the mix as it advances. A fixed share of human
/// demonstration states (<see cref="BlockDudeExpertIterationOptions.DemoShare"/>) keeps a permanent anchor in
/// true 1-to-909 distance data — the range the value head was measured to be blind in.</para>
///
/// <para><b>Distinct checkpoint ids</b> (<c>policy-xit*</c>), so the phase-1 net is never touched and stays
/// available as a fallback tier.</para>
/// </remarks>
public sealed class BlockDudeExpertIterationCampaign : ITrainingCampaign, INetworkTelemetrySource
{
    private const int BatchSize = 128;

    private readonly BlockDudeExpertIterationOptions _options;
    private readonly ILogger? _logger;
    private readonly BlockDudeIds.NetIds _ids = BlockDudeIds.ForPhase(2);

    private BlockDudePolicyNet _net = null!;
    private Adam _adam = null!;
    private Xoshiro256StarStar _rng = null!;

    /// <summary>Moves-from-the-door each level is currently asked to solve. Persisted — a frontier is the run's
    /// real progress, and losing it on restart would drop the campaign back to the opening depth.</summary>
    private Dictionary<string, int> _frontier = [];

    private long _totalSamples;
    private int _rounds;
    private double _liveLoss = double.NaN, _liveAcc = double.NaN;
    private int _windowSolved, _windowAttempts;

    /// <summary>Demonstration states, materialised once — they never change.</summary>
    private BlockDudeDemoState[] _demos = [];

    public BlockDudeExpertIterationCampaign(BlockDudeExpertIterationOptions options, ILogger? logger = null)
    {
        _options = options;
        _logger = logger;
    }

    public string Environment => BlockDudeIds.Environment;

    public bool IsComplete => _options.TargetSamples > 0 && _totalSamples >= _options.TargetSamples;

    public bool Resume(IModelStore store)
    {
        if (_options.Fresh)
            foreach (string id in new[] { _ids.Policy, _ids.PolicyAdam, _ids.State, _ids.PolicyBest })
                if (store.Delete(BlockDudeIds.Environment, id)) Log($"--fresh: deleted {BlockDudeIds.Environment}.{id}");

        _demos = [.. BlockDudeDemonstrations.LabelledStates()];
        _rng = new Xoshiro256StarStar(_options.Seed ^ 0x5EEDF00DUL);

        bool resumed = false;
        using (var stream = store.TryOpenRead(BlockDudeIds.Environment, _ids.Policy))
        {
            if (stream is not null)
            {
                try { _net = BlockDudePolicyNet.Load(stream); resumed = true; Log($"resumed the phase-2 net (trunk [{string.Join(",", _net.Trunk)}])"); }
                catch (InvalidDataException ex) { Log($"STALE phase-2 net ignored: {ex.Message}"); }
            }
        }

        if (!resumed && _options.WarmStart)
        {
            // Phase 1 already learned the local mechanics on small boards. Searched data is expensive; spending
            // it to re-learn how to climb would be wasteful.
            using var phase1 = store.TryOpenRead(BlockDudeIds.Environment, BlockDudeIds.ForPhase(1).PolicyBest);
            if (phase1 is not null)
            {
                try { _net = BlockDudePolicyNet.Load(phase1); Log($"warm-started from the phase-1 best net (trunk [{string.Join(",", _net.Trunk)}])"); }
                catch (InvalidDataException ex) { Log($"phase-1 net unusable for warm start: {ex.Message}"); }
            }
            else Log("no phase-1 net to warm-start from — starting phase 2 from random weights");
        }

        _net ??= new BlockDudePolicyNet(new Xoshiro256StarStar(_options.Seed ^ 0xDEADBEEF), BlockDudeGrowth.Ladder.TrunkFor(0));
        _adam = AdamState.LoadOrInit(store, BlockDudeIds.Environment, _ids.PolicyAdam, _net.Parameters(), _options.LearningRate, Log);

        _frontier = LoadFrontier(store) ?? BlockDudeSolutions.All.ToDictionary(s => s.Name, _ => _options.InitialFrontier);
        Log($"frontier: {Describe(_frontier)}");

        return resumed;
    }

    public long TrainChunk()
    {
        var samples = new List<Sample>();
        _rounds++;

        foreach (var solution in BlockDudeSolutions.All)
        {
            int depth = _frontier[solution.Name];
            int solved = 0;

            for (int attempt = 0; attempt < _options.AttemptsPerLevel; attempt++)
            {
                // Jitter the depth so the net does not overfit one exact position per level: every start is
                // still a genuine state on the demonstrated path, just a different distance along it.
                int jittered = Math.Max(1, depth - _rng.NextInt(Math.Max(1, depth / 4)));
                var start = BlockDudeDemonstrations.ReverseCurriculumStarts(jittered, solution.Name).SingleOrDefault();
                if (start is null) continue;

                _windowAttempts++;
                var found = BlockDudeSearch.Solve(_net, start, _options.Expansions, _options.Weight,
                                                 TimeSpan.FromSeconds(_options.SearchSeconds));
                if (!found.Solved) continue;

                solved++;
                _windowSolved++;
                Collect(start, found.Moves!, samples);
            }

            // Advance only on a clear majority: a frontier pushed outward on one lucky search produces tasks the
            // net cannot do, which yields no data at all and stalls the level.
            if (solved >= _options.AdvanceRate * _options.AttemptsPerLevel)
            {
                int next = Math.Max(depth + 1, (int)(depth * _options.FrontierGrowth));
                int cap = solution.Moves.Length;
                _frontier[solution.Name] = Math.Min(next, cap);
                if (_frontier[solution.Name] != depth)
                    Log($"{solution.Name}: frontier {depth} → {_frontier[solution.Name]}" +
                        (_frontier[solution.Name] == cap ? " (the whole level)" : ""));
            }
        }

        AddDemonstrationStates(samples);
        if (samples.Count < BatchSize) return _totalSamples;

        Shuffle(samples);
        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = TrainStep(samples, offset, BatchSize);
            _liveLoss = ce + huber;
            _liveAcc = acc;
            _totalSamples += BatchSize;
        }

        return _totalSamples;
    }

    /// <summary>Turns a found solution into one sample per state along it, labelled with the move taken and the
    /// moves remaining FROM THERE — the same two labels the demonstrations carry.</summary>
    private void Collect(BlockDudeBoard start, IReadOnlyList<int> moves, List<Sample> into)
    {
        var board = start;
        for (int i = 0; i < moves.Count; i++)
        {
            into.Add(Sample.From(board, (BlockDudeAction)moves[i], moves.Count - i));
            board = board.Apply((BlockDudeAction)moves[i]);
        }
    }

    /// <summary>Mixes in human states so the far-distance labels never fade out as the frontier advances.</summary>
    private void AddDemonstrationStates(List<Sample> into)
    {
        if (_demos.Length == 0 || _options.DemoShare <= 0) return;

        int wanted = (int)(into.Count * _options.DemoShare / Math.Max(0.01, 1 - _options.DemoShare));
        wanted = Math.Max(wanted, BatchSize / 4);

        for (int i = 0; i < wanted; i++)
        {
            var demo = _demos[_rng.NextInt(_demos.Length)];
            into.Add(Sample.From(demo.Board, demo.Action, demo.Remaining));
        }
    }

    public CampaignEval Evaluate()
    {
        double rate = _windowAttempts > 0 ? _windowSolved / (double)_windowAttempts : double.NaN;
        int minFrontier = _frontier.Count > 0 ? _frontier.Values.Min() : 0;
        int maxFrontier = _frontier.Count > 0 ? _frontier.Values.Max() : 0;
        int whole = _frontier.Count(kv => kv.Value >= BlockDudeSolutions.For(kv.Key)!.Moves.Length);

        _windowSolved = _windowAttempts = 0;

        var report = new StringBuilder()
            .Append($"round {_rounds} | {_totalSamples:N0} samples | loss {_liveLoss:F4} | acc {_liveAcc:P0} | ")
            .Append($"search {(double.IsNaN(rate) ? "-" : rate.ToString("P0"))} | ")
            .Append($"frontier {minFrontier}–{maxFrontier} | {whole}/{_frontier.Count} levels whole");

        return new CampaignEval(
        [
            new("samples", _totalSamples, "N0"),
            new("loss", _liveLoss, "F4"),
            new("acc", _liveAcc, "P0"),
            new("search_rate", rate, "P0"),
            new("frontier_min", minFrontier, "N0"),
            new("frontier_max", maxFrontier, "N0"),
            new("levels_whole", whole, "N0"),
        ], report.ToString());
    }

    public void Checkpoint(IModelStore store)
    {
        store.Save(BlockDudeIds.Environment, _ids.Policy, _net.Save);
        AdamState.Save(store, BlockDudeIds.Environment, _ids.PolicyAdam, _adam);
        SaveFrontier(store);
    }

    public void Dispose() { }

    // ── the supervised update ─────────────────────────────────────────────────────────────────────────────────
    // Single-action cross-entropy, unlike phase 1's soft target over ALL optimal actions: a searched or
    // demonstrated path gives ONE move per state and makes no claim that the others are worse. Distance is the
    // remaining length of that path — an upper bound on the optimum, never the optimum.

    private readonly record struct Sample(float[] Obs, float[] MaskOffsets, int Action, float Distance)
    {
        public static Sample From(BlockDudeBoard board, BlockDudeAction action, int remaining)
        {
            var obs = new float[BlockDudeBoard.ObservationSize];
            board.WriteObservation(obs);

            const float illegal = -1e9f;   // finite, not -inf: 0 * -inf is NaN and poisons the logged loss
            var mask = new float[BlockDudeBoard.ActionCount];
            for (int a = 0; a < BlockDudeBoard.ActionCount; a++)
                mask[a] = board.IsLegal((BlockDudeAction)a) ? 0f : illegal;

            return new(obs, mask, (int)action, remaining);
        }
    }

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
            weights[i * actions + s.Action] = 1f;
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
            if (argmax == samples[offset + i].Action) correct++;
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    private void Shuffle(List<Sample> samples)
    {
        for (int i = samples.Count - 1; i > 0; i--)
        {
            int j = _rng.NextInt(i + 1);
            (samples[i], samples[j]) = (samples[j], samples[i]);
        }
    }

    // ── frontier persistence ──────────────────────────────────────────────────────────────────────────────────
    // Deliberately its own tiny sidecar rather than a field on the phase-1 state: the two phases have different
    // progress, different ids and different lifetimes, and sharing a format would couple them for no gain.

    private void SaveFrontier(IModelStore store)
        => store.Save(BlockDudeIds.Environment, _ids.State, stream =>
        {
            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
            writer.Write(1);                       // format version
            writer.Write(_totalSamples);
            writer.Write(_rounds);
            writer.Write(_frontier.Count);
            foreach (var (name, depth) in _frontier) { writer.Write(name); writer.Write(depth); }
        });

    private Dictionary<string, int>? LoadFrontier(IModelStore store)
    {
        using var stream = store.TryOpenRead(BlockDudeIds.Environment, _ids.State);
        if (stream is null) return null;

        try
        {
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            if (reader.ReadInt32() != 1) return null;
            _totalSamples = reader.ReadInt64();
            _rounds = reader.ReadInt32();

            int count = reader.ReadInt32();
            var frontier = new Dictionary<string, int>(count);
            for (int i = 0; i < count; i++) frontier[reader.ReadString()] = reader.ReadInt32();

            // A level added to the pack since the sidecar was written starts at the opening depth rather than
            // throwing the whole frontier away.
            foreach (var solution in BlockDudeSolutions.All)
                frontier.TryAdd(solution.Name, _options.InitialFrontier);

            return frontier;
        }
        catch (Exception ex) when (ex is EndOfStreamException or InvalidDataException)
        {
            Log("phase-2 frontier sidecar unreadable — starting the curriculum from the opening depth");
            return null;
        }
    }

    private static string Describe(Dictionary<string, int> frontier)
        => string.Join(", ", frontier.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"));

    private void Log(string message) => _logger?.LogInformation("[blockdude-xit] {Message}", message);

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "blockdude-policy-xit";

    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];

    /// <summary>
    /// Eval is the share of the shipped levels whose frontier has reached the whole level — this phase's actual
    /// objective. Not the search success rate, which is held near the advance threshold BY the curriculum and so
    /// says more about how fast the frontier is moving than about how good the net is.
    /// </summary>
    NetworkMetrics INetworkTelemetrySource.Sample()
    {
        double whole = _frontier.Count == 0
            ? double.NaN
            : _frontier.Count(kv => kv.Value >= (BlockDudeSolutions.For(kv.Key)?.Moves.Length ?? int.MaxValue))
              / (double)_frontier.Count;

        return new(_totalSamples, _options.TargetSamples, _liveLoss, whole, double.NaN);
    }

    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => ["Left", "Right", "Climb", "Grab"];

    // The probe is the start of shipped level 1 — the same board phase 1 uses, so the two phases' viewers show
    // the same position and their policies can be compared by eye.
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
}
