using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Fully-resolved cube-davi campaign settings (code defaults → appsettings.json → CLI, resolved in
/// <see cref="CubeDaviLab"/>). A plain carrier so the campaign's constructor stays readable despite the knob count.
/// </summary>
public sealed record CubeDaviSettings
{
    public required ulong Seed { get; init; }
    public required string LogDirectory { get; init; }
    public required bool Residual { get; init; }
    public required int Width { get; init; }
    public required int HiddenLayers { get; init; }
    public required int Blocks { get; init; }
    public required int BatchSize { get; init; }
    public required float LearningRate { get; init; }
    public required float EpsSync { get; init; }
    public required int TargetSyncInterval { get; init; }
    public required float Beta2 { get; init; }
    public required bool FrontierBias { get; init; }
    public required int GrowToWidth { get; init; }
    public required long GrowAtSamples { get; init; }
    public required int SetCurriculumDepth { get; init; }
    public required double AdvanceRatio { get; init; }
    public required bool AutoWiden { get; init; }
    public required int MaxWidth { get; init; }
    public required long WidenStallSamples { get; init; }
    public required int MaxDepthCap { get; init; }
    public required long TargetSamples { get; init; }
    public int[]? ProbeOverride { get; init; }

    // eval-only knobs
    public required bool UseSearch { get; init; }
    public required float SearchWeight { get; init; }
    public required int MaxExpansions { get; init; }
    public required bool BatchedSearch { get; init; }
    public required bool VsKociemba { get; init; }
    public required double TimeBudgetSec { get; init; }
    public required bool ValueCurve { get; init; }
    public required int EvalEpisodes { get; init; }
}

/// <summary>
/// Teacher-free cube campaign (`--game cube-davi`, PLAN M18–M21) as an <see cref="ITrainingCampaign"/> driven by
/// <see cref="CampaignRunner"/> (PLAN M25): trains a value net by deep approximate value iteration
/// (<see cref="ValueIterationTrainer{TState}"/>) over <see cref="CubeModel"/> — no Kociemba, no oracle, just the
/// goal and a cost objective. A solve-rate / value-accuracy curriculum grows the scramble depth as the net masters
/// each level; <c>--auto-widen</c> and <c>--grow-to</c> add trunk capacity mid-run (Net2WiderNet warm start).
/// <para>
/// Owns the device-resident GPU stack (<see cref="AdaptiveBackend"/> + resident successor-eval forward + resident
/// residual trainer), torn down in <see cref="Dispose"/>. The per-1000-iteration curriculum/widen/grow/cap-probe
/// machinery lives in <see cref="TrainChunk"/> (so its cadence is independent of the runner's time-based eval);
/// <see cref="IsComplete"/> is the <c>--samples</c> hard stop; <see cref="TryRunStandaloneEval"/> handles the five
/// eval-only modes (greedy / A* / BWAS / time-budget / value-curve). The two depth-column CSVs are campaign-owned.
/// </para>
/// </summary>
public sealed class CubeDaviCampaign(AdaptiveBackend adaptive, CubeDaviSettings settings) : ITrainingCampaign, INetworkTelemetrySource
{
    private const int TrainChunkIterations = 1000; // P.8: train in 1000-iter chunks so the GPU-idle eval runs less
    private const long ProbeEvery = 15_000;        // iters between in-loop BWAS capability probes
    private const long StallWarnSamples = 4_000_000; // informational: note when the frontier hasn't advanced

    private readonly CubeDaviSettings _s = settings;
    private readonly string _valueId = settings.Residual ? "value-davi-res" : "value-davi";
    private readonly string _stateId = (settings.Residual ? "value-davi-res" : "value-davi") + "-state";
    private readonly string _csvPath = Path.Combine(settings.LogDirectory, settings.Residual ? "cube-davi-res.csv" : "cube-davi.csv");
    private readonly string _capCsvPath = Path.Combine(settings.LogDirectory, settings.Residual ? "cube-davi-res-cap.csv" : "cube-davi-cap.csv");
    private readonly int[] _probeDepths = settings.ProbeOverride ?? [8, 10, 12, 14, 16];

    private readonly AdaptiveBackend _backend = adaptive;
    private CubeModel _model = null!;
    private ValueIterationOptions _options = null!;
    private IValueNet _net = null!;
    private Adam _adam = null!;
    private ITargetForward? _targetForward;
    private DeviceResidualTrainer? _residentTrain;
    private ValueIterationTrainer<FaceletCube> _trainer = null!;

    private int _curriculumDepth = 2;
    private long _totalIterations;
    private Xoshiro256StarStar _sampleRng = null!;
    private float _lastLoss;
    private long _samplesSinceAdvance;
    private double _bestLossSinceReset = double.MaxValue;
    private long _samplesAtBestLoss;
    private long _nextProbe;
    private IModelStore _store = null!; // set in Resume; used by an in-chunk widen/grow to persist its new shape

    public string Environment => CubeIds.Environment;

    public long Samples => _totalIterations * _s.BatchSize;
    public bool IsComplete => _s.TargetSamples > 0 && Samples >= _s.TargetSamples;

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains.
    // On the GPU-resident path the CPU net is the periodically-synced master, so this reads the last synced weights. ---
    string INetworkTelemetrySource.NetKind => _valueId;
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];
    NetworkMetrics INetworkTelemetrySource.Sample() => new(Samples, _s.TargetSamples, _lastLoss, _curriculumDepth, double.NaN);
    // Single scalar output = the learned cost-to-go; forward a fixed scramble so its estimate + hidden activations
    // are visible live (the GPU-resident weights are mirrored by this CPU master, read lock-free).
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => ["Estimated distance to solved (quarter-turn moves)"];
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo() => CubeViz.SampleValueIo(_net, ref _probeObs);
    float[][]? INetworkTelemetrySource.SampleActivations() => CubeViz.SampleValueActivations(_net, ref _probeObs);
    private float[]? _probeObs;

    public bool Resume(IModelStore store)
    {
        _store = store;
        Directory.CreateDirectory(_s.LogDirectory);
        // Residual and plain nets use distinct checkpoint ids + CSVs so the two campaigns resume
        // independently and never collide on format or column count.
        if (!File.Exists(_csvPath))
            File.AppendAllText(_csvPath, "utc,iterations,curriculumDepth,loss," + string.Join(',', Enumerable.Range(1, _s.MaxDepthCap).Select(d => $"d{d}")) + "\n");
        if (!File.Exists(_capCsvPath))
            File.AppendAllText(_capCsvPath, "utc,iterations," + string.Join(',', _probeDepths.Select(d => $"d{d}")) + "\n");

        // Route the autograd's GEMMs through the (DI-owned) adaptive backend; CPU-only hosts degrade gracefully.
        Backend.Current = _backend;
        Log($"compute backend: {_backend.Describe()}");

        bool resumed;
        using (var existing = store.TryOpenRead(CubeIds.Environment, _valueId))
        {
            if (existing is not null)
            {
                _net = _s.Residual ? ResidualMlpCheckpoint.Load(existing) : MlpCheckpoint.Load(existing);
                Log($"resumed DAVI value net ({DescribeNet(_net)})");
                resumed = true;
            }
            else if (_s.Residual)
            {
                _net = new ResidualMlp(RubiksCubeEnv.ObservationSize, _s.Width, _s.Blocks, new Xoshiro256StarStar(_s.Seed ^ 0xDA71));
                Log($"fresh DAVI value net ({DescribeNet(_net)})");
                resumed = false;
            }
            else
            {
                // Net shape is configurable — hiddenLayers × width. Resumed nets keep their stored shape regardless.
                var sizes = new int[_s.HiddenLayers + 2];
                sizes[0] = RubiksCubeEnv.ObservationSize;
                for (int i = 1; i <= _s.HiddenLayers; i++) sizes[i] = _s.Width;
                sizes[^1] = 1;
                _net = new Mlp(sizes, new Xoshiro256StarStar(_s.Seed ^ 0xDA71), Activation.Relu);
                Log($"fresh DAVI value net ({DescribeNet(_net)})");
                resumed = false;
            }
        }

        _model = new CubeModel();
        _options = new ValueIterationOptions
        {
            BatchSize = _s.BatchSize,
            LearningRate = _s.LearningRate,
            DistanceScale = 1f,
            TargetUpdateInterval = _s.TargetSyncInterval,
            TargetUpdateLossThreshold = _s.EpsSync,
            AdamBeta2 = _s.Beta2,
        };

        // Resume the FULL learned state: Adam's moments, the curriculum depth, the iteration count and the
        // sampler RNG position. DAVI regenerates states from scrambles for free, so only the learned state matters.
        _sampleRng = new Xoshiro256StarStar(_s.Seed);
        using (var state = store.TryOpenRead(CubeIds.Environment, _stateId))
        {
            if (state is not null)
            {
                using var reader = new BinaryReader(state, System.Text.Encoding.UTF8, leaveOpen: true);
                CheckpointFormat.ReadHeader(reader, "cube-davi-state", 1);
                _curriculumDepth = reader.ReadInt32();
                _totalIterations = reader.ReadInt64();
                _sampleRng = CheckpointFormat.ReadRngState(reader);
                _adam = AdamCheckpoint.Read(_net.Parameters(), reader);
                _adam.LearningRate = _options.LearningRate;
                Log($"resumed training state: curriculum depth {_curriculumDepth}, {_totalIterations:N0} iterations done");
            }
            else
            {
                _adam = new Adam(_net.Parameters(), _options.LearningRate, beta2: _options.AdamBeta2);
            }
        }

        // Consolidation override: re-pin the curriculum to the accuracy frontier (pair with --max-depth) so every
        // sample trains states whose bootstrap targets are still meaningful, not capped deep states.
        if (_s.SetCurriculumDepth > 0 && _curriculumDepth != _s.SetCurriculumDepth)
        {
            Log($"curriculum depth overridden {_curriculumDepth} → {_s.SetCurriculumDepth} (--set-curriculum-depth)");
            _curriculumDepth = _s.SetCurriculumDepth;
        }

        BuildStack();
        Log(_residentTrain is not null
            ? "training: fully device-resident step (fwd+bwd+Adam on GPU); successor eval resident"
            : _targetForward is not null
                ? "successor evaluation: device-resident GPU forward (resident weights); train step on CPU"
                : "successor/train: CPU autograd");

        _bestLossSinceReset = double.MaxValue;
        _samplesAtBestLoss = Samples;
        _samplesSinceAdvance = 0;
        _nextProbe = _totalIterations + ProbeEvery;
        return resumed;
    }

    public long TrainChunk()
    {
        _trainer.Train(Sample, iterations: TrainChunkIterations, onIteration: (_, loss) => _lastLoss = loss);
        _totalIterations += TrainChunkIterations;
        _samplesSinceAdvance += (long)TrainChunkIterations * _s.BatchSize;

        if (_totalIterations >= _nextProbe)
        {
            try { CapabilityProbe(); }
            catch (Exception ex) { Log($"[cap] probe failed (non-fatal): {ex.Message}"); }
            _nextProbe = _totalIterations + ProbeEvery;
        }

        AdvanceCurriculumOrGrow();
        return Samples;
    }

    /// <summary>Per-depth greedy solve rate up to one level beyond the curriculum — the periodic CSV/console row.</summary>
    public CampaignEval Evaluate()
    {
        int evalUpTo = Math.Min(_curriculumDepth + 1, _s.MaxDepthCap);
        return ReportEval(_totalIterations, _curriculumDepth, _lastLoss, evalUpTo);
    }

    public void Checkpoint(IModelStore store) => SaveCheckpoint(store);

    /// <summary>`--eval-only`: dispatch among the five eval modes (value-curve / time-budget / A*-BWAS / greedy).</summary>
    public bool TryRunStandaloneEval(IModelStore store)
    {
        // Heuristic calibration: mean predicted V(start) per depth. Where it flattens is the accuracy ceiling —
        // separates "search-bound" (V still climbs) from "accuracy-bound / under-trained" (V saturated).
        if (_s.ValueCurve)
        {
            int[] depths = _s.ProbeOverride ?? [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26];
            ValueCurve(depths, Math.Max(_s.EvalEpisodes, 50));
            return true;
        }
        // Verify the DEPLOYED time-bounded solver (the exact code /api/cube/solve-davi runs), per-cube timed.
        if (_s.TimeBudgetSec > 0)
        {
            int[] depths = _s.ProbeOverride ?? [10, 12, 14, 16, 18];
            Func<FaceletCube, CubeValueSearch.SearchResult> solve = _targetForward is not null
                ? c => CubeValueSearch.Solve(_targetForward.Forward, c, _s.MaxExpansions, _s.SearchWeight, TimeSpan.FromSeconds(_s.TimeBudgetSec))
                : c => CubeValueSearch.Solve((ResidualMlp)_net, c, _s.MaxExpansions, _s.SearchWeight, TimeSpan.FromSeconds(_s.TimeBudgetSec));
            VerifyTimeBudget(solve, depths, _s.EvalEpisodes, _s.MaxExpansions, _s.SearchWeight, _s.TimeBudgetSec, _targetForward is not null);
            return true;
        }
        if (_s.UseSearch) Log($"eval via {(_s.BatchedSearch ? "batched " : "")}value-guided A* (weight {_s.SearchWeight}, ≤{_s.MaxExpansions:N0} expansions)");
        if (_s.VsKociemba) { CubeSolver.WarmUp(); Log("Tier-2 gate: comparing mean QTM length vs Kociemba"); }
        ReportEval(_totalIterations, _curriculumDepth, loss: 0, evalUpTo: _s.MaxDepthCap,
            _s.UseSearch, _s.SearchWeight, _s.MaxExpansions, _s.BatchedSearch, _s.VsKociemba, onlyDepths: _s.ProbeOverride, episodes: _s.EvalEpisodes);
        return true;
    }

    public void Dispose()
    {
        // The device-resident stack is campaign-owned (rebuilt on every widen/grow), so it is torn down here.
        // The AdaptiveBackend itself is owned by the DI container and disposed when the host is.
        (_targetForward as IDisposable)?.Dispose();
        _residentTrain?.Dispose();
    }

    // ── training internals ───────────────────────────────────────────────────────────────────────

    // The device stack (resident successor-eval forward + resident train step) and the trainer are rebuilt from
    // the current net by BuildStack() — kept in mutable fields (not `using`) so progressive growing / auto-widen
    // can dispose and recreate them at the new width mid-run. CPU-only machines fall back to the autograd path.
    private void BuildStack()
    {
        (_targetForward as IDisposable)?.Dispose();
        _residentTrain?.Dispose();
        _targetForward = _backend.Gpus.FirstOrDefault() is { } gpu
            ? _net switch
            {
                Mlp m => gpu.CreateResidentForward(m),
                ResidualMlp r => gpu.CreateResidentForward(r),
                _ => (ITargetForward?)null,
            }
            : null;
        _residentTrain = _backend.Gpus.FirstOrDefault() is { } gpu2 && _net is ResidualMlp resNet
            ? gpu2.CreateResidentTrainer(resNet, _options.BatchSize, _options.LearningRate, _options.GradClipNorm, _options.HuberDelta, _options.AdamBeta2)
            : null;
        _trainer = new ValueIterationTrainer<FaceletCube>(_model, Featurize, _net, _adam, _options, _targetForward, _residentTrain);
    }

    // Depth sampling within [1, curriculumDepth]: uniform (default), or triangular toward the frontier
    // (--frontier-bias) so samples concentrate where the value signal is still moving.
    private FaceletCube Sample()
    {
        int depth = _s.FrontierBias
            ? 1 + Math.Max(_sampleRng.NextInt(_curriculumDepth), _sampleRng.NextInt(_curriculumDepth))
            : 1 + _sampleRng.NextInt(_curriculumDepth);
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(_sampleRng, depth, quarterTurnsOnly: true));
        return cube;
    }

    private void SaveCheckpoint(IModelStore store)
    {
        store.Save(CubeIds.Environment, _valueId, s =>
        {
            if (_net is ResidualMlp res) ResidualMlpCheckpoint.Save(res, s);
            else MlpCheckpoint.Save((Mlp)_net, s);
        });
        store.Save(CubeIds.Environment, _stateId, s =>
        {
            using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
            CheckpointFormat.WriteHeader(writer, "cube-davi-state", 1);
            writer.Write(_curriculumDepth);
            writer.Write(_totalIterations);
            CheckpointFormat.WriteRngState(writer, _sampleRng);
            AdamCheckpoint.Write(_adam, writer);
        });
    }

    // Curriculum advancement: VALUE-ACCURACY gate, NO forced advance (OPTIMIZATIONS.md "train outward, gate on
    // mastery"). Advance d→d+1 only once mean predicted V at depth d is within `advanceRatio` of d, so V tracks true
    // cost-to-go there and its one-step targets for d+1 are trustworthy. A persistent stall is the honest "needs
    // more training / capacity" signal — surfaced as a note (or an --auto-widen) rather than papered over.
    private void AdvanceCurriculumOrGrow()
    {
        long currentSamples = Samples;
        if (_lastLoss < _bestLossSinceReset * 0.98f) { _bestLossSinceReset = _lastLoss; _samplesAtBestLoss = currentSamples; } // ≥2% improvement resets the plateau timer
        long lossStagnantSamples = currentSamples - _samplesAtBestLoss;

        double frontierRatio = MeanValueAtDepth(_curriculumDepth, episodes: 64) / _curriculumDepth;
        if (_curriculumDepth < _s.MaxDepthCap && frontierRatio >= _s.AdvanceRatio)
        {
            _curriculumDepth++;
            _samplesSinceAdvance = 0;
            _bestLossSinceReset = double.MaxValue; _samplesAtBestLoss = currentSamples; // new shell → track its loss afresh
            Log($"curriculum advanced → scramble depth {_curriculumDepth} (frontier V/d {frontierRatio:F2} ≥ {_s.AdvanceRatio:F2})");
        }
        else if (_s.AutoWiden && _net is ResidualMlp wNet && wNet.Width < _s.MaxWidth && lossStagnantSamples >= _s.WidenStallSamples)
        {
            // Loss flatlined at the frontier AND still can't clear the gate → capacity-bound. Widen the trunk
            // (function-preserving warm start: accuracy preserved, frontier doesn't move) to gain capacity for V
            // to climb past the gate. On NEED (plateau), not a timer; distinct from --grow-at.
            int oldW = wNet.Width, newW = Math.Min(_s.MaxWidth, wNet.Width * 2);
            double plateauLoss = _bestLossSinceReset;
            _net = wNet.WidenTo(newW, new Xoshiro256StarStar(_s.Seed ^ ((ulong)newW * 0x9E3779B1u)), symmetryNoise: 1e-3f);
            _adam = new Adam(_net.Parameters(), _options.LearningRate, beta2: _options.AdamBeta2);
            BuildStack();
            _bestLossSinceReset = double.MaxValue; _samplesAtBestLoss = currentSamples; _samplesSinceAdvance = 0;
            SaveCheckpoint(_store);
            Log($"auto-widen {oldW}→{newW}: frontier d{_curriculumDepth} loss plateaued (~{plateauLoss:F4} for {lossStagnantSamples:N0} samples) → added capacity (Net2WiderNet warm start) → {DescribeNet(_net)}");
        }
        else if (_curriculumDepth < _s.MaxDepthCap && _samplesSinceAdvance >= StallWarnSamples)
        {
            Log($"frontier d{_curriculumDepth} not yet mastered (V/d {frontierRatio:F2} < {_s.AdvanceRatio:F2}) after {_samplesSinceAdvance:N0} samples — needs longer training{(_net is ResidualMlp nw && nw.Width < _s.MaxWidth ? " or more capacity (enable --auto-widen)" : "")} (no forced advance)");
            _samplesSinceAdvance = 0; // throttle the note
        }

        // Progressive growing: train cheap at the narrow width, then widen once enough samples are in
        // (Net2WiderNet warm start; Adam restarts and the device stack rebuilds at the new width).
        if (_s.GrowToWidth > 0 && _net is ResidualMlp toGrow && toGrow.Width < _s.GrowToWidth && Samples >= _s.GrowAtSamples)
        {
            int oldWidth = toGrow.Width;
            _net = toGrow.WidenTo(_s.GrowToWidth, new Xoshiro256StarStar(_s.Seed ^ 0x67707D), symmetryNoise: 1e-3f);
            _adam = new Adam(_net.Parameters(), _options.LearningRate, beta2: _options.AdamBeta2);
            BuildStack();
            SaveCheckpoint(_store); // persist the widened shape immediately so a resume picks it up
            Log($"net widened {oldWidth}→{_s.GrowToWidth} (Net2WiderNet warm start) at {Samples:N0} samples → {DescribeNet(_net)}");
        }
    }

    // ── eval / probe internals (ported from CubeDaviLab) ─────────────────────────────────────────

    /// <summary>
    /// Per-depth greedy (or A*) solve rate for depths 1..<paramref name="evalUpTo"/>, written to the campaign's CSV
    /// (columns 1..MaxDepthCap, unevaluated depths left blank) and returned as a <see cref="CampaignEval"/>.
    /// </summary>
    private CampaignEval ReportEval(
        long iterations, int curriculumDepth, float loss, int evalUpTo,
        bool useSearch = false, float searchWeight = 2f, int maxExpansions = 50_000, bool batched = false, bool vsKociemba = false,
        int[]? onlyDepths = null, int episodes = 12)
    {
        var metrics = new List<CampaignMetric>
        {
            new("iterations", iterations, "0"),
            new("curriculumDepth", curriculumDepth, "0"),
            new("loss", loss, "F5"),
        };
        var report = new System.Text.StringBuilder();
        report.Append($"[eval{(useSearch ? "/A*" : "")}] iters {iterations:N0}, curr.depth {curriculumDepth}, loss {loss:F4} | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{iterations}", $"{curriculumDepth}", $"{loss:F5}" };

        for (int depth = 1; depth <= _s.MaxDepthCap; depth++)
        {
            // Skip depths beyond the curriculum frontier, or — when an explicit set is given — any depth not in it.
            if (depth > evalUpTo || (onlyDepths is not null && !onlyDepths.Contains(depth))) { cells.Add(""); continue; }
            int solved = 0;
            long totalLen = 0;          // Σ net solution QTM over cubes the net solved
            long kociembaLen = 0;       // Σ Kociemba QTM over the SAME solved cubes (Tier-2 baseline)
            int beatsKociemba = 0;      // #cubes where the net's QTM ≤ Kociemba's
            for (int episode = 0; episode < episodes; episode++)
            {
                var evalRng = new Xoshiro256StarStar((ulong)(700_000 + 1_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { solved++; continue; }

                var solution = useSearch
                    ? (batched ? _trainer.SolveWithSearchBatched(cube, maxExpansions, searchWeight)
                               : _trainer.SolveWithSearch(cube, maxExpansions, searchWeight))
                    : _trainer.Solve(cube, maxSteps: 2 * depth + 8);
                if (solution is not null)
                {
                    solved++; totalLen += solution.Count;
                    if (vsKociemba)
                    {
                        int kqt = KociembaQtm(cube);
                        kociembaLen += kqt;
                        if (solution.Count <= kqt) beatsKociemba++;
                    }
                }
            }
            double rate = solved / (double)episodes;
            cells.Add($"{rate:F3}");
            metrics.Add(new($"d{depth}", rate, "F3"));
            // For A* the solution LENGTH matters (the shortest-move objective): show mean QTM over solves.
            string lenTag = useSearch && solved > 0 ? $" ({totalLen / (double)solved:F1}qt)" : "";
            string kTag = vsKociemba && solved > 0 ? $" [vs Koc {kociembaLen / (double)solved:F1}qt, ≤{beatsKociemba}/{solved}]" : "";
            report.Append($"d{depth} {solved}/{episodes}{lenTag} | ");
            if (useSearch) Log($"  d{depth}: {solved}/{episodes} solved{lenTag}{kTag}"); // incremental progress (A* is slow)
        }

        File.AppendAllText(_csvPath, string.Join(',', cells) + "\n");
        return new CampaignEval(metrics, report.ToString());
    }

    /// <summary>
    /// BWAS capability probe at a few discriminating depths (8 cubes each, small budget). The live greedy eval
    /// plateaus ~depth 10 and understates the net; this records the true solve-rate-over-time during a run.
    /// </summary>
    private void CapabilityProbe()
    {
        const int episodes = 8;
        var report = new System.Text.StringBuilder($"[cap] iters {_totalIterations:N0} (BWAS w2.5 ≤8k) | ");
        var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{_totalIterations}" };
        foreach (int depth in _probeDepths)
        {
            int solved = 0;
            long totalLen = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(900_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { solved++; continue; }
                var sol = _trainer.SolveWithSearchBatched(cube, maxExpansions: 8000, weight: 2.5f);
                if (sol is not null) { solved++; totalLen += sol.Count; }
            }
            report.Append($"d{depth} {solved}/{episodes}{(solved > 0 ? $" ({totalLen / (double)solved:F1}qt)" : "")} | ");
            cells.Add($"{solved / (double)episodes:F3}");
        }
        Log(report.ToString());
        File.AppendAllText(_capCsvPath, string.Join(',', cells) + "\n");
    }

    private static float[] Featurize(FaceletCube cube)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);
        return obs;
    }

    /// <summary>
    /// Mean predicted cost-to-go <c>V(start)</c> over <paramref name="episodes"/> fixed-seed random scrambles at
    /// <paramref name="depth"/> (forward passes only). The curriculum's value-accuracy gate compares
    /// <c>this / depth</c> against the advance ratio.
    /// </summary>
    private double MeanValueAtDepth(int depth, int episodes)
    {
        double sum = 0;
        for (int e = 0; e < episodes; e++)
        {
            var rng = new Xoshiro256StarStar((ulong)(860_000 + 1_000 * depth + e));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
            sum += _trainer.Value(cube);
        }
        return sum / episodes;
    }

    /// <summary>
    /// Heuristic-calibration probe: mean predicted cost-to-go <c>V(start)</c> (in moves) per depth. A calibrated
    /// value tracks the scramble depth; where the mean flattens is the accuracy ceiling gating search reach.
    /// </summary>
    private void ValueCurve(int[] depths, int episodes)
    {
        Log($"value calibration: mean predicted V(start) vs scramble depth ({episodes} cubes/depth)");
        foreach (int depth in depths)
        {
            double sum = 0, sumSq = 0; float min = float.MaxValue, max = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(800_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));
                float v = _trainer.Value(cube);
                sum += v; sumSq += (double)v * v; min = MathF.Min(min, v); max = MathF.Max(max, v);
            }
            double mean = sum / episodes;
            double std = Math.Sqrt(Math.Max(0, sumSq / episodes - mean * mean));
            Log($"  d{depth,2}: mean V {mean,6:F2}  (std {std:F2}, range {min:F1}–{max:F1})  ratio V/d {mean / depth:F2}");
        }
    }

    /// <summary>
    /// Benchmark the deployed time-bounded solver: per depth, solve fixed-seed scrambles through
    /// <paramref name="solve"/> and report solve rate, mean solution length, and mean/worst wall-clock.
    /// </summary>
    private void VerifyTimeBudget(
        Func<FaceletCube, CubeValueSearch.SearchResult> solve, int[] depths, int episodes,
        int maxExpansions, float weight, double seconds, bool gpu)
    {
        Log($"time-bounded solve verification ({(gpu ? "resident GPU forward" : "CPU forward")}): weight {weight:g}, ≤{maxExpansions:N0} exp ceiling, {seconds:F0}s budget, {episodes} cubes/depth");
        foreach (int depth in depths)
        {
            int solved = 0; long totalLen = 0; double totalMs = 0, worstMs = 0;
            for (int e = 0; e < episodes; e++)
            {
                var rng = new Xoshiro256StarStar((ulong)(950_000 + 1_000 * depth + e));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(rng, depth, quarterTurnsOnly: true));

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var result = solve(cube);
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                totalMs += ms; worstMs = Math.Max(worstMs, ms);
                if (result.Solved) { solved++; totalLen += result.Moves.Length; }
            }
            string len = solved > 0 ? $", mean {totalLen / (double)solved:F1}qt" : "";
            Log($"  d{depth}: {solved}/{episodes} solved{len} | mean {totalMs / episodes:F0} ms, worst {worstMs:F0} ms");
        }
    }

    /// <summary>Quarter-turn length of Kociemba's (half-turn-metric) solution — a "X2" face turn is 2 quarter-turns.</summary>
    private static int KociembaQtm(FaceletCube cube)
    {
        var result = CubeSolver.Solve(cube);
        if (!result.Solved) return 1000;
        int qt = 0;
        foreach (var move in result.Moves) qt += move.EndsWith('2') ? 2 : 1;
        return qt;
    }

    private static string DescribeNet(IValueNet net) => net switch
    {
        Mlp mlp => string.Join('→', mlp.Sizes),
        ResidualMlp res => $"residual {res.InputSize}→{res.Width}×{res.Blocks}blocks→1",
        _ => net.GetType().Name,
    };

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
