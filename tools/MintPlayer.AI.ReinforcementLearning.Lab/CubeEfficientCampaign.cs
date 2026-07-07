using System.Globalization;
using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

/// <summary>
/// EfficientCube campaign (`--game cube-policy`) as an <see cref="ITrainingCampaign"/> on
/// <see cref="CampaignRunner"/> (PLAN M25). Teacher-FREE: the label is the cube's own scramble reversal (no
/// Kociemba), trained into the two-headed <see cref="CubePolicyNet"/> (CE on the reversing move + Huber on path
/// length) and solved by policy beam search. Eval routes the beam's bulk forwards through a GPU-resident
/// <see cref="DeviceMlp"/> over the policy head, using the injected <see cref="AdaptiveBackend"/> (registered via
/// the Ilgpu.Hosting <c>AddGpuBackend()</c>; the container owns its lifetime). Resumes net + Adam + a persisted
/// (samples, round) counter under distinct `policy-efficient*` ids, so the imitation net is never touched.
/// </summary>
internal sealed class CubeEfficientCampaign(AdaptiveBackend adaptive, ulong seed, float learningRate, int width, int maxScramble, int beamWidth, int evalEpisodes, bool optimalProbe = false, int[]? beamSweep = null, string? sweepCsvPath = null, double[]? valueSweep = null, string? valueCsvPath = null)
    : ITrainingCampaign
{
    private const string PolicyId = "policy-efficient";
    private const string PolicyAdamId = "policy-efficient-adam";
    private const string PolicyProgressId = "policy-efficient-progress";
    private const int BatchSize = 1000;
    private const int SamplesPerRound = 50_000;
    private static readonly int[] EvalDepths = [4, 8, 12, 14, 16, 18, 20, 22, 24, 26];
    // BFS-optimal is the ground-truth reference for the provable-optimality probe, but the radius-d ball grows
    // ~9× per quarter-turn (d6 ≈ 1M states, d7 ≈ 9M), so the probe caps at a depth that stays sub-minute per cube.
    private static readonly int[] OptimalProbeDepths = [1, 2, 3, 4, 5, 6];

    private readonly Xoshiro256StarStar _rng = new(seed);
    private readonly int _generators = Math.Max(1, System.Environment.ProcessorCount - 2);
    private readonly AdaptiveBackend _adaptive = adaptive;

    private CubePolicyNet _net = null!;
    private Adam _adam = null!;
    private long _round, _totalSamples;
    private double _windowCe, _windowHuber, _windowAcc;
    private long _windowCount;

    public string Environment => CubeIds.Environment;

    public bool Resume(IModelStore store)
    {
        // Route the autograd's GEMMs through the (DI-owned) adaptive backend; CPU-only hosts degrade gracefully.
        Backend.Current = _adaptive;
        Log($"compute backend: {_adaptive.Describe()}");

        bool resumed;
        using (var existing = store.TryOpenRead(CubeIds.Environment, PolicyId))
        {
            if (existing is not null)
            {
                _net = CubePolicyNet.Load(existing);
                Log($"resumed EfficientCube net '{PolicyId}' from the model store");
                resumed = true;
            }
            else
            {
                _net = new CubePolicyNet(new Xoshiro256StarStar(seed ^ 0xC0FFEE), hidden: width);
                Log($"initialized a fresh EfficientCube net '{PolicyId}' (trunk width {width})");
                resumed = false;
            }
        }
        using (var adamState = store.TryOpenRead(CubeIds.Environment, PolicyAdamId))
        {
            if (adamState is not null)
            {
                using var reader = new BinaryReader(adamState, Encoding.UTF8, leaveOpen: true);
                _adam = AdamCheckpoint.Read(_net.Parameters(), reader);
                _adam.LearningRate = learningRate;
                Log($"resumed Adam state (lr set to {learningRate:E1})");
            }
            else _adam = new Adam(_net.Parameters(), learningRate);
        }
        // Persisted progress: cumulative samples (displayed count) + the round counter (so the scramble RNG
        // continues its stream on resume rather than regenerating the same scrambles).
        using (var progress = store.TryOpenRead(CubeIds.Environment, PolicyProgressId))
        {
            if (progress is not null)
            {
                using var reader = new BinaryReader(progress, Encoding.UTF8, leaveOpen: true);
                _totalSamples = reader.ReadInt64();
                _round = reader.ReadInt64();
                Log($"resumed progress: {_totalSamples:N0} samples generated, data stream at round {_round}");
            }
        }
        Log($"teacher-free (no Kociemba), max scramble {maxScramble}, beam {beamWidth}");
        return resumed;
    }

    public long TrainChunk()
    {
        // Self-supervised data gen on all cores: scrambling is independent and (unlike Kociemba imitation) no
        // solver bounds throughput — generation is nearly free.
        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 256);
        var perWorker = new List<CubeOracle.LabeledState>[_generators];
        ulong roundBase = unchecked(seed + (ulong)(++_round) * 1_000_003UL);
        Parallel.For(0, _generators, worker =>
        {
            var workerRng = new Xoshiro256StarStar(unchecked(roundBase + 0x9E3779B97F4A7C15UL * (ulong)(worker + 1)));
            var local = new List<CubeOracle.LabeledState>(SamplesPerRound / _generators + 64);
            while (local.Count < SamplesPerRound / _generators)
                local.AddRange(CubeSelfSupervised.LabelScramblePath(workerRng, maxScramble));
            perWorker[worker] = local;
        });
        foreach (var local in perWorker) samples.AddRange(local);
        CubePolicyTraining.Shuffle(samples, _rng);

        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = CubePolicyTraining.TrainStep(_net, _adam, samples, offset, BatchSize);
            _windowCe += ce;
            _windowHuber += huber;
            _windowAcc += acc;
            _windowCount++;
            _totalSamples += BatchSize;
        }
        return _totalSamples;
    }

    public CampaignEval Evaluate()
    {
        double ce = _windowCount > 0 ? _windowCe / _windowCount : 0;
        double acc = _windowCount > 0 ? _windowAcc / _windowCount : 0;
        double huber = _windowCount > 0 ? _windowHuber / _windowCount : 0;
        _windowCe = _windowHuber = _windowAcc = 0;
        _windowCount = 0;

        var metrics = new List<CampaignMetric>
        {
            new("samples", _totalSamples, "0"),
            new("ce", ce, "F4"),
            new("acc", acc, "F4"),
            new("huber", huber, "F5"),
        };
        var report = new StringBuilder();
        report.Append($"samples {_totalSamples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

        var (beamLogits, device) = BuildBeamForward();
        try
        {
            foreach (var d in EvaluateDepths(cube => CubePolicySearch.BeamSearch(beamLogits, cube, beamWidth), includeGreedy: true))
            {
                metrics.Add(new($"d{d.Depth}_greedy", d.GreedySolved / (double)d.Episodes, "F3"));
                metrics.Add(new($"d{d.Depth}_beam", d.BeamSolved / (double)d.Episodes, "F3"));
                // Solution quality: mean beam length (quarter-turns) and its slack over the scramble depth, which
                // upper-bounds the optimal (accidental cancellations only shorten it). Slack → 0 means near-optimal.
                metrics.Add(new($"d{d.Depth}_beamlen", d.MeanLen, "F2"));
                metrics.Add(new($"d{d.Depth}_slack", d.BeamSolved > 0 ? d.MeanLen - d.Depth : 0, "F2"));
                string lenTag = d.BeamSolved > 0 ? $" ({d.MeanLen:F1}qt)" : "";
                report.Append($"d{d.Depth}: {d.GreedySolved}/{d.Episodes}g {d.BeamSolved}/{d.Episodes}b{lenTag} | ");
            }
        }
        finally
        {
            device?.Dispose();
        }
        return new CampaignEval(metrics, report.ToString());
    }

    /// <summary>One depth's beam (and optional greedy) eval over the fixed episode seeds — the shared inner loop of
    /// the periodic eval and the beam-width sweep. <see cref="MeanExpansions"/> (net-forward count) is the
    /// machine-independent search-cost proxy the sweep trades off against solution length.</summary>
    private readonly record struct DepthEval(int Depth, int Episodes, int GreedySolved, int BeamSolved, double MeanLen, double MeanExpansions);

    private List<DepthEval> EvaluateDepths(Func<FaceletCube, CubePolicySearch.SearchResult> solve, bool includeGreedy)
    {
        var results = new List<DepthEval>(EvalDepths.Length);
        foreach (int depth in EvalDepths)
        {
            int greedySolved = 0, beamSolved = 0, beamLen = 0;
            long expansions = 0;
            for (int episode = 0; episode < evalEpisodes; episode++)
            {
                var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { greedySolved++; beamSolved++; continue; }

                if (includeGreedy && CubePolicySearch.GreedyRollout(_net, cube).Solved) greedySolved++;
                var beam = solve(cube);
                expansions += beam.Expansions;
                if (beam.Solved) { beamSolved++; beamLen += beam.Moves.Length; }
            }
            double meanLen = beamSolved > 0 ? beamLen / (double)beamSolved : 0;
            results.Add(new(depth, evalEpisodes, greedySolved, beamSolved, meanLen, expansions / (double)evalEpisodes));
        }
        return results;
    }

    /// <summary>
    /// The beam-search forward: a GPU-resident <see cref="DeviceMlp"/> over the policy head (weights uploaded
    /// once) when a device is present, else a CPU autograd forward. The returned <see cref="DeviceMlp"/> (if any)
    /// is owned by the caller and must be disposed once the beam work is done.
    /// </summary>
    private (Func<float[], int, float[]> Forward, DeviceMlp? Device) BuildBeamForward()
    {
        DeviceMlp? device = _adaptive.Gpu is { } gpu ? gpu.CreateResidentForward(_net.PolicyAsMlp()) : null;
        Func<float[], int, float[]> forward = device is not null
            ? device.Forward
            : (features, rows) =>
            {
                using (GradMode.NoGrad())
                    return _net.Forward(new Tensor(features, rows, RubiksCubeEnv.ObservationSize)).Logits.Data;
            };
        return (forward, device);
    }

    /// <summary>
    /// Eval-only standalone modes whose output does not fit the generic metric CSV (so the runner trains nothing and
    /// treats the run as handled): the beam-width sweep (`--beam-sweep`) and the provable-optimality probe
    /// (`--optimal-probe`). The sweep takes precedence if both are set.
    /// </summary>
    public bool TryRunStandaloneEval(IModelStore store)
    {
        if (valueSweep is { Length: > 0 }) { RunValueSweep(); return true; }
        if (beamSweep is { Length: > 0 }) { RunBeamSweep(); return true; }
        if (optimalProbe) { RunOptimalProbe(); return true; }
        return false;
    }

    /// <summary>
    /// The value-guided beam forward: like <see cref="BuildBeamForward"/> but over the COMBINED policy+value head
    /// (<see cref="CubePolicyNet.PolicyAndValueAsMlp"/>), so each row returns [12 logits ‖ 1 value]. The returned
    /// <see cref="DeviceMlp"/> (if any) is owned by the caller.
    /// </summary>
    private (Func<float[], int, float[]> Forward, DeviceMlp? Device) BuildPolicyValueForward()
    {
        const int stride = RubiksCubeEnv.ActionCount + 1;
        DeviceMlp? device = _adaptive.Gpu is { } gpu ? gpu.CreateResidentForward(_net.PolicyAndValueAsMlp()) : null;
        Func<float[], int, float[]> forward = device is not null
            ? device.Forward
            : (features, rows) =>
            {
                using (GradMode.NoGrad())
                {
                    var (logits, value) = _net.Forward(new Tensor(features, rows, RubiksCubeEnv.ObservationSize));
                    var packed = new float[rows * stride];
                    for (int r = 0; r < rows; r++)
                    {
                        Array.Copy(logits.Data, r * RubiksCubeEnv.ActionCount, packed, r * stride, RubiksCubeEnv.ActionCount);
                        packed[r * stride + RubiksCubeEnv.ActionCount] = value.Data[r];
                    }
                    return packed;
                }
            };
        return (forward, device);
    }

    /// <summary>
    /// `--value-sweep λ1,λ2,…`: run the depth eval with value-GUIDED beam search at each λ (fixed beam width) and
    /// log length + expansions per λ × depth (M34 W3). Tests whether the (already-trained) value head shortens
    /// solutions and at what search cost — compared against the pure-policy W2 curve by expansions (states
    /// forwarded), NOT beam width. λ = 0 is the pure-policy baseline. Writes its own self-describing CSV.
    /// </summary>
    private void RunValueSweep()
    {
        var (forward, device) = BuildPolicyValueForward();
        try
        {
            if (valueCsvPath is not null)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(valueCsvPath)!);
                if (!File.Exists(valueCsvPath))
                    File.AppendAllText(valueCsvPath, "utc,lambda,beam,depth,episodes,beam_solved,beamlen,slack,mean_expansions\n");
            }
            Log($"value-guided sweep: λ [{string.Join(", ", valueSweep!)}], beam {beamWidth}, {evalEpisodes} cubes/depth over d[{string.Join(",", EvalDepths)}]");
            foreach (double lambda in valueSweep!)
            {
                var report = new StringBuilder($"  λ {lambda}: ");
                Func<FaceletCube, CubePolicySearch.SearchResult> solve = lambda <= 0
                    ? cube => CubePolicySearch.BeamSearchValueGuided(forward, cube, beamWidth, 0.0) // still forwards children (matched cost)
                    : cube => CubePolicySearch.BeamSearchValueGuided(forward, cube, beamWidth, lambda);
                foreach (var d in EvaluateDepths(solve, includeGreedy: false))
                {
                    double slack = d.BeamSolved > 0 ? d.MeanLen - d.Depth : 0;
                    report.Append($"d{d.Depth} {d.BeamSolved}/{d.Episodes}@{d.MeanLen:F1}qt(x{d.MeanExpansions:F0}) | ");
                    if (valueCsvPath is not null)
                        File.AppendAllText(valueCsvPath, string.Create(CultureInfo.InvariantCulture,
                            $"{DateTime.UtcNow:u},{lambda},{beamWidth},{d.Depth},{d.Episodes},{d.BeamSolved},{d.MeanLen:F2},{slack:F2},{d.MeanExpansions:F0}\n"));
                }
                Log(report.ToString());
            }
        }
        finally
        {
            device?.Dispose();
        }
    }

    /// <summary>
    /// `--beam-sweep w1,w2,…`: re-run the depth eval at each beam width and log the length ↔ width ↔ solve-rate ↔
    /// expansions curve (M34 W2). Finds the smallest width that still holds solve-rate + length (a latency win) and
    /// whether a wider beam buys shorter solutions. Writes its own self-describing CSV (one row per width × depth)
    /// since the varying width doesn't fit the fixed-column metric log.
    /// </summary>
    private void RunBeamSweep()
    {
        var (beamLogits, device) = BuildBeamForward();
        try
        {
            if (sweepCsvPath is not null)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(sweepCsvPath)!);
                if (!File.Exists(sweepCsvPath))
                    File.AppendAllText(sweepCsvPath, "utc,beam,depth,episodes,beam_solved,beamlen,slack,mean_expansions\n");
            }
            Log($"beam-width sweep: widths [{string.Join(", ", beamSweep!)}], {evalEpisodes} cubes/depth over d[{string.Join(",", EvalDepths)}]");
            foreach (int width in beamSweep!)
            {
                var report = new StringBuilder($"  beam {width}: ");
                foreach (var d in EvaluateDepths(cube => CubePolicySearch.BeamSearch(beamLogits, cube, width), includeGreedy: false))
                {
                    double slack = d.BeamSolved > 0 ? d.MeanLen - d.Depth : 0;
                    report.Append($"d{d.Depth} {d.BeamSolved}/{d.Episodes}@{d.MeanLen:F1}qt(x{d.MeanExpansions:F0}) | ");
                    if (sweepCsvPath is not null)
                        // Invariant formatting — this machine's locale uses comma decimals, which would corrupt a
                        // comma-separated CSV (the generic CampaignCli writer takes the same care).
                        File.AppendAllText(sweepCsvPath, string.Create(CultureInfo.InvariantCulture,
                            $"{DateTime.UtcNow:u},{width},{d.Depth},{d.Episodes},{d.BeamSolved},{d.MeanLen:F2},{slack:F2},{d.MeanExpansions:F0}\n"));
                }
                Log(report.ToString());
            }
        }
        finally
        {
            device?.Dispose();
        }
    }

    private void RunOptimalProbe()
    {
        int episodes = Math.Min(evalEpisodes, 10); // BFS dominates the cost here; a handful of cubes/depth suffices.
        var model = new CubeModel();
        var (beamLogits, device) = BuildBeamForward();
        Log($"provable-optimality probe: beam (width {beamWidth}) vs BFS-optimal, {episodes} cubes/depth (quarter-turns)");
        try
        {
            foreach (int depth in OptimalProbeDepths)
            {
                int solved = 0, provablyOptimal = 0;
                double beamLenSum = 0, optLenSum = 0;
                for (int episode = 0; episode < episodes; episode++)
                {
                    var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
                    var cube = new FaceletCube();
                    cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));

                    // The scramble is itself a solution of length ≤ depth, so an optimum within `depth` always exists.
                    int optLen = BreadthFirstPlanner.FindOptimal(model, cube, depth)?.Count ?? depth;
                    var beam = CubePolicySearch.BeamSearch(beamLogits, cube, beamWidth);
                    if (!beam.Solved) continue;

                    solved++;
                    beamLenSum += beam.Moves.Length;
                    optLenSum += optLen;
                    if (beam.Moves.Length == optLen) provablyOptimal++; // beam ≥ optimal always; equality ⇒ optimal
                }
                double beamMean = solved > 0 ? beamLenSum / solved : 0;
                double optMean = solved > 0 ? optLenSum / solved : 0;
                double pct = solved > 0 ? 100.0 * provablyOptimal / solved : 0;
                Log($"  d{depth}: {solved}/{episodes} solved | beam {beamMean:F2}qt vs optimal {optMean:F2}qt | provably-optimal {provablyOptimal}/{solved} ({pct:F0}%)");
            }
        }
        finally
        {
            device?.Dispose();
        }
    }

    public void Checkpoint(IModelStore store)
    {
        store.Save(CubeIds.Environment, PolicyId, s => _net.Save(s));
        store.Save(CubeIds.Environment, PolicyAdamId, s =>
        {
            using var writer = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
            AdamCheckpoint.Write(_adam, writer);
        });
        store.Save(CubeIds.Environment, PolicyProgressId, s =>
        {
            using var writer = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
            writer.Write(_totalSamples); // cumulative samples generated
            writer.Write(_round);        // data-stream round counter
        });
    }

    // The AdaptiveBackend is owned by the DI container (disposed with the host), not the campaign. The per-eval
    // device-resident DeviceMlp is created and disposed inside Evaluate, so there is nothing campaign-owned here.
    public void Dispose() { }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
