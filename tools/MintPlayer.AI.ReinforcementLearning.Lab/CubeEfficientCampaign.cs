using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
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
internal sealed class CubeEfficientCampaign(AdaptiveBackend adaptive, ulong seed, float learningRate, int width, int maxScramble, int beamWidth, int evalEpisodes)
    : ITrainingCampaign, INetworkTelemetrySource
{
    private const string PolicyId = "policy-efficient";
    private const string PolicyAdamId = "policy-efficient-adam";
    private const string PolicyProgressId = "policy-efficient-progress";
    private const int BatchSize = 1000;
    private const int SamplesPerRound = 50_000;
    private static readonly int[] EvalDepths = [4, 8, 12, 14, 16, 18, 20, 22, 24, 26];

    private readonly Xoshiro256StarStar _rng = new(seed);
    private readonly int _generators = Math.Max(1, System.Environment.ProcessorCount - 2);
    private readonly AdaptiveBackend _adaptive = adaptive;

    private CubePolicyNet _net = null!;
    private Adam _adam = null!;
    private long _round, _totalSamples;
    private double _windowCe, _windowHuber, _windowAcc;
    private long _windowCount;
    private double _liveLoss = double.NaN, _liveAcc = double.NaN; // most-recent batch, for the live viewer

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
            _liveLoss = ce + huber;
            _liveAcc = acc;
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

        // Beam search runs the bulk of the forwards — route them through a GPU-resident DeviceMlp over the
        // policy head (weights uploaded once per eval) when a GPU is present; CPU autograd otherwise.
        DeviceMlp? device = _adaptive.Gpu is { } gpu ? gpu.CreateResidentForward(_net.PolicyAsMlp()) : null;
        Func<float[], int, float[]> beamLogits = device is not null
            ? device.Forward
            : (features, rows) =>
            {
                using (GradMode.NoGrad())
                    return _net.Forward(new Tensor(features, rows, RubiksCubeEnv.ObservationSize)).Logits.Data;
            };
        try
        {
            foreach (int depth in EvalDepths)
            {
                int greedySolved = 0, beamSolved = 0, beamLen = 0;
                for (int episode = 0; episode < evalEpisodes; episode++)
                {
                    var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
                    var cube = new FaceletCube();
                    cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
                    if (cube.IsSolved) { greedySolved++; beamSolved++; continue; }

                    if (CubePolicySearch.GreedyRollout(_net, cube).Solved) greedySolved++;
                    var beam = CubePolicySearch.BeamSearch(beamLogits, cube, beamWidth);
                    if (beam.Solved) { beamSolved++; beamLen += beam.Moves.Length; }
                }
                metrics.Add(new($"d{depth}_greedy", greedySolved / (double)evalEpisodes, "F3"));
                metrics.Add(new($"d{depth}_beam", beamSolved / (double)evalEpisodes, "F3"));
                string lenTag = beamSolved > 0 ? $" ({beamLen / (double)beamSolved:F1}qt)" : "";
                report.Append($"d{depth}: {greedySolved}/{evalEpisodes}g {beamSolved}/{evalEpisodes}b{lenTag} | ");
            }
        }
        finally
        {
            device?.Dispose();
        }
        return new CampaignEval(metrics, report.ToString());
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

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "cube-policy";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];
    NetworkMetrics INetworkTelemetrySource.Sample() => new(_totalSamples, 0, _liveLoss, _liveAcc, double.NaN);

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
