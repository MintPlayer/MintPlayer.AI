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

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// EfficientCube campaign (`--game cube-policy`) as an <see cref="ITrainingCampaign"/> on
/// <see cref="CampaignRunner"/> (PLAN M25). Teacher-FREE: the label is the cube's own scramble reversal (no
/// Kociemba), trained into the two-headed <see cref="CubePolicyNet"/> (CE on the reversing move + Huber on path
/// length) and solved by policy beam search. Eval routes the beam's bulk forwards through a GPU-resident
/// <see cref="DeviceMlp"/> over the policy head, using the injected <see cref="AdaptiveBackend"/> (registered via
/// the Ilgpu.Hosting <c>AddGpuBackend()</c>; the container owns its lifetime). Resumes net + Adam + a persisted
/// (samples, round) counter under distinct `policy-efficient*` ids, so the imitation net is never touched.
/// </summary>
public sealed class CubeEfficientCampaign(AdaptiveBackend adaptive, ulong seed, float learningRate, int width, int maxScramble, int beamWidth, int evalEpisodes, bool grow = false, int growEvery = 50_000)
    : ITrainingCampaign, INetworkTelemetrySource
{
    private readonly Xoshiro256StarStar _growRng = new(seed ^ 0x6C0FFEEUL); // dedicated stream for growth
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
    private TrainWindow _window;
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
                var initRng = new Xoshiro256StarStar(seed ^ 0xC0FFEE);
                _net = grow ? new CubePolicyNet(initRng, DqnGrowth.Start) : new CubePolicyNet(initRng, hidden: width);
                Log(grow
                    ? $"initialized a fresh GROWING EfficientCube net '{PolicyId}' (start trunk [{string.Join(",", DqnGrowth.Start)}])"
                    : $"initialized a fresh EfficientCube net '{PolicyId}' (trunk width {width})");
                resumed = false;
            }
        }
        _adam = AdamState.LoadOrInit(store, CubeIds.Environment, PolicyAdamId, _net.Parameters(), learningRate, Log);
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
        // solver bounds throughput — generation is nearly free. DeterministicParallel derives each generator's RNG
        // from (roundBase, worker+1) — byte-identical to the old hand-rolled `roundBase + φ·(worker+1)` seeding.
        ulong roundBase = unchecked(seed + (ulong)(++_round) * 1_000_003UL);
        int per = SamplesPerRound / _generators;
        var perWorker = DeterministicParallel.Generate(_generators, roundBase, baseIndex: 1, (worker, rng) =>
        {
            var local = new List<CubeOracle.LabeledState>(per + 64);
            while (local.Count < per)
                local.AddRange(CubeSelfSupervised.LabelScramblePath(rng, maxScramble));
            return local;
        }, parallel: true);

        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 256);
        foreach (var local in perWorker) samples.AddRange(local);
        CubePolicyTraining.Shuffle(samples, _rng);

        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = CubePolicyTraining.TrainStep(_net, _adam, samples, offset, BatchSize);
            _window.Add(ce, huber, acc);
            _totalSamples += BatchSize;
            _liveLoss = ce + huber;
            _liveAcc = acc;
        }
        if (PolicyGrowth.Maybe(_net, _totalSamples, grow, growEvery, learningRate, _growRng, Log) is var g && g.HasValue)
            (_net, _adam) = (g.Value.Net, g.Value.Adam);
        return _totalSamples;
    }

    public CampaignEval Evaluate()
    {
        var (ce, huber, acc) = _window.MeanAndReset();

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
        DeviceMlp? device = _adaptive.Gpus.FirstOrDefault() is { } gpu ? gpu.CreateResidentForward(_net.PolicyAsMlp()) : null;
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
        AdamState.Save(store, CubeIds.Environment, PolicyAdamId, _adam);
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
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => RubiksCubeEnv.ActionLabels; // 12 quarter-turns
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo() => CubeViz.SampleIo(_net, ref _probeObs);
    float[][]? INetworkTelemetrySource.SampleActivations() => CubeViz.SampleActivations(_net, ref _probeObs);
    private float[]? _probeObs;

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
