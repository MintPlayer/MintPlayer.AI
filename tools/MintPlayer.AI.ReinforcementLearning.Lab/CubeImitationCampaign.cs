using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Cube imitation campaign (PLAN M16, `--game cube`) as an <see cref="ITrainingCampaign"/> driven by
/// <see cref="CampaignRunner"/> (PLAN M25): streams random scrambles through the Kociemba oracle (one solve = a
/// whole labeled solution path), trains the two-headed <see cref="CubePolicyNet"/> supervised (CE on the next
/// quarter-turn + Huber on distance-to-go), and reports per-depth greedy/search solve rates. Resumes the net +
/// Adam from `cube.policy` / `cube.policy-adam` (width-ladder ids via <see cref="CubeIds.ForWidth"/>).
/// </summary>
internal sealed class CubeImitationCampaign(ulong seed, float learningRate, int width) : ITrainingCampaign, INetworkTelemetrySource
{
    private const int BatchSize = 256;
    private const int SamplesPerRound = 4096;
    private static readonly int[] EvalDepths = [2, 4, 6, 8, 10, 12, 16, 20];

    private readonly CubeIds.NetIds _ids = CubeIds.ForWidth(width);
    private readonly Xoshiro256StarStar _rng = new(seed);
    private readonly int _generators = Math.Max(1, System.Environment.ProcessorCount - 2);

    private CubePolicyNet _net = null!;
    private Adam _adam = null!;
    private long _round, _totalSamples, _totalSolves;
    private double _windowCe, _windowHuber, _windowAcc;
    private long _windowCount;
    private double _liveLoss = double.NaN, _liveAcc = double.NaN; // most-recent batch, for the live viewer

    public string Environment => CubeIds.Environment;

    public bool Resume(IModelStore store)
    {
        bool resumed;
        using (var existing = store.TryOpenRead(CubeIds.Environment, _ids.Policy))
        {
            if (existing is not null)
            {
                _net = CubePolicyNet.Load(existing);
                Log($"resumed cube policy net '{_ids.Policy}' from the model store");
                resumed = true;
            }
            else
            {
                _net = new CubePolicyNet(new Xoshiro256StarStar(seed ^ 0xDEADBEEF), hidden: width);
                Log($"initialized a fresh cube policy net '{_ids.Policy}' (trunk width {width})");
                resumed = false;
            }
        }
        using (var adamState = store.TryOpenRead(CubeIds.Environment, _ids.PolicyAdam))
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
        Log("warming the Kociemba tables…");
        CubeSolver.WarmUp();
        return resumed;
    }

    public long TrainChunk()
    {
        // One round: parallel Kociemba data-gen (the oracle, not the NN math, bounds throughput on CPU) → shuffle
        // → supervised batches. Window-mean loss accumulates across rounds until the runner calls Evaluate.
        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 64);
        var perWorker = new List<CubeOracle.LabeledState>[_generators];
        ulong roundBase = unchecked(seed + (ulong)(++_round) * 1_000_003UL);
        long solvesThisRound = 0;
        Parallel.For(0, _generators, worker =>
        {
            var workerRng = new Xoshiro256StarStar(unchecked(roundBase + 0x9E3779B97F4A7C15UL * (ulong)(worker + 1)));
            var local = new List<CubeOracle.LabeledState>(SamplesPerRound / _generators + 40);
            while (local.Count < SamplesPerRound / _generators)
            {
                var path = CubeOracle.LabelScramblePath(workerRng);
                if (path is null) continue;
                local.AddRange(path);
                Interlocked.Increment(ref solvesThisRound);
            }
            perWorker[worker] = local;
        });
        foreach (var local in perWorker) samples.AddRange(local);
        _totalSolves += solvesThisRound;
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
            new("solves", _totalSolves, "0"),
            new("samples", _totalSamples, "0"),
            new("ce", ce, "F4"),
            new("acc", acc, "F4"),
            new("huber", huber, "F5"),
        };
        var report = new StringBuilder();
        report.Append($"solves {_totalSolves:N0}, samples {_totalSamples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

        const int episodes = 20;
        foreach (int depth in EvalDepths)
        {
            var (greedy, search) = SolveCounts(depth, episodes);
            metrics.Add(new($"d{depth}_greedy", greedy / (double)episodes, "F3"));
            metrics.Add(new($"d{depth}_search", search / (double)episodes, "F3"));
            report.Append($"d{depth}: {greedy}/{episodes}g {search}/{episodes}s | ");
        }
        return new CampaignEval(metrics, report.ToString());
    }

    public void Checkpoint(IModelStore store)
    {
        store.Save(CubeIds.Environment, _ids.Policy, s => _net.Save(s));
        store.Save(CubeIds.Environment, _ids.PolicyAdam, s =>
        {
            using var writer = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
            AdamCheckpoint.Write(_adam, writer);
        });
    }

    /// <summary>`--eval-only`: per-depth eval report + the pre-registered M16 gate. No training, no checkpoint.</summary>
    public bool TryRunStandaloneEval(IModelStore store)
    {
        Log($"[eval] {Evaluate().Summary}");
        EvaluateGate();
        return true;
    }

    public void Dispose() { }

    /// <summary>Small budgets on purpose — this is a progress tracker; the full-budget check is the gate.</summary>
    private (int Greedy, int Search) SolveCounts(int depth, int episodes)
    {
        int greedy = 0, search = 0;
        for (int episode = 0; episode < episodes; episode++)
        {
            var evalRng = new Xoshiro256StarStar((ulong)(100_000 * depth + episode));
            var cube = new FaceletCube();
            cube.Apply(FaceletCube.ScrambleMoves(evalRng, depth, quarterTurnsOnly: true));
            if (cube.IsSolved) { greedy++; search++; continue; }

            if (CubePolicySearch.GreedyRollout(_net, cube).Solved) { greedy++; search++; }
            else if (CubePolicySearch.Solve(_net, cube, maxExpansions: 2_000).Solved) search++;
        }
        return (greedy, search);
    }

    /// <summary>The pre-registered M16 gate: ≥ 90% of 100 random scrambles (depths 1–10) solved (greedy or full search).</summary>
    private void EvaluateGate()
    {
        int totalSolved = 0, totalGreedy = 0;
        for (int depth = 1; depth <= 10; depth++)
        {
            int greedySolved = 0, searchSolved = 0;
            for (int episode = 0; episode < 10; episode++)
            {
                var gateRng = new Xoshiro256StarStar((ulong)(900_000 + 1_000 * depth + episode));
                var cube = new FaceletCube();
                cube.Apply(FaceletCube.ScrambleMoves(gateRng, depth, quarterTurnsOnly: true));
                if (cube.IsSolved) { greedySolved++; continue; }

                if (CubePolicySearch.GreedyRollout(_net, cube).Solved) greedySolved++;
                else if (CubePolicySearch.Solve(_net, cube).Solved) searchSolved++;
            }
            totalGreedy += greedySolved;
            totalSolved += greedySolved + searchSolved;
            Log($"  gate depth {depth}: {greedySolved}/10 greedy, +{searchSolved} with lookahead = {greedySolved + searchSolved}/10");
        }
        Log($"gate: {totalSolved}/100 solved ({totalSolved}%, target >= 90%); greedy alone {totalGreedy}%");
    }

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "cube-policy";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];
    NetworkMetrics INetworkTelemetrySource.Sample() => new(_totalSamples, 0, _liveLoss, _liveAcc, double.NaN);

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
}
