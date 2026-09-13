using System.Text;
using Microsoft.Extensions.Logging;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Cube imitation campaign (PLAN M16, `--game cube`) as an <see cref="ITrainingCampaign"/> driven by
/// <see cref="CampaignRunner"/> (PLAN M25): streams random scrambles through the Kociemba oracle (one solve = a
/// whole labeled solution path), trains the two-headed <see cref="CubePolicyNet"/> supervised (CE on the next
/// quarter-turn + Huber on distance-to-go), and reports per-depth greedy/search solve rates. Resumes the net +
/// Adam from `cube.policy` / `cube.policy-adam` (width-ladder ids via <see cref="CubeIds.ForWidth"/>).
/// </summary>
public sealed class CubeImitationCampaign(CubeImitationOptions options, ILogger? logger = null) : ITrainingCampaign, INetworkTelemetrySource
{
    private readonly Xoshiro256StarStar _growRng = new(options.Seed ^ 0x6C0FFEEUL); // dedicated stream for growth

    // Rooted at THIS run's configured width, so --grow starts where a plain run starts. It used to climb the
    // shared DqnGrowth ladder, which tops out at [128,128,128] — below any --width worth training — so growth
    // shrank the net it was meant to enlarge.
    private GrowthLadder Ladder => GrowthLadder.FromTrunk([options.Width, options.Width], steps: 3);
    private const int BatchSize = 256;
    private const int SamplesPerRound = 4096;
    private static readonly int[] EvalDepths = [2, 4, 6, 8, 10, 12, 16, 20];

    private readonly CubeIds.NetIds _ids = CubeIds.ForWidth(options.Width);
    private readonly Xoshiro256StarStar _rng = new(options.Seed);
    private readonly int _generators = Math.Max(1, System.Environment.ProcessorCount - 2);

    private CubePolicyNet _net = null!;
    private Adam _adam = null!;
    private long _round, _totalSamples, _totalSolves;
    private TrainWindow _window;
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
                var initRng = new Xoshiro256StarStar(options.Seed ^ 0xDEADBEEF);
                _net = new CubePolicyNet(initRng, Ladder.TrunkFor(0));
                Log(options.Grow
                    ? $"initialized a fresh GROWING cube policy net '{_ids.Policy}' (rung 0, trunk [{string.Join(",", Ladder.TrunkFor(0))}])"
                    : $"initialized a fresh cube policy net '{_ids.Policy}' (trunk width {options.Width})");
                resumed = false;
            }
        }
        _adam = AdamState.LoadOrInit(store, CubeIds.Environment, _ids.PolicyAdam, _net.Parameters(), options.LearningRate, Log);

        // Restore progress counters and the owner-thread RNG streams. Without this a restarted run replayed the
        // same scrambles from round zero AND reset PolicyGrowth's stage target (it keys off _totalSamples).
        // Absent sidecar = zeros, which is the pre-fix behaviour, so older stores still resume.
        var progress = CampaignProgressState.TryLoad(store, CubeIds.Environment, ProgressId, ProgressKind, rngCount: 2, Log);
        if (progress is not null)
        {
            _totalSamples = progress.Samples;
            _round = progress.Units;
            _totalSolves = (long)progress.LastMetric; // an exact counter below 2^53, carried in the metric slot
            CampaignProgressState.RestoreInto(progress.Rngs[0], _rng);
            CampaignProgressState.RestoreInto(progress.Rngs[1], _growRng);
            Log($"resumed progress: {_totalSamples:N0} samples over {_round:N0} rounds");
        }
        else if (resumed)
        {
            Log("no progress sidecar found — the net resumed but counters restart at zero (pre-M58 checkpoint)");
        }

        Log("warming the Kociemba tables…");
        CubeSolver.WarmUp();
        return resumed;
    }

    public long TrainChunk()
    {
        // One round: parallel Kociemba data-gen (the oracle, not the NN math, bounds throughput on CPU) → shuffle
        // → supervised batches. Window-mean loss accumulates across rounds until the runner calls Evaluate.
        // DeterministicParallel derives each generator's RNG from (roundBase, worker+1) — byte-identical to the old
        // hand-rolled `roundBase + φ·(worker+1)` seeding, and each returns its own list (no shared window/counter).
        ulong roundBase = unchecked(options.Seed + (ulong)(++_round) * 1_000_003UL);
        int per = SamplesPerRound / _generators;
        var perWorker = DeterministicParallel.Generate(_generators, roundBase, baseIndex: 1, (worker, rng) =>
        {
            var local = new List<CubeOracle.LabeledState>(per + 40);
            int solves = 0;
            while (local.Count < per)
            {
                var path = CubeOracle.LabelScramblePath(rng);
                if (path is null) continue;
                local.AddRange(path);
                solves++;
            }
            return (Samples: local, Solves: solves);
        }, parallel: true);

        var samples = new List<CubeOracle.LabeledState>(SamplesPerRound + 64);
        long solvesThisRound = 0;
        foreach (var w in perWorker) { samples.AddRange(w.Samples); solvesThisRound += w.Solves; }
        _totalSolves += solvesThisRound;
        CubePolicyTraining.Shuffle(samples, _rng);

        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = CubePolicyTraining.TrainStep(_net, _adam, samples, offset, BatchSize);
            _window.Add(ce, huber, acc);
            _totalSamples += BatchSize;
            _liveLoss = ce + huber;
            _liveAcc = acc;
        }
        if (PolicyGrowth.Maybe(_net, _totalSamples, options.Grow, options.GrowEvery, options.LearningRate, Ladder, _growRng, Log) is var g && g.HasValue)
            (_net, _adam) = (g.Value.Net, g.Value.Adam);
        return _totalSamples;
    }

    public CampaignEval Evaluate()
    {
        var (ce, huber, acc) = _window.MeanAndReset();

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
        AdamState.Save(store, CubeIds.Environment, _ids.PolicyAdam, _adam);
        CampaignProgressState.Save(store, CubeIds.Environment, ProgressId, ProgressKind,
            _totalSamples, _round, _totalSolves, _rng, _growRng);
    }

    // Namespaced per width rung, exactly as the net and Adam ids are, so one rung's progress never overwrites
    // another's (CubeIds.ForWidth).
    private string ProgressId => $"{_ids.Policy}-progress";
    private const string ProgressKind = "cube-imitation-progress";

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
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => RubiksCubeEnv.ActionLabels; // 12 quarter-turns
    // No running env, so the viewer forwards a FIXED scramble each frame — you watch the net's move preferences +
    // hidden activations for that one board evolve as it learns.
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo() => CubeViz.SampleIo(_net, ref _probeObs);
    float[][]? INetworkTelemetrySource.SampleActivations() => CubeViz.SampleActivations(_net, ref _probeObs);
    private float[]? _probeObs;

    private void Log(string message)
    {
        if (logger is null) Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");
        else logger.LogInformation("{Message}", message);
    }
}
