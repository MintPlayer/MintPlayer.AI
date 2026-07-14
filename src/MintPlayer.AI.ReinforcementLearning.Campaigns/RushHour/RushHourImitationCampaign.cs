using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Rush Hour imitation campaign (PLAN M16, the Lab's default `--game rushhour`) as an
/// <see cref="ITrainingCampaign"/> driven by <see cref="CampaignRunner"/> (PLAN M25). Each round streams a random
/// configuration through the BFS oracle (exact optimal action + distance-to-goal for EVERY reachable state), then
/// trains the two-headed <see cref="RushHourPolicyNet"/> supervised (soft CE over all optimal moves + Huber on
/// distance) on a DAgger mix of on-policy and stratified samples. Eval tracks the held-out official ThinkFun cards
/// (1, 38, 39, 40) with reactive play and policy-guided A*, plus a 30-puzzle random hold-out set.
/// </summary>
public sealed class RushHourImitationCampaign(ulong seed, float learningRate, bool grow = false, int growEvery = 2048) : ITrainingCampaign, INetworkTelemetrySource
{
    private readonly Xoshiro256StarStar _growRng = new(seed ^ 0x6C0FFEEUL); // dedicated stream for growth
    private const int BatchSize = 256;
    private const int SamplesPerConfig = 1024;
    private const int MaxStatesPerConfig = 150_000;

    private readonly Xoshiro256StarStar _rng = new(seed);

    private RushHourPolicyNet _net = null!;
    private Adam _adam = null!;
    private long _totalSamples;
    private int _totalConfigs;
    private TrainWindow _window;
    private long _windowOnPolicy, _windowDrawn;
    private double _liveLoss = double.NaN, _liveAcc = double.NaN; // most-recent batch, for the live viewer

    // Held-out official ThinkFun cards (never produced by the random generator).
    private readonly (string Name, RushHourPuzzle Puzzle, int Optimal)[] _cards =
    [
        ("level1", new RushHourPuzzle([
            new Vehicle(2, 1, 2, true), new Vehicle(0, 0, 2, true), new Vehicle(0, 5, 3, false),
            new Vehicle(1, 0, 3, false), new Vehicle(1, 3, 3, false), new Vehicle(4, 0, 2, false),
            new Vehicle(4, 4, 2, true), new Vehicle(5, 2, 3, true)]), 16),
        ("card38", new RushHourPuzzle([
            new Vehicle(2, 0, 2, true), new Vehicle(0, 0, 2, false), new Vehicle(0, 3, 3, true),
            new Vehicle(1, 1, 2, true), new Vehicle(1, 3, 2, false), new Vehicle(2, 2, 2, false),
            new Vehicle(2, 5, 3, false), new Vehicle(3, 3, 2, true), new Vehicle(4, 2, 2, false),
            new Vehicle(4, 3, 2, true), new Vehicle(5, 3, 3, true)]), 77),
        ("card39", new RushHourPuzzle([
            new Vehicle(2, 0, 2, true), new Vehicle(0, 2, 2, false), new Vehicle(0, 3, 3, true),
            new Vehicle(1, 3, 2, false), new Vehicle(2, 2, 2, false), new Vehicle(2, 5, 3, false),
            new Vehicle(3, 0, 2, true), new Vehicle(3, 3, 2, true), new Vehicle(4, 0, 2, false),
            new Vehicle(4, 1, 2, false), new Vehicle(4, 2, 2, true), new Vehicle(5, 2, 2, true)]), 82),
        ("card40", new RushHourPuzzle([
            new Vehicle(2, 3, 2, true), new Vehicle(0, 0, 3, false), new Vehicle(0, 1, 2, true),
            new Vehicle(0, 4, 2, false), new Vehicle(1, 1, 2, false), new Vehicle(1, 2, 2, false),
            new Vehicle(1, 5, 3, false), new Vehicle(3, 0, 3, true), new Vehicle(3, 3, 2, false),
            new Vehicle(4, 2, 2, false), new Vehicle(4, 4, 2, true), new Vehicle(5, 0, 2, true),
            new Vehicle(5, 3, 2, true)]), 81),
    ];
    private readonly IReadOnlyList<RushHourPuzzle> _randomEval = RushHourGenerator.Generate(
        seed: 777, count: 30, minOptimal: 4, maxOptimal: 20, minVehicles: 3, maxVehicles: 10, varyRedLength: true);

    public string Environment => "rushhour";

    public bool Resume(IModelStore store)
    {
        bool resumed;
        using (var existing = store.TryOpenRead("rushhour", "policy"))
        {
            if (existing is not null)
            {
                _net = RushHourPolicyNet.Load(existing);
                Log("resumed policy net from the model store");
                resumed = true;
            }
            else
            {
                var initRng = new Xoshiro256StarStar(seed ^ 0xDEADBEEF);
                _net = grow ? new RushHourPolicyNet(initRng, DqnGrowth.Start) : new RushHourPolicyNet(initRng);
                Log(grow ? $"initialized a fresh GROWING policy net (start trunk [{string.Join(",", DqnGrowth.Start)}])"
                         : "initialized a fresh policy net");
                resumed = false;
            }
        }
        // Restore Adam's moment estimates when continuing a campaign — without them, resumed
        // training spends its first minutes re-estimating gradient statistics from zero.
        _adam = AdamState.LoadOrInit(store, "rushhour", "policy-adam", _net.Parameters(), learningRate, Log);
        return resumed;
    }

    public long TrainChunk()
    {
        // One round = one random config labeled by the BFS oracle, then supervised batches over a DAgger mix.
        // The runner calls this repeatedly; window-mean loss accumulates across rounds until Evaluate.
        RushHourPuzzle? puzzle = null;
        List<RushHourOracle.LabeledState>? labeled = null;
        while (labeled is null || labeled.Count < 50)
        {
            puzzle = RushHourGenerator.RandomLayout(_rng, minVehicles: 4, maxVehicles: 11, varyRedLength: true);
            if (puzzle is null) continue;
            labeled = RushHourOracle.LabelReachableStates(puzzle, MaxStatesPerConfig);
        }

        _totalConfigs++;
        var samples = BuildSamples(puzzle!, labeled);
        Shuffle(samples, _rng);

        for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
        {
            var (ce, huber, acc) = TrainStep(samples, offset, BatchSize);
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
        if (_windowDrawn > 0)
            Log($"[mix] on-policy share this window: {_windowOnPolicy / (double)_windowDrawn:P1}");
        _windowOnPolicy = _windowDrawn = 0;

        var metrics = new List<CampaignMetric>
        {
            new("configs", _totalConfigs, "0"),
            new("samples", _totalSamples, "0"),
            new("ce", ce, "F4"),
            new("acc", acc, "F4"),
            new("huber", huber, "F5"),
        };
        var report = new StringBuilder();
        report.Append($"configs {_totalConfigs:N0}, samples {_totalSamples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

        foreach (var (name, puzzle, optimal) in _cards)
        {
            var search = RushHourPolicySearch.Solve(_net, puzzle, maxExpansions: 150_000);
            if (name == "level1")
            {
                var greedy = RushHourPolicySearch.GreedyRollout(_net, puzzle, Math.Max(60, 2 * optimal));
                metrics.Add(new("l1_greedy", greedy.Solved ? greedy.Actions.Count : -1, "0"));
                report.Append($"{name}: greedy {(greedy.Solved ? greedy.Actions.Count + "mv" : "fail")}, ");
            }
            metrics.Add(new($"{ColumnPrefix(name)}_search", search.Solved ? search.Actions.Length : -1, "0"));
            metrics.Add(new($"{ColumnPrefix(name)}_exp", search.Expansions, "0"));
            report.Append($"{name} search {(search.Solved ? $"{search.Actions.Length}mv/{search.Expansions}exp" : $"FAIL/{search.Expansions}exp")} (opt {optimal}) | ");
        }

        int greedySolved = 0, searchSolved = 0;
        foreach (var puzzle in _randomEval)
        {
            if (RushHourPolicySearch.GreedyRollout(_net, puzzle, Math.Max(60, 2 * puzzle.OptimalMoves)).Solved) greedySolved++;
            if (RushHourPolicySearch.Solve(_net, puzzle, 50_000).Solved) searchSolved++;
        }
        metrics.Add(new("rand_greedy", greedySolved / (double)_randomEval.Count, "F3"));
        metrics.Add(new("rand_search", searchSolved / (double)_randomEval.Count, "F3"));
        report.Append($"random30: greedy {greedySolved}/30, search {searchSolved}/30");

        return new CampaignEval(metrics, report.ToString());
    }

    public void Checkpoint(IModelStore store)
    {
        store.Save("rushhour", "policy", s => _net.Save(s));
        AdamState.Save(store, "rushhour", "policy-adam", _adam);
    }

    public void Dispose() { }

    /// <summary>The CSV column stem for a held-out card: "l1" for level1, otherwise the bare "card##" → "c##".</summary>
    private static string ColumnPrefix(string name)
        => name == "level1" ? "l1" : name.Replace("card", "c");

    private (double Ce, double Huber, double Acc) TrainStep(List<Sample> samples, int offset, int batch)
    {
        var obs = new float[batch * RushHourBoard.ObservationSize];
        var maskOffsets = new float[batch * RushHourBoard.ActionCount];
        var weights = new float[batch * RushHourBoard.ActionCount];
        var targets = new float[batch];
        for (int i = 0; i < batch; i++)
        {
            var s = samples[offset + i];
            s.Obs.CopyTo(obs.AsSpan(i * RushHourBoard.ObservationSize));
            s.MaskOffsets.CopyTo(maskOffsets.AsSpan(i * RushHourBoard.ActionCount));
            // Soft target: uniform over ALL optimal actions — a single arbitrary label
            // penalizes the other equally-good moves and flattens the policy.
            float w = 1f / System.Numerics.BitOperations.PopCount(s.LabelMask);
            for (uint bits = s.LabelMask; bits != 0; bits &= bits - 1)
                weights[i * RushHourBoard.ActionCount + System.Numerics.BitOperations.TrailingZeroCount(bits)] = w;
            targets[i] = s.Distance / RushHourPolicyNet.DistanceScale;
        }

        var (logits, value) = _net.Forward(new Tensor(obs, batch, RushHourBoard.ObservationSize));
        var logProbs = logits.Add(new Tensor(maskOffsets, batch, RushHourBoard.ActionCount)).LogSoftmax();
        var ce = logProbs.Mul(new Tensor(weights, batch, RushHourBoard.ActionCount)).Sum().MulScalar(-1f / batch);
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
            for (int a = 1; a < RushHourBoard.ActionCount; a++)
                if (logProbs.Data[i * RushHourBoard.ActionCount + a] > logProbs.Data[i * RushHourBoard.ActionCount + argmax])
                    argmax = a;
            if ((samples[offset + i].LabelMask >> argmax & 1) != 0) correct++; // any optimal action counts
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    // DAgger-style mix: up to half the budget is the ON-POLICY state distribution — the
    // states the current net actually visits when it plays this config. Its loops and
    // detours are exactly what stratified oracle sampling never shows it, and because the
    // oracle labeled the WHOLE reachable graph, relabeling a visited state is a dictionary
    // lookup. The remainder stays stratified-by-distance for coverage.
    private List<Sample> BuildSamples(RushHourPuzzle puzzle, List<RushHourOracle.LabeledState> labeled)
    {
        var byKey = new Dictionary<ulong, RushHourOracle.LabeledState>(labeled.Count);
        foreach (var state in labeled) byKey[RushHourSolver.Encode(state.Positions)] = state;

        // Roll out from the canonical start plus a few deep states — depths that exist in
        // every mid-size graph even though random START generation can't produce them.
        var deep = labeled.OrderByDescending(s => s.DistanceToGoal)
            .Take(Math.Max(1, labeled.Count / 4)).ToArray();
        // Eight rollouts per config: solved rollouts visit only ~distance states each, so
        // fewer starts leave the on-policy pool nearly empty (~7% share observed with 4).
        var rolloutStarts = new List<int[]> { RushHourBoard.InitialPositions(puzzle) };
        for (int i = 0; i < 7; i++) rolloutStarts.Add(deep[_rng.NextInt(deep.Length)].Positions);

        var pool = new List<RushHourOracle.LabeledState>();
        foreach (var rolloutStart in rolloutStarts)
        {
            int d = byKey.TryGetValue(RushHourSolver.Encode(rolloutStart), out var s0) ? s0.DistanceToGoal : 20;
            var visited = new List<int[]>();
            var (solved, _) = RushHourPolicySearch.GreedyRolloutFrom(_net, puzzle, rolloutStart, Math.Max(60, 2 * d), visited);
            foreach (var position in visited)
                if (byKey.TryGetValue(RushHourSolver.Encode(position), out var label))
                {
                    pool.Add(label);
                    if (!solved) pool.Add(label); // failed rollouts ARE the distribution gap — double weight
                }
        }

        Shuffle(pool, _rng);
        var samples = new List<Sample>(SamplesPerConfig);
        foreach (var state in pool.Take(SamplesPerConfig / 2))
            samples.Add(MakeSample(puzzle, state));
        _windowOnPolicy += samples.Count;
        samples.AddRange(StratifiedSample(puzzle, labeled, SamplesPerConfig - samples.Count, _rng));
        _windowDrawn += samples.Count;
        return samples;
    }

    private static List<Sample> StratifiedSample(RushHourPuzzle puzzle, List<RushHourOracle.LabeledState> labeled, int budget, Xoshiro256StarStar rng)
    {
        var byDistance = labeled.GroupBy(s => s.DistanceToGoal).ToList();
        int perBucket = Math.Max(8, budget / byDistance.Count);
        var samples = new List<Sample>(Math.Min(budget + perBucket, labeled.Count));

        foreach (var bucket in byDistance)
        {
            var states = bucket.ToArray();
            Shuffle(states, rng);
            foreach (var state in states.Take(perBucket))
                samples.Add(MakeSample(puzzle, state));
        }
        return samples;
    }

    private static Sample MakeSample(RushHourPuzzle puzzle, RushHourOracle.LabeledState state)
    {
        var obs = new float[RushHourBoard.ObservationSize];
        RushHourBoard.WriteObservation(puzzle, state.Positions, obs);
        var mask = RushHourBoard.ActionMask(puzzle, state.Positions);
        var offsets = new float[RushHourBoard.ActionCount];
        for (int a = 0; a < offsets.Length; a++)
            if (!mask[a]) offsets[a] = -1e9f;
        return new Sample(obs, offsets, state.OptimalActionsMask, state.DistanceToGoal);
    }

    private static void Shuffle<T>(IList<T> list, Xoshiro256StarStar rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");

    // --- Live telemetry (INetworkTelemetrySource): read-only; a viewer samples the current net as it trains. ---
    string INetworkTelemetrySource.NetKind => "rushhour-policy";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => ReferenceEquals(_net, null) ? null : [.. _net.Parameters()];
    NetworkMetrics INetworkTelemetrySource.Sample() => new(_totalSamples, 0, _liveLoss, _liveAcc, double.NaN);
    IReadOnlyList<string>? INetworkTelemetrySource.OutputLabels => RushHourBoard.ActionLabels; // 32 vehicle×dir moves
    // No running env, so the viewer forwards ONE fixed puzzle (the level-1 card's start) each frame — you watch the
    // net's move preferences + hidden activations for that board evolve. Read-only forward; CPU (imitation has no GPU).
    private float[]? _probeObs;
    private float[] ProbeObs()
    {
        if (_probeObs is null)
        {
            var puzzle = _cards[0].Puzzle;
            var obs = new float[RushHourBoard.ObservationSize];
            RushHourBoard.WriteObservation(puzzle, RushHourBoard.InitialPositions(puzzle), obs);
            _probeObs = obs;
        }
        return _probeObs;
    }
    (float[] Input, float[] Output)? INetworkTelemetrySource.SampleIo()
    {
        if (ReferenceEquals(_net, null)) return null;
        try { var obs = ProbeObs(); var (logits, _) = _net.Forward(new Tensor((float[])obs.Clone(), 1, obs.Length)); return ((float[])obs.Clone(), [.. logits.Data]); }
        catch { return null; }
    }
    float[][]? INetworkTelemetrySource.SampleActivations()
    {
        if (ReferenceEquals(_net, null)) return null;
        try { var obs = ProbeObs(); return _net.LayerActivations(new Tensor((float[])obs.Clone(), 1, obs.Length)); }
        catch { return null; }
    }

    private sealed record Sample(float[] Obs, float[] MaskOffsets, uint LabelMask, float Distance);
}
