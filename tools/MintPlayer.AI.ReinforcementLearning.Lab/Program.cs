using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

// MintPlayer.AI.ReinforcementLearning.Lab — long-running imitation-learning campaign for Rush Hour.
// Streams random configurations through the BFS oracle (exact optimal action +
// distance-to-goal for EVERY reachable state), trains the two-headed policy/value
// net supervised, checkpoints to the model store every eval, and tracks held-out
// official ThinkFun cards (1, 38, 39, 40) with both reactive play and policy-guided A*.
//
// Usage: MintPlayer.AI.ReinforcementLearning.Lab [--hours H] [--data DIR] [--seed S] [--lr LR] [--eval-only]

double hours = 9;
string dataDir = "data";
ulong seed = 1;
float learningRate = 3e-4f;
bool evalOnly = false;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--hours" && i + 1 < args.Length) hours = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
    else if (args[i] == "--data" && i + 1 < args.Length) dataDir = args[++i];
    else if (args[i] == "--seed" && i + 1 < args.Length) seed = ulong.Parse(args[++i]);
    else if (args[i] == "--lr" && i + 1 < args.Length) learningRate = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
    else if (args[i] == "--eval-only") evalOnly = true;
}

const int BatchSize = 256;
const int SamplesPerConfig = 1024;
const int MaxStatesPerConfig = 150_000;
var evalEvery = TimeSpan.FromMinutes(10);

var store = new FileModelStore(dataDir);
string logPath = Path.Combine(store.RootDirectory, "logs");
Directory.CreateDirectory(logPath);
string csvPath = Path.Combine(logPath, "imitation.csv");
if (!File.Exists(csvPath))
    File.AppendAllText(csvPath,
        "utc,configs,samples,ce,acc,huber," +
        "l1_greedy,l1_search,l1_exp,c38_search,c38_exp,c39_search,c39_exp,c40_search,c40_exp,rand_greedy,rand_search\n");

var rng = new Xoshiro256StarStar(seed);
RushHourPolicyNet net;
using (var existing = store.TryOpenRead("rushhour", "policy"))
{
    if (existing is not null)
    {
        net = RushHourPolicyNet.Load(existing);
        Log("resumed policy net from the model store");
    }
    else
    {
        net = new RushHourPolicyNet(new Xoshiro256StarStar(seed ^ 0xDEADBEEF));
        Log("initialized a fresh policy net");
    }
}
// Restore Adam's moment estimates when continuing a campaign — without them, resumed
// training spends its first minutes re-estimating gradient statistics from zero.
Adam adam;
using (var adamState = store.TryOpenRead("rushhour", "policy-adam"))
{
    if (adamState is not null)
    {
        using var reader = new BinaryReader(adamState, System.Text.Encoding.UTF8, leaveOpen: true);
        adam = AdamCheckpoint.Read(net.Parameters(), reader);
        adam.LearningRate = learningRate; // CLI overrides the stored schedule position
        Log($"resumed Adam state (lr set to {learningRate:E1})");
    }
    else
    {
        adam = new Adam(net.Parameters(), learningRate);
    }
}

// Held-out official ThinkFun cards (never produced by the random generator).
var cards = new (string Name, RushHourPuzzle Puzzle, int Optimal)[]
{
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
};
var randomEval = RushHourGenerator.Generate(seed: 777, count: 30, minOptimal: 4, maxOptimal: 20,
    minVehicles: 3, maxVehicles: 10, varyRedLength: true);

if (evalOnly)
{
    Evaluate(0, 0, 0, 0, 0);
    return;
}

var deadline = DateTime.UtcNow.AddHours(hours);
var nextEval = DateTime.UtcNow + TimeSpan.FromMinutes(2); // early first eval as a baseline
long totalSamples = 0;
int totalConfigs = 0;
double windowCe = 0, windowHuber = 0, windowAcc = 0;
long windowCount = 0;
long windowOnPolicy = 0, windowDrawn = 0;

Log($"training until {deadline:u} (~{hours:F1} h), data dir: {store.RootDirectory}");

while (DateTime.UtcNow < deadline)
{
    var puzzle = RushHourGenerator.RandomLayout(rng, minVehicles: 4, maxVehicles: 11, varyRedLength: true);
    if (puzzle is null) continue;
    var labeled = RushHourOracle.LabelReachableStates(puzzle, MaxStatesPerConfig);
    if (labeled is null || labeled.Count < 50) continue;

    totalConfigs++;
    var samples = BuildSamples(puzzle, labeled);
    Shuffle(samples, rng);

    for (int offset = 0; offset + BatchSize <= samples.Count; offset += BatchSize)
    {
        var (ce, huber, acc) = TrainStep(samples, offset, BatchSize);
        windowCe += ce;
        windowHuber += huber;
        windowAcc += acc;
        windowCount++;
        totalSamples += BatchSize;
    }

    if (DateTime.UtcNow >= nextEval)
    {
        double meanCe = windowCount > 0 ? windowCe / windowCount : 0;
        double meanHuber = windowCount > 0 ? windowHuber / windowCount : 0;
        double meanAcc = windowCount > 0 ? windowAcc / windowCount : 0;
        windowCe = windowHuber = windowAcc = 0;
        windowCount = 0;
        if (windowDrawn > 0)
            Log($"[mix] on-policy share this window: {windowOnPolicy / (double)windowDrawn:P1}");
        windowOnPolicy = windowDrawn = 0;
        Evaluate(totalConfigs, totalSamples, meanCe, meanAcc, meanHuber);
        nextEval = DateTime.UtcNow + evalEvery;
    }
}

Evaluate(totalConfigs, totalSamples,
    windowCount > 0 ? windowCe / windowCount : 0,
    windowCount > 0 ? windowAcc / windowCount : 0,
    windowCount > 0 ? windowHuber / windowCount : 0);
Log("time budget reached — final checkpoint saved.");
return;

(double Ce, double Huber, double Acc) TrainStep(List<Sample> samples, int offset, int batch)
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

    var (logits, value) = net.Forward(new Tensor(obs, batch, RushHourBoard.ObservationSize));
    var logProbs = logits.Add(new Tensor(maskOffsets, batch, RushHourBoard.ActionCount)).LogSoftmax();
    var ce = logProbs.Mul(new Tensor(weights, batch, RushHourBoard.ActionCount)).Sum().MulScalar(-1f / batch);
    var huber = value.Reshape(batch).HuberLoss(new Tensor(targets, batch));
    var loss = ce.Add(huber);

    adam.ZeroGrad();
    loss.Backward();
    adam.ClipGradNorm(5f);
    adam.Step();

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

void Evaluate(int configs, long samples, double ce, double acc, double huber)
{
    var cells = new List<string> { $"{DateTime.UtcNow:u}", $"{configs}", $"{samples}", $"{ce:F4}", $"{acc:F4}", $"{huber:F5}" };
    var report = new System.Text.StringBuilder();
    report.Append($"[eval] configs {configs:N0}, samples {samples:N0}, CE {ce:F3}, acc {acc:P1}, value {huber:F4} | ");

    foreach (var (name, puzzle, optimal) in cards)
    {
        var greedy = RushHourPolicySearch.GreedyRollout(net, puzzle, Math.Max(60, 2 * optimal));
        var search = RushHourPolicySearch.Solve(net, puzzle, maxExpansions: 150_000);
        if (name == "level1")
        {
            cells.Add(greedy.Solved ? $"{greedy.Actions.Count}" : "-1");
            report.Append($"{name}: greedy {(greedy.Solved ? greedy.Actions.Count + "mv" : "fail")}, ");
        }
        cells.Add(search.Solved ? $"{search.Actions.Length}" : "-1");
        cells.Add($"{search.Expansions}");
        report.Append($"{name} search {(search.Solved ? $"{search.Actions.Length}mv/{search.Expansions}exp" : $"FAIL/{search.Expansions}exp")} (opt {optimal}) | ");
    }

    int greedySolved = 0, searchSolved = 0;
    foreach (var puzzle in randomEval)
    {
        if (RushHourPolicySearch.GreedyRollout(net, puzzle, Math.Max(60, 2 * puzzle.OptimalMoves)).Solved) greedySolved++;
        if (RushHourPolicySearch.Solve(net, puzzle, 50_000).Solved) searchSolved++;
    }
    cells.Add($"{greedySolved / (double)randomEval.Count:F3}");
    cells.Add($"{searchSolved / (double)randomEval.Count:F3}");
    report.Append($"random30: greedy {greedySolved}/30, search {searchSolved}/30");

    Log(report.ToString());
    File.AppendAllText(csvPath, string.Join(',', cells) + "\n");
    store.Save("rushhour", "policy", s => net.Save(s));
    store.Save("rushhour", "policy-adam", s =>
    {
        using var writer = new BinaryWriter(s, System.Text.Encoding.UTF8, leaveOpen: true);
        AdamCheckpoint.Write(adam, writer);
    });
}

// DAgger-style mix: up to half the budget is the ON-POLICY state distribution — the
// states the current net actually visits when it plays this config. Its loops and
// detours are exactly what stratified oracle sampling never shows it, and because the
// oracle labeled the WHOLE reachable graph, relabeling a visited state is a dictionary
// lookup. The remainder stays stratified-by-distance for coverage.
List<Sample> BuildSamples(RushHourPuzzle puzzle, List<RushHourOracle.LabeledState> labeled)
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
    for (int i = 0; i < 7; i++) rolloutStarts.Add(deep[rng.NextInt(deep.Length)].Positions);

    var pool = new List<RushHourOracle.LabeledState>();
    foreach (var rolloutStart in rolloutStarts)
    {
        int d = byKey.TryGetValue(RushHourSolver.Encode(rolloutStart), out var s0) ? s0.DistanceToGoal : 20;
        var visited = new List<int[]>();
        var (solved, _) = RushHourPolicySearch.GreedyRolloutFrom(net, puzzle, rolloutStart, Math.Max(60, 2 * d), visited);
        foreach (var position in visited)
            if (byKey.TryGetValue(RushHourSolver.Encode(position), out var label))
            {
                pool.Add(label);
                if (!solved) pool.Add(label); // failed rollouts ARE the distribution gap — double weight
            }
    }

    Shuffle(pool, rng);
    var samples = new List<Sample>(SamplesPerConfig);
    foreach (var state in pool.Take(SamplesPerConfig / 2))
        samples.Add(MakeSample(puzzle, state));
    windowOnPolicy += samples.Count;
    samples.AddRange(StratifiedSample(puzzle, labeled, SamplesPerConfig - samples.Count, rng));
    windowDrawn += samples.Count;
    return samples;
}

static List<Sample> StratifiedSample(RushHourPuzzle puzzle, List<RushHourOracle.LabeledState> labeled, int budget, Xoshiro256StarStar rng)
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

static Sample MakeSample(RushHourPuzzle puzzle, RushHourOracle.LabeledState state)
{
    var obs = new float[RushHourBoard.ObservationSize];
    RushHourBoard.WriteObservation(puzzle, state.Positions, obs);
    var mask = RushHourBoard.ActionMask(puzzle, state.Positions);
    var offsets = new float[RushHourBoard.ActionCount];
    for (int a = 0; a < offsets.Length; a++)
        if (!mask[a]) offsets[a] = -1e9f;
    return new Sample(obs, offsets, state.OptimalActionsMask, state.DistanceToGoal);
}

static void Shuffle<T>(IList<T> list, Xoshiro256StarStar rng)
{
    for (int i = list.Count - 1; i > 0; i--)
    {
        int j = rng.NextInt(i + 1);
        (list[i], list[j]) = (list[j], list[i]);
    }
}

static void Log(string message)
    => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");

internal sealed record Sample(float[] Obs, float[] MaskOffsets, uint LabelMask, float Distance);
