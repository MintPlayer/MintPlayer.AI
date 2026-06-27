using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// FruitCake <b>planner-distillation</b> campaign (`--game fruitcake --distill`, PRD FRUITCAKE_IMPROVE lever F6)
/// as an <see cref="ITrainingCampaign"/> on <see cref="CampaignRunner"/>. The reactive DQN plateaued at pineapple
/// because it cannot plan; the depth-N <see cref="FruitCakeSearch"/> (with the shipped DQN's max-Q as its leaf)
/// plans and reaches watermelons, but is too slow to run deep at serve time. This campaign uses that search as a
/// <b>teacher / oracle</b> — it self-plays games choosing each column by the search, labels every drop with the
/// search's choice + the realized (discounted) remaining score, and trains the two-headed
/// <see cref="FruitCakePolicyNet"/> supervised (CE on the column + Huber on the return). The student absorbs the
/// planner's strength: its policy head plays strongly in a single forward pass, and its value head is a
/// planning-aware leaf for serve-time search.
///
/// <para>Data-gen runs the (expensive) search per drop, so it's the throughput bottleneck — parallelize it across
/// workers with the compute backend forced single-threaded (the games are the parallelism; see
/// docs/OPTIMIZATIONS.md). Behaviour is the teacher with a small <see cref="_daggerProb"/> chance of a random
/// column (DAgger-lite: the label is still the teacher's choice, so the net also learns to recover the boards a
/// pure-teacher trajectory never visits). Persists the net + Adam under `fruitcake`/`policy`(`-adam`), leaving the
/// shipped `fruitcake`/`dqn` (the teacher) untouched.</para>
/// </summary>
internal sealed class FruitCakeImitationCampaign(
    ulong seed, float learningRate, int hidden, double gamma,
    int teacherDepth, int teacherTopK, int teacherTopK2, double daggerProb, int epochs = 2)
    : ITrainingCampaign
{
    private const string PolicyId = "policy";
    private const string PolicyAdamId = "policy-adam";
    private const string TeacherNetId = "dqn"; // the shipped DuelingQNet, used as the search teacher's leaf
    private const int BatchSize = 256;
    private const int DropsPerRound = 4096;   // labeled drops generated per TrainChunk (each = one full search — expensive)
    private const int BufferCap = 150_000;    // rolling sample buffer: the teacher is costly, so reuse data across rounds

    private readonly double _daggerProb = daggerProb;
    private readonly int _epochs = Math.Max(1, epochs); // supervised passes over the buffer per round (data reuse)
    private readonly List<Sample> _buffer = new(BufferCap + DropsPerRound);
    private readonly Xoshiro256StarStar _rng = new(seed);
    private readonly int _generators = Math.Max(1, System.Environment.ProcessorCount - 2);

    private DuelingQNet _teacher = null!;
    private FruitCakePolicyNet _net = null!;
    private Adam _adam = null!;
    private long _round, _totalDrops, _totalGenerated, _totalGames;
    private double _windowCe, _windowHuber, _windowAcc;
    private long _windowCount;

    public string Environment => "fruitcake";

    public bool Resume(IModelStore store)
    {
        // Data-gen (search + physics) is the bottleneck and is parallelized across games, so the games ARE the
        // parallelism — force the backend single-threaded so the tiny net GEMMs don't oversubscribe the cores.
        Backend.Current = new ManagedBackend(maxDegreeOfParallelism: 1);

        using (var t = store.TryOpenRead(Environment, TeacherNetId))
        {
            _teacher = t is not null
                ? DuelingQNetCheckpoint.Load(t)
                : throw new FileNotFoundException(
                    $"planner distillation needs the shipped teacher net '{Environment}.{TeacherNetId}' to label data");
        }

        bool resumed;
        using (var existing = store.TryOpenRead(Environment, PolicyId))
        {
            if (existing is not null)
            {
                _net = FruitCakePolicyNet.Load(existing);
                Log($"resumed FruitCake policy net '{PolicyId}'");
                resumed = true;
            }
            else
            {
                _net = new FruitCakePolicyNet(new Xoshiro256StarStar(seed ^ 0xF00DCAFEUL), hidden);
                Log($"initialized a fresh FruitCake policy net '{PolicyId}' (trunk width {hidden})");
                resumed = false;
            }
        }
        using (var adamState = store.TryOpenRead(Environment, PolicyAdamId))
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
        Log($"teacher = depth-{teacherDepth} search (topK {teacherTopK}, topK2 {teacherTopK2}, DQN-maxQ leaf), dagger {_daggerProb:P0}, {_epochs} pass(es)/round, buffer cap {BufferCap:N0}");
        return resumed;
    }

    public long TrainChunk()
    {
        // One round: parallel teacher self-play (the search, not the NN math, bounds throughput) → fold into the
        // rolling buffer → a few supervised passes over the whole buffer. The teacher is expensive (a full search
        // per labeled drop), so a sample is reused across rounds instead of trained once-and-discarded.
        var fresh = new List<Sample>(DropsPerRound + 512);
        var perWorker = new List<Sample>[_generators];
        ulong roundBase = unchecked(seed + (ulong)(++_round) * 1_000_003UL);
        long gamesThisRound = 0;
        Parallel.For(0, _generators, worker =>
        {
            var workerRng = new Xoshiro256StarStar(unchecked(roundBase + 0x9E3779B97F4A7C15UL * (ulong)(worker + 1)));
            var search = MakeTeacher();
            var env = new FruitCakeEnv();
            var local = new List<Sample>(DropsPerRound / _generators + 64);
            while (local.Count < DropsPerRound / _generators)
            {
                PlayGame(env, search, workerRng, local);
                Interlocked.Increment(ref gamesThisRound);
            }
            perWorker[worker] = local;
        });
        foreach (var local in perWorker) fresh.AddRange(local);
        _totalGames += gamesThisRound;
        _totalGenerated += fresh.Count;

        _buffer.AddRange(fresh);
        if (_buffer.Count > BufferCap)
            _buffer.RemoveRange(0, _buffer.Count - BufferCap); // evict the oldest

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            Shuffle(_buffer, _rng);
            for (int offset = 0; offset + BatchSize <= _buffer.Count; offset += BatchSize)
            {
                var (ce, huber, acc) = TrainStep(_buffer, offset, BatchSize);
                _windowCe += ce;
                _windowHuber += huber;
                _windowAcc += acc;
                _windowCount++;
                _totalDrops += BatchSize;
            }
        }
        return _totalGenerated;
    }

    public bool IsComplete => false; // score-maximizing: stops on the runner's time budget

    public CampaignEval Evaluate()
    {
        double ce = _windowCount > 0 ? _windowCe / _windowCount : 0;
        double acc = _windowCount > 0 ? _windowAcc / _windowCount : 0;
        double huber = _windowCount > 0 ? _windowHuber / _windowCount : 0;
        _windowCe = _windowHuber = _windowAcc = 0;
        _windowCount = 0;

        // Greedy policy-head play (no search): the cheap-serving capability the distillation is meant to lift.
        var (score, maxTier, melon) = EvalPolicy(20);
        var metrics = new List<CampaignMetric>
        {
            new("gen", _totalGenerated, "0"),
            new("trained", _totalDrops, "0"),
            new("buffer", _buffer.Count, "0"),
            new("games", _totalGames, "0"),
            new("ce", ce, "F4"),
            new("acc", acc, "F4"),
            new("huber", huber, "F5"),
            new("polScore", score, "F1"),
            new("polMaxTier", maxTier, "F2"),
            new("polMelon", melon, "0"),
        };
        return new CampaignEval(metrics,
            $"gen {_totalGenerated:N0} | trained {_totalDrops:N0} (buf {_buffer.Count:N0}) | games {_totalGames:N0} | CE {ce:F3} acc {acc:P1} value {huber:F4} | policy: score {score:F1} maxTier {maxTier:F2} melon {melon}/20");
    }

    public void Checkpoint(IModelStore store)
    {
        // Imitation loss is stable (not noisy like DQN eval), so save the latest net every checkpoint — both the
        // deployable policy net and the Adam state for a lossless resume.
        store.Save(Environment, PolicyId, s => _net.Save(s));
        store.Save(Environment, PolicyAdamId, s =>
        {
            using var writer = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
            AdamCheckpoint.Write(_adam, writer);
        });
    }

    public bool TryRunStandaloneEval(IModelStore store)
    {
        var (score, maxTier, melon) = EvalPolicy(100);
        Log($"[eval] policy-head greedy over 100 games: score {score:F1}, maxTier {maxTier:F2}, watermelon {melon}/100");
        return true;
    }

    public void Dispose() { }

    // ── teacher / data generation ────────────────────────────────────────────────────────────────────────────

    private FruitCakeSearch MakeTeacher()
    {
        var agent = new GreedyQAgent(_teacher, FruitCakeEnv.ColumnCount);
        // Leaf = the DQN's sense of the board, marginalized over the unknown upcoming fruit (avg max-Q over the
        // droppable tiers) — the exact leaf the shipped serving search uses (FruitCakeSearchEval.NetValue).
        double Leaf(FruitCakeWorld w)
        {
            double sum = 0;
            foreach (var d in FruitCatalog.Droppable)
                sum += Max(agent.QValues(FruitCakeEnv.BuildObservation(w, d.Tier, d.Tier)));
            return sum / FruitCatalog.Droppable.Count;
        }
        return new FruitCakeSearch(Leaf) { MaxDepth = teacherDepth, TopK = teacherTopK, TopK2 = teacherTopK2 };
    }

    // Play one teacher self-play game, appending one labeled Sample per drop. Label = the search's chosen column;
    // value target = the discounted realized remaining score (raw merge points) from that drop. Behaviour follows
    // the teacher, with a small DAgger chance of a random column for state coverage.
    private void PlayGame(FruitCakeEnv env, FruitCakeSearch search, Xoshiro256StarStar rng, List<Sample> outSamples)
    {
        var (_, _) = env.Reset(rng.NextUInt64());
        var obsList = new List<float[]>();
        var colList = new List<int>();
        var pointList = new List<int>();
        while (true)
        {
            var obs = FruitCakeEnv.BuildObservation(env.World, env.CurrentTier, env.NextTier);
            int teacherCol = search.ChooseColumn(env.World, env.CurrentTier, env.NextTier);
            int actCol = rng.NextDouble() < _daggerProb ? rng.NextInt(FruitCakeEnv.ColumnCount) : teacherCol;

            int before = env.Score;
            var step = env.Step(actCol);
            obsList.Add(obs);
            colList.Add(teacherCol);          // always label with the expert's choice (DAgger)
            pointList.Add(env.Score - before); // raw merge points scored by this drop
            if (step.Done) break;
        }

        // Discounted return-to-go in raw-score units (the value head's target, ÷ ValueScale at train time).
        double ret = 0;
        var returns = new float[obsList.Count];
        for (int t = obsList.Count - 1; t >= 0; t--)
        {
            ret = pointList[t] + gamma * ret;
            returns[t] = (float)ret;
        }
        for (int t = 0; t < obsList.Count; t++)
            outSamples.Add(new Sample(obsList[t], colList[t], returns[t]));
    }

    private readonly record struct Sample(float[] Obs, int Column, float Return);

    // ── supervised training step (CE on the expert column + Huber on the return) ─────────────────────────────

    private (double Ce, double Huber, double Acc) TrainStep(List<Sample> samples, int offset, int batch)
    {
        const int obsSize = FruitCakeEnv.ObservationSize;
        const int cols = FruitCakeEnv.ColumnCount;
        var obs = new float[batch * obsSize];
        var weights = new float[batch * cols];
        var targets = new float[batch];
        for (int i = 0; i < batch; i++)
        {
            var s = samples[offset + i];
            s.Obs.CopyTo(obs.AsSpan(i * obsSize, obsSize));
            weights[i * cols + s.Column] = 1f;
            targets[i] = s.Return / FruitCakePolicyNet.ValueScale;
        }

        var (logits, value) = _net.Forward(new Tensor(obs, batch, obsSize));
        var logProbs = logits.LogSoftmax();
        var ce = logProbs.Mul(new Tensor(weights, batch, cols)).Sum().MulScalar(-1f / batch);
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
            for (int a = 1; a < cols; a++)
                if (logProbs.Data[i * cols + a] > logProbs.Data[i * cols + argmax]) argmax = a;
            if (argmax == samples[offset + i].Column) correct++;
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    // ── eval: greedy policy-head play (no search) ────────────────────────────────────────────────────────────

    private (double Score, double MaxTier, int Watermelon) EvalPolicy(int episodes)
    {
        var env = new FruitCakeEnv();
        double totalScore = 0, totalMaxTier = 0;
        int melon = 0;
        for (int e = 0; e < episodes; e++)
        {
            env.Reset((ulong)(5_000 + e));
            int epMaxTier = 0;
            while (true)
            {
                int col = _net.ChooseColumn(env.World, env.CurrentTier, env.NextTier);
                var step = env.Step(col);
                foreach (var b in env.World.Bodies) if (b.Tier > epMaxTier) epMaxTier = b.Tier;
                if (step.Done) break;
            }
            totalScore += env.Score;
            totalMaxTier += epMaxTier;
            if (epMaxTier >= FruitCatalog.TopTier) melon++;
        }
        return (totalScore / episodes, totalMaxTier / (double)episodes, melon);
    }

    private static float Max(float[] xs)
    {
        float m = float.NegativeInfinity;
        foreach (var x in xs) if (x > m) m = x;
        return m;
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
}
