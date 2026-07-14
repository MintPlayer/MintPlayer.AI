using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// AlphaZero-style self-play campaign (PLAN M39) as an <see cref="ITrainingCampaign"/> over any
/// <see cref="IZeroSumGame{TState}"/>. Each chunk plays a batch of games where BOTH sides are the current net guided
/// by <see cref="Mcts"/>, records <c>(observation, MCTS visit-count π, game outcome z)</c> into a rolling window, and
/// trains the two-headed <see cref="PolicyValueNet"/> on that window (soft-CE policy + value regression). The net
/// bootstraps from random init — no oracle, no human data. Reuses the model store + checkpoint format, the M38
/// <see cref="AdamState"/>/<see cref="TrainWindow"/> plumbing, and the live-network telemetry seam.
/// </summary>
/// <typeparam name="TState">The game state type of <see cref="IZeroSumGame{TState}"/>.</typeparam>
internal sealed class SelfPlayCampaign<TState> : ITrainingCampaign, INetworkTelemetrySource
{
    private const string NetId = "az";
    private const string AdamId = "az-adam";
    private readonly int _batchSize;
    private readonly int _epochsPerChunk;
    private readonly float _gradClipNorm;

    private readonly IZeroSumGame<TState> _game;
    private readonly string _environmentId;
    private readonly float _learningRate;
    private readonly int[] _hidden;
    private readonly int _gamesPerChunk, _tempMoves, _evalGames, _windowCapacity, _maxPlies;
    private readonly long _targetGames;
    private readonly double _opponentRandomFrac; // fraction of self-play games where the learner faces a random opponent
    private readonly Mcts.Config _selfPlayCfg, _evalCfg;

    private readonly SeedSequence _seeds;
    private readonly Xoshiro256StarStar _shuffleRng, _evalRng;

    // Parallel self-play generation (M41.2): games are independent, so a chunk fans out across cores via
    // DeterministicParallel — each game derives its OWN RNG from its global index (never execution order) and returns
    // its samples, which are merged into the window in ascending index. The trained checkpoint is therefore
    // bitwise-identical at any degree of parallelism (gated by a SHA dop-1==dop-N test). Generation reads a stable
    // read-only net (training runs on the owner thread AFTER the join), and net inference is concurrent-safe
    // ([ThreadStatic] NoGrad + fresh buffers), so no per-thread net copy is needed.
    private readonly bool _parallel;
    private readonly int? _maxDop;

    // Auto-difficulty-ladder (M40.4a): produce a reliably-ordered ladder of increasingly-strong nets, written straight
    // into the web app's models dir. A new tier is promoted only when the live net beats the last-promoted CHAMPION by
    // a margin in a net-vs-net arena (so Level K+1 provably beats Level K). Runs on its OWN RNG (`_arenaRng`, seeded
    // independently of the training/eval streams) and is inference-only, so enabling it does NOT change trained weights
    // for a given seed. Disabled when `_ladder` is null.
    private readonly LadderOptions? _ladder;
    private readonly Xoshiro256StarStar _arenaRng;
    private IPolicyValueNet? _champion;            // frozen snapshot of the last-promoted net (null until tier 1)
    private int _tierCount;
    private readonly List<TierInfo> _tiers = [];

    // Material-shaped value target (fixes the draw-saturation plateau): blend the sparse game outcome with a DENSE
    // per-position material advantage so the value head gets a gradient on every capture — not just win/loss/draw,
    // which is ~always a draw (→0) until the net can force mate. `_material` = the game's optional IMaterialScore
    // (null → pure outcome, unchanged); `_materialWeight` (α) = blend; MaterialScale squashes pawns → [-1,1] via tanh.
    private readonly IMaterialScore<TState>? _material;
    private readonly float _materialWeight;
    // Weight on the value (MSE) loss relative to the policy (CE) loss (1 = equal). Down-weighting is the standard fix
    // for value-head overfitting → strength regression at small scale (Leela Zero cut it 1.0→0.25).
    private readonly float _valueWeight;
    // Leaf-inference batch size for self-play MCTS. 1 = sequential (bitwise back-compat). >1 uses Mcts.SearchBatched
    // (virtual loss) so each net.Forward sees a batch of leaves — the only way self-play keeps a GPU busy.
    private readonly int _leafBatch;
    // Optional compute backend (e.g. the GPU AdaptiveBackend). When set, installed as Backend.Current in Resume — a
    // plain (non-thread-static) global, so the parallel self-play worker threads route their GEMMs through it too.
    private readonly IComputeBackend? _backend;
    private const float MaterialScale = 5f;

    // How the net is built + reloaded (arch-agnostic seam: MLP by default, conv when a ConvNetBuilder is passed).
    private readonly IPolicyValueNetBuilder _builder;
    private readonly string _checkpointKind;
    private IPolicyValueNet _net = null!;
    // Inference forward for batched self-play (the GPU-resident seam, M43). Default = the autograd forward over _net;
    // a device-resident impl can be installed via _forwardFactory. Built in Resume once _net exists, re-synced per chunk.
    private IPolicyValueForward _forward = null!;
    // Optional factory (from the lab entry point, which owns the GPU knowledge) that builds a resident forward for the
    // loaded net; null → the autograd default. Keeps this generic campaign free of any Ilgpu dependency.
    private readonly Func<IPolicyValueNet, IPolicyValueForward>? _forwardFactory;
    private Adam _adam = null!;
    private readonly List<Sample> _window;
    private TrainWindow _lossWindow;
    private long _totalSamples, _totalGames;
    private double _liveLoss = double.NaN, _lastWinRate = double.NaN;

    /// <param name="options">All tunable config (see <see cref="SelfPlayOptions"/>). Defaults reproduce the previous
    /// constructor defaults exactly.</param>
    /// <param name="netBuilder">Net architecture factory (default = the flat MLP, sized from <c>options.Hidden</c>).</param>
    /// <param name="backend">Optional compute backend (e.g. the GPU AdaptiveBackend) installed as Backend.Current.</param>
    public SelfPlayCampaign(IZeroSumGame<TState> game, string environmentId, SelfPlayOptions options,
        IPolicyValueNetBuilder? netBuilder = null, IComputeBackend? backend = null,
        Func<IPolicyValueNet, IPolicyValueForward>? forwardFactory = null)
    {
        _backend = backend;
        _forwardFactory = forwardFactory;
        _parallel = options.Parallel;
        _maxDop = options.MaxDop;
        _ladder = options.Ladder;
        _material = game as IMaterialScore<TState>; // dense material shaping when the game supports it
        _materialWeight = options.MaterialWeight;
        _valueWeight = options.ValueWeight;
        _leafBatch = options.LeafBatch < 1 ? 1 : options.LeafBatch;
        _batchSize = options.BatchSize;
        _epochsPerChunk = options.EpochsPerChunk;
        _gradClipNorm = options.GradClipNorm;
        // Independent stream (not from SeedSequence) so the arena can't perturb the training/eval RNG → reproducible.
        _arenaRng = new Xoshiro256StarStar(unchecked(options.Seed * 0x9E3779B97F4A7C15UL + 0xD1B54A32D192ED03UL));
        _opponentRandomFrac = options.OpponentRandomFrac;
        _game = game;
        _environmentId = environmentId;
        _learningRate = options.LearningRate;
        _hidden = [options.Hidden, options.Hidden];
        _builder = netBuilder ?? new MlpNetBuilder(_hidden); // default = the flat MLP (back-compat)
        _checkpointKind = _builder.CheckpointKind;
        _selfPlayCfg = options.Search;
        _evalCfg = options.Search with { RootNoiseFrac = 0f }; // eval plays deterministically (no exploration noise)
        _gamesPerChunk = options.GamesPerChunk;
        _tempMoves = options.TempMoves;
        _evalGames = options.EvalGames;
        _windowCapacity = options.WindowCapacity;
        _maxPlies = options.MaxPlies;
        _targetGames = options.TargetGames;
        _seeds = new SeedSequence(options.Seed);
        // The training-window shuffle runs on the owner thread; it gets the Buffer stream so it never collides with
        // the Policy stream that per-game generation derives its RNGs from (game 0 would otherwise share a seed).
        _shuffleRng = _seeds.CreateRng(RngStreams.Buffer);
        _evalRng = _seeds.CreateRng(RngStreams.Evaluation);
        _window = new List<Sample>(options.WindowCapacity);
    }

    public string Environment => _environmentId;

    public bool Resume(IModelStore store)
    {
        if (_backend is not null) Backend.Current = _backend; // route all Tensor ops (incl. worker threads) to it
        bool resumed = false;
        using (var s = store.TryOpenRead(_environmentId, NetId))
        {
            if (s is not null)
            {
                _net = _builder.Load(s, _game.ObservationSize, _game.PolicySize);
                Log($"resumed {_environmentId} self-play net ({_net.Describe()})");
                resumed = true;
            }
        }
        if (!resumed)
        {
            _net = _builder.CreateFresh(_game.ObservationSize, _game.PolicySize, _seeds.CreateRng(RngStreams.Init));
            Log($"starting fresh {_environmentId} self-play ({_net.Describe()}, {_selfPlayCfg.Simulations} sims/move)");
        }
        _adam = AdamState.LoadOrInit(store, _environmentId, AdamId, _net.Parameters(), _learningRate, Log);
        _forward = _forwardFactory?.Invoke(_net) ?? new AutogradPolicyValueForward(_net, _game.ObservationSize);
        if (_ladder is not null) LoadLadderState();
        return resumed;
    }

    // M44.1 gate: when CHESS_CHUNK_TIMING is set, log the generation-vs-training wall-time split per chunk. This is the
    // measurement that decides whether a GPU-resident TRAINER (M44.3) is worth building: if generation dominates (the
    // ply-cap straggler), the resident forward + --leaf-batch already shipped is the real lever and a resident trainer
    // buys little. Off by default (no per-chunk log noise in real runs); a couple of chunks under it answer the gate.
    private static readonly bool _chunkTiming = System.Environment.GetEnvironmentVariable("CHESS_CHUNK_TIMING") is not null;

    public long TrainChunk()
    {
        var swGen = _chunkTiming ? System.Diagnostics.Stopwatch.StartNew() : null;
        // Generate the chunk's games (each on its own index-derived RNG, over the stable read-only net), then merge
        // their samples into the rolling window in ascending game index — identical to the old sequential add order,
        // and independent of how many cores ran it.
        var perGame = DeterministicParallel.Generate(
            _gamesPerChunk, _seeds, RngStreams.Policy, _totalGames,
            (i, rng) => GenerateGame(_totalGames + i, rng), _parallel, _maxDop);
        foreach (var samples in perGame)
            foreach (var sample in samples) AddSample(sample);
        _totalGames += _gamesPerChunk;
        swGen?.Stop();

        var swTrain = _chunkTiming ? System.Diagnostics.Stopwatch.StartNew() : null;
        int trainBatches = 0;
        // Train EpochsPerChunk shuffled passes over the current window (skip until a full batch has accumulated).
        if (_window.Count >= _batchSize)
        {
            int obsSize = _game.ObservationSize, actions = _game.PolicySize;
            for (int epoch = 0; epoch < _epochsPerChunk; epoch++)
            {
                CubePolicyTraining.Shuffle(_window, _shuffleRng);
                for (int offset = 0; offset + _batchSize <= _window.Count; offset += _batchSize)
                {
                    var obs = new float[_batchSize * obsSize];
                    var pi = new float[_batchSize * actions];
                    var z = new float[_batchSize];
                    for (int i = 0; i < _batchSize; i++)
                    {
                        var sample = _window[offset + i];
                        sample.Obs.CopyTo(obs.AsSpan(i * obsSize));
                        sample.Pi.CopyTo(pi.AsSpan(i * actions));
                        z[i] = sample.Z;
                    }
                    var (pl, vl) = PolicyValueTraining.TrainStep(_net, _adam, obs, pi, z, _batchSize, obsSize, actions, _valueWeight, _gradClipNorm);
                    _lossWindow.Add(pl, vl, 0);
                    _liveLoss = pl + vl;
                    _totalSamples += _batchSize;
                    trainBatches++;
                }
            }
        }
        swTrain?.Stop();
        if (_chunkTiming)
        {
            double gen = swGen!.Elapsed.TotalMilliseconds, train = swTrain!.Elapsed.TotalMilliseconds, total = gen + train;
            Log($"chunk-timing: gen {gen:F0} ms ({gen / total:P1}) | train {train:F0} ms ({train / total:P1}, {trainBatches} batches) | window {_window.Count}");
        }
        // Re-sync the inference forward to the just-trained weights, so next chunk's generation uses them. For the
        // autograd default this re-points the same net (a no-op); for a GPU-resident forward it re-uploads the weights.
        _forward.OnWeightsSynced(_net);
        return _totalGames;
    }

    /// <summary>Score-maximizing: run to the runner's time budget, or an optional absolute game cap.</summary>
    public bool IsComplete => _targetGames > 0 && _totalGames >= _targetGames;

    public CampaignEval Evaluate()
    {
        if (_net is null) return new CampaignEval([new("games", 0, "0")], "no model yet (train first)");

        double winRate = ArenaVsRandom();
        _lastWinRate = winRate;
        var (policyLoss, valueLoss, _) = _lossWindow.MeanAndReset();

        var metrics = new List<CampaignMetric>
        {
            new("games", _totalGames, "0"),
            new("samples", _totalSamples, "0"),
            new("winRate", winRate, "F3"),
            new("policyLoss", policyLoss, "F4"),
            new("valueLoss", valueLoss, "F4"),
        };
        return new CampaignEval(metrics,
            $"games {_totalGames:N0} | winRate-vs-random {winRate:P1} | policy {policyLoss:F4} | value {valueLoss:F4}");
    }

    public void Checkpoint(IModelStore store)
    {
        if (_net is null) return;
        store.Save(_environmentId, NetId, s => _net.Save(s, _checkpointKind));
        AdamState.Save(store, _environmentId, AdamId, _adam);
        if (_ladder is not null) MaybePromoteDifficulty();
    }

    public void Dispose() { }

    // ── Self-play ────────────────────────────────────────────────────────────────────────────────────────────────
    // One self-play game, pure w.r.t. its own RNG + the shared read-only net → safe to run concurrently. Returns its
    // samples (the caller merges them into the window on the owner thread) instead of touching shared state.
    private List<Sample> GenerateGame(long globalIndex, Xoshiro256StarStar rng)
    {
        // A fraction of games pit the learner (MCTS) against a RANDOM opponent, so the net trains on the
        // off-distribution positions a weak/unexpected move reaches — the direct fix for "a novel move disorients
        // the AI" (PLAN M39.3). The rest are pure self-play. Opening variety otherwise comes from temperature.
        // The learner's colour keys off the global game index (not a shared counter) so it's parallelism-independent.
        return _opponentRandomFrac > 0 && rng.NextDouble() < _opponentRandomFrac
            ? PlayVsRandom(learnerFirst: (globalIndex & 1) == 0, rng)
            : PlaySelfPlay(rng);
    }

    // Both sides are the current net + MCTS; every position is a training sample. The outcome z alternates sign each
    // ply (zero-sum): the terminal result is for the side to move there, so the LAST position played gets its negation.
    private List<Sample> PlaySelfPlay(Xoshiro256StarStar rng)
    {
        var state = _game.Root();
        var obsHistory = new List<float[]>(64);
        var piHistory = new List<float[]>(64);
        var matHistory = new List<float>(64); // per-position material target (side-to-move relative)

        int ply = 0;
        while (_game.Result(state) == GameResult.Ongoing && ply < _maxPlies)
        {
            float[] pi = _leafBatch > 1
                ? Mcts.SearchBatched(_game, state, EvaluateBatch, _selfPlayCfg, rng, _leafBatch)
                : Mcts.Search(_game, state, Evaluate, _selfPlayCfg, rng);
            var obs = new float[_game.ObservationSize];
            _game.WriteObservation(state, obs);
            obsHistory.Add(obs);
            piHistory.Add(pi);
            matHistory.Add(MaterialTarget(state));
            state = _game.Apply(state, SelectMove(pi, ply, rng));
            ply++;
        }

        var samples = new List<Sample>(obsHistory.Count);
        float zTerminalMover = _game.Result(state) switch
        {
            GameResult.Loss => -1f,
            GameResult.Win => 1f,
            GameResult.Ongoing => AdjudicateCapped(state), // hit the ply cap → decide by material, not a forced draw
            _ => 0f,                                       // true draw (stalemate / repetition / insufficient material)
        };
        float z = -zTerminalMover;
        for (int i = obsHistory.Count - 1; i >= 0; i--)
        {
            samples.Add(new Sample(obsHistory[i], piHistory[i], Blend(z, matHistory[i])));
            z = -z;
        }
        return samples;
    }

    // The learner (net + MCTS) plays one colour and records its positions; the opponent plays random-legal moves. The
    // learner's colour is constant, so every recorded position takes the same outcome z (from the learner's view).
    private List<Sample> PlayVsRandom(bool learnerFirst, Xoshiro256StarStar rng)
    {
        var state = _game.Root();
        var obsHistory = new List<float[]>(64);
        var piHistory = new List<float[]>(64);
        var matHistory = new List<float>(64); // per learner-position material target (learner relative)

        bool learnerToMove = learnerFirst;
        int ply = 0;
        GameResult result;
        while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
        {
            int move;
            if (learnerToMove)
            {
                float[] pi = Mcts.Search(_game, state, Evaluate, _selfPlayCfg, rng);
                var obs = new float[_game.ObservationSize];
                _game.WriteObservation(state, obs);
                obsHistory.Add(obs);
                piHistory.Add(pi);
                matHistory.Add(MaterialTarget(state)); // state's side-to-move = learner here
                move = SelectMove(pi, ply, rng);
            }
            else move = RandomLegalMove(state, rng);
            state = _game.Apply(state, move);
            learnerToMove = !learnerToMove;
            ply++;
        }

        // learnerToMove now = the side to move in the terminal state. Result is for that side.
        float z = result switch
        {
            GameResult.Loss => learnerToMove ? -1f : 1f, // side-to-move lost → learner lost iff it was to move
            GameResult.Win => learnerToMove ? 1f : -1f,
            GameResult.Ongoing => learnerToMove ? AdjudicateCapped(state) : -AdjudicateCapped(state), // ply cap → material
            _ => 0f,                                     // true draw
        };
        var samples = new List<Sample>(obsHistory.Count);
        for (int i = 0; i < obsHistory.Count; i++)
            samples.Add(new Sample(obsHistory[i], piHistory[i], Blend(z, matHistory[i])));
        return samples;
    }

    // Per-position material target from the side-to-move's view, squashed to [-1,1] (0 when the game has no material).
    private float MaterialTarget(TState state) => _material is null ? 0f : MathF.Tanh(_material.MaterialAdvantage(state) / MaterialScale);

    // A game that hit the ply cap is NOT a genuine draw — training a materially winning position as z=0 starves the
    // outcome signal and collapses the net onto passive, shuffle-to-the-cap play (observed overnight: the net's
    // material vs its own baseline slid −2 → −9 pawns while value loss fell to ~0). Adjudicate the capped position by
    // material from the side-to-move's view: a decisive edge (≥ margin pawns) trains as a win/loss, otherwise a true
    // 0. No-op when the game has no material notion (_material null), so Connect-4 etc. are unaffected.
    private const float AdjudicationMargin = 1.5f; // pawns — a clearly decisive material edge
    private float AdjudicateCapped(TState state)
    {
        if (_material is null) return 0f;
        float m = _material.MaterialAdvantage(state);
        return m > AdjudicationMargin ? 1f : m < -AdjudicationMargin ? -1f : 0f;
    }

    // Blend the sparse game outcome z with the dense material target (α = _materialWeight); pure z when no material.
    private float Blend(float z, float mat) => _material is null ? z : (1f - _materialWeight) * z + _materialWeight * mat;

    private void AddSample(Sample sample)
    {
        if (_window.Count >= _windowCapacity)
            _window.RemoveAt(0); // drop the oldest — a rolling replay window
        _window.Add(sample);
    }

    // MCTS leaf evaluator: masked-softmax policy priors + tanh value, read-only (no autograd).
    private (float[] Priors, float Value) Evaluate(TState state) => EvaluateWith(_net, state);

    private (float[] Priors, float Value) EvaluateWith(IPolicyValueNet net, TState state)
    {
        var obs = new float[_game.ObservationSize];
        _game.WriteObservation(state, obs);
        using (GradMode.NoGrad())
        {
            var (logits, value) = net.Forward(new Tensor(obs, 1, obs.Length));
            var priors = MaskedSoftmax(logits.Data, _game.LegalMoves(state));
            return (priors, MathF.Tanh(value.Data[0]));
        }
    }

    // Batched leaf evaluator for Mcts.SearchBatched: stack all leaf observations into one [B, obsSize] tensor, run a
    // SINGLE net.Forward, and split the result per leaf (masked-softmax priors + tanh value). This is what turns
    // self-play's batch-1 inference into batch-B — read-only over the shared net, so it's safe on the game threads.
    private IReadOnlyList<(float[] Priors, float Value)> EvaluateBatch(IReadOnlyList<TState> states)
    {
        int b = states.Count, obsSize = _game.ObservationSize, policy = _game.PolicySize;
        var obs = new float[b * obsSize];
        for (int i = 0; i < b; i++) _game.WriteObservation(states[i], obs.AsSpan(i * obsSize, obsSize));
        // One batched forward through the inference seam (autograd default, or a GPU-resident impl). Returns raw
        // logits [b*policy] + linear value [b]; masked-softmax + tanh stay here (they need each state's legal moves).
        var (logits, value) = _forward.Forward(obs, b);
        var results = new (float[], float)[b];
        for (int i = 0; i < b; i++)
        {
            var row = logits.AsSpan(i * policy, policy).ToArray();
            results[i] = (MaskedSoftmax(row, _game.LegalMoves(states[i])), MathF.Tanh(value[i]));
        }
        return results;
    }

    private float[] MaskedSoftmax(float[] logits, IReadOnlyList<int> legal)
    {
        var priors = new float[_game.PolicySize];
        float max = float.NegativeInfinity;
        foreach (int m in legal) if (logits[m] > max) max = logits[m];
        float sum = 0f;
        foreach (int m in legal) { float e = MathF.Exp(logits[m] - max); priors[m] = e; sum += e; }
        if (sum > 0f) foreach (int m in legal) priors[m] /= sum;
        return priors;
    }

    private int SelectMove(float[] pi, int ply, Xoshiro256StarStar rng)
    {
        if (ply >= _tempMoves) // late game: play the most-visited move
        {
            int best = 0;
            for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
            return best;
        }
        // early game: sample proportional to visit counts, for opening variety
        double r = rng.NextDouble(), acc = 0;
        for (int a = 0; a < pi.Length; a++) { acc += pi[a]; if (r <= acc && pi[a] > 0) return a; }
        // numerical fallback: the last legal (highest-visited) move
        int fallback = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[fallback]) fallback = a;
        return fallback;
    }

    // ── Eval: the model (net + MCTS, no root noise, argmax) vs a random-legal opponent, colors alternated ──
    // These games are independent and inference-only, so they run across cores via the same DeterministicParallel
    // primitive self-play uses — with a heavy net (conv) this sequential loop otherwise dominates wall time and, since
    // it runs on the owner thread between training chunks, stalls training itself. One base seed is drawn per call so
    // each game's RNG is a pure function of (cycle, game index): reproducible and identical at any degree of
    // parallelism. Inference-only ⇒ it never touches trained weights, so checkpoints stay bitwise-identical per seed.
    private double ArenaVsRandom()
    {
        ulong baseSeed = _evalRng.NextUInt64();
        var scores = DeterministicParallel.Generate(_evalGames, baseSeed, baseIndex: 0,
            (game, rng) => PlayEvalGame(game, rng), _parallel, _maxDop);
        double score = 0;
        foreach (double s in scores) score += s;
        return score / _evalGames;
    }

    // One eval game: model (net + MCTS) vs a random-legal opponent. Returns win = 1, draw = 0.5, loss = 0.
    private double PlayEvalGame(int game, Xoshiro256StarStar rng)
    {
        int modelSide = game % 2 == 0 ? 1 : 2; // model is player 1 on even games, player 2 on odd
        var state = _game.Root();
        int mover = 1, ply = 0;
        GameResult result;
        while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
        {
            int move = mover == modelSide ? ModelMove(state, rng) : RandomLegalMove(state, rng);
            state = _game.Apply(state, move);
            mover = 3 - mover;
            ply++;
        }
        // result is for the side to move in the terminal state (who did NOT just move). Loss ⇒ the last mover won.
        if (result == GameResult.Loss) { int lastMover = 3 - mover; return lastMover == modelSide ? 1 : 0; }
        return result != GameResult.Win ? 0.5 : 0; // draw (ply cap or full board), else a win for the mover (not model)
    }

    private int ModelMove(TState state, Xoshiro256StarStar rng)
    {
        float[] pi = Mcts.Search(_game, state, Evaluate, _evalCfg, rng);
        int best = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
        return best;
    }

    private int RandomLegalMove(TState state, Xoshiro256StarStar rng)
    {
        var legal = _game.LegalMoves(state);
        return legal[rng.NextInt(legal.Count)];
    }

    // ── Auto-difficulty ladder (M40.4a) ──────────────────────────────────────────────────────────────────────────
    // Called after each Checkpoint. Promotes a new difficulty tier when the live net is "significantly stronger" than
    // the last-promoted champion — measured by a net-vs-net arena, so tiers are reliably ordered by construction.
    // Promote when the live net is "significantly stronger" than the last-promoted champion, judged by ANY signal:
    //  • average material margin in the net-vs-net arena ≥ PromoteMaterial — the PRIMARY signal: dense and
    //    non-saturating, it separates two nets that only ever draw (the whole point of material shaping); OR
    //  • winRate-vs-random improved by ≥ PromoteMargin (fallback while very weak); OR
    //  • head-to-head win/draw score ≥ ArenaMargin (fallback once winRate-vs-random saturates near 100%).
    private void MaybePromoteDifficulty()
    {
        if (_net is null || _ladder is null) return;
        if (_champion is null) { PromoteTier("baseline (first checkpoint)"); return; } // Level 1 = a weak baseline

        double championWinRate = _tiers.Count > 0 ? _tiers[^1].WinRate : double.NaN;
        double delta = _lastWinRate - championWinRate;
        var (arena, material) = ArenaVsNet(_net, _champion, _ladder.ArenaGames);
        bool byMaterial = _material is not null && material >= _ladder.PromoteMaterial;
        bool byWinRate = !double.IsNaN(championWinRate) && !double.IsNaN(_lastWinRate) && delta >= _ladder.PromoteMargin;
        bool byArena = arena >= _ladder.ArenaMargin;
        Log($"ladder: vs Level {_tierCount} — material {material:+0.00;-0.00} pawns (need +{_ladder.PromoteMaterial:0.00}) | winRate-vs-random {_lastWinRate:P0} vs {championWinRate:P0} (Δ {delta:P0}, +{_ladder.PromoteMargin:P0}) | head-to-head {arena:P0} of {_ladder.ArenaGames} ({_ladder.ArenaMargin:P0})");
        if (byMaterial || byWinRate || byArena)
            PromoteTier(byMaterial ? $"material {material:+0.00;-0.00} pawns" : byWinRate ? $"winRate-vs-random +{delta:P0}" : $"head-to-head {arena:P0}");
    }

    private void PromoteTier(string reason)
    {
        _tierCount++;
        Directory.CreateDirectory(_ladder!.Dir);
        string ckptName = $"{_environmentId}.az.d{_tierCount}.ckpt";
        string ckptPath = Path.Combine(_ladder.Dir, ckptName);
        string tmp = ckptPath + ".tmp";
        using (var fs = File.Create(tmp)) _net.Save(fs, _checkpointKind);
        if (File.Exists(ckptPath)) File.Delete(ckptPath);
        File.Move(tmp, ckptPath);

        _champion = Freeze(_net); // frozen snapshot; continued training on _net won't mutate the champion
        _tiers.Add(new TierInfo($"Level {_tierCount}", $"/models/{ckptName}", _ladder.Sims, 0.0, 1.5, _lastWinRate, _totalGames));
        WriteManifest();
        Log($"ladder: PROMOTED Level {_tierCount} → {ckptName} ({reason}); winRate-vs-random {_lastWinRate:P1}, games {_totalGames:N0}");
    }

    // `challenger` vs `champion` over `games` full games, colours alternated, both picking deterministically (eval
    // MCTS argmax, no root noise); games diversified by a short randomized opening drawn from `_arenaRng` (independent
    // of the training/eval streams). Returns BOTH the win/draw Score (win 1 / draw 0.5) AND the challenger's average
    // end-of-game material margin in pawns — the non-saturating signal that separates two nets that only ever draw.
    // Parallelized like ArenaVsRandom (independent, inference-only games); one base seed per call keeps the arena
    // reproducible and DOP-invariant, and it never mutates trained weights.
    private (double Score, double Material) ArenaVsNet(IPolicyValueNet challenger, IPolicyValueNet champion, int games)
    {
        ulong baseSeed = _arenaRng.NextUInt64();
        var results = DeterministicParallel.Generate(games, baseSeed, baseIndex: 0,
            (g, rng) => PlayArenaGame(g, rng, challenger, champion), _parallel, _maxDop);
        double score = 0, material = 0;
        foreach (var (s, m) in results) { score += s; material += m; }
        return (score / games, material / games);
    }

    // One arena game: challenger vs champion, colours by index, a short randomized opening from the per-game RNG.
    // Returns the win/draw Score (win 1 / draw 0.5 / loss 0) and the challenger's end-of-game material margin in pawns.
    private (double Score, double Material) PlayArenaGame(int g, Xoshiro256StarStar rng, IPolicyValueNet challenger, IPolicyValueNet champion)
    {
        int challengerSide = g % 2 == 0 ? 1 : 2;
        int openingPlies = _ladder!.OpeningPlies == 0 ? 0 : rng.NextInt(_ladder.OpeningPlies + 1);
        var state = _game.Root();
        int mover = 1, ply = 0;
        GameResult result;
        while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
        {
            int move = ply < openingPlies
                ? RandomLegalMove(state, rng)                                        // neutral randomized opening
                : ModelMoveWith(mover == challengerSide ? challenger : champion, state, rng);
            state = _game.Apply(state, move);
            mover = 3 - mover;
            ply++;
        }
        double score = 0;
        if (result == GameResult.Loss) { int lastMover = 3 - mover; if (lastMover == challengerSide) score = 1; }
        else if (result != GameResult.Win) score = 0.5; // draw / ply-cap
        double material = 0;
        if (_material is not null)
        {
            // MaterialAdvantage is side-to-move relative; at the terminal state the side to move is `mover`.
            float mat = _material.MaterialAdvantage(state);
            material = mover == challengerSide ? mat : -mat;
        }
        return (score, material);
    }

    private int ModelMoveWith(IPolicyValueNet net, TState state, Xoshiro256StarStar rng)
    {
        float[] pi = Mcts.Search(_game, state, s => EvaluateWith(net, s), _evalCfg, rng);
        int best = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
        return best;
    }

    // A frozen deep copy (save→reload) so the champion is stable while _net keeps training.
    private IPolicyValueNet Freeze(IPolicyValueNet net)
    {
        using var ms = new MemoryStream();
        net.Save(ms, _checkpointKind);
        ms.Position = 0;
        return _builder.Load(ms, _game.ObservationSize, _game.PolicySize);
    }

    private void WriteManifest()
    {
        string path = Path.Combine(_ladder!.Dir, $"{_environmentId}-difficulties.json");
        var payload = _tiers.Select(t => new
        {
            label = t.Label, ckpt = t.Ckpt, sims = t.Sims,
            temperature = t.Temperature, cpuct = t.Cpuct, winRateVsRandom = t.WinRate, games = t.Games,
        });
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
    }

    // Resume an existing ladder: adopt the highest tier on disk as the champion and rebuild the manifest list, so a
    // resumed run continues the ladder (higher tiers) instead of restarting at Level 1.
    private void LoadLadderState()
    {
        var dir = _ladder!.Dir;
        if (!Directory.Exists(dir)) return;
        int highest = 0;
        foreach (var f in Directory.EnumerateFiles(dir, $"{_environmentId}.az.d*.ckpt"))
        {
            string name = Path.GetFileNameWithoutExtension(f); // env.az.dK
            int dot = name.LastIndexOf(".d", StringComparison.Ordinal);
            if (dot >= 0 && int.TryParse(name[(dot + 2)..], out int k) && k > highest) highest = k;
        }
        if (highest == 0) return;

        try { RebuildTiersFromManifest(); } catch { _tiers.Clear(); }
        _tierCount = highest;
        string championPath = Path.Combine(dir, $"{_environmentId}.az.d{highest}.ckpt");
        using var s = File.OpenRead(championPath);
        _champion = _builder.Load(s, _game.ObservationSize, _game.PolicySize);
        Log($"ladder: resumed at Level {_tierCount} (champion {Path.GetFileName(championPath)})");
    }

    private void RebuildTiersFromManifest()
    {
        string path = Path.Combine(_ladder!.Dir, $"{_environmentId}-difficulties.json");
        if (!File.Exists(path)) return;
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            _tiers.Add(new TierInfo(
                e.GetProperty("label").GetString() ?? "",
                e.GetProperty("ckpt").GetString() ?? "",
                e.TryGetProperty("sims", out var si) ? si.GetInt32() : _ladder.Sims,
                e.TryGetProperty("temperature", out var te) ? te.GetDouble() : 0.0,
                e.TryGetProperty("cpuct", out var cp) ? cp.GetDouble() : 1.5,
                e.TryGetProperty("winRateVsRandom", out var wr) ? wr.GetDouble() : double.NaN,
                e.TryGetProperty("games", out var ga) ? ga.GetInt64() : 0));
        }
    }

    // ── Live telemetry (INetworkTelemetrySource): read-only snapshot of the current net ──
    string INetworkTelemetrySource.NetKind => "policy-value";
    IReadOnlyList<Tensor>? INetworkTelemetrySource.SnapshotParameters()
        => _net is null ? null : [.. _net.Parameters()];
    NetworkMetrics INetworkTelemetrySource.Sample()
        => new(_totalGames, _targetGames, _liveLoss, _lastWinRate, double.NaN);

    private static void Log(string message) => Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} {message}");

    private sealed record Sample(float[] Obs, float[] Pi, float Z);
    private sealed record TierInfo(string Label, string Ckpt, int Sims, double Temperature, double Cpuct, double WinRate, long Games);
}

/// <summary>Config for the auto-difficulty ladder (M40.4a). <paramref name="Dir"/> is where tier checkpoints + the
/// <c>{env}-difficulties.json</c> manifest are written (the web app's models dir). A tier is promoted when the live
/// net beats the last champion by EITHER: <paramref name="PromoteMargin"/> = the required rise in winRate-vs-random
/// (the signal while nets are weak/drawish), OR <paramref name="ArenaMargin"/> = the head-to-head net-vs-net score
/// (the signal once winRate-vs-random saturates). <paramref name="OpeningPlies"/> is the max random opening length
/// used to diversify arena games; <paramref name="Sims"/> is the default per-tier search budget written to the manifest.</summary>
internal sealed record LadderOptions(string Dir, double PromoteMaterial, double PromoteMargin, double ArenaMargin, int ArenaGames, int Sims, int OpeningPlies);
