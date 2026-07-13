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
    private const string CheckpointKind = "selfplay-pv";
    private const string NetId = "az";
    private const string AdamId = "az-adam";
    private const int BatchSize = 128;

    private readonly IZeroSumGame<TState> _game;
    private readonly string _environmentId;
    private readonly float _learningRate;
    private readonly int[] _hidden;
    private readonly int _gamesPerChunk, _tempMoves, _evalGames, _windowCapacity, _maxPlies;
    private readonly long _targetGames;
    private readonly double _opponentRandomFrac; // fraction of self-play games where the learner faces a random opponent
    private readonly Mcts.Config _selfPlayCfg, _evalCfg;

    private readonly SeedSequence _seeds;
    private readonly Xoshiro256StarStar _searchRng, _evalRng;

    // Auto-difficulty-ladder (M40.4a): produce a reliably-ordered ladder of increasingly-strong nets, written straight
    // into the web app's models dir. A new tier is promoted only when the live net beats the last-promoted CHAMPION by
    // a margin in a net-vs-net arena (so Level K+1 provably beats Level K). Runs on its OWN RNG (`_arenaRng`, seeded
    // independently of the training/eval streams) and is inference-only, so enabling it does NOT change trained weights
    // for a given seed. Disabled when `_ladder` is null.
    private readonly LadderOptions? _ladder;
    private readonly Xoshiro256StarStar _arenaRng;
    private PolicyValueNet? _champion;             // frozen snapshot of the last-promoted net (null until tier 1)
    private int _tierCount;
    private readonly List<TierInfo> _tiers = [];

    private PolicyValueNet _net = null!;
    private Adam _adam = null!;
    private readonly List<Sample> _window;
    private TrainWindow _lossWindow;
    private long _totalSamples, _totalGames;
    private double _liveLoss = double.NaN, _lastWinRate = double.NaN;

    public SelfPlayCampaign(IZeroSumGame<TState> game, string environmentId, ulong seed, float learningRate,
        int hidden, Mcts.Config selfPlayCfg, int gamesPerChunk = 32, int tempMoves = 8, int evalGames = 20,
        int windowCapacity = 40_000, int maxPlies = 512, long targetGames = 0, double opponentRandomFrac = 0,
        LadderOptions? ladder = null)
    {
        _ladder = ladder;
        // Independent stream (not from SeedSequence) so the arena can't perturb the training/eval RNG → reproducible.
        _arenaRng = new Xoshiro256StarStar(unchecked(seed * 0x9E3779B97F4A7C15UL + 0xD1B54A32D192ED03UL));
        _opponentRandomFrac = opponentRandomFrac;
        _game = game;
        _environmentId = environmentId;
        _learningRate = learningRate;
        _hidden = [hidden, hidden];
        _selfPlayCfg = selfPlayCfg;
        _evalCfg = selfPlayCfg with { RootNoiseFrac = 0f }; // eval plays deterministically (no exploration noise)
        _gamesPerChunk = gamesPerChunk;
        _tempMoves = tempMoves;
        _evalGames = evalGames;
        _windowCapacity = windowCapacity;
        _maxPlies = maxPlies;
        _targetGames = targetGames;
        _seeds = new SeedSequence(seed);
        _searchRng = _seeds.CreateRng(RngStreams.Policy);
        _evalRng = _seeds.CreateRng(RngStreams.Evaluation);
        _window = new List<Sample>(windowCapacity);
    }

    public string Environment => _environmentId;

    public bool Resume(IModelStore store)
    {
        bool resumed = false;
        using (var s = store.TryOpenRead(_environmentId, NetId))
        {
            if (s is not null)
            {
                _net = PolicyValueNet.Load(s, CheckpointKind, _game.ObservationSize, _game.PolicySize);
                Log($"resumed {_environmentId} self-play net (trunk [{string.Join(",", _net.Trunk)}])");
                resumed = true;
            }
        }
        if (!resumed)
        {
            _net = new PolicyValueNet(_game.ObservationSize, _hidden, _game.PolicySize, _seeds.CreateRng(RngStreams.Init));
            Log($"starting fresh {_environmentId} self-play (trunk [{string.Join(",", _hidden)}], {_selfPlayCfg.Simulations} sims/move)");
        }
        _adam = AdamState.LoadOrInit(store, _environmentId, AdamId, _net.Parameters(), _learningRate, Log);
        if (_ladder is not null) LoadLadderState();
        return resumed;
    }

    public long TrainChunk()
    {
        for (int g = 0; g < _gamesPerChunk; g++) PlayGame();

        // Train one shuffled pass over the current window (skip until a full batch has accumulated).
        if (_window.Count >= BatchSize)
        {
            CubePolicyTraining.Shuffle(_window, _searchRng);
            int obsSize = _game.ObservationSize, actions = _game.PolicySize;
            for (int offset = 0; offset + BatchSize <= _window.Count; offset += BatchSize)
            {
                var obs = new float[BatchSize * obsSize];
                var pi = new float[BatchSize * actions];
                var z = new float[BatchSize];
                for (int i = 0; i < BatchSize; i++)
                {
                    var sample = _window[offset + i];
                    sample.Obs.CopyTo(obs.AsSpan(i * obsSize));
                    sample.Pi.CopyTo(pi.AsSpan(i * actions));
                    z[i] = sample.Z;
                }
                var (pl, vl) = PolicyValueTraining.TrainStep(_net, _adam, obs, pi, z, BatchSize, obsSize, actions);
                _lossWindow.Add(pl, vl, 0);
                _liveLoss = pl + vl;
                _totalSamples += BatchSize;
            }
        }
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
        store.Save(_environmentId, NetId, s => _net.Save(s, CheckpointKind));
        AdamState.Save(store, _environmentId, AdamId, _adam);
        if (_ladder is not null) MaybePromoteDifficulty();
    }

    public void Dispose() { }

    // ── Self-play ────────────────────────────────────────────────────────────────────────────────────────────────
    private void PlayGame()
    {
        // A fraction of games pit the learner (MCTS) against a RANDOM opponent, so the net trains on the
        // off-distribution positions a weak/unexpected move reaches — the direct fix for "a novel move disorients
        // the AI" (PLAN M39.3). The rest are pure self-play. Opening variety otherwise comes from temperature.
        if (_opponentRandomFrac > 0 && _searchRng.NextDouble() < _opponentRandomFrac)
            PlayVsRandom(learnerFirst: (_totalGames & 1) == 0);
        else
            PlaySelfPlay();
        _totalGames++;
    }

    // Both sides are the current net + MCTS; every position is a training sample. The outcome z alternates sign each
    // ply (zero-sum): the terminal result is for the side to move there, so the LAST position played gets its negation.
    private void PlaySelfPlay()
    {
        var state = _game.Root();
        var obsHistory = new List<float[]>(64);
        var piHistory = new List<float[]>(64);

        int ply = 0;
        while (_game.Result(state) == GameResult.Ongoing && ply < _maxPlies)
        {
            float[] pi = Mcts.Search(_game, state, Evaluate, _selfPlayCfg, _searchRng);
            var obs = new float[_game.ObservationSize];
            _game.WriteObservation(state, obs);
            obsHistory.Add(obs);
            piHistory.Add(pi);
            state = _game.Apply(state, SelectMove(pi, ply));
            ply++;
        }

        float zTerminalMover = _game.Result(state) switch { GameResult.Loss => -1f, GameResult.Win => 1f, _ => 0f };
        float z = -zTerminalMover;
        for (int i = obsHistory.Count - 1; i >= 0; i--)
        {
            AddSample(new Sample(obsHistory[i], piHistory[i], z));
            z = -z;
        }
    }

    // The learner (net + MCTS) plays one colour and records its positions; the opponent plays random-legal moves. The
    // learner's colour is constant, so every recorded position takes the same outcome z (from the learner's view).
    private void PlayVsRandom(bool learnerFirst)
    {
        var state = _game.Root();
        var obsHistory = new List<float[]>(64);
        var piHistory = new List<float[]>(64);

        bool learnerToMove = learnerFirst;
        int ply = 0;
        GameResult result;
        while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
        {
            int move;
            if (learnerToMove)
            {
                float[] pi = Mcts.Search(_game, state, Evaluate, _selfPlayCfg, _searchRng);
                var obs = new float[_game.ObservationSize];
                _game.WriteObservation(state, obs);
                obsHistory.Add(obs);
                piHistory.Add(pi);
                move = SelectMove(pi, ply);
            }
            else move = RandomLegalMove(state, _searchRng);
            state = _game.Apply(state, move);
            learnerToMove = !learnerToMove;
            ply++;
        }

        // learnerToMove now = the side to move in the terminal state. Result is for that side.
        float z = result switch
        {
            GameResult.Loss => learnerToMove ? -1f : 1f, // side-to-move lost → learner lost iff it was to move
            GameResult.Win => learnerToMove ? 1f : -1f,
            _ => 0f,                                     // draw or ply-cap
        };
        for (int i = 0; i < obsHistory.Count; i++)
            AddSample(new Sample(obsHistory[i], piHistory[i], z));
    }

    private void AddSample(Sample sample)
    {
        if (_window.Count >= _windowCapacity)
            _window.RemoveAt(0); // drop the oldest — a rolling replay window
        _window.Add(sample);
    }

    // MCTS leaf evaluator: masked-softmax policy priors + tanh value, read-only (no autograd).
    private (float[] Priors, float Value) Evaluate(TState state) => EvaluateWith(_net, state);

    private (float[] Priors, float Value) EvaluateWith(PolicyValueNet net, TState state)
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

    private int SelectMove(float[] pi, int ply)
    {
        if (ply >= _tempMoves) // late game: play the most-visited move
        {
            int best = 0;
            for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
            return best;
        }
        // early game: sample proportional to visit counts, for opening variety
        double r = _searchRng.NextDouble(), acc = 0;
        for (int a = 0; a < pi.Length; a++) { acc += pi[a]; if (r <= acc && pi[a] > 0) return a; }
        // numerical fallback: the last legal (highest-visited) move
        int fallback = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[fallback]) fallback = a;
        return fallback;
    }

    // ── Eval: the model (net + MCTS, no root noise, argmax) vs a random-legal opponent, colors alternated ──
    private double ArenaVsRandom()
    {
        double score = 0; // win = 1, draw = 0.5
        for (int game = 0; game < _evalGames; game++)
        {
            int modelSide = game % 2 == 0 ? 1 : 2; // model is player 1 on even games, player 2 on odd
            var state = _game.Root();
            int mover = 1, ply = 0;
            GameResult result;
            while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
            {
                int move = mover == modelSide ? ModelMove(state) : RandomLegalMove(state, _evalRng);
                state = _game.Apply(state, move);
                mover = 3 - mover;
                ply++;
            }
            // result is for the side to move in the terminal state (who did NOT just move). Loss ⇒ the last mover won.
            if (result == GameResult.Loss) { int lastMover = 3 - mover; if (lastMover == modelSide) score += 1; }
            else if (result != GameResult.Win) score += 0.5; // draw (ply cap or full board)
        }
        return score / _evalGames;
    }

    private int ModelMove(TState state)
    {
        float[] pi = Mcts.Search(_game, state, Evaluate, _evalCfg, _evalRng);
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
    // Promote when the live net is "significantly stronger" than the last-promoted champion, judged by EITHER signal:
    //  • winRate-vs-random improved by ≥ PromoteMargin — the discriminating signal while nets are WEAK (two weak nets
    //    draw each other head-to-head, so a net-vs-net score stays ~50% and can't tell them apart; beating *random*
    //    more can); and
    //  • head-to-head arena score ≥ ArenaMargin — the signal once nets are STRONG enough that winRate-vs-random
    //    saturates near 100% (delta ≈ 0) but they can actually convert wins against each other.
    private void MaybePromoteDifficulty()
    {
        if (_net is null || _ladder is null) return;
        if (_champion is null) { PromoteTier("baseline (first checkpoint)"); return; } // Level 1 = a weak baseline

        double championWinRate = _tiers.Count > 0 ? _tiers[^1].WinRate : double.NaN;
        double delta = _lastWinRate - championWinRate;
        double arena = ArenaVsNet(_net, _champion, _ladder.ArenaGames);
        bool byWinRate = !double.IsNaN(championWinRate) && !double.IsNaN(_lastWinRate) && delta >= _ladder.PromoteMargin;
        bool byArena = arena >= _ladder.ArenaMargin;
        Log($"ladder: vs Level {_tierCount} — winRate-vs-random {_lastWinRate:P0} vs champion {championWinRate:P0} (Δ {delta:P0}, need +{_ladder.PromoteMargin:P0}) | head-to-head {arena:P0} over {_ladder.ArenaGames} (need {_ladder.ArenaMargin:P0})");
        if (byWinRate || byArena) PromoteTier(byWinRate ? $"winRate-vs-random +{delta:P0}" : $"head-to-head {arena:P0}");
    }

    private void PromoteTier(string reason)
    {
        _tierCount++;
        Directory.CreateDirectory(_ladder!.Dir);
        string ckptName = $"{_environmentId}.az.d{_tierCount}.ckpt";
        string ckptPath = Path.Combine(_ladder.Dir, ckptName);
        string tmp = ckptPath + ".tmp";
        using (var fs = File.Create(tmp)) _net.Save(fs, CheckpointKind);
        if (File.Exists(ckptPath)) File.Delete(ckptPath);
        File.Move(tmp, ckptPath);

        _champion = Freeze(_net); // frozen snapshot; continued training on _net won't mutate the champion
        _tiers.Add(new TierInfo($"Level {_tierCount}", $"/models/{ckptName}", _ladder.Sims, 0.0, 1.5, _lastWinRate, _totalGames));
        WriteManifest();
        Log($"ladder: PROMOTED Level {_tierCount} → {ckptName} ({reason}); winRate-vs-random {_lastWinRate:P1}, games {_totalGames:N0}");
    }

    // Score for `challenger` vs `champion` over `games` full games (win 1 / draw 0.5), colours alternated. Both nets
    // pick deterministically (eval MCTS, argmax, no root noise); games are diversified by a short randomized opening
    // drawn from `_arenaRng` (independent of the training/eval streams).
    private double ArenaVsNet(PolicyValueNet challenger, PolicyValueNet champion, int games)
    {
        double score = 0;
        for (int g = 0; g < games; g++)
        {
            int challengerSide = g % 2 == 0 ? 1 : 2;
            int openingPlies = _ladder!.OpeningPlies == 0 ? 0 : _arenaRng.NextInt(_ladder.OpeningPlies + 1);
            var state = _game.Root();
            int mover = 1, ply = 0;
            GameResult result;
            while ((result = _game.Result(state)) == GameResult.Ongoing && ply < _maxPlies)
            {
                int move = ply < openingPlies
                    ? RandomLegalMove(state, _arenaRng)                                   // neutral randomized opening
                    : ModelMoveWith(mover == challengerSide ? challenger : champion, state);
                state = _game.Apply(state, move);
                mover = 3 - mover;
                ply++;
            }
            if (result == GameResult.Loss) { int lastMover = 3 - mover; if (lastMover == challengerSide) score += 1; }
            else if (result != GameResult.Win) score += 0.5; // draw / ply-cap
        }
        return score / games;
    }

    private int ModelMoveWith(PolicyValueNet net, TState state)
    {
        float[] pi = Mcts.Search(_game, state, s => EvaluateWith(net, s), _evalCfg, _arenaRng);
        int best = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
        return best;
    }

    // A frozen deep copy (save→reload) so the champion is stable while _net keeps training.
    private PolicyValueNet Freeze(PolicyValueNet net)
    {
        using var ms = new MemoryStream();
        net.Save(ms, CheckpointKind);
        ms.Position = 0;
        return PolicyValueNet.Load(ms, CheckpointKind, _game.ObservationSize, _game.PolicySize);
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
        _champion = PolicyValueNet.Load(s, CheckpointKind, _game.ObservationSize, _game.PolicySize);
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
internal sealed record LadderOptions(string Dir, double PromoteMargin, double ArenaMargin, int ArenaGames, int Sims, int OpeningPlies);
