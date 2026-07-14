using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// A NON-SATURATING strength eval for a trained chess net (M42 follow-up). Plays the net (net + MCTS, argmax, no
/// root noise) against a fixed <b>material alpha-beta</b> opponent of tunable depth, over N colour-alternated games
/// with a short random opening for variety, and reports the net's win/draw/loss, score, and average end-material.
/// <para>
/// Why this exists: <c>winRate-vs-random</c> SATURATES (~50% for any net that draws-but-can't-mate random) and
/// material-in-self-play is self-referential — neither can tell whether training actually made the net <i>play</i>
/// better. A depth-D material player is a FIXED, EXTERNAL, monotonically-stronger yardstick: a genuinely improving
/// net beats a deeper opponent over time, so raising <c>--minimax-depth</c> keeps the metric informative as the net
/// grows. Depth 1 ≈ grab hanging material; depth 2 ≈ don't hang to an immediate recapture; depth 3+ ≈ short tactics.
/// </para>
/// </summary>
internal static class ChessStrengthEval
{
    public static void Run(string ckptPath, string arch, int filters, int blocks, int[] hidden, int sims, int depth,
        int games, int maxPlies, int openingPlies, ulong seed)
    {
        if (!File.Exists(ckptPath)) { Console.WriteLine($"[strength] checkpoint not found: {ckptPath}"); return; }

        var game = new ChessGame();
        var material = (IMaterialScore<ChessState>)game;
        IPolicyValueNetBuilder builder = arch.ToLowerInvariant() == "conv"
            ? new ConvNetBuilder(18, 8, 8, filters, blocks)
            : new MlpNetBuilder(hidden);
        IPolicyValueNet net;
        using (var fs = File.OpenRead(ckptPath))
            net = builder.Load(fs, game.ObservationSize, game.PolicySize);
        Console.WriteLine($"[strength] loaded {arch} net ({net.Describe()}) from {ckptPath}");
        Console.WriteLine($"[strength] net (+MCTS {sims} sims, argmax) vs material minimax depth {depth} | {games} games | max {maxPlies} plies");

        var cfg = new Mcts.Config(Simulations: sims, RootNoiseFrac: 0f); // eval: deterministic, no exploration noise
        var opponent = new MaterialMinimaxPlayer<ChessState>(game, material, depth);
        var rng = new Xoshiro256StarStar(seed);

        // Leaf evaluator: masked-softmax policy priors over legal moves + tanh value (mirrors SelfPlayCampaign.Evaluate).
        Mcts.Evaluate<ChessState> evaluate = s =>
        {
            var obs = new float[game.ObservationSize];
            game.WriteObservation(s, obs);
            using (GradMode.NoGrad())
            {
                var (logits, value) = net.Forward(new Tensor(obs, 1, obs.Length));
                return (MaskedSoftmax(logits.Data, game.LegalMoves(s), game.PolicySize), MathF.Tanh(value.Data[0]));
            }
        };

        int wins = 0, draws = 0, losses = 0;
        double materialSum = 0;
        for (int g = 0; g < games; g++)
        {
            int netSide = g % 2 == 0 ? 1 : 2;                                  // alternate colours
            int opening = openingPlies == 0 ? 0 : rng.NextInt(openingPlies + 1); // short random opening for variety
            var state = game.Root();
            int mover = 1, ply = 0;
            GameResult result;
            while ((result = game.Result(state)) == GameResult.Ongoing && ply < maxPlies)
            {
                int move = ply < opening ? RandomMove(game, state, rng)
                         : mover == netSide ? NetMove(game, state, evaluate, cfg, rng)
                         : opponent.SelectMove(state);
                state = game.Apply(state, move);
                mover = 3 - mover;
                ply++;
            }
            // result is for the side to move in the terminal state (who did NOT just move). Loss ⇒ the last mover won.
            if (result == GameResult.Loss) { int last = 3 - mover; if (last == netSide) wins++; else losses++; }
            else if (result == GameResult.Win) { if (mover == netSide) wins++; else losses++; } // rare (mover already won)
            else draws++;                                                       // true draw or ply-cap
            float mat = material.MaterialAdvantage(state);                       // side-to-move relative → net relative
            materialSum += mover == netSide ? mat : -mat;
        }

        double score = (wins + 0.5 * draws) / games;
        Console.WriteLine($"[strength] net vs minimax-d{depth}: {wins}W {draws}D {losses}L | score {score:P1} | avg end-material {materialSum / games:+0.00;-0.00} pawns");
    }

    private static int NetMove(ChessGame game, ChessState state, Mcts.Evaluate<ChessState> evaluate, Mcts.Config cfg, Xoshiro256StarStar rng)
    {
        float[] pi = Mcts.Search(game, state, evaluate, cfg, rng);
        int best = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
        return best;
    }

    private static int RandomMove(ChessGame game, ChessState state, Xoshiro256StarStar rng)
    {
        var legal = game.LegalMoves(state);
        return legal[rng.NextInt(legal.Count)];
    }

    private static float[] MaskedSoftmax(float[] logits, IReadOnlyList<int> legal, int policySize)
    {
        var priors = new float[policySize];
        float max = float.NegativeInfinity;
        foreach (int m in legal) if (logits[m] > max) max = logits[m];
        float sum = 0f;
        foreach (int m in legal) { float e = MathF.Exp(logits[m] - max); priors[m] = e; sum += e; }
        if (sum > 0f) foreach (int m in legal) priors[m] /= sum;
        return priors;
    }
}

/// <summary>
/// A fixed, deterministic material player: negamax (alpha-beta) to <paramref name="depth"/> plies over any
/// <see cref="IZeroSumGame{TState}"/>, scoring leaves by the game's <see cref="IMaterialScore{TState}"/> (side-to-move
/// relative). Checkmate = ±Mate, draw = 0. This is the fixed external yardstick the strength eval measures against;
/// deeper = stronger, so it doesn't saturate as the net improves.
/// </summary>
internal sealed class MaterialMinimaxPlayer<TState>(IZeroSumGame<TState> game, IMaterialScore<TState> material, int depth)
{
    private const float Mate = 1_000_000f;

    public int SelectMove(TState state)
    {
        var moves = game.LegalMoves(state);
        int best = moves[0];
        float bestScore = float.NegativeInfinity;
        foreach (int m in moves)
        {
            float score = -Negamax(game.Apply(state, m), depth - 1, float.NegativeInfinity, float.PositiveInfinity);
            if (score > bestScore) { bestScore = score; best = m; }
        }
        return best;
    }

    private float Negamax(TState s, int d, float alpha, float beta)
    {
        switch (game.Result(s))
        {
            case GameResult.Loss: return -Mate; // side to move is checkmated
            case GameResult.Win: return Mate;
            case GameResult.Draw: return 0f;
        }
        if (d <= 0) return material.MaterialAdvantage(s);
        float best = float.NegativeInfinity;
        foreach (int m in game.LegalMoves(s))
        {
            float score = -Negamax(game.Apply(s, m), d - 1, -beta, -alpha);
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break; // alpha-beta cutoff
        }
        return best;
    }
}
