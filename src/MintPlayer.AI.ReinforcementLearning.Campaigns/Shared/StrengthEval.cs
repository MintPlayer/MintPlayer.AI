using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>The outcome of a <see cref="StrengthEval"/> match, from the net's side. <paramref name="AvgEndMaterial"/>
/// is the average terminal material advantage in the game's units (chess: pawns; draughts: man-units).</summary>
public sealed record StrengthResult(int Wins, int Draws, int Losses, double Score, double AvgEndMaterial);

/// <summary>
/// A NON-SATURATING strength eval for a trained self-play net: plays it (net + MCTS, argmax, no root noise)
/// against a fixed <see cref="MaterialMinimaxPlayer{TState}"/> of tunable depth, over N colour-alternated games
/// with a short random opening for variety. Why: <c>winRate-vs-random</c> SATURATES and material-in-self-play is
/// self-referential — neither can tell whether training actually made the net <i>play</i> better. A depth-D
/// material player is a FIXED, EXTERNAL, monotonically-stronger yardstick, so raising the depth keeps the metric
/// informative as the net grows. Generalized out of the Lab's chess eval (M42 → M47) — works for any
/// <see cref="IZeroSumGame{TState}"/> with an <see cref="IMaterialScore{TState}"/>.
/// </summary>
public static class StrengthEval
{
    public static StrengthResult Run<TState>(IZeroSumGame<TState> game, IMaterialScore<TState> material,
        IPolicyValueNet net, int sims, int depth, int games, int maxPlies, int openingPlies, ulong seed)
    {
        var cfg = new Mcts.Config(Simulations: sims, RootNoiseFrac: 0f); // eval: deterministic, no exploration noise
        var opponent = new MaterialMinimaxPlayer<TState>(game, material, depth);
        var rng = new Xoshiro256StarStar(seed);

        // Leaf evaluator: masked-softmax policy priors over legal moves + tanh value (mirrors SelfPlayCampaign.Evaluate).
        Mcts.Evaluate<TState> evaluate = s =>
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
            int netSide = g % 2 == 0 ? 1 : 2;                                    // alternate colours
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
            else draws++;                                                         // true draw or ply-cap
            float mat = material.MaterialAdvantage(state);                        // side-to-move relative → net relative
            materialSum += mover == netSide ? mat : -mat;
        }

        return new StrengthResult(wins, draws, losses, (wins + 0.5 * draws) / games, materialSum / games);
    }

    private static int NetMove<TState>(IZeroSumGame<TState> game, TState state, Mcts.Evaluate<TState> evaluate,
        Mcts.Config cfg, Xoshiro256StarStar rng)
    {
        float[] pi = Mcts.Search(game, state, evaluate, cfg, rng);
        int best = 0;
        for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[best]) best = a;
        return best;
    }

    private static int RandomMove<TState>(IZeroSumGame<TState> game, TState state, Xoshiro256StarStar rng)
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
