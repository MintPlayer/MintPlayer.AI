namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// A fixed, deterministic material player: negamax (alpha-beta) to <paramref name="depth"/> plies over any
/// <see cref="IZeroSumGame{TState}"/>, scoring leaves by the game's <see cref="IMaterialScore{TState}"/>
/// (side-to-move relative). A terminal loss/win counts as ∓Mate, a draw as 0. Originally the Lab's chess
/// strength yardstick (M42), promoted to the library for any material game (draughts M47): deeper = stronger,
/// so it doesn't saturate as a trained net improves.
/// </summary>
public sealed class MaterialMinimaxPlayer<TState>(IZeroSumGame<TState> game, IMaterialScore<TState> material, int depth)
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
            case GameResult.Loss: return -Mate; // side to move has lost
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
