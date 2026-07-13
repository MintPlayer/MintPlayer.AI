using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

/// <summary>
/// A depth-limited negamax solver for Connect-4 — the exact oracle the self-play tests measure MCTS against (the same
/// role Kociemba plays for the cube and BFS for Rush Hour). Scores are from the side-to-move's perspective:
/// <c>+1</c> = a forced win within the horizon, <c>-1</c> = a forced loss, <c>0</c> = a draw or undecided within the
/// depth. Full-game solving is exponential, so callers pass a modest <c>maxDepth</c> and use it on tactical positions.
/// </summary>
public static class Connect4Solver
{
    private static readonly Connect4Game Game = new();

    /// <summary>The best move for the side to move (preferring a forced win, else a draw, avoiding a loss) and its
    /// negamax score in {-1, 0, +1} within <paramref name="maxDepth"/> plies.</summary>
    public static (int Score, int BestMove) Solve(Connect4State state, int maxDepth)
    {
        int bestScore = int.MinValue, bestMove = -1;
        foreach (int move in Game.LegalMoves(state))
        {
            int score = -Negamax(Game.Apply(state, move), maxDepth - 1);
            if (score > bestScore) { bestScore = score; bestMove = move; }
        }
        return (bestScore == int.MinValue ? 0 : bestScore, bestMove);
    }

    /// <summary>The forced result of <paramref name="state"/> for its side to move within <paramref name="depth"/> plies.</summary>
    public static int Negamax(Connect4State state, int depth)
    {
        switch (Game.Result(state))
        {
            case GameResult.Loss: return -1; // side to move has already lost
            case GameResult.Win: return 1;
            case GameResult.Draw: return 0;
        }
        if (depth <= 0) return 0; // undecided within the horizon → treat as neutral

        int best = -2; // below the -1..+1 range
        foreach (int move in Game.LegalMoves(state))
        {
            int value = -Negamax(Game.Apply(state, move), depth - 1);
            if (value > best) best = value;
            if (best == 1) break; // a forced win can't be improved on — prune
        }
        return best;
    }
}
