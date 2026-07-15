using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

/// <summary>A Connect-4 position: 7×6 cells (0 = empty, 1 / 2 = the two players) plus the side to move.
/// Immutable from the caller's view — <see cref="Connect4Game.Apply"/> always returns a fresh state.</summary>
public sealed class Connect4State
{
    public const int Columns = 7;
    public const int Rows = 6;

    /// <summary>Row-major, row 0 = bottom: cell (row, col) = <c>Cells[row * Columns + col]</c>.</summary>
    public byte[] Cells { get; }
    public int ToMove { get; } // 1 or 2

    public Connect4State(byte[] cells, int toMove) { Cells = cells; ToMove = toMove; }
}

/// <summary>
/// Connect-4 as an <see cref="IZeroSumGame{TState}"/> — the cheap first consumer of the self-play stack (tiny rules,
/// converges from random self-play in minutes on CPU, and a shallow negamax gives an exact test oracle). The action
/// index is the column (0..6); the observation is a side-to-move-relative two-plane (mine / theirs) board.
/// </summary>
[Register(typeof(IZeroSumGame<Connect4State>), ServiceLifetime.Singleton, "ReinforcementLearningGames")]
public sealed class Connect4Game : IZeroSumGame<Connect4State>
{
    public int PolicySize => Connect4State.Columns;              // one action per column
    public int ObservationSize => 2 * Connect4State.Columns * Connect4State.Rows; // mine + theirs planes

    public Connect4State Root(ulong? seed = null) => new(new byte[Connect4State.Columns * Connect4State.Rows], toMove: 1);

    public IReadOnlyList<int> LegalMoves(Connect4State state)
    {
        var moves = new List<int>(Connect4State.Columns);
        int topRowBase = (Connect4State.Rows - 1) * Connect4State.Columns;
        for (int col = 0; col < Connect4State.Columns; col++)
            if (state.Cells[topRowBase + col] == 0) moves.Add(col); // top cell empty → column not full
        return moves;
    }

    public Connect4State Apply(Connect4State state, int move)
    {
        var cells = (byte[])state.Cells.Clone();
        for (int row = 0; row < Connect4State.Rows; row++)
        {
            int idx = row * Connect4State.Columns + move;
            if (cells[idx] == 0) { cells[idx] = (byte)state.ToMove; break; }
        }
        return new Connect4State(cells, toMove: 3 - state.ToMove);
    }

    public GameResult Result(Connect4State state)
    {
        int opponent = 3 - state.ToMove;
        if (HasFour(state.Cells, opponent)) return GameResult.Loss; // the player who just moved completed a line
        if (HasFour(state.Cells, state.ToMove)) return GameResult.Win; // defensive; unreachable in normal play
        return IsFull(state.Cells) ? GameResult.Draw : GameResult.Ongoing;
    }

    public void WriteObservation(Connect4State state, Span<float> destination)
    {
        int n = Connect4State.Columns * Connect4State.Rows;
        int mine = state.ToMove, theirs = 3 - state.ToMove;
        for (int i = 0; i < n; i++)
        {
            destination[i] = state.Cells[i] == mine ? 1f : 0f;
            destination[n + i] = state.Cells[i] == theirs ? 1f : 0f;
        }
    }

    private static bool IsFull(byte[] cells)
    {
        int topRowBase = (Connect4State.Rows - 1) * Connect4State.Columns;
        for (int col = 0; col < Connect4State.Columns; col++)
            if (cells[topRowBase + col] == 0) return false;
        return true;
    }

    // True if player p has any four-in-a-row (horizontal, vertical, or either diagonal).
    private static bool HasFour(byte[] cells, int p)
    {
        const int C = Connect4State.Columns, R = Connect4State.Rows;
        for (int row = 0; row < R; row++)
            for (int col = 0; col < C; col++)
            {
                if (cells[row * C + col] != p) continue;
                if (col + 3 < C && Line(cells, p, row, col, 0, 1)) return true;   // →
                if (row + 3 < R && Line(cells, p, row, col, 1, 0)) return true;   // ↑
                if (row + 3 < R && col + 3 < C && Line(cells, p, row, col, 1, 1)) return true;   // ↗
                if (row + 3 < R && col - 3 >= 0 && Line(cells, p, row, col, 1, -1)) return true;  // ↖
            }
        return false;
    }

    private static bool Line(byte[] cells, int p, int row, int col, int dRow, int dCol)
    {
        for (int step = 1; step < 4; step++)
            if (cells[(row + step * dRow) * Connect4State.Columns + (col + step * dCol)] != p) return false;
        return true;
    }
}
