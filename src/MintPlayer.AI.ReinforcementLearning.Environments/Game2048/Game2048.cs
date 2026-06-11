using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

/// <summary>
/// Pure 2048 board mechanics on a 16-byte exponent grid (0 = empty, k = tile 2^k),
/// row-major 4×4. Shared by the generic RL environment and the n-tuple learner.
/// Standard rules: tiles slide and equal neighbors merge once per move (a merged tile
/// never re-merges within the same move); after every changing move a new tile spawns
/// in a uniformly random empty cell — 2 with probability 0.9, 4 with probability 0.1.
/// </summary>
public static class Board2048
{
    public const int ActionLeft = 0;
    public const int ActionDown = 1;
    public const int ActionRight = 2;
    public const int ActionUp = 3;
    public const int ActionCount = 4;

    // Cell visit order per (action, line): the slide direction is "toward index 0".
    private static readonly int[][][] Lines = BuildLines();

    private static int[][][] BuildLines()
    {
        var lines = new int[4][][];
        lines[ActionLeft] = [.. Enumerable.Range(0, 4).Select(r => Enumerable.Range(0, 4).Select(c => r * 4 + c).ToArray())];
        lines[ActionRight] = [.. Enumerable.Range(0, 4).Select(r => Enumerable.Range(0, 4).Select(c => r * 4 + (3 - c)).ToArray())];
        lines[ActionUp] = [.. Enumerable.Range(0, 4).Select(c => Enumerable.Range(0, 4).Select(r => r * 4 + c).ToArray())];
        lines[ActionDown] = [.. Enumerable.Range(0, 4).Select(c => Enumerable.Range(0, 4).Select(r => (3 - r) * 4 + c).ToArray())];
        return lines;
    }

    /// <summary>
    /// Applies a move in place. Returns false (board untouched) if the move is illegal.
    /// <paramref name="mergedValueSum"/> is the classic game-score increment (sum of the
    /// VALUES of tiles created by merges); <paramref name="mergedExponentSum"/> is the
    /// log2 variant used as a well-scaled RL reward.
    /// </summary>
    public static bool ApplyMove(Span<byte> board, int action, out int mergedExponentSum, out int mergedValueSum)
    {
        mergedExponentSum = 0;
        mergedValueSum = 0;
        bool moved = false;
        Span<byte> line = stackalloc byte[4];

        foreach (var map in Lines[action])
        {
            for (int i = 0; i < 4; i++) line[i] = board[map[i]];
            if (SlideLine(line, ref mergedExponentSum, ref mergedValueSum))
            {
                moved = true;
                for (int i = 0; i < 4; i++) board[map[i]] = line[i];
            }
        }
        return moved;
    }

    /// <summary>Slides one 4-cell line toward index 0, merging equal pairs once. True if it changed.</summary>
    internal static bool SlideLine(Span<byte> line, ref int mergedExponentSum, ref int mergedValueSum)
    {
        Span<byte> result = stackalloc byte[4];
        int write = 0;
        byte pending = 0;

        for (int read = 0; read < 4; read++)
        {
            byte tile = line[read];
            if (tile == 0) continue;

            if (pending == 0)
            {
                pending = tile;
            }
            else if (pending == tile)
            {
                byte merged = (byte)Math.Min(pending + 1, 15); // cap exponent at 4 bits
                result[write++] = merged;
                mergedExponentSum += merged;
                mergedValueSum += 1 << merged;
                pending = 0;
            }
            else
            {
                result[write++] = pending;
                pending = tile;
            }
        }
        if (pending != 0) result[write++] = pending;

        bool changed = false;
        for (int i = 0; i < 4; i++)
        {
            if (line[i] != result[i]) changed = true;
            line[i] = result[i];
        }
        return changed;
    }

    public static bool[] ValidMoves(ReadOnlySpan<byte> board)
    {
        var mask = new bool[ActionCount];
        Span<byte> scratch = stackalloc byte[16];
        for (int action = 0; action < ActionCount; action++)
        {
            board.CopyTo(scratch);
            mask[action] = ApplyMove(scratch, action, out _, out _);
        }
        return mask;
    }

    public static bool AnyMoveAvailable(ReadOnlySpan<byte> board)
    {
        // Cheap checks first: an empty cell or an equal neighbor means a move exists.
        for (int i = 0; i < 16; i++)
            if (board[i] == 0) return true;
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                if (c < 3 && board[r * 4 + c] == board[r * 4 + c + 1]) return true;
                if (r < 3 && board[r * 4 + c] == board[(r + 1) * 4 + c]) return true;
            }
        return false;
    }

    /// <summary>Spawns a 2 (p=0.9) or 4 (p=0.1) in a uniformly random empty cell.</summary>
    public static void Spawn(Span<byte> board, Xoshiro256StarStar rng)
    {
        int empty = 0;
        for (int i = 0; i < 16; i++)
            if (board[i] == 0) empty++;
        if (empty == 0)
            throw new InvalidOperationException("Cannot spawn on a full board.");

        int pick = rng.NextInt(empty);
        byte tile = rng.NextDouble() < 0.9 ? (byte)1 : (byte)2;
        for (int i = 0; i < 16; i++)
            if (board[i] == 0 && pick-- == 0)
            {
                board[i] = tile;
                return;
            }
    }

    public static int MaxExponent(ReadOnlySpan<byte> board)
    {
        byte max = 0;
        foreach (byte b in board) max = Math.Max(max, b);
        return max;
    }

    public static string Render(ReadOnlySpan<byte> board)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("┌──────┬──────┬──────┬──────┐");
        for (int r = 0; r < 4; r++)
        {
            sb.Append('│');
            for (int c = 0; c < 4; c++)
            {
                byte e = board[r * 4 + c];
                sb.Append(e == 0 ? "      " : $"{1 << e,5} ").Append('│');
            }
            sb.AppendLine();
            sb.AppendLine(r < 3 ? "├──────┼──────┼──────┼──────┤" : "└──────┴──────┴──────┴──────┘");
        }
        return sb.ToString();
    }
}
