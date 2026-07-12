namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// The AlphaZero move encoding: a fixed <c>64 × 73 = 4672</c> action space (from-square × move-type plane), which is
/// the policy-head width. The 73 planes are 56 "queen" moves (8 directions × 7 distances — this also carries
/// queen-promotions and every sliding/king/pawn move), 8 knight moves, and 9 underpromotions (Knight/Bishop/Rook ×
/// {capture-left, straight, capture-right}). Distinct legal moves always map to distinct indices, so the search can
/// tell them apart; queen-promotions ride the queen planes (decoded promotion is inferred as Queen when a pawn
/// reaches the last rank — see <see cref="ChessGame.Apply"/>).
/// </summary>
public static class ChessMoveEncoding
{
    public const int PlanesPerSquare = 73;
    public const int Size = 64 * PlanesPerSquare; // 4672

    // 8 sliding directions (df, dr), indices 0..7 — shared by encode and decode.
    private static readonly (int df, int dr)[] Dirs =
        [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
    // 8 knight deltas (df, dr), indices 0..7.
    private static readonly (int df, int dr)[] Knights =
        [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

    private static int File(int sq) => sq & 7;
    private static int Rank(int sq) => sq >> 3;

    /// <summary>The action index (0..4671) for a legal move. Queen-promotions use the queen planes; N/B/R
    /// promotions use the underpromotion planes.</summary>
    public static int Encode(ChessMove move)
    {
        int df = File(move.To) - File(move.From);
        int dr = Rank(move.To) - Rank(move.From);
        int plane;

        if (move.Promotion is PieceType.Knight or PieceType.Bishop or PieceType.Rook)
        {
            int pieceIdx = move.Promotion switch { PieceType.Knight => 0, PieceType.Bishop => 1, _ => 2 };
            plane = 64 + pieceIdx * 3 + (df + 1); // df ∈ {-1,0,1} → {capture-left, straight, capture-right}
        }
        else if (IsKnight(df, dr))
        {
            plane = 56 + KnightIndex(df, dr);
        }
        else
        {
            int dir = DirIndex(Math.Sign(df), Math.Sign(dr));
            int dist = Math.Max(Math.Abs(df), Math.Abs(dr));
            plane = dir * 7 + (dist - 1);
        }
        return move.From * PlanesPerSquare + plane;
    }

    /// <summary>Decodes an action index to a move. A queen-promotion decodes with <see cref="PieceType.None"/> — the
    /// caller promotes a pawn reaching the last rank to a Queen by default.</summary>
    public static ChessMove Decode(int index)
    {
        int from = index / PlanesPerSquare, plane = index % PlanesPerSquare;
        int f = File(from), r = Rank(from);

        if (plane >= 64)
        {
            int p = plane - 64;
            var promo = (p / 3) switch { 0 => PieceType.Knight, 1 => PieceType.Bishop, _ => PieceType.Rook };
            int df = (p % 3) - 1;
            int dr = r == 6 ? 1 : -1; // a pawn on the 7th rank promotes upward, on the 2nd rank downward
            return new ChessMove((byte)from, (byte)((r + dr) * 8 + (f + df)), promo);
        }
        if (plane >= 56)
        {
            var (df, dr) = Knights[plane - 56];
            return new ChessMove((byte)from, (byte)((r + dr) * 8 + (f + df)));
        }
        {
            var (df, dr) = Dirs[plane / 7];
            int dist = plane % 7 + 1;
            return new ChessMove((byte)from, (byte)((r + dr * dist) * 8 + (f + df * dist)));
        }
    }

    private static bool IsKnight(int df, int dr)
    {
        int a = Math.Abs(df), b = Math.Abs(dr);
        return (a == 1 && b == 2) || (a == 2 && b == 1);
    }

    private static int KnightIndex(int df, int dr)
    {
        for (int i = 0; i < Knights.Length; i++) if (Knights[i] == (df, dr)) return i;
        throw new ArgumentException($"Not a knight move: ({df},{dr}).");
    }

    private static int DirIndex(int sdf, int sdr)
    {
        for (int i = 0; i < Dirs.Length; i++) if (Dirs[i] == (sdf, sdr)) return i;
        throw new ArgumentException($"Not a straight/diagonal direction: ({sdf},{sdr}).");
    }
}
