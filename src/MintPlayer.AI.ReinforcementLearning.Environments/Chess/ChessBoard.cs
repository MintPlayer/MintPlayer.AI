namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>Piece type (colour-agnostic). <see cref="None"/> = empty square.</summary>
public enum PieceType : byte { None = 0, Pawn = 1, Knight = 2, Bishop = 3, Rook = 4, Queen = 5, King = 6 }

/// <summary>A move: from/to squares (0..63, = rank*8 + file, a1 = 0), plus the promotion piece for a pawn reaching
/// the last rank (<see cref="PieceType.None"/> otherwise). Castling, en passant, and double-push are inferred from
/// the board when the move is made, so they need no extra flag here.</summary>
public readonly record struct ChessMove(byte From, byte To, PieceType Promotion = PieceType.None);

/// <summary>
/// A chess position (mailbox board + side to move + castling rights + en-passant target + halfmove clock) and the
/// full rules: legal move generation (castling, en passant, promotion, pins/checks), make-move, attack detection,
/// and terminal detection (checkmate, stalemate, 50-move, insufficient material). Correctness-first (clone-per-move,
/// no bitboards); verified by <c>perft</c>. Squares are 0..63 with file = sq &amp; 7, rank = sq &gt;&gt; 3; White moves
/// toward higher ranks. Board cells: 0 = empty, +1..+6 = White P,N,B,R,Q,K, −1..−6 = Black (sign = colour).
/// <para>Threefold-repetition is intentionally not modelled in v1 (see PLAN M39); the 50-move rule and the self-play
/// ply cap bound looping games.</para>
/// </summary>
public sealed class ChessState
{
    public sbyte[] Squares { get; }   // length 64
    public bool WhiteToMove { get; }
    public byte Castling { get; }     // bit 0 = White O-O, 1 = White O-O-O, 2 = Black O-O, 3 = Black O-O-O
    public sbyte EnPassant { get; }   // target square a pawn could capture onto, or -1
    public byte HalfmoveClock { get; }

    public const byte CastleWK = 1, CastleWQ = 2, CastleBK = 4, CastleBQ = 8;

    public ChessState(sbyte[] squares, bool whiteToMove, byte castling, sbyte enPassant, byte halfmoveClock)
    {
        Squares = squares;
        WhiteToMove = whiteToMove;
        Castling = castling;
        EnPassant = enPassant;
        HalfmoveClock = halfmoveClock;
    }

    public static ChessState StartPosition() => ChessFen.Parse(ChessFen.StartFen);
}

/// <summary>Static chess rules over <see cref="ChessState"/>.</summary>
public static class ChessRules
{
    private static int File(int sq) => sq & 7;
    private static int Rank(int sq) => sq >> 3;
    private static PieceType Type(sbyte piece) => (PieceType)Math.Abs(piece);
    private static bool IsWhite(sbyte piece) => piece > 0;

    private static readonly (int df, int dr)[] KnightDeltas =
        [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
    private static readonly (int df, int dr)[] KingDeltas =
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    private static readonly (int df, int dr)[] BishopDirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)];
    private static readonly (int df, int dr)[] RookDirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    // ── Attack detection ────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Whether <paramref name="square"/> is attacked by the given colour's pieces.</summary>
    public static bool IsSquareAttacked(ChessState s, int square, bool byWhite)
    {
        int tf = File(square), tr = Rank(square);
        var b = s.Squares;

        // Pawns: a pawn on the attacking side attacks diagonally toward its forward direction.
        int pawnDir = byWhite ? 1 : -1;
        sbyte pawn = (sbyte)(byWhite ? 1 : -1);
        foreach (int df in stackalloc[] { -1, 1 })
        {
            int f = tf + df, r = tr - pawnDir; // a pawn on (r) attacking (tr) sits one rank behind in its dir
            if ((uint)f < 8 && (uint)r < 8 && b[r * 8 + f] == pawn) return true;
        }

        foreach (var (df, dr) in KnightDeltas)
        {
            int f = tf + df, r = tr + dr;
            if ((uint)f < 8 && (uint)r < 8 && b[r * 8 + f] == (sbyte)(byWhite ? 2 : -2)) return true;
        }

        foreach (var (df, dr) in KingDeltas)
        {
            int f = tf + df, r = tr + dr;
            if ((uint)f < 8 && (uint)r < 8 && b[r * 8 + f] == (sbyte)(byWhite ? 6 : -6)) return true;
        }

        // Sliders: bishops/queens on diagonals, rooks/queens on ranks/files.
        if (RayHits(b, tf, tr, BishopDirs, byWhite, PieceType.Bishop)) return true;
        if (RayHits(b, tf, tr, RookDirs, byWhite, PieceType.Rook)) return true;
        return false;
    }

    private static bool RayHits(sbyte[] b, int tf, int tr, (int df, int dr)[] dirs, bool byWhite, PieceType slider)
    {
        foreach (var (df, dr) in dirs)
        {
            int f = tf + df, r = tr + dr;
            while ((uint)f < 8 && (uint)r < 8)
            {
                sbyte piece = b[r * 8 + f];
                if (piece != 0)
                {
                    if (IsWhite(piece) == byWhite)
                    {
                        var t = Type(piece);
                        if (t == slider || t == PieceType.Queen) return true;
                    }
                    break; // blocked
                }
                f += df; r += dr;
            }
        }
        return false;
    }

    public static int KingSquare(ChessState s, bool white)
    {
        sbyte king = (sbyte)(white ? 6 : -6);
        for (int sq = 0; sq < 64; sq++) if (s.Squares[sq] == king) return sq;
        return -1;
    }

    public static bool InCheck(ChessState s, bool white)
        => IsSquareAttacked(s, KingSquare(s, white), byWhite: !white);

    // ── Move generation ─────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>All fully-legal moves for the side to move (own king not left in check).</summary>
    public static List<ChessMove> LegalMoves(ChessState s)
    {
        var pseudo = PseudoLegal(s);
        var legal = new List<ChessMove>(pseudo.Count);
        foreach (var move in pseudo)
        {
            var next = MakeMove(s, move);
            if (!InCheck(next, white: s.WhiteToMove)) legal.Add(move); // the side that just moved must be safe
        }
        return legal;
    }

    private static List<ChessMove> PseudoLegal(ChessState s)
    {
        var moves = new List<ChessMove>(48);
        bool white = s.WhiteToMove;
        var b = s.Squares;

        for (int sq = 0; sq < 64; sq++)
        {
            sbyte piece = b[sq];
            if (piece == 0 || IsWhite(piece) != white) continue;
            int f = File(sq), r = Rank(sq);
            switch (Type(piece))
            {
                case PieceType.Pawn: PawnMoves(s, sq, f, r, white, moves); break;
                case PieceType.Knight: StepMoves(b, sq, f, r, white, KnightDeltas, moves); break;
                case PieceType.King: StepMoves(b, sq, f, r, white, KingDeltas, moves); CastleMoves(s, white, moves); break;
                case PieceType.Bishop: SlideMoves(b, sq, f, r, white, BishopDirs, moves); break;
                case PieceType.Rook: SlideMoves(b, sq, f, r, white, RookDirs, moves); break;
                case PieceType.Queen:
                    SlideMoves(b, sq, f, r, white, BishopDirs, moves);
                    SlideMoves(b, sq, f, r, white, RookDirs, moves);
                    break;
            }
        }
        return moves;
    }

    private static void StepMoves(sbyte[] b, int sq, int f, int r, bool white, (int df, int dr)[] deltas, List<ChessMove> moves)
    {
        foreach (var (df, dr) in deltas)
        {
            int nf = f + df, nr = r + dr;
            if ((uint)nf >= 8 || (uint)nr >= 8) continue;
            int to = nr * 8 + nf;
            sbyte target = b[to];
            if (target == 0 || IsWhite(target) != white) moves.Add(new ChessMove((byte)sq, (byte)to));
        }
    }

    private static void SlideMoves(sbyte[] b, int sq, int f, int r, bool white, (int df, int dr)[] dirs, List<ChessMove> moves)
    {
        foreach (var (df, dr) in dirs)
        {
            int nf = f + df, nr = r + dr;
            while ((uint)nf < 8 && (uint)nr < 8)
            {
                int to = nr * 8 + nf;
                sbyte target = b[to];
                if (target == 0) moves.Add(new ChessMove((byte)sq, (byte)to));
                else { if (IsWhite(target) != white) moves.Add(new ChessMove((byte)sq, (byte)to)); break; }
                nf += df; nr += dr;
            }
        }
    }

    private static void PawnMoves(ChessState s, int sq, int f, int r, bool white, List<ChessMove> moves)
    {
        var b = s.Squares;
        int dir = white ? 1 : -1;
        int startRank = white ? 1 : 6;
        int lastRank = white ? 7 : 0;

        // Single / double push.
        int one = (r + dir) * 8 + f;
        if (b[one] == 0)
        {
            AddPawn(moves, sq, one, r + dir == lastRank);
            if (r == startRank)
            {
                int two = (r + 2 * dir) * 8 + f;
                if (b[two] == 0) moves.Add(new ChessMove((byte)sq, (byte)two));
            }
        }

        // Captures (incl. en passant).
        foreach (int df in stackalloc[] { -1, 1 })
        {
            int nf = f + df, nr = r + dir;
            if ((uint)nf >= 8 || (uint)nr >= 8) continue;
            int to = nr * 8 + nf;
            sbyte target = b[to];
            if (target != 0 && IsWhite(target) != white) AddPawn(moves, sq, to, nr == lastRank);
            else if (to == s.EnPassant) moves.Add(new ChessMove((byte)sq, (byte)to)); // en passant onto the empty ep square
        }
    }

    private static void AddPawn(List<ChessMove> moves, int from, int to, bool promotion)
    {
        if (promotion)
        {
            moves.Add(new ChessMove((byte)from, (byte)to, PieceType.Queen));
            moves.Add(new ChessMove((byte)from, (byte)to, PieceType.Rook));
            moves.Add(new ChessMove((byte)from, (byte)to, PieceType.Bishop));
            moves.Add(new ChessMove((byte)from, (byte)to, PieceType.Knight));
        }
        else moves.Add(new ChessMove((byte)from, (byte)to));
    }

    private static void CastleMoves(ChessState s, bool white, List<ChessMove> moves)
    {
        var b = s.Squares;
        int rank = white ? 0 : 7;
        int kingSq = rank * 8 + 4;
        if (b[kingSq] != (sbyte)(white ? 6 : -6)) return;
        if (IsSquareAttacked(s, kingSq, byWhite: !white)) return; // can't castle out of check

        byte kSide = white ? ChessState.CastleWK : ChessState.CastleBK;
        byte qSide = white ? ChessState.CastleWQ : ChessState.CastleBQ;

        // King-side: squares f,g empty; king doesn't pass through/into attack.
        if ((s.Castling & kSide) != 0 && b[rank * 8 + 5] == 0 && b[rank * 8 + 6] == 0
            && !IsSquareAttacked(s, rank * 8 + 5, !white) && !IsSquareAttacked(s, rank * 8 + 6, !white))
            moves.Add(new ChessMove((byte)kingSq, (byte)(rank * 8 + 6)));

        // Queen-side: squares b,c,d empty; king passes over d,c (b need not be safe, only empty).
        if ((s.Castling & qSide) != 0 && b[rank * 8 + 1] == 0 && b[rank * 8 + 2] == 0 && b[rank * 8 + 3] == 0
            && !IsSquareAttacked(s, rank * 8 + 3, !white) && !IsSquareAttacked(s, rank * 8 + 2, !white))
            moves.Add(new ChessMove((byte)kingSq, (byte)(rank * 8 + 2)));
    }

    // ── Make move ───────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>The position after <paramref name="move"/> (a fresh state; <paramref name="s"/> is unchanged).
    /// Castling / en passant / promotion / double-push are inferred from the board.</summary>
    public static ChessState MakeMove(ChessState s, ChessMove move)
    {
        var b = (sbyte[])s.Squares.Clone();
        bool white = s.WhiteToMove;
        sbyte piece = b[move.From];
        var type = Type(piece);
        bool capture = b[move.To] != 0;
        sbyte newEp = -1;
        byte castling = s.Castling;

        b[move.From] = 0;

        // En passant capture: the taken pawn sits beside the destination, not on it.
        if (type == PieceType.Pawn && move.To == s.EnPassant && s.EnPassant >= 0)
        {
            int capturedSq = move.To + (white ? -8 : 8);
            b[capturedSq] = 0;
            capture = true;
        }

        // Place the piece (promotion swaps in the chosen piece).
        b[move.To] = move.Promotion != PieceType.None
            ? (sbyte)((white ? 1 : -1) * (int)move.Promotion)
            : piece;

        // Double push sets the en-passant target square.
        if (type == PieceType.Pawn && Math.Abs(Rank(move.To) - Rank(move.From)) == 2)
            newEp = (sbyte)((move.From + move.To) / 2);

        // Castling: move the rook too.
        if (type == PieceType.King && Math.Abs(File(move.To) - File(move.From)) == 2)
        {
            int rank = Rank(move.From);
            if (File(move.To) == 6) { b[rank * 8 + 5] = b[rank * 8 + 7]; b[rank * 8 + 7] = 0; } // king-side
            else { b[rank * 8 + 3] = b[rank * 8 + 0]; b[rank * 8 + 0] = 0; }                     // queen-side
        }

        // Update castling rights: king move clears both; rook move/capture clears that corner.
        if (type == PieceType.King) castling &= (byte)(white ? ~(ChessState.CastleWK | ChessState.CastleWQ) : ~(ChessState.CastleBK | ChessState.CastleBQ));
        castling &= CastlingMask(move.From);
        castling &= CastlingMask(move.To); // a rook captured on its home square loses that right too

        byte halfmove = (byte)(type == PieceType.Pawn || capture ? 0 : s.HalfmoveClock + 1);
        return new ChessState(b, !white, castling, newEp, halfmove);
    }

    // Clears the castling right whose rook/king home square is 'sq' (identity mask otherwise).
    private static byte CastlingMask(int sq) => sq switch
    {
        0 => unchecked((byte)~ChessState.CastleWQ),   // a1 rook
        7 => unchecked((byte)~ChessState.CastleWK),   // h1 rook
        56 => unchecked((byte)~ChessState.CastleBQ),  // a8 rook
        63 => unchecked((byte)~ChessState.CastleBK),  // h8 rook
        _ => 0xFF,
    };

    // ── Terminal detection ──────────────────────────────────────────────────────────────────────────────────────

    public static bool IsFiftyMove(ChessState s) => s.HalfmoveClock >= 100;

    public static bool IsInsufficientMaterial(ChessState s)
    {
        int knights = 0, bishops = 0;
        foreach (sbyte p in s.Squares)
        {
            switch (Type(p))
            {
                case PieceType.None or PieceType.King: break;
                case PieceType.Knight: knights++; break;
                case PieceType.Bishop: bishops++; break;
                default: return false; // a pawn/rook/queen = enough material
            }
        }
        return knights + bishops <= 1; // K vs K, K+N vs K, K+B vs K (not exhaustive, but the common draws)
    }

    /// <summary>Perft: the number of leaf nodes at <paramref name="depth"/> — the movegen correctness oracle.</summary>
    public static long Perft(ChessState s, int depth)
    {
        if (depth == 0) return 1;
        var moves = LegalMoves(s);
        if (depth == 1) return moves.Count;
        long nodes = 0;
        foreach (var move in moves) nodes += Perft(MakeMove(s, move), depth - 1);
        return nodes;
    }
}
