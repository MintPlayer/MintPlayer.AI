namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>Piece type (colour-agnostic). <see cref="None"/> = empty square.</summary>
public enum PieceType : byte { None = 0, Pawn = 1, Knight = 2, Bishop = 3, Rook = 4, Queen = 5, King = 6 }

/// <summary>A move: from/to squares (0..63, = rank*8 + file, a1 = 0), plus the promotion piece for a pawn reaching
/// the last rank (<see cref="PieceType.None"/> otherwise). Castling, en passant, and double-push are inferred from
/// the board when the move is made, so they need no extra flag here.</summary>
public readonly record struct ChessMove(byte From, byte To, PieceType Promotion = PieceType.None);

/// <summary>
/// A chess position — the public, ergonomic view (mailbox <see cref="Squares"/>, side to move, castling rights,
/// en-passant target, halfmove clock). This is a thin facade over the single-source engine transpiled from
/// <c>polyglot/chess_solver.pg</c> (<c>PgChessState</c>) — the same source the browser client runs. The board is
/// immutable: every move produces a fresh state. Squares are 0..63 (file = sq % 8, rank = sq / 8); White moves
/// toward higher ranks. Cells: 0 = empty, +1..+6 = White P,N,B,R,Q,K, −1..−6 = Black (sign = colour).
/// <para>Threefold-repetition is intentionally not modelled (see PLAN M39); the 50-move rule and the self-play
/// ply cap bound looping games.</para>
/// </summary>
public sealed class ChessState
{
    internal readonly PgChessState Core;
    private sbyte[]? _squares;

    public const byte CastleWK = 1, CastleWQ = 2, CastleBK = 4, CastleBQ = 8;

    internal ChessState(PgChessState core) => Core = core;

    public ChessState(sbyte[] squares, bool whiteToMove, byte castling, sbyte enPassant, byte halfmoveClock)
    {
        var cells = new List<int>(64);
        for (int i = 0; i < 64; i++) cells.Add(squares[i]);
        Core = new PgChessState(cells, whiteToMove, castling, enPassant, halfmoveClock);
    }

    /// <summary>The 64-cell mailbox (a cached snapshot of the core board — read-only).</summary>
    public sbyte[] Squares
    {
        get
        {
            if (_squares is null)
            {
                _squares = new sbyte[64];
                for (int i = 0; i < 64; i++) _squares[i] = (sbyte)Core.squares[i];
            }
            return _squares;
        }
    }

    public bool WhiteToMove => Core.whiteToMove;
    public byte Castling => (byte)Core.castling;   // bit 0 = White O-O, 1 = White O-O-O, 2 = Black O-O, 3 = Black O-O-O
    public sbyte EnPassant => (sbyte)Core.enPassant;
    public byte HalfmoveClock => (byte)Core.halfmoveClock;

    public static ChessState StartPosition() => ChessFen.Parse(ChessFen.StartFen);

    // Move ↔ generated-move mapping (promotion piece codes coincide with PieceType's byte values, so the cast is
    // a no-op renaming). Shared by ChessRules and ChessMoveEncoding — the two facades over the single source.
    internal static PgChessMove ToPg(ChessMove m) => new(m.From, m.To, (int)m.Promotion);
    internal static ChessMove FromPg(PgChessMove m) => new((byte)m.from, (byte)m.to, (PieceType)m.promotion);
}

/// <summary>
/// The chess rules — a public facade over the single-source engine (<c>chess_solver.pg</c> → <c>PgChessState</c>):
/// legal move generation (castling, en passant, promotion, pins/checks), make-move, attack detection, and terminal
/// detection (checkmate, stalemate, 50-move, insufficient material). Correctness is pinned by <c>perft</c>, which
/// recurses entirely inside the generated core (no per-node marshalling).
/// </summary>
public static class ChessRules
{
    public static bool IsSquareAttacked(ChessState s, int square, bool byWhite) => s.Core.isSquareAttacked(square, byWhite);

    public static int KingSquare(ChessState s, bool white) => s.Core.kingSquare(white);

    public static bool InCheck(ChessState s, bool white) => s.Core.inCheck(white);

    /// <summary>All fully-legal moves for the side to move (own king not left in check).</summary>
    public static List<ChessMove> LegalMoves(ChessState s)
    {
        var core = s.Core.legalMoves();
        var moves = new List<ChessMove>(core.Count);
        foreach (var m in core) moves.Add(ChessState.FromPg(m));
        return moves;
    }

    /// <summary>The position after <paramref name="move"/> (a fresh state; <paramref name="s"/> is unchanged).</summary>
    public static ChessState MakeMove(ChessState s, ChessMove move) => new(s.Core.makeMove(ChessState.ToPg(move)));

    public static bool IsFiftyMove(ChessState s) => s.Core.isFiftyMove();

    public static bool IsInsufficientMaterial(ChessState s) => s.Core.isInsufficientMaterial();

    /// <summary>Perft: the number of leaf nodes at <paramref name="depth"/> — the movegen correctness oracle. The
    /// count is produced entirely inside the generated engine (i32 there; widened to long here).</summary>
    public static long Perft(ChessState s, int depth) => s.Core.perft(depth);
}
