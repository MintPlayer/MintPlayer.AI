namespace MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

/// <summary>The two rule sets the single-source engine is parameterized over (PLAN M47): international
/// 10×10 "dammen" (flying kings, men capture backward, majority rule) — the showcase variant — and
/// english 8×8 checkers (one-step kings, men capture forward only, any complete capture allowed).</summary>
public enum DraughtsVariant { International10, English8 }

/// <summary>A complete move: a quiet step, or a full capture sequence — from/to squares (0..N²−1,
/// = rank·N + file) plus the captured squares in jump order. A multi-jump is ONE move; a capturing
/// king may end where it started (<paramref name="From"/> == <paramref name="To"/> is legal).</summary>
public sealed record DraughtsMove(int From, int To, IReadOnlyList<int> Captures);

/// <summary>
/// A draughts position — the public, ergonomic view (mailbox <see cref="Squares"/>, side to move,
/// no-progress clock). This is a thin facade over the single-source engine transpiled from
/// <c>polyglot/draughts_solver.pg</c> (<c>PgDraughtsState</c>), which also carries the variant's rule
/// flags. The board is immutable: every move produces a fresh state. Squares are 0..N²−1
/// (file = sq % N, rank = sq / N); play is on the (file+rank)-even squares; White moves toward higher
/// ranks and moves first in both variants. Cells: 0 = empty, +1/+2 = White man/king, −1/−2 = Black.
/// </summary>
public sealed class DraughtsState
{
    internal readonly PgDraughtsState Core;
    private sbyte[]? _squares;

    internal DraughtsState(PgDraughtsState core) => Core = core;

    public DraughtsState(DraughtsVariant variant, sbyte[] squares, bool whiteToMove, int noProgress = 0)
    {
        var template = TemplateFor(variant);
        if (squares.Length != template.size * template.size)
            throw new ArgumentException($"Expected {template.size * template.size} cells for {variant}, got {squares.Length}.", nameof(squares));
        var cells = new List<int>(squares.Length);
        for (int i = 0; i < squares.Length; i++) cells.Add(squares[i]);
        Core = new PgDraughtsState(template.size, template.flyingKings, template.menCaptureBackward,
            template.majorityCapture, template.noProgressLimit, cells, whiteToMove, noProgress);
    }

    /// <summary>The N×N mailbox (a cached snapshot of the core board — read-only).</summary>
    public sbyte[] Squares
    {
        get
        {
            if (_squares is null)
            {
                _squares = new sbyte[Core.squares.Count];
                for (int i = 0; i < _squares.Length; i++) _squares[i] = (sbyte)Core.squares[i];
            }
            return _squares;
        }
    }

    public int Size => Core.size;
    public bool WhiteToMove => Core.whiteToMove;
    /// <summary>Plies without a capture or man move; <see cref="NoProgressLimit"/> ⇒ draw by rule.</summary>
    public int NoProgress => Core.noProgress;
    public int NoProgressLimit => Core.noProgressLimit;

    public static DraughtsState StartPosition(DraughtsVariant variant) => new(TemplateFor(variant));

    // The rule flags live in ONE place — the .pg variant factories; a custom-board state copies
    // them off a template start state instead of re-tabulating them here.
    private static PgDraughtsState TemplateFor(DraughtsVariant variant)
        => variant == DraughtsVariant.English8 ? PgDraughtsState.english() : PgDraughtsState.international();

    internal static PgDraughtsMove ToPg(DraughtsMove m) => new(m.From, m.To, [.. m.Captures]);
    internal static DraughtsMove FromPg(PgDraughtsMove m) => new(m.from, m.to, m.captures);
}

/// <summary>
/// The draughts rules — a public facade over the single-source engine (<c>draughts_solver.pg</c> →
/// <c>PgDraughtsState</c>): complete-capture-sequence move generation (forced captures, majority rule,
/// flying kings, Turkish strike, promotion only at move end), make-move, and the no-progress draw
/// rule. Correctness is pinned by <c>perft</c>, which recurses entirely inside the generated core.
/// </summary>
public static class DraughtsRules
{
    /// <summary>All legal complete moves for the side to move (capture sequences when any capture
    /// exists — deduped by FMJD move identity (from, to, captured set) — else quiet moves).</summary>
    public static List<DraughtsMove> LegalMoves(DraughtsState s)
    {
        var core = s.Core.legalMoves();
        var moves = new List<DraughtsMove>(core.Count);
        foreach (var m in core) moves.Add(DraughtsState.FromPg(m));
        return moves;
    }

    /// <summary>The position after <paramref name="move"/> (a fresh state; <paramref name="s"/> is unchanged).</summary>
    public static DraughtsState MakeMove(DraughtsState s, DraughtsMove move) => new(s.Core.makeMove(DraughtsState.ToPg(move)));

    /// <summary>Perft: the number of leaf nodes at <paramref name="depth"/> — the movegen correctness oracle
    /// (M47.1 gate). The count is produced entirely inside the generated engine (i32 there; widened here).</summary>
    public static long Perft(DraughtsState s, int depth) => s.Core.perft(depth);
}
