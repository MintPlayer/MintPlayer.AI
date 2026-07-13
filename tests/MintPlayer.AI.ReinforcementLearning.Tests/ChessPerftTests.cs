using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Perft (performance test): count the legal-move tree to a fixed depth and match the published node counts for
/// standard positions. This is the non-negotiable correctness gate for the chess move generator (PLAN M39.2) —
/// it exercises castling, en passant, promotion, pins, and check evasion in exactly the ways hand-written tests
/// miss. Shallow depths run in the fast bucket; the multi-million-node depths are marked Slow.
/// </summary>
public class ChessPerftTests
{
    private const string StartPos = ChessFen.StartFen;
    private const string Kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    private const string Position3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    private const string Position4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    private const string Position5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    private const string Position6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    [Theory]
    // Startpos — the baseline.
    [InlineData(StartPos, 1, 20L)]
    [InlineData(StartPos, 2, 400L)]
    [InlineData(StartPos, 3, 8902L)]
    // Kiwipete — dense with castling, pins and captures.
    [InlineData(Kiwipete, 1, 48L)]
    [InlineData(Kiwipete, 2, 2039L)]
    [InlineData(Kiwipete, 3, 97862L)]
    // Position 3 — en passant and rook endgame checks.
    [InlineData(Position3, 1, 14L)]
    [InlineData(Position3, 2, 191L)]
    [InlineData(Position3, 3, 2812L)]
    // Position 4 — promotions and discovered checks.
    [InlineData(Position4, 1, 6L)]
    [InlineData(Position4, 2, 264L)]
    [InlineData(Position4, 3, 9467L)]
    // Position 5 — castling rights and promotions.
    [InlineData(Position5, 1, 44L)]
    [InlineData(Position5, 2, 1486L)]
    [InlineData(Position5, 3, 62379L)]
    // Position 6 — a quiet middlegame.
    [InlineData(Position6, 1, 46L)]
    [InlineData(Position6, 2, 2079L)]
    [InlineData(Position6, 3, 89890L)]
    public void Perft_matches_published_counts(string fen, int depth, long expected)
        => Assert.Equal(expected, ChessRules.Perft(ChessFen.Parse(fen), depth));

    [Theory]
    [Trait("Category", "Slow")]
    [InlineData(StartPos, 4, 197281L)]
    [InlineData(StartPos, 5, 4865609L)]
    [InlineData(Kiwipete, 4, 4085603L)]
    [InlineData(Position3, 4, 43238L)]
    [InlineData(Position3, 5, 674624L)]
    [InlineData(Position4, 4, 422333L)]
    [InlineData(Position5, 4, 2103487L)]
    public void Perft_deep_matches_published_counts(string fen, int depth, long expected)
        => Assert.Equal(expected, ChessRules.Perft(ChessFen.Parse(fen), depth));
}
