using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Hand-verified capture-dense positions — the second half of the M47.1 gate, exercising exactly the
/// rules perft-from-startpos reaches only statistically: the majority rule, the Turkish strike (a
/// captured piece stays on the board, blocks, and cannot be re-jumped), flying-king landing choice,
/// promotion only at move end (a man jumping THROUGH the crown row stays a man), forced/complete
/// captures, men's move-vs-capture direction rules, the no-progress draw, and loss-when-blocked.
/// Squares are rank·N + file; pieces: +1/+2 White man/king, −1/−2 Black.
/// </summary>
public class DraughtsRulesTests
{
    private const sbyte WM = 1, WK = 2, BM = -1, BK = -2;

    private static DraughtsState State(DraughtsVariant v, bool whiteToMove, params (int File, int Rank, sbyte Piece)[] pieces)
    {
        int size = v == DraughtsVariant.English8 ? 8 : 10;
        var board = new sbyte[size * size];
        foreach (var (f, r, p) in pieces) board[r * size + f] = p;
        return new DraughtsState(v, board, whiteToMove);
    }

    [Fact]
    public void Start_positions_have_the_right_armies()
    {
        var intl = DraughtsState.StartPosition(DraughtsVariant.International10).Squares;
        Assert.Equal(20, intl.Count(p => p == WM));
        Assert.Equal(20, intl.Count(p => p == BM));
        var eng = DraughtsState.StartPosition(DraughtsVariant.English8).Squares;
        Assert.Equal(12, eng.Count(p => p == WM));
        Assert.Equal(12, eng.Count(p => p == BM));
    }

    // White man on (2,2); black men on (3,3), (1,3), (1,5). Right branch captures 1 (over (3,3) to
    // (4,4)); left branch captures 2 (over (1,3) to (0,4), then over (1,5) to (2,6)). International
    // majority rule: ONLY the 2-capture is legal.
    [Fact]
    public void International_majority_rule_keeps_only_the_maximum_capture()
    {
        var s = State(DraughtsVariant.International10, true, (2, 2, WM), (3, 3, BM), (1, 3, BM), (1, 5, BM));
        var move = Assert.Single(DraughtsRules.LegalMoves(s));
        Assert.Equal(22, move.From);
        Assert.Equal(62, move.To);
        Assert.Equal(new[] { 31, 51 }, move.Captures.Order());
    }

    // The same pattern on the english board: no majority rule, so BOTH complete sequences are legal —
    // but a capture must still run to completion (stopping after the first jump of the 2-jump branch
    // is not a move).
    [Fact]
    public void English_allows_any_capture_but_it_must_run_to_completion()
    {
        var s = State(DraughtsVariant.English8, true, (2, 2, WM), (3, 3, BM), (1, 3, BM), (1, 5, BM));
        var moves = DraughtsRules.LegalMoves(s);
        Assert.Equal(2, moves.Count);
        Assert.All(moves, m => Assert.NotEmpty(m.Captures));          // captures are forced
        Assert.DoesNotContain(moves, m => m.To == 4 * 8 + 0);         // no stopping mid-sequence at (0,4)
        Assert.Contains(moves, m => m.Captures.Count == 1 && m.To == 4 * 8 + 4);
        Assert.Contains(moves, m => m.Captures.Count == 2 && m.To == 6 * 8 + 2);
    }

    // White man between two black men, one ahead, one behind. English men capture forward only
    // (1 move); international men capture in all four directions (2 moves, both single captures —
    // the majority rule keeps both).
    [Fact]
    public void Men_capture_direction_differs_between_variants()
    {
        var eng = State(DraughtsVariant.English8, true, (2, 2, WM), (3, 3, BM), (1, 1, BM));
        var move = Assert.Single(DraughtsRules.LegalMoves(eng));
        Assert.Equal(4 * 8 + 4, move.To);

        var intl = State(DraughtsVariant.International10, true, (2, 2, WM), (3, 3, BM), (1, 1, BM));
        Assert.Equal(2, DraughtsRules.LegalMoves(intl).Count);
    }

    // The Turkish strike, end to end: a white flying king circles a diamond of four black men
    // ((1,1), (1,3), (3,3), (3,1)) and returns to its start square. The two 4-capture paths
    // (clockwise/counterclockwise) are ONE move by FMJD identity (same from, to, captured set);
    // majority discards every shorter variant; the fourth jump is only stoppable at the origin
    // because the first captured man still blocks the ray (captured pieces stay on the board);
    // and nothing can be re-jumped. Applying it wipes Black out — a loss for the side to move.
    [Fact]
    public void Turkish_strike_flying_king_loops_the_diamond_in_one_deduped_move()
    {
        var s = State(DraughtsVariant.International10, true,
            (2, 0, WK), (1, 1, BM), (1, 3, BM), (3, 3, BM), (3, 1, BM));
        var move = Assert.Single(DraughtsRules.LegalMoves(s));
        Assert.Equal(2, move.From);
        Assert.Equal(2, move.To);                                     // a full loop: from == to
        Assert.Equal(new[] { 11, 13, 31, 33 }, move.Captures.Order());

        var next = DraughtsRules.MakeMove(s, move);
        Assert.Equal(WK, next.Squares[2]);
        Assert.DoesNotContain(next.Squares, p => p < 0);
        Assert.Equal(GameResult.Loss, new DraughtsGame(DraughtsVariant.International10).Result(next));
    }

    // International promotion happens only where the move ENDS: a man jumping over (6,8) lands on
    // the crown row at (7,9), must continue capturing over (8,8), and finishes on (9,7) — still a
    // man. Ending ON the crown row does promote.
    [Fact]
    public void International_man_jumping_through_the_crown_row_stays_a_man()
    {
        var through = State(DraughtsVariant.International10, true, (5, 7, WM), (6, 8, BM), (8, 8, BM));
        var move = Assert.Single(DraughtsRules.LegalMoves(through));
        Assert.Equal(2, move.Captures.Count);
        Assert.Equal(7 * 10 + 9, move.To);
        Assert.Equal(WM, DraughtsRules.MakeMove(through, move).Squares[7 * 10 + 9]);

        var onto = State(DraughtsVariant.International10, true, (5, 7, WM), (6, 8, BM));
        var crowning = Assert.Single(DraughtsRules.LegalMoves(onto));
        Assert.Equal(9 * 10 + 7, crowning.To);
        Assert.Equal(WK, DraughtsRules.MakeMove(onto, crowning).Squares[9 * 10 + 7]);
    }

    // A flying king sliding at a lone black man may land on ANY empty square beyond it — and
    // because a capture exists, no quiet slide is legal.
    [Fact]
    public void Flying_king_chooses_its_landing_square_and_captures_are_forced()
    {
        var s = State(DraughtsVariant.International10, true, (0, 0, WK), (3, 3, BM));
        var moves = DraughtsRules.LegalMoves(s);
        Assert.Equal(6, moves.Count);
        Assert.All(moves, m => Assert.Single(m.Captures, 33));
        Assert.Equal(new[] { 44, 55, 66, 77, 88, 99 }, moves.Select(m => m.To).Order());
    }

    [Fact]
    public void No_progress_rule_draws_king_shuffles_and_resets_on_man_moves()
    {
        int size = 10;
        var board = new sbyte[size * size];
        board[0] = WK; board[99] = BK;
        var shuffle = new DraughtsState(DraughtsVariant.International10, board, whiteToMove: true, noProgress: 79);
        var game = new DraughtsGame(DraughtsVariant.International10);
        Assert.Equal(GameResult.Ongoing, game.Result(shuffle));

        var moves = DraughtsRules.LegalMoves(shuffle);
        Assert.All(moves, m => Assert.Empty(m.Captures));
        var drawn = DraughtsRules.MakeMove(shuffle, moves[0]);
        Assert.Equal(80, drawn.NoProgress);
        Assert.Equal(GameResult.Draw, game.Result(drawn));

        board = new sbyte[size * size];
        board[0] = WM; board[99] = BK;
        var manMove = new DraughtsState(DraughtsVariant.International10, board, whiteToMove: true, noProgress: 79);
        var reset = DraughtsRules.MakeMove(manMove, DraughtsRules.LegalMoves(manMove)[0]);
        Assert.Equal(0, reset.NoProgress);
        Assert.Equal(GameResult.Ongoing, game.Result(reset));
    }

    // A side with pieces but no legal move loses: the white man on (0,6) can neither step (both
    // forward squares are off-board or occupied) nor jump (the landing square is off-board).
    [Fact]
    public void Side_with_no_moves_loses_even_with_pieces_on_the_board()
    {
        var s = State(DraughtsVariant.English8, true, (0, 6, WM), (1, 7, BM));
        Assert.Empty(DraughtsRules.LegalMoves(s));
        Assert.Equal(GameResult.Loss, new DraughtsGame(DraughtsVariant.English8).Result(s));
    }
}
