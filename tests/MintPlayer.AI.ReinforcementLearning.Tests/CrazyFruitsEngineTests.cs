using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M49.1 gate (docs/prd/CRAZY_FRUITS_PRD.md §5): engine invariants (init board clean + playable; every
/// post-move board stable + playable; mask == brute force), hand-computed scoring on directed boards,
/// reshuffle multiset preservation, minstd RNG correctness, and determinism. The generated core is internal
/// (global namespace); everything here goes through the public <see cref="CrazyFruitsBoard"/> facade.
/// </summary>
public class CrazyFruitsEngineTests
{
    private const int Size = CrazyFruitsBoard.Size;

    // ── RNG ─────────────────────────────────────────────────────────────────────────────────────────────────

    // The C++ standard's contract for std::minstd_rand: seeded with 1, the 10,000th value is 399268537.
    // Pins that the Schrage implementation is the real minstd generator (and therefore exact on both sides).
    [Fact]
    public void Rng_MatchesMinstdReference()
    {
        var rng = new PgCfRng(1);
        int last = 0;
        for (int i = 0; i < 10_000; i++) last = rng.next();
        Assert.Equal(399268537, last);
    }

    // ── Board invariants (PRD §3.3/§3.5) ────────────────────────────────────────────────────────────────────

    private static void AssertPackedValid(CrazyFruitsBoard board)
    {
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
        {
            Assert.InRange(board.Kind(i), 0, 4); // never empty, never armed on a stable board
            if (board.Kind(i) == 4) Assert.Equal(0, board.Fruit(i));
            else Assert.InRange(board.Fruit(i), 1, CrazyFruitsBoard.FruitTypes);
        }
    }

    [Fact]
    public void InitialBoard_HasNoMatches_AndALegalSwap()
    {
        for (ulong seed = 1; seed <= 50; seed++)
        {
            var board = new CrazyFruitsBoard();
            board.Reset(seed);
            Assert.False(board.AnyMatchOnBoard());
            Assert.True(board.HasLegalSwap());
            AssertPackedValid(board);
        }
    }

    [Fact]
    public void AfterEveryMove_BoardIsStable_AndHasALegalSwap()
    {
        for (ulong seed = 1; seed <= 20; seed++)
        {
            var board = new CrazyFruitsBoard();
            board.Reset(seed);
            int scoreBefore = 0;
            for (int move = 0; move < 30; move++)
            {
                int action = board.RandomAction(seed * 1000, move);
                Assert.InRange(action, 0, CrazyFruitsBoard.ActionCount - 1);
                int points = board.ApplySwap(action);
                Assert.True(points > 0);                       // every legal swap clears at least a 3-line (30 pts)
                Assert.Equal(scoreBefore + points, board.Score);
                scoreBefore = board.Score;
                Assert.False(board.AnyMatchOnBoard());         // cascades fully resolved
                Assert.True(board.HasLegalSwap());             // deadlock defined out of existence
                AssertPackedValid(board);
            }
            Assert.Equal(30, board.MovesMade);
        }
    }

    [Fact]
    public void IllegalSwap_ReturnsMinusOne_AndChangesNothing()
    {
        var board = new CrazyFruitsBoard();
        board.Reset(7);
        var mask = board.LegalMask();
        int illegal = Array.IndexOf(mask, false);
        Assert.True(illegal >= 0);
        var before = board.GridSnapshot();
        Assert.Equal(-1, board.ApplySwap(illegal));
        Assert.Equal(before, board.GridSnapshot());
        Assert.Equal(0, board.MovesMade);
    }

    // ── Mask == brute force (PRD M49.1 gate) ────────────────────────────────────────────────────────────────

    // Independent re-implementation of the legality rule over PACKED values: typewise 3+ line after the
    // swap, OR either cell is a bomb, OR both cells are specials. On a stable board the line check through
    // the whole board is equivalent to the engine's through-the-swapped-cells test.
    private static bool BruteforceLegal(int[] g, int action)
    {
        static int F(int v) => v % 16;
        static int K(int v) => v / 16;
        int a, b;
        if (action < 56) { int r = action / 7, c = action % 7; a = r * Size + c; b = a + 1; }
        else { int v = action - 56; int r = v / Size, c = v % Size; a = r * Size + c; b = a + Size; }
        if (K(g[a]) == 4 || K(g[b]) == 4) return true;                      // a bomb consumes any swap
        if (K(g[a]) != 0 && K(g[b]) != 0) return true;                      // special + special = combo
        if (F(g[a]) == F(g[b])) return false;
        var copy = (int[])g.Clone();
        (copy[a], copy[b]) = (copy[b], copy[a]);
        for (int r = 0; r < Size; r++)
            for (int c = 0; c + 2 < Size; c++)
                if (F(copy[r * Size + c]) != 0 && F(copy[r * Size + c]) == F(copy[r * Size + c + 1]) && F(copy[r * Size + c]) == F(copy[r * Size + c + 2]))
                    return true;
        for (int c = 0; c < Size; c++)
            for (int r = 0; r + 2 < Size; r++)
                if (F(copy[r * Size + c]) != 0 && F(copy[r * Size + c]) == F(copy[(r + 1) * Size + c]) && F(copy[r * Size + c]) == F(copy[(r + 2) * Size + c]))
                    return true;
        return false;
    }

    [Fact]
    public void LegalMask_MatchesBruteForce_OnFreshAndPlayedBoards()
    {
        for (ulong seed = 1; seed <= 10; seed++)
        {
            var board = new CrazyFruitsBoard();
            board.Reset(seed);
            for (int move = 0; move < 5; move++)
            {
                var grid = board.GridSnapshot();
                var mask = board.LegalMask();
                for (int a = 0; a < CrazyFruitsBoard.ActionCount; a++)
                    Assert.Equal(BruteforceLegal(grid, a), mask[a]);
                board.ApplySwap(board.RandomAction(seed, move));
            }
        }
    }

    // ── Hand-computed scoring (PRD §3.5 locks: 10·(k+1)/fruit, +20/+50 line bonuses) ───────────────────────
    // Directed boards sit on a period-3 no-match base pattern (fruits 1..3); the acted fruits are 4/5 so no
    // accidental runs form. Exact assertions use the DETERMINISTIC cascade (gravity, no refill) — the full
    // ApplySwap path adds random refill on top, covered by the invariant tests above.

    private static int[] BaseGrid()
    {
        var g = new int[CrazyFruitsBoard.Cells];
        for (int r = 0; r < Size; r++)
            for (int c = 0; c < Size; c++)
                g[r * Size + c] = (r + c) % 3 + 1;
        return g;
    }

    private static CrazyFruitsBoard BoardWith(params (int R, int C, int Fruit)[] cells)
    {
        var g = BaseGrid();
        foreach (var (r, c, f) in cells) g[r * Size + c] = f;
        var board = new CrazyFruitsBoard();
        board.Reset(1);
        board.LoadGrid(g);
        Assert.False(board.AnyMatchOnBoard()); // directed boards must start stable
        return board;
    }

    private const int SwapDown_6_2 = 56 + 6 * Size + 2;  // vertical swap (6,2)↔(7,2)
    private const int SwapRight_7_2 = 7 * 7 + 2;         // horizontal swap (7,2)↔(7,3)

    [Fact]
    public void ThreeLine_Scores30()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (6, 2, 4));
        Assert.True(board.IsLegal(SwapDown_6_2));
        Assert.Equal(30, board.ImmediateScore(SwapDown_6_2));
        Assert.Equal(30, board.DeterministicValue(SwapDown_6_2));
    }

    [Fact]
    public void FourLine_Scores60()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4));
        Assert.Equal(4 * 10 + 20, board.ImmediateScore(SwapDown_6_2));
        Assert.Equal(60, board.DeterministicValue(SwapDown_6_2));
    }

    [Fact]
    public void FiveLine_Scores100()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (7, 4, 4), (6, 2, 4));
        Assert.Equal(5 * 10 + 50, board.ImmediateScore(SwapDown_6_2));
        Assert.Equal(100, board.DeterministicValue(SwapDown_6_2));
    }

    // Swap (7,2)↔(7,3) clears three 4s in row 7 (step 0: 30); gravity then drops the two 5s at (5,1),(5,2)
    // next to the 5 at (6,3), forming a 3-run in the new row 6 (step 1: 3·10·2 = 60). Total 90.
    [Fact]
    public void TwoStepCascade_DoublesStepOne_Scores90()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (5, 1, 5), (5, 2, 5), (6, 3, 5));
        Assert.Equal(30, board.ImmediateScore(SwapRight_7_2));      // step 0 only
        Assert.Equal(90, board.DeterministicValue(SwapRight_7_2));  // step 0 + doubled step 1
    }

    [Fact]
    public void Baselines_GreedyPicksTheBiggerMatch_ExpectimaxSeesTheCascade()
    {
        // The cascade board: greedy (step-0 only) is indifferent between 30-point swaps and picks the lowest
        // index; expectimax must pick the 90-point cascade swap.
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (5, 1, 5), (5, 2, 5), (6, 3, 5));
        Assert.Equal(90, board.DeterministicValue(board.ExpectimaxAction()));

        // Greedy prefers a 4-line (60) over a 3-line (30).
        var board2 = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4), (0, 0, 5), (0, 1, 5), (1, 2, 5));
        Assert.Equal(60, board2.ImmediateScore(board2.GreedyAction()));
    }

    // ── Reshuffle (PRD §3.5) ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Reshuffle_PreservesMultiset_YieldsStablePlayableBoard()
    {
        for (ulong seed = 1; seed <= 10; seed++)
        {
            var board = new CrazyFruitsBoard();
            board.Reset(seed);
            var before = board.GridSnapshot().OrderBy(f => f).ToArray();
            board.Reshuffle();
            Assert.Equal(before, board.GridSnapshot().OrderBy(f => f).ToArray());
            Assert.False(board.AnyMatchOnBoard());
            Assert.True(board.HasLegalSwap());
        }
    }

    // ── Observation (PRD §3.4) ──────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Observation_OneHotPlanesMatchGrid_WouldMatchPlaneMatchesMask()
    {
        var board = new CrazyFruitsBoard();
        board.Reset(42);
        var obs = board.BuildObservation();
        Assert.Equal(CrazyFruitsBoard.ObservationSize, obs.Length);

        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
        {
            for (int f = 1; f <= CrazyFruitsBoard.FruitTypes; f++)
                Assert.Equal(board.Fruit(i) == f ? 1f : 0f, obs[(f - 1) * CrazyFruitsBoard.Cells + i]);
            for (int k = 1; k <= CrazyFruitsBoard.SpecialKinds; k++)
                Assert.Equal(board.Kind(i) == k ? 1f : 0f, obs[(CrazyFruitsBoard.FruitTypes + k - 1) * CrazyFruitsBoard.Cells + i]);
        }

        var expected = new bool[CrazyFruitsBoard.Cells];
        var mask = board.LegalMask();
        for (int a = 0; a < CrazyFruitsBoard.ActionCount; a++)
        {
            if (!mask[a]) continue;
            var (cellA, cellB) = board.SwapCells(a);
            expected[cellA] = true;
            expected[cellB] = true;
        }
        int wmOffset = (CrazyFruitsBoard.FruitTypes + CrazyFruitsBoard.SpecialKinds) * CrazyFruitsBoard.Cells;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            Assert.Equal(expected[i] ? 1f : 0f, obs[wmOffset + i]);

        // Per-action feature planes: immediate score and deterministic cascade value, ÷300, matching the
        // facade's scripted-baseline queries exactly.
        int baseOffset = (CrazyFruitsBoard.FruitTypes + CrazyFruitsBoard.SpecialKinds + 1) * CrazyFruitsBoard.Cells;
        for (int a = 0; a < CrazyFruitsBoard.ActionCount; a++)
        {
            Assert.Equal(board.ImmediateScore(a) / 300f, obs[baseOffset + a], 5);
            float expectedDet = mask[a] ? board.DeterministicValue(a) / 300f : 0f;
            Assert.Equal(expectedDet, obs[baseOffset + CrazyFruitsBoard.ActionCount + a], 5);
            float expectedShaped = mask[a] ? board.DeterministicValueShaped(a) / 300f : 0f;
            Assert.Equal(expectedShaped, obs[baseOffset + 2 * CrazyFruitsBoard.ActionCount + a], 5);
        }
    }

    // ── Determinism + state round-trip ──────────────────────────────────────────────────────────────────────

    [Fact]
    public void SameSeed_SameGreedyEpisode()
    {
        var a = new CrazyFruitsBoard();
        var b = new CrazyFruitsBoard();
        a.Reset(123);
        b.Reset(123);
        for (int move = 0; move < 30; move++)
        {
            int actA = a.GreedyAction();
            Assert.Equal(actA, b.GreedyAction());
            a.ApplySwap(actA);
            b.ApplySwap(actA);
        }
        Assert.Equal(a.Score, b.Score);
        Assert.Equal(a.GridSnapshot(), b.GridSnapshot());
    }

    [Fact]
    public void StateRoundTrip_ContinuesIdentically()
    {
        var board = new CrazyFruitsBoard();
        board.Reset(99);
        for (int move = 0; move < 10; move++) board.ApplySwap(board.GreedyAction());

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            board.WriteState(writer);

        var restored = new CrazyFruitsBoard();
        stream.Position = 0;
        using (var reader = new BinaryReader(stream))
            restored.ReadState(reader);

        Assert.Equal(board.Score, restored.Score);
        Assert.Equal(board.GridSnapshot(), restored.GridSnapshot());
        for (int move = 0; move < 10; move++)
        {
            int expected = board.GreedyAction();
            Assert.Equal(expected, restored.GreedyAction());
            Assert.Equal(board.ApplySwap(expected), restored.ApplySwap(expected));
        }
        Assert.Equal(board.GridSnapshot(), restored.GridSnapshot());
    }

    // ── Action encoding ─────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void ActionEncoding_MatchesTheDocumentedLayout()
    {
        var board = new CrazyFruitsBoard();
        board.Reset(1);
        Assert.Equal((0, 1), board.SwapCells(0));                       // (0,0)↔(0,1)
        Assert.Equal((7 * Size + 6, 7 * Size + 7), board.SwapCells(55)); // last horizontal
        Assert.Equal((0, Size), board.SwapCells(56));                   // (0,0)↔(1,0)
        Assert.Equal((6 * Size + 7, 7 * Size + 7), board.SwapCells(111)); // last vertical
    }
}
