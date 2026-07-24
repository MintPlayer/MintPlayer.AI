using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M50.1 gate (docs/prd/CRAZY_FRUITS_SPECIALS_PRD.md §5): directed tests for every special's creation shape
/// (incl. priority + cascade spawn), every activation (incl. the two-step wrapped timeline), every combo,
/// chain reactions, the new legality rule, and the planning purity (no RNG, no mutation) on specials boards.
/// Directed boards sit on the period-3 no-match base pattern (fruits 1..3); placed pieces use fruits 4/5.
/// ImmediateScore is the exact-assertion workhorse: it runs a REAL step-0 clearStep (combos, chains,
/// creations included) and restores the board.
/// </summary>
public class CrazyFruitsSpecialsTests
{
    private const int Size = CrazyFruitsBoard.Size;

    // Packed values (kind·16 + type).
    private const int SH4 = 16 + 4;   // striped (row blast), fruit 4
    private const int SV4 = 32 + 4;
    private const int SH5 = 16 + 5;
    private const int WR4 = 48 + 4;
    private const int WR5 = 48 + 5;
    private const int BOMB = 64;

    private static int[] BaseGrid()
    {
        var g = new int[CrazyFruitsBoard.Cells];
        for (int r = 0; r < Size; r++)
            for (int c = 0; c < Size; c++)
                g[r * Size + c] = (r + c) % 3 + 1;
        return g;
    }

    private static CrazyFruitsBoard BoardWith(params (int R, int C, int Packed)[] cells)
    {
        var g = BaseGrid();
        foreach (var (r, c, v) in cells) g[r * Size + c] = v;
        var board = new CrazyFruitsBoard();
        board.Reset(1);
        board.LoadGrid(g);
        Assert.False(board.AnyMatchOnBoard());
        return board;
    }

    private static int HSwap(int r, int c) => r * (Size - 1) + c;          // (r,c)↔(r,c+1)
    private static int VSwap(int r, int c) => 56 + r * Size + c;           // (r,c)↔(r+1,c)

    // ── Creation ────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Horizontal4_CreatesRowStriped_AtTheSwappedCell()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4));
        Assert.Equal(60, board.ImmediateScore(VSwap(6, 2)));               // 4·10 + 20 — creation scores nothing
        Assert.True(board.ApplySwap(VSwap(6, 2)) >= 60);
        Assert.Equal(1, board.MoveCreatedStriped);
        Assert.Equal(1, board.Kind(7 * Size + 2));                         // stripedH (blast ∥ the horizontal match)
        Assert.Equal(4, board.Fruit(7 * Size + 2));
    }

    [Fact]
    public void Vertical4_CreatesColumnStriped()
    {
        var board = BoardWith((4, 2, 4), (5, 2, 4), (7, 2, 4), (6, 3, 4));
        board.ApplySwap(HSwap(6, 2));                                      // brings the 4 into (6,2)
        Assert.Equal(1, board.MoveCreatedStriped);
        // The created stripedV falls to the bottom of its column after the collapse.
        Assert.Equal(2, board.Kind(7 * Size + 2));
        Assert.Equal(4, board.Fruit(7 * Size + 2));
    }

    [Fact]
    public void Straight5_CreatesBomb_AndOutranksTheStriped()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (7, 4, 4), (6, 2, 4));
        Assert.Equal(100, board.ImmediateScore(VSwap(6, 2)));              // 5·10 + 50
        board.ApplySwap(VSwap(6, 2));
        Assert.Equal(1, board.MoveCreatedBombs);
        Assert.Equal(0, board.MoveCreatedStriped);                         // the 5-run is consumed by the bomb
        Assert.Equal(4, board.Kind(7 * Size + 2));                         // bomb at the swapped cell
        Assert.Equal(0, board.Fruit(7 * Size + 2));                        // colorless
    }

    [Fact]
    public void LShape_CreatesWrapped_AtTheIntersection()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (5, 2, 4), (6, 2, 4), (7, 3, 4));
        Assert.Equal(50, board.ImmediateScore(HSwap(7, 2)));               // 5 distinct cells, no line bonus
        board.ApplySwap(HSwap(7, 2));
        Assert.Equal(1, board.MoveCreatedWrapped);
        Assert.Equal(3, board.Kind(7 * Size + 2));                         // wrapped at the intersection
        Assert.Equal(4, board.Fruit(7 * Size + 2));
    }

    [Fact]
    public void StraightFiveBeatsLShape_WhenBothQualify()
    {
        // Horizontal 5-run through (5,2) + vertical 3-run through it: bomb wins, the v-run creates nothing.
        var board = BoardWith((5, 0, 4), (5, 1, 4), (5, 3, 4), (5, 4, 4), (3, 2, 4), (4, 2, 4), (6, 2, 4));
        board.ApplySwap(VSwap(5, 2));                                      // pulls the (6,2) 4 up into (5,2)
        Assert.Equal(1, board.MoveCreatedBombs);
        Assert.Equal(0, board.MoveCreatedWrapped);
        Assert.Equal(4, board.Kind(5 * Size + 2));                         // spawned at the swapped cell, nothing below cleared
    }

    [Fact]
    public void CascadeMatch_CreatesSpecialToo()
    {
        // The M49 cascade board extended to a 4-run of 5s formed by GRAVITY (cascade step 1): the swap itself
        // only makes a 3-run of 4s, so any striped-5 on the final board proves cascade creation.
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (5, 1, 5), (5, 2, 5), (6, 3, 5), (6, 4, 5));
        board.ApplySwap(HSwap(7, 2));
        Assert.Equal(1, board.MoveCreatedStriped);
        bool found = false;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            if (board.Fruit(i) == 5 && (board.Kind(i) == 1 || board.Kind(i) == 2)) found = true;
        Assert.True(found, "expected a cascade-created striped-5 on the board");
    }

    // ── Activation ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Striped_ClearsItsWholeRow_WhenMatched()
    {
        // Match c3..c5 of row 7 includes the stripedH at (7,5) → the whole row fires: 8 cells, no bonus.
        var board = BoardWith((7, 3, 4), (6, 4, 4), (7, 5, SH4));
        Assert.Equal(80, board.ImmediateScore(VSwap(6, 4)));
        Assert.Equal(80, board.DeterministicValue(VSwap(6, 4)));           // uniform base shift → no follow-up
    }

    [Fact]
    public void Wrapped_ExplodesTwice_AndNeverSurvivesArmed()
    {
        var board = BoardWith((6, 3, 4), (5, 4, 4), (6, 5, WR4));
        // Step 0: 3-match (3 cells) ∪ 3×3 box around (6,5) (9 cells, 2 overlapping) = 10 cells.
        Assert.Equal(100, board.ImmediateScore(VSwap(5, 4)));
        // The armed shell falls and fires again on the next step — the full move is worth more than step 0.
        Assert.True(board.DeterministicValue(VSwap(5, 4)) > 100);
        board.ApplySwap(VSwap(5, 4));
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            Assert.NotEqual(5, board.Kind(i));                             // a stable board never holds an armed cell
    }

    [Fact]
    public void ChainReaction_StripedFiresStriped()
    {
        // Row-striped at (7,5) fires row 7, hitting the column-striped at (7,0) → its column fires too:
        // row 7 (8) + column 0 (8) − overlap = 15 cells.
        var board = BoardWith((7, 3, 4), (6, 4, 4), (7, 5, SH4), (7, 0, SV4));
        Assert.Equal(150, board.ImmediateScore(VSwap(6, 4)));
    }

    // ── Combos + the bomb swap ──────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void BombPlusPlain_IsLegalWithoutAMatch_AndClearsThatType()
    {
        var board = BoardWith((5, 5, BOMB));
        int action = HSwap(5, 5);                                          // bomb ↔ plain neighbour (5,6)
        Assert.True(board.IsLegal(action));
        int targetType = board.Fruit(5 * Size + 6);
        int count = 0;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) if (board.Fruit(i) == targetType) count++;
        Assert.Equal((count + 1) * 10, board.ImmediateScore(action));      // every cell of that type + the bomb
        board.ApplySwap(action);
        Assert.Equal(1, board.MoveSpecialsFired);
    }

    [Fact]
    public void StripedPlusPlain_WithoutAMatch_StaysIllegal()
    {
        var board = BoardWith((5, 5, SH4));
        Assert.False(board.IsLegal(HSwap(5, 5)));
        Assert.Equal(-1, board.ApplySwap(HSwap(5, 5)));
    }

    [Fact]
    public void BombPlusBomb_ClearsTheBoard()
    {
        var board = BoardWith((5, 5, BOMB), (5, 6, BOMB));
        Assert.True(board.IsLegal(HSwap(5, 5)));
        Assert.Equal(640, board.ImmediateScore(HSwap(5, 5)));              // all 64 cells
    }

    [Fact]
    public void StripedPlusStriped_FiresRowAndColumnThroughTheSwap()
    {
        var board = BoardWith((5, 5, SH4), (5, 6, SV4 + 1));               // striped-4 + striped-5
        Assert.Equal(150, board.ImmediateScore(HSwap(5, 5)));              // row 5 + column 5 = 15 cells
    }

    [Fact]
    public void StripedPlusWrapped_FiresTheGiantCross()
    {
        var board = BoardWith((5, 5, SH4), (5, 6, WR5));
        Assert.Equal(390, board.ImmediateScore(HSwap(5, 5)));              // 3 rows + 3 cols − 9 overlap = 39
    }

    [Fact]
    public void WrappedPlusWrapped_Fires5x5_ThenTheArmedShells()
    {
        var board = BoardWith((5, 5, WR4), (5, 6, WR5));
        Assert.Equal(250, board.ImmediateScore(HSwap(5, 5)));              // 5×5 = 25 cells (fully on-board)
        Assert.True(board.DeterministicValue(HSwap(5, 5)) > 250);          // both shells re-fire after the settle
        board.ApplySwap(HSwap(5, 5));
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) Assert.NotEqual(5, board.Kind(i));
    }

    [Fact]
    public void BombPlusStriped_ConvertsThatColorAndFiresEverything()
    {
        // Type-4 pieces: the striped itself + two plains → conversions fire row 2 ((2,2), (r+c) even → row
        // blast), column 4 ((3,4), odd → column blast) and row 5 (the original): 24 − 2 overlaps = 22 cells.
        var board = BoardWith((5, 5, BOMB), (5, 6, SH4), (2, 2, 4), (3, 4, 4));
        Assert.Equal(220, board.ImmediateScore(HSwap(5, 5)));
    }

    [Fact]
    public void PassiveBomb_ZapsTheMostFrequentType()
    {
        // The row-striped's blast hits the bomb at (5,7): the bomb zaps the board's most frequent fruit type
        // (deterministic stand-in for the canonical random color).
        var board = BoardWith((5, 3, 4), (4, 4, 4), (5, 5, SH4), (5, 7, BOMB));
        var g = board.GridSnapshot();
        (g[4 * Size + 4], g[5 * Size + 4]) = (g[5 * Size + 4], g[4 * Size + 4]); // simulate the swap
        var counts = new int[7];
        foreach (var v in g) counts[v % 16 <= 6 ? v % 16 : 0]++;
        int most = 1;
        for (int t = 2; t <= 6; t++) if (counts[t] > counts[most]) most = t;
        int outsideRow5 = 0;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            if (i / Size != 5 && g[i] % 16 == most && g[i] < 16) outsideRow5++;
        Assert.Equal((8 + outsideRow5) * 10, board.ImmediateScore(VSwap(4, 4)));
    }

    // ── Baseline tiers (M50.2 gates) ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Greedy_ProvablyPicksTheBombSwap()
    {
        // A plain 3-match (30 pts) exists, but any bomb+neighbour swap clears ~22 cells (~220 pts).
        var board = BoardWith((5, 5, BOMB), (7, 0, 4), (7, 1, 4), (6, 2, 4));
        int action = board.GreedyAction();
        var (a, b) = board.SwapCells(action);
        Assert.True(board.Kind(a) == 4 || board.Kind(b) == 4, "greedy must take a bomb swap over the 3-match");
    }

    [Fact]
    public void SpecialsGreedy_PrefersBuildingAStriped_WherePlainGreedyTakesTheRowBlast()
    {
        // Two options: fire an existing striped via a 3-match (80 pts immediate; shaped 80) vs make a 4-run
        // (60 pts immediate; shaped 60+40=100). Plain greedy takes the blast, specials-greedy builds.
        var board = BoardWith(
            (5, 3, 4), (4, 4, 4), (5, 5, SH4),                  // striped row-fire option (immediate 80)
            (7, 0, 5), (7, 1, 5), (7, 3, 5), (6, 2, 5));        // 4-run creation option (immediate 60)
        Assert.Equal(80, board.ImmediateScore(VSwap(4, 4)));
        Assert.Equal(60, board.ImmediateScore(VSwap(6, 2)));
        Assert.Equal(100, board.ImmediateScoreShaped(VSwap(6, 2)));
        Assert.Equal(VSwap(4, 4), board.GreedyAction());
        Assert.Equal(VSwap(6, 2), board.SpecialsGreedyAction());
    }

    [Fact]
    public void Expectimax2_ReturnsALegalAction_AndRestoresTheBoard()
    {
        var board = BoardWith((5, 5, BOMB), (5, 6, SH4), (2, 2, WR4));
        var before = board.GridSnapshot();
        int action = board.Expectimax2Action();
        Assert.True(board.IsLegal(action));
        Assert.Equal(before, board.GridSnapshot());
    }

    // ── Planning purity + invariants ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void PlanningOnSpecialsBoards_ConsumesNoRng_AndRestoresEverything()
    {
        var board = BoardWith((5, 5, BOMB), (5, 6, SH4), (2, 2, WR4), (7, 0, SV4));
        byte[] Snapshot()
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms);
            board.WriteState(w);
            return ms.ToArray();
        }
        var before = Snapshot();
        for (int a = 0; a < CrazyFruitsBoard.ActionCount; a++)
        {
            board.ImmediateScore(a);
            board.DeterministicValue(a);
        }
        Assert.Equal(before, Snapshot());
    }

    [Fact]
    public void RandomEpisodes_WithSpecials_KeepEveryInvariant()
    {
        int specialsSeen = 0;
        for (ulong seed = 1; seed <= 20; seed++)
        {
            var board = new CrazyFruitsBoard();
            board.Reset(seed);
            for (int move = 0; move < 30; move++)
            {
                int action = board.RandomAction(seed * 77, move);
                Assert.True(board.ApplySwap(action) > 0);
                Assert.False(board.AnyMatchOnBoard());
                Assert.True(board.HasLegalSwap());
                for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
                {
                    int kind = board.Kind(i);
                    int fruit = board.Fruit(i);
                    Assert.InRange(kind, 0, 4);                            // never armed on a stable board
                    if (kind == 4) Assert.Equal(0, fruit);                 // bombs are colorless
                    else Assert.InRange(fruit, 1, CrazyFruitsBoard.FruitTypes);
                    if (kind > 0) specialsSeen++;
                }
            }
        }
        Assert.True(specialsSeen > 0, "specials never occurred across 600 random moves — creation is broken");
    }

    [Fact]
    public void Observation_CarriesKindPlanes_AndBothPlanesForColoredSpecials()
    {
        var board = BoardWith((5, 5, BOMB), (5, 6, SH4), (2, 2, WR5));
        var obs = board.BuildObservation();
        Assert.Equal(CrazyFruitsBoard.ObservationSize, obs.Length);
        int cells = CrazyFruitsBoard.Cells;
        // Colored special: fruit plane AND kind plane both set.
        Assert.Equal(1f, obs[(4 - 1) * cells + 5 * Size + 6]);             // fruit-4 plane at the striped
        Assert.Equal(1f, obs[(6 + 0) * cells + 5 * Size + 6]);             // stripedH plane (kind 1 → plane 6)
        Assert.Equal(1f, obs[(6 + 2) * cells + 2 * Size + 2]);             // wrapped plane (kind 3 → plane 8)
        // Bomb: colorless — kind plane only.
        Assert.Equal(1f, obs[(6 + 3) * cells + 5 * Size + 5]);             // bomb plane (kind 4 → plane 9)
        for (int f = 0; f < 6; f++) Assert.Equal(0f, obs[f * cells + 5 * Size + 5]);
    }
}
