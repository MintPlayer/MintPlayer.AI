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
    public void Horizontal4_CreatesColumnStriped_AtTheSwappedCell()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4));
        Assert.Equal(60, board.ImmediateScore(VSwap(6, 2)));               // 4·10 + 20 — creation scores nothing
        Assert.True(board.ApplySwap(VSwap(6, 2)) >= 60);
        Assert.Equal(1, board.MoveCreatedStriped);
        // Blast ⊥ match (the real Candy Crush rule, owner-corrected): horizontal 4-match → column-striped.
        Assert.Equal(2, board.Kind(7 * Size + 2));
        Assert.Equal(4, board.Fruit(7 * Size + 2));
    }

    [Fact]
    public void Vertical4_CreatesRowStriped()
    {
        var board = BoardWith((4, 2, 4), (5, 2, 4), (7, 2, 4), (6, 3, 4));
        board.ApplySwap(HSwap(6, 2));                                      // brings the 4 into (6,2)
        Assert.Equal(1, board.MoveCreatedStriped);
        // The created row-striped (vertical match → row blast) falls to the bottom of its column.
        Assert.Equal(1, board.Kind(7 * Size + 2));
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

    // ── Creation-cell collision (owner bug 2026-07-24, two rounds) ─────────────────────────────────────────
    // Dragging a special into its own match must FIRE the special and relocate the new special to the
    // nearest plain cell of the shape (ties toward the lower index); a shape with no plain cell creates
    // nothing and everything fires. Round two (M50.6): the RELOCATED creation is SHIELDED from blasts for
    // the rest of the move — the fired special must not consume its own replacement (the wrapped 3×3 always
    // covered the nearest run cell, so the player never kept the promised special). Matches in later steps
    // still consume it; in-place creations keep the form-then-trigger chain rule (the 190-point test).

    private static PgCrazyFruits CoreWith(params (int R, int C, int Packed)[] cells)
    {
        var core = new PgCrazyFruits();
        core.reset(1);
        var g = BaseGrid();
        foreach (var (r, c, v) in cells) g[r * Size + c] = v;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) core.grid[i] = g[i];
        return core;
    }

    private static (int Pts, List<bool> Marked) StepZero(PgCrazyFruits core, int action)
    {
        Assert.True(core.stageSwap(action, core.cellB(action)));
        var marked = new List<bool>();
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) marked.Add(false);
        return (core.clearStep(0, marked), marked);
    }

    [Fact]
    public void Swap_StripedIntoMatch4_ExistingFires_ShieldedNewStripedSurvives()
    {
        // The reported bug. The dragged stripedH lands at (7,2) completing a horizontal 4-run: it fires its
        // ROW, and the relocated fresh striped at (7,1) is SHIELDED from that very blast — the player keeps
        // it: row 7 minus the shielded cell (7) + consumed (1) = 8 → 100 with the 4-bonus.
        var core = CoreWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, SH4));
        var (pts, marked) = StepZero(core, VSwap(6, 2));
        Assert.Equal(100, pts);
        Assert.Equal(SV4, core.grid[7 * Size + 1]);                        // the promised striped SURVIVES
        Assert.False(marked[7 * Size + 1]);
        Assert.Equal(0, core.grid[7 * Size + 2]);                          // the dragged striped fired and died
        Assert.Equal(1, core.moveCreatedStriped);
        Assert.Equal(1, core.moveSpecialsFired);
    }

    [Fact]
    public void Swap_WrappedIntoMatch4_FiresAndArms_ShieldedNewStripedSurvivesBothExplosions()
    {
        // Same collision with a wrapped: its 3×3 covers the relocated striped at (7,1), and so does the
        // armed second explosion a step later — the shield holds for the WHOLE move:
        // run (3, shielded cell excluded) + box (3 new) + consumed (1) = 7 → 90 with the 4-bonus.
        var core = CoreWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, WR4));
        var (pts, marked) = StepZero(core, VSwap(6, 2));
        Assert.Equal(90, pts);
        Assert.Equal(SV4, core.grid[7 * Size + 1]);
        Assert.False(marked[7 * Size + 1]);
        Assert.Equal(80 + 4, core.grid[7 * Size + 2]);                     // the wrapped ARMED in place (kind 5)
        core.collapseColumns(false);
        var marked1 = new List<bool>();
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) marked1.Add(false);
        core.clearStep(1, marked1);                                        // the armed second explosion
        Assert.Equal(SV4, core.grid[7 * Size + 1]);                        // still standing
        Assert.Equal(2, core.moveSpecialsFired);
    }

    [Fact]
    public void NonSpawnSpecialInLine_StillFires_AndCreationStaysAtTheMovedCell()
    {
        // Regression guard: the moved cell is PLAIN, an old striped sits elsewhere in the line — creation
        // stays at the moved cell exactly as before, and the old striped fires via the normal seed pass.
        var core = CoreWith((7, 0, SV4), (7, 1, 4), (7, 3, 4), (6, 2, 4));
        var (pts, marked) = StepZero(core, VSwap(6, 2));
        Assert.Equal(130, pts);                                            // run (3) + column 0 (7) + consumed (1)
        Assert.Equal(SV4, core.grid[7 * Size + 2]);                        // fresh striped at the moved cell
        Assert.False(marked[7 * Size + 2]);                                // untriggered → survives
    }

    [Fact]
    public void Run4_AllSpecials_NoCreation_EverythingFires()
    {
        // Degenerate: every cell of the 4-run is special → no plain host, no creation, all four fire:
        // row 7 (×2, dup) + columns 0 and 3 = 8 + 7 + 7 = 22 cells, nothing consumed.
        var core = CoreWith((7, 0, SV4), (7, 1, SH4), (7, 3, SV4), (6, 2, SH4));
        var (pts, _) = StepZero(core, VSwap(6, 2));
        Assert.Equal(240, pts);
        Assert.Equal(0, core.moveCreatedStriped);
        Assert.Equal(4, core.moveSpecialsFired);
    }

    [Fact]
    public void RelocatedCreation_TieBreaksTowardTheLowerIndex()
    {
        // The dragged striped lands at (7,1) with plain run cells at equal distance on BOTH sides — the
        // new striped must take the lower index (7,0). Its column-1 blast touches neither, so it survives.
        var core = CoreWith((7, 0, 4), (7, 2, 4), (7, 3, 4), (6, 1, SV4));
        var (pts, marked) = StepZero(core, VSwap(6, 1));
        Assert.Equal(130, pts);                                            // run (3) + column 1 (7) + consumed (1)
        Assert.Equal(SV4, core.grid[7 * Size + 0]);                        // relocated to the LOWER equidistant cell
        Assert.False(marked[7 * Size + 0]);
        Assert.Equal(0, core.grid[7 * Size + 2]);                          // the higher candidate just cleared
        Assert.Equal(1, core.moveCreatedStriped);
    }

    [Fact]
    public void Match5_WithSpecialAtTheSpawnCell_BombRelocates_AndTheSpecialFires()
    {
        // Bomb priority survives the collision: the dragged stripedV completes a 5-run, fires its column,
        // and the bomb forms on the nearest plain neighbour instead of overwriting it.
        var core = CoreWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (7, 4, 4), (6, 2, SV4));
        var (pts, _) = StepZero(core, VSwap(6, 2));
        Assert.Equal(170, pts);                                            // run (4) + column 2 (7) + consumed (1) + 50
        Assert.Equal(BOMB, core.grid[7 * Size + 1]);
        Assert.Equal(1, core.moveCreatedBombs);
        Assert.Equal(0, core.moveCreatedStriped);
    }

    [Fact]
    public void WrappedPivot_Special_NewWrappedRelocatesShielded_AndSurvives()
    {
        // L-shape whose intersection is the dragged wrapped: it fires (and arms), the NEW wrapped forms on
        // the nearest plain arm cell (6,2) and is shielded from the pivot's 3×3 — the player keeps a
        // wrapped: match (4, shielded cell excluded... it was never match-marked) + box (3 new) + consumed
        // (1) = 8 → 80 (3-runs carry no line bonus).
        var core = CoreWith((7, 0, 4), (7, 1, 4), (5, 2, 4), (6, 2, 4), (7, 3, WR4));
        var (pts, marked) = StepZero(core, HSwap(7, 2));
        Assert.Equal(80, pts);
        Assert.Equal(WR4, core.grid[6 * Size + 2]);                        // fresh wrapped survives, UNARMED
        Assert.False(marked[6 * Size + 2]);
        Assert.Equal(1, core.moveCreatedWrapped);
        Assert.Equal(1, core.moveSpecialsFired);                           // only the dragged pivot fired
    }

    [Fact]
    public void CascadeRun_SpecialAtTheLowestCell_CreationRelocatesShielded()
    {
        // Gravity drops an old striped-5 into the lowest cell of a cascade-formed 4-run: the striped fires,
        // the cascade creation relocates (shielded from the row blast) instead of being overwritten.
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (5, 1, SH5), (5, 2, 5), (6, 3, 5), (6, 4, 5));
        board.ApplySwap(HSwap(7, 2));
        Assert.Equal(1, board.MoveCreatedStriped);
        Assert.True(board.MoveSpecialsFired >= 1);
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
        // AI moves centre on the action's bottom/right cell — (5,5) here — so the 5×5 sits fully on-board.
        var board = BoardWith((5, 4, WR4), (5, 5, WR5));
        Assert.Equal(250, board.ImmediateScore(HSwap(5, 4)));              // 5×5 = 25 cells
        Assert.True(board.DeterministicValue(HSwap(5, 4)) > 250);          // both shells re-fire after the settle
        board.ApplySwap(HSwap(5, 4));
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++) Assert.NotEqual(5, board.Kind(i));
    }

    [Fact]
    public void ComboBlast_CentresOnTheLastSelectedCell()
    {
        // The same striped+striped swap, staged toward each end: the cleared COLUMN follows the gesture's
        // last-selected cell (owner rule), not a fixed corner of the action.
        var core = new PgCrazyFruits();
        core.reset(1);
        var g = BaseGrid();
        g[5 * Size + 5] = SH4;
        g[5 * Size + 6] = SV4 + 1;
        int action = HSwap(5, 5);

        foreach (int target in new[] { 5 * Size + 6, 5 * Size + 5 })
        {
            for (int i = 0; i < CrazyFruitsBoard.Cells; i++) core.grid[i] = g[i];
            Assert.True(core.stageSwap(action, target));
            var marked = new List<bool>();
            for (int i = 0; i < CrazyFruitsBoard.Cells; i++) marked.Add(false);
            core.clearStep(0, marked);
            int targetCol = target % Size;
            int otherCol = targetCol == 5 ? 6 : 5;
            Assert.True(marked[0 * Size + targetCol], "the last-selected cell's column must fire");
            Assert.False(marked[0 * Size + otherCol], "the other cell's column must NOT fire");
            core.finishMove(0);
        }
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
    public void PassiveBomb_ZapsTheMostFrequentType_AndTheTriggeringStripedIsRemoved()
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

        // Owner scenario: the striped that set the bomb off — and the bomb — must be OFF the board afterward.
        board.ApplySwap(VSwap(4, 4));
        Assert.True(board.MoveSpecialsFired >= 2, "striped AND bomb must both fire");
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
        {
            Assert.False(board.Kind(i) == 1 && board.Fruit(i) == 4, "the triggering striped must be removed");
            Assert.NotEqual(4, board.Kind(i)); // no bomb left (none can re-form: refill never makes specials without a 5-run)
        }
    }

    // Owner scenario: a WRAPPED fruit whose explosion triggers a sugar bomb. The wrapped arms (its canonical
    // second explosion fires after the settle), so by the END of the move both the wrapped and the bomb are
    // off the board — and no armed shell survives on the stable board.
    [Fact]
    public void Wrapped_TriggersBomb_AndBothAreRemovedByMoveEnd()
    {
        var board = BoardWith((6, 3, 4), (5, 4, 4), (6, 5, WR4), (7, 6, BOMB));
        var g = board.GridSnapshot();
        (g[5 * Size + 4], g[6 * Size + 4]) = (g[6 * Size + 4], g[5 * Size + 4]); // simulate the swap
        // Step-0 marks: 3-match(row 6, c3..c5) ∪ wrapped 3×3(rows 5-7 × cols 4-6, incl. the bomb) ∪ the
        // bomb's most-frequent-type zap over the rest of the board.
        var union = new bool[CrazyFruitsBoard.Cells];
        for (int c = 3; c <= 5; c++) union[6 * Size + c] = true;
        for (int r = 5; r <= 7; r++) for (int c = 4; c <= 6; c++) union[r * Size + c] = true;
        var counts = new int[7];
        foreach (var v in g) counts[v % 16 <= 6 ? v % 16 : 0]++;
        int most = 1;
        for (int t = 2; t <= 6; t++) if (counts[t] > counts[most]) most = t;
        int marked = 0;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            if (union[i] || (g[i] < 16 && g[i] % 16 == most)) marked++;
        Assert.Equal(marked * 10, board.ImmediateScore(VSwap(5, 4)));

        board.ApplySwap(VSwap(5, 4));
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
        {
            Assert.False(board.Kind(i) == 3 && board.Fruit(i) == 4, "the triggering wrapped must be removed by move end");
            Assert.NotEqual(5, board.Kind(i));  // no armed shell on a stable board
            Assert.NotEqual(4, board.Kind(i));  // the bomb is gone
        }
    }

    // A double-match swap: the down-moving 4 completes a horizontal 4-run (fresh stripedV at the swapped
    // cell), while the up-moving 5 completes a 3-run containing a wrapped-5 whose 3×3 blast covers the fresh
    // striped's cell — the FRESH special must fire in the SAME step (owner rule: form first, then trigger).
    // Exact count: 4-run(4, one consumed into the creation) ∪ 3-run(3) ∪ wrapped box(9) ∪ fresh striped's
    // column 2 (8) = 16 distinct marked + 1 consumed = 17 → 170 + 20 line bonus = 190.
    [Fact]
    public void FreshSpecial_FiresImmediately_WhenBlastedInTheSameStep()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4), (7, 2, 5), (6, 3, WR5), (6, 4, 5));
        Assert.Equal(190, board.ImmediateScore(VSwap(6, 2)));
        board.ApplySwap(VSwap(6, 2));
        Assert.Equal(1, board.MoveCreatedStriped);
        Assert.True(board.MoveSpecialsFired >= 2, "the wrapped AND the fresh striped must both fire");
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            Assert.False(board.Kind(i) == 2 && board.Fruit(i) == 4, "the fresh striped must not survive the blast");
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

    // The web page drives moves through the STEPWISE protocol (stageSwap → clearStep/collapse loop →
    // finishMove) so it can animate between steps; applySwap drains the same loop atomically. This pins the
    // two paths byte-identical — grid, score, RNG stream, telemetry — over full games with specials and
    // combos, so the browser's game functionality is the tested engine functionality by construction
    // (C#↔TS equality is separately pinned by the parity checksum).
    [Fact]
    public void StepwiseHostProtocol_IsByteIdenticalTo_ApplySwap()
    {
        for (ulong seed = 1; seed <= 10; seed++)
        {
            var atomic = new PgCrazyFruits();
            var stepwise = new PgCrazyFruits();
            atomic.reset(CrazyFruitsBoard.SeedToInt(seed));
            stepwise.reset(CrazyFruitsBoard.SeedToInt(seed));

            for (int move = 0; move < 30; move++)
            {
                // Alternate policies for coverage; both boards are identical, so the action is valid on both.
                int action = move % 2 == 0 ? atomic.greedyAction() : atomic.randomAction(new PgCfRng((int)(seed * 100 + (ulong)move)));
                int expectedPoints = 0;

                // Stepwise (the web host's exact calls, minus animation):
                Assert.True(stepwise.stageSwap(action, stepwise.cellB(action)));
                for (int k = 0; ; k++)
                {
                    var marked = new List<bool>();
                    for (int i = 0; i < CrazyFruitsBoard.Cells; i++) marked.Add(false);
                    int pts = stepwise.clearStep(k, marked);
                    if (pts == 0) break;
                    expectedPoints += pts;
                    stepwise.collapseColumns(true);
                }
                stepwise.finishMove(expectedPoints);

                // Atomic:
                Assert.Equal(expectedPoints, atomic.applySwap(action));

                Assert.Equal(atomic.score, stepwise.score);
                Assert.Equal(atomic.rng.state, stepwise.rng.state);
                Assert.Equal(atomic.reshuffles, stepwise.reshuffles);
                Assert.Equal(atomic.moveSpecialsFired, stepwise.moveSpecialsFired);
                for (int i = 0; i < CrazyFruitsBoard.Cells; i++) Assert.Equal(atomic.grid[i], stepwise.grid[i]);
            }
        }
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

    // ── deterministicValueShaped (RANKING PRD lever B) ─────────────────────────────────────────────────────

    [Fact]
    public void DeterministicValueShaped_AddsCreationWeight_ForAStripedCreatingSwap()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4)); // VSwap(6,2) → horizontal 4 → striped
        int action = VSwap(6, 2);
        Assert.True(board.DeterministicValueShaped(action) >= board.DeterministicValue(action) + 40,
            "a striped-creating swap's shaped deterministic value must carry at least the +40 creation weight");
    }

    [Fact]
    public void DeterministicValueShaped_EqualsDeterministicValue_WhenNothingIsCreated()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (6, 2, 4));            // VSwap(6,2) → plain 3-match
        int action = VSwap(6, 2);
        Assert.Equal(board.DeterministicValue(action), board.DeterministicValueShaped(action));
    }

    [Fact]
    public void DeterministicValueShaped_IsPure_AndZeroForIllegal()
    {
        var board = BoardWith((7, 0, 4), (7, 1, 4), (7, 3, 4), (6, 2, 4));
        var before = board.GridSnapshot();
        int scoreBefore = board.Score;
        _ = board.DeterministicValueShaped(VSwap(6, 2));
        Assert.Equal(before, board.GridSnapshot());
        Assert.Equal(scoreBefore, board.Score);
        Assert.Equal(0, board.DeterministicValueShaped(HSwap(0, 0)));      // base pattern: no match, illegal
    }
}
