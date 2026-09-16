using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M54.1 engine gates (docs/prd/TETRIS_PRD.md §5): rules invariants (piece tables, line clears, garbage,
/// micro==macro lock parity — risk 4), pinned feature values on a hand-drawn board (risk 3), and the
/// env-validation bars: the C# engine must reproduce the spike's measured behavior
/// (docs/prd/tetris-spike/tetris_spike.mjs) — random ~25.6 pieces to top-out, Dellacherie ~197.4 lines
/// per 500-piece episode, and the 18× garbage-survival separation.
/// </summary>
public class TetrisEngineTests
{
    private const int FullRow = 1023;

    [Fact]
    public void PieceTables_AllRotationsHaveFourDistinctCellsInsideTheBoundingBox()
    {
        var t = new PgTetris();
        int[] expectedRotations = [2, 1, 4, 2, 2, 4, 4]; // I O T S Z L J
        for (int p = 0; p < TetrisBoard.PieceCount; p++)
        {
            Assert.Equal(expectedRotations[p], t.rotCount[p]);
            for (int r = 0; r < t.rotCount[p]; r++)
            {
                int ri = p * 4 + r;
                var seen = new HashSet<(int, int)>();
                for (int k = 0; k < 4; k++)
                {
                    int x = t.cellX[ri * 4 + k];
                    int y = t.cellY[ri * 4 + k];
                    Assert.InRange(x, 0, t.rotW[ri] - 1);
                    Assert.InRange(y, 0, t.rotH[ri] - 1);
                    Assert.True(seen.Add((x, y)), $"piece {p} rot {r} duplicates cell ({x},{y})");
                }
            }
        }
    }

    [Fact]
    public void LineClear_VerticalIIntoASingleGapClearsOneRow()
    {
        var board = new TetrisBoard();
        board.Reset(1);
        var rows = new int[20];
        rows[19] = FullRow - 1; // bottom row, gap at column 0
        board.LoadRows(rows);
        board.LoadPieces(current: 0, next: 0); // I

        int cleared = board.ApplyPlacement(1 * 10 + 0); // vertical I in column 0
        Assert.Equal(1, cleared);
        Assert.Equal(1, board.Lines);
        Assert.Equal(40, board.Score); // NES single
        // The three surviving I cells slid down one row: column 0 filled in rows 17..19.
        Assert.True(board.Cell(0, 19));
        Assert.True(board.Cell(0, 17));
        Assert.False(board.Cell(0, 16));
    }

    [Fact]
    public void LineClear_TetrisScoresTwelveHundredAndCounts()
    {
        var board = new TetrisBoard();
        board.Reset(1);
        var rows = new int[20];
        for (int y = 16; y < 20; y++) rows[y] = FullRow - (1 << 9); // four rows, gap in column 9
        board.LoadRows(rows);
        board.LoadPieces(current: 0, next: 0);

        int cleared = board.ApplyPlacement(1 * 10 + 9); // vertical I in column 9
        Assert.Equal(4, cleared);
        Assert.Equal(1200, board.Score); // NES tetris
        Assert.Equal(1, board.Tetrises);
        for (int y = 0; y < 20; y++) Assert.Equal(0, board.Row(y));
    }

    [Fact]
    public void Garbage_InsertsAFullRowWithOneGapAndShiftsTheStackUp()
    {
        var board = new TetrisBoard();
        board.Reset(7, sevenBag: false, garbageEvery: 1);
        var mask = board.LegalMask();
        int action = Array.IndexOf(mask, true);
        board.ApplyPlacement(action);

        int bottom = board.Row(19);
        int gapBits = FullRow - bottom;
        Assert.NotEqual(0, gapBits);
        Assert.Equal(0, gapBits & (gapBits - 1)); // exactly one gap bit
    }

    [Fact]
    public void GarbageMasks_TrackExactlyTheGarbageCells_ThroughFillsAndClears()
    {
        var t = new PgTetris();
        t.reset(5, false, 0);
        t.insertGarbageRow();
        int gap = FullRow - t.rows[19];
        Assert.Equal(t.rows[19], t.garbageMasks[19]); // fresh garbage row: mask == cells

        // A player piece filling the gap must NOT be marked as garbage.
        int gapCol = System.Numerics.BitOperations.TrailingZeroCount(gap);
        t.current = 0; // vertical I into the gap column
        int cleared = t.applyPlacement(1 * 10 + gapCol);
        Assert.Equal(1, cleared); // the garbage row completes and clears
        // The cleared garbage row is gone; the mask must carry no stale garbage bits anywhere.
        for (int y = 0; y < 20; y++) Assert.Equal(0, t.garbageMasks[y] & ~t.rows[y] & FullRow);
        for (int y = 0; y < 20; y++) Assert.Equal(0, t.garbageMasks[y]); // nothing garbage remains

        // Two stacked garbage rows shift together and keep their masks aligned with their cells.
        t.insertGarbageRow();
        t.insertGarbageRow();
        Assert.Equal(t.rows[19], t.garbageMasks[19]);
        Assert.True((t.garbageMasks[18] & t.rows[18]) == t.garbageMasks[18]);
        Assert.NotEqual(0, t.garbageMasks[18]);
    }

    [Fact]
    public void Garbage_OverflowEndsTheGame()
    {
        var t = new PgTetris();
        t.reset(5, false, 0);
        t.rows[0] = 1;
        t.insertGarbageRow();
        Assert.True(t.gameOver);
    }

    [Fact]
    public void MicroPath_ReachesTheSameBoardAsTheMacroPlacement()
    {
        // Risk 4: hard drop from spawn == applyPlacement whenever the micro moves fit.
        var macro = new TetrisBoard();
        macro.Reset(42);
        var junk = new int[20];
        junk[19] = 0b0000011111; // half-filled bottom row
        macro.LoadRows(junk);
        macro.LoadPieces(current: 2, next: 1); // T

        var micro = new TetrisBoard();
        micro.Reset(42);
        micro.LoadRows(junk);
        micro.LoadPieces(current: 2, next: 1);

        const int rot = 2, col = 5;
        Assert.Equal(0, macro.ApplyPlacement(rot * 10 + col));

        Assert.True(micro.MicroSpawn());   // NES spawn: Td, box x=4 (origin 5)
        Assert.True(micro.MicroRotate());  // → Tl (uses the y=−1 head-room)
        Assert.True(micro.MicroRotate());  // → Tu = macro rot 2
        Assert.True(micro.MicroShift(+1)); // box x=4 → 5
        Assert.True(micro.MicroHardDrop());

        for (int y = 0; y < 20; y++) Assert.Equal(macro.Row(y), micro.Row(y));
        Assert.Equal(macro.Lines, micro.Lines);
        Assert.Equal(macro.PiecesPlaced, micro.PiecesPlaced);
    }

    [Theory]
    // piece order I O T S Z L J; rotCount 2 1 4 2 2 4 4.
    [InlineData(2)] // T
    [InlineData(5)] // L
    [InlineData(6)] // J
    public void MicroRotateCcw_FourTimesIsIdentity_ForTheFourStatePieces(int piece)
    {
        // M62.1: CCW walks the same NRS cycle backwards, so a full loop must restore rot, x and y exactly.
        var t = new TetrisBoard();
        t.Reset(7);
        t.LoadPieces(current: piece, next: 1);
        Assert.True(t.MicroSpawn());
        int rot0 = t.ActiveRot, x0 = t.ActiveX;

        for (int i = 0; i < 4; i++) Assert.True(t.MicroRotateCcw());

        Assert.Equal(rot0, t.ActiveRot);
        Assert.Equal(x0, t.ActiveX);
    }

    [Fact]
    public void MicroRotateCcw_IsTheInverseOfMicroRotate()
    {
        // CW then CCW returns to the spawn state — the property that makes Z a genuine "undo" of X.
        foreach (int piece in new[] { 0, 1, 2, 3, 4, 5, 6 })
        {
            var t = new TetrisBoard();
            t.Reset(7);
            t.LoadPieces(current: piece, next: 1);
            Assert.True(t.MicroSpawn());
            int rot0 = t.ActiveRot, x0 = t.ActiveX;

            Assert.True(t.MicroRotate());
            Assert.True(t.MicroRotateCcw());

            Assert.Equal(rot0, t.ActiveRot);
            Assert.Equal(x0, t.ActiveX);
        }
    }

    [Theory]
    [InlineData(0)] // I — rotCount 2
    [InlineData(1)] // O — rotCount 1, a no-op
    [InlineData(3)] // S — rotCount 2
    [InlineData(4)] // Z — rotCount 2
    public void MicroRotateCcw_MatchesMicroRotate_WhenTheCycleIsTwoOrOne(int piece)
    {
        // With rotCount ≤ 2 the cycle is its own inverse, so CCW must land on exactly the same state as CW.
        var cw = new TetrisBoard();
        cw.Reset(7);
        cw.LoadPieces(current: piece, next: 1);
        Assert.True(cw.MicroSpawn());
        Assert.True(cw.MicroRotate());

        var ccw = new TetrisBoard();
        ccw.Reset(7);
        ccw.LoadPieces(current: piece, next: 1);
        Assert.True(ccw.MicroSpawn());
        Assert.True(ccw.MicroRotateCcw());

        Assert.Equal(cw.ActiveRot, ccw.ActiveRot);
        Assert.Equal(cw.ActiveX, ccw.ActiveX);
    }

    [Fact]
    public void GravityCurve_IsTheAuthenticNtscTable_AndFlatAboveLevel29()
    {
        // M62: pinned value-by-value against the ROM table at $898E. Level 29 is the LAST speed change —
        // NTSC gravity is a flat 1 frame/row from 29 to 255. This test exists to stop a well-meaning
        // "the speed should keep increasing past 29" change: it should not.
        var t = new PgTetris();
        t.reset(1, false, 0);
        int[] expected = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3];
        for (int lvl = 0; lvl <= 18; lvl++) Assert.Equal(expected[lvl], t.gravityFrames(lvl));
        for (int lvl = 19; lvl <= 28; lvl++) Assert.Equal(2, t.gravityFrames(lvl));
        for (int lvl = 29; lvl <= 255; lvl++) Assert.Equal(1, t.gravityFrames(lvl));
    }

    [Fact]
    public void LevelProgression_From18Start_ReachesL19At130LinesAndL29At230()
    {
        // M62: this is the fact behind the "speed changes at 130 and 230" report — they are LINE
        // thresholds of the 18-start ROM progression, not levels and not speed steps.
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.setStartLevel(18);

        Assert.Equal(18, t.levelForLines(129));
        Assert.Equal(19, t.levelForLines(130));
        Assert.Equal(28, t.levelForLines(229));
        Assert.Equal(29, t.levelForLines(230));
    }

    [Theory]
    // M62: the ROM rule is first level-up at min(start*10 + 10, max(100, start*10 - 50)) lines, then every
    // 10. The shape looks arbitrary enough to invite "simplification", so the CTWC-relevant starts are
    // pinned.
    //
    // The punchline is the last three rows, and it is an INVARIANT rather than a coincidence: for every
    // start level from 15 upward the kill screen arrives at EXACTLY 230 lines. Above 15 the first
    // threshold is start*10 - 50, so the total is (start*10 - 50) + 10*(28 - start) = 230 — the start
    // level cancels out. That is why 230 is the number every commentator quotes, and why it sounds like a
    // property of the game rather than of a particular start. Below 15 it does vary (start 14 → 240).
    [InlineData(0, 10, 290)]
    [InlineData(9, 100, 290)]
    [InlineData(14, 100, 240)]
    [InlineData(15, 100, 230)]
    [InlineData(18, 130, 230)]
    [InlineData(19, 140, 230)]
    public void LevelProgression_MatchesTheRomRule_AtEveryCtwcStart(int start, int firstUp, int killScreenLines)
    {
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.setStartLevel(start);

        // Stays on the start level until the (irregular) first threshold, then steps exactly there.
        Assert.Equal(start, t.levelForLines(firstUp - 1));
        Assert.Equal(start + 1, t.levelForLines(firstUp));
        // Every 10 lines thereafter.
        Assert.Equal(start + 2, t.levelForLines(firstUp + 10));
        // And the kill screen lands where the tournament expects it.
        Assert.Equal(28, t.levelForLines(killScreenLines - 1));
        Assert.Equal(29, t.levelForLines(killScreenLines));
    }

    [Fact]
    public void LevelProgression_StartZeroSpecialCase_AgreesWithTheGeneralRule()
    {
        // levelForLines short-circuits startLevel 0 to floor(lines/10). The general branch produces the same
        // answer (first = min(10, 100) = 10), so the special case is redundant — pinned so that if anyone
        // removes it, the equivalence is what is being asserted rather than assumed.
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.setStartLevel(0);
        foreach (int lines in new[] { 0, 9, 10, 11, 29, 30, 99, 100, 229, 230, 289, 290 })
            Assert.Equal(lines / 10, t.levelForLines(lines));
    }

    [Fact]
    public void KillscreenVariants_DefaultIsAuthenticAndCostsNothing()
    {
        // Mode 0 is the shipped behaviour: one row per step at every level, never halted. The default must
        // stay this way or every checkpoint and the parity checksum move.
        var t = new PgTetris();
        t.reset(1, false, 0);
        Assert.Equal(0, t.killscreenMode);
        foreach (int lvl in new[] { 0, 18, 29, 39, 100, 255 }) Assert.Equal(1, t.gravityRowsPerStep(lvl));
        Assert.False(t.killscreenHalted());
    }

    [Fact]
    public void KillscreenVariants_2xksDoublesRowsPerStepFrom39()
    {
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.setKillscreenMode(2);
        foreach (int lvl in new[] { 0, 29, 38 }) Assert.Equal(1, t.gravityRowsPerStep(lvl));
        foreach (int lvl in new[] { 39, 40, 255 }) Assert.Equal(2, t.gravityRowsPerStep(lvl));
        Assert.False(t.killscreenHalted()); // 2xks never halts; it just becomes unsurvivable
    }

    [Fact]
    public void KillscreenVariants_HaltStopsAt39_AndOnlyAt39()
    {
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.setKillscreenMode(1);
        t.setStartLevel(38);
        Assert.False(t.killscreenHalted());
        Assert.Equal(1, t.gravityRowsPerStep(39)); // halt mode never changes gravity

        t.setStartLevel(39);
        Assert.True(t.killscreenHalted());

        t.forceGameOver();
        Assert.True(t.gameOver);
    }

    [Fact]
    public void Reachability_OffByDefault_IsExactlyPlacementLegal()
    {
        // M62.3b D7: enforcement is opt-in precisely so the default path — and therefore every shipped
        // checkpoint and the parity checksum — is untouched. If this ever fails, the opt-in leaked.
        var t = new TetrisBoard();
        t.Reset(11);
        for (int a = 0; a < TetrisBoard.ActionCount; a++)
            Assert.Equal(t.IsLegal(a), t.PlacementReachable(a));
    }

    [Fact]
    public void Reachability_AtLevelZero_EverythingLegalIsAlsoReachable()
    {
        // 48 frames per row is so slow that even DAS crosses the whole board with frames to spare. This is
        // why an A/B at the default start level shows nothing — the dial cannot bite until gravity is fast.
        var t = new TetrisBoard();
        t.Reset(11);
        t.SetReachEnforced(true);
        t.SetTapModel(6, 16); // DAS, the most restrictive
        for (int a = 0; a < TetrisBoard.ActionCount; a++)
            Assert.Equal(t.IsLegal(a), t.PlacementReachable(a));
    }

    [Theory]
    [InlineData(19)]
    [InlineData(29)]
    public void Reachability_RollingReachesASupersetOfDas(int startLevel)
    {
        // The core ordering property: faster hands never reach LESS. Checked across pieces and seeds so a
        // sign error or an off-by-one in the shift schedule cannot hide in one lucky board.
        for (ulong seed = 1; seed <= 12; seed++)
        {
            var das = new TetrisBoard();
            das.Reset(seed);
            das.SetStartLevel(startLevel);
            das.SetReachEnforced(true);
            das.SetTapModel(6, 16);

            var roll = new TetrisBoard();
            roll.Reset(seed);
            roll.SetStartLevel(startLevel);
            roll.SetReachEnforced(true);
            roll.SetTapModel(3, 3);

            for (int a = 0; a < TetrisBoard.ActionCount; a++)
                if (das.PlacementReachable(a))
                    Assert.True(roll.PlacementReachable(a),
                        $"seed {seed}, action {a}: DAS reached it but rolling did not");
        }
    }

    [Fact]
    public void Reachability_AtTheKillScreen_DasLosesPlacementsThatRollingKeeps()
    {
        // The measurable claim behind the whole technique dial. At 1 frame per row DAS pays a 16-frame
        // charge before its second shift, which is 16 rows of fall — so the far columns go out of reach.
        var das = new TetrisBoard();
        das.Reset(5);
        das.SetStartLevel(29);
        das.SetReachEnforced(true);
        das.SetTapModel(6, 16);

        var roll = new TetrisBoard();
        roll.Reset(5);
        roll.SetStartLevel(29);
        roll.SetReachEnforced(true);
        roll.SetTapModel(3, 3);

        int dasCount = 0, rollCount = 0;
        for (int a = 0; a < TetrisBoard.ActionCount; a++)
        {
            if (das.PlacementReachable(a)) dasCount++;
            if (roll.PlacementReachable(a)) rollCount++;
        }

        Assert.True(dasCount > 0, "DAS must always keep the placements near spawn — an empty mask is a top-out");
        Assert.True(rollCount > dasCount, $"rolling {rollCount} should beat DAS {dasCount} at the kill screen");
    }

    [Fact]
    public void ReachTimeline_ReplaysToExactlyTheChosenPlacement()
    {
        // D9's guarantee, asserted rather than assumed: replaying the emitted inputs through the MICRO api,
        // against the same gravity clock, must land the piece on the same board the MACRO placement gives.
        // This is what makes "the AI chose it" and "the player saw it" the same statement.
        var t = new TetrisBoard();
        t.Reset(5);
        t.SetStartLevel(19);
        t.SetReachEnforced(true);
        t.SetTapModel(3, 3);

        int checkedCount = 0;
        for (int a = 0; a < TetrisBoard.ActionCount && checkedCount < 6; a++)
        {
            if (!t.PlacementReachable(a)) continue;
            var timeline = t.ReachTimelineFor(a);

            var macro = new TetrisBoard();
            macro.Reset(5);
            macro.SetStartLevel(19);
            Assert.Equal(0, macro.ApplyPlacement(a));

            var micro = new TetrisBoard();
            micro.Reset(5);
            micro.SetStartLevel(19);
            Assert.True(micro.MicroSpawn());

            int g = micro.GravityFramesAt(19);
            int rows = micro.GravityRowsPerStep();
            int targetRot = a / 10, targetX = a % 10;
            for (int frame = 0, ev = 0; frame < 1200; frame++)
            {
                while (ev + 1 < timeline.Count && timeline[ev] == frame)
                {
                    switch (timeline[ev + 1])
                    {
                        case 1: micro.MicroShift(-1); break;
                        case 2: micro.MicroShift(1); break;
                        case 3: micro.MicroRotate(); break;
                        case 4: micro.MicroRotateCcw(); break;
                    }
                    ev += 2;
                }
                if (micro.ActiveRot == targetRot && micro.ActiveX == targetX) break;
                if ((frame + 1) % g == 0)
                    for (int r = 0; r < rows; r++) micro.MicroDropStep();
            }

            Assert.Equal(targetRot, micro.ActiveRot);
            Assert.Equal(targetX, micro.ActiveX);
            micro.MicroHardDrop();
            for (int y = 0; y < 20; y++) Assert.Equal(macro.Row(y), micro.Row(y));
            checkedCount++;
        }
        Assert.True(checkedCount > 0, "no reachable placement was exercised");
    }

    [Theory]
    [InlineData(6, 6)] // DAS
    [InlineData(3, 3)] // rolling
    public void ReachSpan_CoversTheWholeBoard_WhenNothingIsMasked(int frames, int charge)
    {
        // REGRESSION (M62.4a). The span must be measured over the COLUMNS A PIECE WOULD OCCUPY, not over
        // action indices. actionCol is the placement's LEFT edge, so a piece of width w can never have
        // actionCol > W - w: an O tops out at 8 and a horizontal I at 6. A span built from action indices
        // therefore reported reachHi < 9 for nearly every piece at every level, `lineout` became
        // permanently true, and EvalReady — the only term rewarding an open well — was switched off.
        // Measured cost at level 0 with rolling, where the census says nothing is masked at all:
        // della-search fell from 17.62 to 0.12 tetrises per episode.
        //
        // Every other gate stayed green through that bug — parity covers the unenforced path, the
        // reachability tests count set sizes, latency was fine. Nothing asserted that the AI still WANTS a
        // well. This is that assertion.
        foreach (int piece in new[] { 0, 1, 2, 3, 4, 5, 6 })
        {
            var t = new PgTetris();
            t.reset(5, false, 0);
            t.current = piece;
            t.next = piece;
            t.setReachEnforced(true);
            t.setTapModel(frames, charge);
            t.refreshReachSpan();

            Assert.Equal(0, t.reachLo);
            Assert.Equal(TetrisBoard.Width - 1, t.reachHi);
        }
    }

    [Fact]
    public void Features_PinnedOnAHandDrawnBoard()
    {
        // Single filled cell at (x=0, y=19): rowT = 19 empty rows × 2 + 2 = 40; colT = 1 (col 0) + 9
        // empty-column floors = 10; no holes; no wells (spike variant).
        var t = new PgTetris();
        t.reset(1, false, 0);
        for (int y = 0; y < 20; y++) t.rows[y] = 0;
        t.rows[19] = 1;
        Assert.Equal(40, t.rowTransitions());
        Assert.Equal(10, t.colTransitions());
        Assert.Equal(0, t.holes());
        Assert.Equal(0, t.wellSum());

        // A buried gap is a hole; a 3-deep open shaft flanked by walls is a 3-well.
        for (int y = 0; y < 20; y++) t.rows[y] = 0;
        t.rows[19] = 0b0000000101; // gap at column 1 between two filled cells
        t.rows[18] = 0b0000000010; // cap above the gap → 1 hole at (1,19)
        Assert.Equal(1, t.holes());
        for (int y = 0; y < 20; y++) t.rows[y] = 0;
        t.rows[17] = 0b0000000101;
        t.rows[18] = 0b0000000101;
        t.rows[19] = 0b0000000111; // column-1 shaft of depth 2 above a floor cell
        Assert.Equal(2, t.wellSum());
    }

    // ── NES rotation (M55.3, ROM-table NRS: in-place pivot, target-cells-only check, NO kicks) ─────────────

    private static PgTetris MicroBoard(int piece, int rot, int x, int y)
    {
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.current = piece;
        t.activeRot = rot;
        t.activeX = x;
        t.activeY = y;
        t.activeLive = true;
        return t;
    }

    private static int[] Cells(PgTetris t)
    {
        var ri = t.current * 4 + t.activeRot;
        return [.. Enumerable.Range(0, 4)
            .Select(k => (t.activeY + t.cellY[ri * 4 + k]) * 10 + t.activeX + t.cellX[ri * 4 + k])
            .OrderBy(c => c)];
    }

    [Fact]
    public void NesSpawn_PiecesAppearAtOriginFiveInTheirNesState()
    {
        // NES spawn origin (5,0): T/J/L/S/Z occupy x 4–6, O x 4–5, I x 3–6; spawn states Td/Jd/Ld/Sh/Zh/Ih.
        var t = new PgTetris();
        t.reset(1, false, 0);
        (int piece, int rot, int x)[] expected =
            [(0, 0, 3), (1, 0, 4), (2, 0, 4), (3, 0, 4), (4, 0, 4), (5, 1, 4), (6, 3, 4)];
        foreach (var (piece, rot, x) in expected)
        {
            t.current = piece;
            Assert.True(t.microSpawn());
            Assert.Equal(rot, t.activeRot);
            Assert.Equal(x, t.activeX);
            Assert.Equal(0, t.activeY);
        }
    }

    [Fact]
    public void NesRotation_TCyclesInPlaceAroundItsPivot()
    {
        // The research sanity example, origin (5,10): Td → Tl → Tu → Tr → Td, pivot never moves.
        var t = MicroBoard(piece: 2, rot: 0, x: 4, y: 10); // Td box top-left = origin(5,10) + (−1,0)
        Assert.Equal(new[] { 104, 105, 106, 115 }, Cells(t));            // (4..6,10) + (5,11)
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 95, 104, 105, 115 }, Cells(t));             // Tl: (5,9)(4,10)(5,10)(5,11)
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 95, 104, 105, 106 }, Cells(t));             // Tu: (5,9)(4..6,10)
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 95, 105, 106, 115 }, Cells(t));             // Tr: (5,9)(5,10)(6,10)(5,11)
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 104, 105, 106, 115 }, Cells(t));            // back to Td — a true pivot
    }

    [Fact]
    public void NesRotation_IWobblesBetweenItsTwoStates()
    {
        // Ih (3..6,10) ⇄ Iv (5,8..11) — the NES I's characteristic asymmetric wobble about column 5.
        var t = MicroBoard(piece: 0, rot: 0, x: 3, y: 10);
        Assert.Equal(new[] { 103, 104, 105, 106 }, Cells(t));
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 85, 95, 105, 115 }, Cells(t));
        Assert.True(t.microRotate());
        Assert.Equal(new[] { 103, 104, 105, 106 }, Cells(t));
    }

    [Fact]
    public void NesRotation_TRotatesWithAllFourDiagonalsOccupied()
    {
        // The owner's scenario: only the TARGET cells matter — filled diagonal corners around the pivot
        // must not block the rotation (the NES "T-slot" feel).
        var t = MicroBoard(piece: 2, rot: 0, x: 4, y: 10); // Td, pivot (5,10)
        t.rows[9] = (1 << 4) | (1 << 6);   // corners above: (4,9) (6,9)
        t.rows[11] = (1 << 4) | (1 << 6);  // corners below: (4,11) (6,11)
        Assert.True(t.microRotate(), "occupied diagonals must not block an NRS rotation");
        Assert.Equal(new[] { 95, 104, 105, 115 }, Cells(t)); // Tl, in place
    }

    [Fact]
    public void NesRotation_NoKicks_WallAndFloorBlockRotation()
    {
        // Pure NES: a vertical I hugging the right wall CANNOT rotate (Ih would need x 7..10)...
        var t = MicroBoard(piece: 0, rot: 1, x: 9, y: 10);
        Assert.False(t.microRotate());
        Assert.Equal(1, t.activeRot);
        Assert.Equal(9, t.activeX);
        Assert.Equal(10, t.activeY);
        // ...and a flat I on the floor cannot go vertical (Iv would need rows 17..22).
        var t2 = MicroBoard(piece: 0, rot: 0, x: 3, y: 19);
        Assert.False(t2.microRotate());
        Assert.Equal(0, t2.activeRot);
    }

    [Fact]
    public void NesRotation_SpawnRowRotationUsesTheVirtualHeadroom()
    {
        // On the NES the validity check accepts cells up to 2 rows above the board — so a T can rotate
        // on its spawn row (Tl's box top lands at y = −1).
        var t = new PgTetris();
        t.reset(1, false, 0);
        t.current = 2;
        Assert.True(t.microSpawn());
        Assert.True(t.microRotate(), "spawn-row rotation must succeed via the y ≥ −2 head-room");
        Assert.Equal(1, t.activeRot);
        Assert.Equal(-1, t.activeY);
    }

    [Fact]
    public void NesRotation_BlockedBySquares_StillFails()
    {
        var t = MicroBoard(piece: 0, rot: 1, x: 9, y: 16); // vertical I at the right wall...
        for (int y = 10; y < 20; y++) t.rows[y] = 0b0111111111; // ...columns 0..8 solid below row 10
        Assert.False(t.microRotate(), "a rotation blocked by real squares must fail in place");
        Assert.Equal(1, t.activeRot);
        Assert.Equal(9, t.activeX);
    }

    [Fact]
    public void NesLevels_AdvanceEveryTenLines_WithPostLevelUpScoringAndTheSpeedCurve()
    {
        var t = new PgTetris();
        t.reset(3, false, 0);
        Assert.Equal(0, t.level);
        t.lines = 9;
        t.afterLock(1); // 10th line ⇒ level 1; NES rule: the points use the level AFTER the level-up
        Assert.Equal(10, t.lines);
        Assert.Equal(1, t.level);
        Assert.Equal(40 * 2, t.score);

        Assert.Equal(48, t.gravityFrames(0));
        Assert.Equal(8, t.gravityFrames(8));
        Assert.Equal(6, t.gravityFrames(9));
        Assert.Equal(2, t.gravityFrames(28));
        Assert.Equal(1, t.gravityFrames(29)); // the kill screen
    }

    [Fact]
    public void SevenBag_DealsEachPieceExactlyOncePerBag()
    {
        var t = new PgTetris();
        t.reset(3, true, 0);
        var firstBag = new List<int> { t.current, t.next };
        for (int i = 0; i < 5; i++) firstBag.Add(t.drawPiece());
        Assert.Equal([0, 1, 2, 3, 4, 5, 6], firstBag.OrderBy(x => x));
    }

    [Fact]
    public void StateRoundTrip_ResumedGameEvolvesIdentically()
    {
        var a = new TetrisBoard();
        a.Reset(11, sevenBag: true, garbageEvery: 10);
        for (int i = 0; i < 30 && !a.GameOver; i++) a.ApplyPlacement(a.DellacherieAction());

        var ms = new MemoryStream();
        a.WriteState(new BinaryWriter(ms));
        ms.Position = 0;
        var b = new TetrisBoard();
        b.ReadState(new BinaryReader(ms));

        for (int i = 0; i < 30 && !a.GameOver; i++)
        {
            int action = a.DellacherieAction();
            Assert.Equal(action, b.DellacherieAction());
            Assert.Equal(a.ApplyPlacement(action), b.ApplyPlacement(action));
        }
        for (int y = 0; y < 20; y++) Assert.Equal(a.Row(y), b.Row(y));
        Assert.Equal(a.Score, b.Score);
    }

    // ── Spike-bar reproduction (the M54.1 env-validation gate) ──────────────────────────────────────────────

    private static int PlayRandomToTopOut(ulong seed, int garbageEvery)
    {
        var board = new TetrisBoard();
        board.Reset(seed, sevenBag: false, garbageEvery: garbageEvery);
        int steps = 0;
        while (!board.GameOver && steps < 100_000)
        {
            board.ApplyPlacement(board.RandomAction(seed, steps));
            steps++;
        }
        return board.PiecesPlaced;
    }

    [Fact]
    public void SpikeBar_RandomPolicyTopsOutNearTwentySixPieces()
    {
        double mean = Enumerable.Range(0, 300).Select(e => (double)PlayRandomToTopOut((ulong)(9000 + e), 0)).Average();
        Assert.InRange(mean, 24.0, 27.5); // spike: 25.6 ± 0.4 (n=200)
    }

    /// <summary>
    /// M57.1 re-pin. Pre-M57.1 this asserted "clears near-maximal lines, never tops out" (197.4 lines,
    /// 0 top-outs) — the signature of an evaluator that flattens the stack and burns singles. The widened
    /// evaluator deliberately trades a little line-count, and a few top-outs, for far more SCORE and
    /// tetrises: the whole point of the milestone. Baselines it replaced: 197.4 lines / ~94.6k score /
    /// 0.26 tetrises / 0 top-outs.
    /// </summary>
    [Fact]
    public void SpikeBar_DellacherieBuildsForTetrisesTradingLinesForScore()
    {
        double totalLines = 0, totalScore = 0, totalTetrises = 0;
        int topOuts = 0;
        for (int e = 0; e < 20; e++)
        {
            var board = new TetrisBoard();
            board.Reset((ulong)(7000 + e));
            for (int i = 0; i < 500 && !board.GameOver; i++) board.ApplyPlacement(board.DellacherieAction());
            if (board.GameOver) topOuts++;
            totalLines += board.Lines;
            totalScore += board.Score;
            totalTetrises += board.Tetrises;
        }
        double lines = totalLines / 20, score = totalScore / 20, tetrises = totalTetrises / 20;
        Assert.True(score >= 110_000, $"mean score {score:F0} < 110,000 (M57.1 measured ~132k; pre-M57.1 94.6k)");
        Assert.True(tetrises >= 2.0, $"mean tetrises {tetrises:F2} < 2.0 (M57.1 measured ~3.4; pre-M57.1 0.26)");
        Assert.True(lines >= 165.0, $"mean lines {lines:F1} < 165 (M57.1 measured ~191; pre-M57.1 197.4)");
        // Stack-and-camp watchdog: building for tetrises costs SOME top-outs, but must not become reckless.
        Assert.True(topOuts <= 6, $"{topOuts}/20 episodes topped out — the tetris terms have gone reckless");
    }

    [Fact]
    public void SpikeBar_GarbageSurvivalSeparatesDellacherieFromRandom()
    {
        double randomMean = Enumerable.Range(0, 100).Select(e => (double)PlayRandomToTopOut((ulong)(4000 + e), 10)).Average();

        double dellaTotal = 0;
        for (int e = 0; e < 30; e++)
        {
            var board = new TetrisBoard();
            board.Reset((ulong)(4200 + e), sevenBag: false, garbageEvery: 10);
            int steps = 0;
            while (!board.GameOver && steps < 5000) { board.ApplyPlacement(board.DellacherieAction()); steps++; }
            dellaTotal += board.PiecesPlaced;
        }
        double dellaMean = dellaTotal / 30;

        Assert.InRange(randomMean, 18.0, 30.0);          // spike: 21.6 ± 0.6
        Assert.True(dellaMean >= 200.0, $"della garbage survival {dellaMean:F1} < 200 (spike: 392.8 ± 45.1)");
        Assert.True(dellaMean >= 5 * randomMean, "expected a wide policy separation under garbage");
    }

    /// <summary>
    /// M57 / gate G7. The NES A-TYPE curve with a selectable start level: the first level-up arrives after
    /// min(start*10+10, max(100, start*10-50)) lines, then every 10. An 18-start therefore reaches 19 at
    /// 130 lines and the level-29 kill screen at 230 — the numbers competitive play is organised around.
    /// </summary>
    [Fact]
    public void StartLevel_FollowsTheNesTransitionCurveAndDrivesGravity()
    {
        var b = new TetrisBoard();
        b.Reset(1);
        Assert.Equal(0, b.Level);                 // default is unchanged: every pre-M57 protocol is intact
        Assert.Equal(48, b.GravityFrames());      // level 0 = 48 frames/row

        b.SetStartLevel(18);
        Assert.Equal(18, b.Level);
        Assert.Equal(3, b.GravityFrames());       // level 18 = 3 frames/row

        b.SetStartLevel(19);
        Assert.Equal(2, b.GravityFrames());       // 19-28 = 2
        b.SetStartLevel(29);
        Assert.Equal(1, b.GravityFrames());       // the kill screen = 1

        // Reset clears the start level so protocols cannot leak into one another.
        b.Reset(2);
        Assert.Equal(0, b.Level);
        Assert.Equal(48, b.GravityFrames());
    }
}
