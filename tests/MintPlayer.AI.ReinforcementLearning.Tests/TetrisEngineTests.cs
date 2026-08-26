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

        Assert.True(micro.MicroSpawn());
        Assert.True(micro.MicroRotate());
        Assert.True(micro.MicroRotate());
        Assert.True(micro.MicroShift(+1)); // spawn x=3 → 5
        Assert.True(micro.MicroShift(+1));
        Assert.True(micro.MicroHardDrop());

        for (int y = 0; y < 20; y++) Assert.Equal(macro.Row(y), micro.Row(y));
        Assert.Equal(macro.Lines, micro.Lines);
        Assert.Equal(macro.PiecesPlaced, micro.PiecesPlaced);
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

    // ── Rotation kicks (owner report 2026-08-26: pieces not blocked by squares must still rotate) ──────────

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

    [Fact]
    public void Rotate_VerticalIAtTheRightWall_WallKicksIntoTheBoard()
    {
        var t = MicroBoard(piece: 0, rot: 1, x: 9, y: 10); // vertical I hugging the right wall, open board
        Assert.True(t.microRotate(), "open-space rotation must succeed via a wall kick");
        Assert.Equal(0, t.activeRot);
        Assert.Equal(6, t.activeX); // shifted just enough for the 4-wide horizontal I
        Assert.Equal(10, t.activeY);
    }

    [Fact]
    public void Rotate_FlatIOnTheFloor_FloorKicksUpward()
    {
        var t = MicroBoard(piece: 0, rot: 0, x: 3, y: 19); // flat I lying on the floor
        Assert.True(t.microRotate(), "open-space rotation must succeed via a floor kick");
        Assert.Equal(1, t.activeRot);
        Assert.Equal(16, t.activeY); // lifted just enough for the 4-tall vertical I
    }

    [Fact]
    public void Rotate_TAgainstTheLeftWall_Succeeds()
    {
        var t = MicroBoard(piece: 2, rot: 1, x: 0, y: 10); // T pointing left, flush with the wall
        Assert.True(t.microRotate());
    }

    [Fact]
    public void Rotate_SZLJOnTheFloor_AllSucceedInOpenSpace()
    {
        // Every piece, every rotation slot, resting on the floor of an empty board: rotation must succeed
        // (kick ladder), except the O piece which has a single rotation and trivially succeeds in place.
        var probe = new PgTetris();
        probe.reset(1, false, 0);
        for (int piece = 0; piece < TetrisBoard.PieceCount; piece++)
        {
            for (int rot = 0; rot < probe.rotCount[piece]; rot++)
            {
                int h = probe.rotH[piece * 4 + rot];
                var t = MicroBoard(piece, rot, x: 4, y: 20 - h); // resting on the floor
                Assert.True(t.microRotate(), $"piece {piece} rot {rot} must rotate in open space");
            }
        }
    }

    [Fact]
    public void Rotate_ActuallyBlockedBySquares_StillFails()
    {
        var t = MicroBoard(piece: 0, rot: 1, x: 9, y: 16); // vertical I at the right wall...
        for (int y = 10; y < 20; y++) t.rows[y] = 0b0111111111; // ...columns 0..8 solid below row 10
        Assert.False(t.microRotate(), "a rotation blocked by real squares must not kick through them");
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

    [Fact]
    public void SpikeBar_DellacherieClearsNearMaximalLinesInCappedEpisodes()
    {
        double totalLines = 0;
        for (int e = 0; e < 20; e++)
        {
            var board = new TetrisBoard();
            board.Reset((ulong)(7000 + e));
            for (int i = 0; i < 500 && !board.GameOver; i++) board.ApplyPlacement(board.DellacherieAction());
            Assert.False(board.GameOver); // spike: 0 top-outs in 50 episodes
            totalLines += board.Lines;
        }
        Assert.True(totalLines / 20 >= 190.0, $"mean lines {totalLines / 20:F1} < 190 (spike: 197.4 ± 0.4)");
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
}
