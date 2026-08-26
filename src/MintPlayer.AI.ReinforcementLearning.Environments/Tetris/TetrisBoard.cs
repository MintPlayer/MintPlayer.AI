namespace MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

// PUBLIC FACADE over the single-source transpiled engine (polyglot/tetris_solver.pg → PgTetris).
// The rules live ONCE in the .pg (shared with the web client's TypeScript); this facade adapts the
// generated internal, camelCase core to the public API the env/Lab/tests consume, plus host-only
// helpers (state serialization, LoadRows for directed tests).

/// <summary>
/// The Tetris board: 10×20, 7 tetrominoes, afterstate macro-actions (action = rot·10 + col, hard
/// vertical drop, no tucks/kicks — TETRIS_PRD.md §1). Reward currency is LINES cleared per placement;
/// the display score (100/300/500/800) accrues separately. Optional 7-bag piece stream and the
/// rising-garbage mode (a gapped bottom row every N placements). Top-out = no legal placement for the
/// new piece, or a garbage shift overflowing the top.
/// </summary>
public sealed class TetrisBoard
{
    public const int Width = 10;
    public const int Height = 20;
    public const int PieceCount = 7; // 0=I 1=O 2=T 3=S 4=Z 5=L 6=J
    public const int ActionCount = 40; // 4 rotations × 10 columns, hard-masked
    // 454: 200 board cells + 7 current + 7 next one-hots + six 40-wide per-action feature planes
    // (landing/20, eroded/8, ΔrowT/20, ΔcolT/20, Δholes/10, Δwells/20 — PRD §3.4).
    public const int ObservationSize = Width * Height + 2 * PieceCount + 6 * ActionCount;

    private readonly PgTetris _core = new();

    /// <summary>Start a fresh game. <paramref name="sevenBag"/> selects the 7-bag stream (web default)
    /// over uniform-random (training/benchmark); <paramref name="garbageEvery"/> = 0 disables garbage.</summary>
    public void Reset(ulong seed, bool sevenBag = false, int garbageEvery = 0)
        => _core.reset(SeedToInt(seed), sevenBag, garbageEvery);

    /// <summary>Map any 64-bit seed onto the engine's minstd state range [1, 2^31−2].</summary>
    internal static int SeedToInt(ulong seed) => (int)(seed % 2147483646UL) + 1;

    public int Score => _core.score;
    public int Lines => _core.lines;
    public int PiecesPlaced => _core.piecesPlaced;
    public bool GameOver => _core.gameOver;
    public int LastLinesCleared => _core.lastLinesCleared;
    public int CurrentPiece => _core.current;
    public int NextPiece => _core.next;

    /// <summary>Row bitmask (bit c = column c filled) of row <paramref name="y"/> (0 = top).</summary>
    public int Row(int y) => _core.rows[y];

    /// <summary>True if the cell at column <paramref name="x"/>, row <paramref name="y"/> is filled.</summary>
    public bool Cell(int x, int y) => ((_core.rows[y] >> x) & 1) == 1;

    // ── Macro (AI/env) step API ──────────────────────────────────────────────────────────────────────────────

    /// <summary>Legality of each of the <see cref="ActionCount"/> placements for the current piece.
    /// All-false means the board has topped out (terminated).</summary>
    public bool[] LegalMask()
    {
        var core = _core.legalMask();
        var mask = new bool[ActionCount];
        for (int i = 0; i < ActionCount; i++) mask[i] = core[i];
        return mask;
    }

    public bool IsLegal(int action) => _core.placementLegal(action);
    public bool HasLegalPlacement() => _core.hasLegalPlacement();

    /// <summary>Apply a placement end-to-end (lock, clear, score, garbage clock, draw). Returns the
    /// LINES cleared (0–4, the reward currency); −1 for an illegal action (no change).</summary>
    public int ApplyPlacement(int action) => _core.applyPlacement(action);

    /// <summary>The <see cref="ObservationSize"/>-dim observation (f64 core → float32 net).</summary>
    public float[] BuildObservation()
    {
        var core = _core.buildObservation();
        var obs = new float[core.Count];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)core[i];
        return obs;
    }

    // ── Micro (human) step API — same lock/scoring/garbage path as the macro one ────────────────────────────

    /// <summary>Spawn the current piece for micro play (rot 0, x 3, y 0). False = blocked ⇒ game over.</summary>
    public bool MicroSpawn() => _core.microSpawn();

    public bool MicroShift(int dx) => _core.microShift(dx);
    public bool MicroRotate() => _core.microRotate();

    /// <summary>One gravity/soft-drop step; true if the piece locked.</summary>
    public bool MicroDropStep() => _core.microDropStep();

    /// <summary>Drop to the floor and lock; true unless the game is over.</summary>
    public bool MicroHardDrop() => _core.microHardDrop();

    public int ActiveRot => _core.activeRot;
    public int ActiveX => _core.activeX;
    public int ActiveY => _core.activeY;
    public bool ActiveLive => _core.activeLive;

    // ── Scripted tiers (single-sourced in the .pg; eval bars + env sanity gates + web watch tiers) ──────────

    /// <summary>Uniform-random legal placement, drawn from a caller-owned policy stream (never the
    /// piece/garbage RNGs). The (seed, step) pair makes each draw independent and reproducible.</summary>
    public int RandomAction(ulong policySeed, int step)
    {
        var rng = new PgTetRng(SeedToInt(policySeed ^ (ulong)(step * 0x9E3779B9L)));
        return _core.randomAction(rng);
    }

    /// <summary>The Dellacherie tier: argmax placement by the canonical hand-tuned evaluator.</summary>
    public int DellacherieAction() => _core.dellacherieAction();

    /// <summary>Dellacherie score of placing the current piece at (rot, col); −1e18 if illegal.</summary>
    public double DellaScore(int rot, int col) => _core.dellaScoreFor(_core.current, rot, col);

    /// <summary>Search tier over the Dellacherie evaluator: beamA first ply, beamB over the known next
    /// piece, expectimax over the unknown third piece (PRD §3.8).</summary>
    public int DellaSearchAction(int beamA = 8, int beamB = 5) => _core.dellaSearchAction(beamA, beamB);

    // ── Host-only helpers ────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Overwrite the board rows for directed tests (bitmasks, row 0 = top). Counters untouched.</summary>
    public void LoadRows(ReadOnlySpan<int> rowMasks)
    {
        if (rowMasks.Length != Height) throw new ArgumentException($"expected {Height} rows", nameof(rowMasks));
        for (int y = 0; y < Height; y++) _core.rows[y] = rowMasks[y];
    }

    /// <summary>Force the current/next pieces for directed tests.</summary>
    public void LoadPieces(int current, int next)
    {
        _core.current = current;
        _core.next = next;
    }

    /// <summary>Serialize the full engine state for env Save.</summary>
    public void WriteState(BinaryWriter writer)
    {
        writer.Write(_core.pieceRng.state);
        writer.Write(_core.garbageRng.state);
        writer.Write(_core.useBag);
        writer.Write(_core.bagPos);
        writer.Write(_core.bag.Count);
        for (int i = 0; i < _core.bag.Count; i++) writer.Write(_core.bag[i]);
        writer.Write(_core.current);
        writer.Write(_core.next);
        writer.Write(_core.score);
        writer.Write(_core.lines);
        writer.Write(_core.piecesPlaced);
        writer.Write(_core.gameOver);
        writer.Write(_core.lastLinesCleared);
        writer.Write(_core.garbageEvery);
        writer.Write(_core.garbageCounter);
        writer.Write(_core.activeRot);
        writer.Write(_core.activeX);
        writer.Write(_core.activeY);
        writer.Write(_core.activeLive);
        for (int y = 0; y < Height; y++) writer.Write(_core.rows[y]);
    }

    /// <summary>Restore the state written by <see cref="WriteState"/>.</summary>
    public void ReadState(BinaryReader reader)
    {
        _core.pieceRng = new PgTetRng(1) { state = reader.ReadInt32() };
        _core.garbageRng = new PgTetRng(1) { state = reader.ReadInt32() };
        _core.useBag = reader.ReadBoolean();
        _core.bagPos = reader.ReadInt32();
        int bagCount = reader.ReadInt32();
        _core.bag.Clear();
        for (int i = 0; i < bagCount; i++) _core.bag.Add(reader.ReadInt32());
        _core.current = reader.ReadInt32();
        _core.next = reader.ReadInt32();
        _core.score = reader.ReadInt32();
        _core.lines = reader.ReadInt32();
        _core.piecesPlaced = reader.ReadInt32();
        _core.gameOver = reader.ReadBoolean();
        _core.lastLinesCleared = reader.ReadInt32();
        _core.garbageEvery = reader.ReadInt32();
        _core.garbageCounter = reader.ReadInt32();
        _core.activeRot = reader.ReadInt32();
        _core.activeX = reader.ReadInt32();
        _core.activeY = reader.ReadInt32();
        _core.activeLive = reader.ReadBoolean();
        _core.rows.Clear();
        for (int y = 0; y < Height; y++) _core.rows.Add(reader.ReadInt32());
    }
}
