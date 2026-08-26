namespace MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

// PUBLIC FACADE over the single-source transpiled engine (polyglot/tetris_solver.pg → PgTetris).
// The rules live ONCE in the .pg (shared with the web client's TypeScript); this facade adapts the
// generated internal, camelCase core to the public API the env/Lab/tests consume, plus host-only
// helpers (state serialization, LoadRows for directed tests).

/// <summary>
/// The Tetris board: 10×20, 7 tetrominoes, afterstate macro-actions (action = rot·10 + col, hard
/// vertical drop, no tucks/kicks — TETRIS_PRD.md §1). Reward currency is LINES cleared per placement
/// (+ the tetris bonus); the NES-style score (40/100/300/1200) accrues separately and is the metric the
/// AI is asked to maximize. Optional 7-bag piece stream and the
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

    /// <summary>Training-reward bonus (lines units) for a 4-line clear — mirror of the engine's
    /// <c>RewardTetrisBonus</c> (owner decision 2026-08-26: build for tetrises).</summary>
    public const int TetrisRewardBonus = 8;

    public int Score => _core.score;
    public int Lines => _core.lines;
    public int Tetrises => _core.tetrises;
    /// <summary>NES level: lines/10 (start level 0); scoring is base × (level+1), gravity follows the NES curve.</summary>
    public int Level => _core.level;
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

    // ── Board features (the Dellacherie basis, on the CURRENT rows — the env's shaping potential reads
    // these; per-placement deltas live in the observation planes) ───────────────────────────────────────────

    public int Holes() => _core.holes();
    public int RowTransitions() => _core.rowTransitions();
    public int ColTransitions() => _core.colTransitions();
    public int WellSum() => _core.wellSum();

    // ── Trained-net tiers through the single-source forward ─────────────────────────────────────────────────
    // The SDK's GreedyQAgent is the reference net player (float32 forward); these run the GENERATED f64
    // forward instead — the exact code the browser executes — so the Lab's net+search row and the
    // net-parity test measure what actually ships.

    private PgTetDuelingNet? _net;

    /// <summary>Input width of the loaded net; −1 when none is loaded.</summary>
    public int NetInputSize => _net?.inputSize ?? -1;

    /// <summary>Parse a dueling-q checkpoint into the single-source net (the line-for-line reference for
    /// <c>tetris-net.ts</c> — keep the two in sync). False = wrong kind or input width (stale checkpoint).</summary>
    public bool LoadNet(Stream checkpoint)
    {
        var net = ParseDuelingQCheckpoint(checkpoint);
        if (net is null || net.inputSize != ObservationSize) return false;
        _net = net;
        return true;
    }

    /// <summary>Masked argmax over the loaded net's Q — the browser's "Trained net" tier.</summary>
    public int NetAction() => _net is null ? -1 : _core.netAction(_net);

    /// <summary>Beam over Q + one-ply rollout + expectimax over the unknown next-next piece — the browser's
    /// "Net + search" tier (PRD §3.8).</summary>
    public int NetSearchAction(int beamWidth = 8) => _net is null ? -1 : _core.netSearchAction(_net, beamWidth);

    internal static PgTetDuelingNet? ParseDuelingQCheckpoint(Stream stream)
    {
        using var r = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        List<double> ReadFloats()
        {
            int n = r.ReadInt32();
            var a = new List<double>(n);
            for (int i = 0; i < n; i++) a.Add(r.ReadSingle()); // float32 → exact f64, like the DataView read
            return a;
        }

        if (r.ReadUInt32() != 0x434E4C52u) return null;      // "RLNC"
        if (r.ReadString() != "dueling-q") return null;
        int version = r.ReadInt32();
        int inputSize = r.ReadInt32();
        int hiddenCount = r.ReadInt32();
        var hidden = new List<int>();
        for (int i = 0; i < hiddenCount; i++) hidden.Add(r.ReadInt32());
        int actions = r.ReadInt32();
        bool noisy = version >= 2 && r.ReadByte() != 0;

        var trunkW = new List<double>();
        var trunkB = new List<double>();
        for (int l = 0; l < hiddenCount; l++)
        {
            trunkW.AddRange(ReadFloats());
            trunkB.AddRange(ReadFloats());
        }

        List<double> valueW, valueB, advW, advB;
        if (!noisy)
        {
            valueW = ReadFloats(); valueB = ReadFloats();
            advW = ReadFloats(); advB = ReadFloats();
        }
        else
        {
            valueW = ReadFloats(); ReadFloats(); valueB = ReadFloats(); ReadFloats();
            advW = ReadFloats(); ReadFloats(); advB = ReadFloats(); ReadFloats();
        }

        return new PgTetDuelingNet(inputSize, actions, hidden, trunkW, trunkB, valueW, valueB, advW, advB);
    }

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
        writer.Write(_core.tetrises);
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
        _core.tetrises = reader.ReadInt32();
        _core.level = _core.lines / 10; // derived — recomputed rather than serialized
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
