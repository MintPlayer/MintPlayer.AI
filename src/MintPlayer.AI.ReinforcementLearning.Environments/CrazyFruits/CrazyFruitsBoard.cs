namespace MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

// PUBLIC FACADE over the single-source transpiled engine (polyglot/crazyfruits_solver.pg → PgCrazyFruits).
// The rules live ONCE in the .pg (shared with the web client's TypeScript); this facade adapts the generated
// internal, camelCase core to the public API the env/Lab/tests consume, plus host-only helpers (state
// serialization, LoadGrid for directed tests) that aren't part of the shared engine.

/// <summary>
/// The Crazy Fruits match-3 board: 8×8 grid, 6 fruit types, 112 adjacent-swap actions (56 horizontal then
/// 56 vertical). Only match-producing swaps are legal; matches clear with proportional cascade scoring
/// (10·(k+1) per fruit at cascade step k, +20/+50 line bonuses), fruit falls and refills from a
/// deterministic minstd stream, and a dead board reshuffles in place — there is always a legal swap.
/// </summary>
public sealed class CrazyFruitsBoard
{
    public const int Size = 8;
    public const int FruitTypes = 6;
    public const int Cells = Size * Size;
    public const int ActionCount = 2 * Size * (Size - 1); // 112
    public const int SpecialKinds = 4; // stripedH, stripedV, wrapped, bomb (armed is internal-only)
    // 1040: 6 fruit planes + 4 kind planes + would-act plane + 3 per-action feature planes (immediate score,
    // deterministic cascade value, creation-shaped deterministic value — ÷300; SPECIALS PRD §3.5 + RANKING
    // PRD lever B).
    public const int ObservationSize = (FruitTypes + SpecialKinds + 1) * Cells + 3 * ActionCount;

    private readonly PgCrazyFruits _core = new();

    /// <summary>Deal a fresh board (no pre-existing matches, ≥1 legal swap) from the given seed.</summary>
    public void Reset(ulong seed) => _core.reset(SeedToInt(seed));

    /// <summary>Map any 64-bit seed onto the engine's minstd state range [1, 2^31−2].</summary>
    internal static int SeedToInt(ulong seed) => (int)(seed % 2147483646UL) + 1;

    public int Score => _core.score;
    public int MovesMade => _core.movesMade;
    public int LastMoveScore => _core.lastMoveScore;
    public int LastCascadeSteps => _core.lastCascadeSteps;
    public int Reshuffles => _core.reshuffles;

    /// <summary>Fruit type (1..6; 0 for the colorless bomb) at row-major <paramref name="cell"/>.</summary>
    public int Fruit(int cell) => PolyglotProgram.cfFruitOf(_core.grid[cell]);

    /// <summary>Special kind at <paramref name="cell"/>: 0 none · 1 stripedH · 2 stripedV · 3 wrapped · 4 bomb.</summary>
    public int Kind(int cell) => PolyglotProgram.cfKindOf(_core.grid[cell]);

    /// <summary>Raw packed value (kind·16 + type) at <paramref name="cell"/> — for tests/serialization.</summary>
    public int Packed(int cell) => _core.grid[cell];

    // Per-move specials telemetry (valid after ApplySwap; the specials-usage gates + training shaping read these).
    public int MoveCreatedStriped => _core.moveCreatedStriped;
    public int MoveCreatedWrapped => _core.moveCreatedWrapped;
    public int MoveCreatedBombs => _core.moveCreatedBombs;
    public int MoveSpecialsFired => _core.moveSpecialsFired;

    /// <summary>Row-major snapshot of the grid (PACKED values — kind·16 + type).</summary>
    public int[] GridSnapshot()
    {
        var grid = new int[Cells];
        for (int i = 0; i < Cells; i++) grid[i] = _core.grid[i];
        return grid;
    }

    /// <summary>The two row-major cells a swap action exchanges (A is left/top).</summary>
    public (int A, int B) SwapCells(int action) => (_core.cellA(action), _core.cellB(action));

    /// <summary>Legality of each of the <see cref="ActionCount"/> swaps (true = produces a match).</summary>
    public bool[] LegalMask()
    {
        var core = _core.legalMask();
        var mask = new bool[ActionCount];
        for (int i = 0; i < ActionCount; i++) mask[i] = core[i];
        return mask;
    }

    public bool IsLegal(int action) => _core.swapIsLegal(action);
    public bool HasLegalSwap() => _core.hasLegalSwap();
    public bool AnyMatchOnBoard() => _core.anyMatchOnBoard();

    /// <summary>Apply a legal swap (resolve cascades, refill, reshuffle-if-dead); returns the points gained.
    /// An illegal action returns −1 and changes nothing.</summary>
    public int ApplySwap(int action) => _core.applySwap(action);

    /// <summary>Re-deal the current fruit multiset until no instant match and ≥1 legal swap (the deadlock rule).</summary>
    public void Reshuffle() => _core.reshuffleBoard();

    /// <summary>The <see cref="ObservationSize"/>-dim observation: fruit/kind/would-match planes + the three
    /// per-action value planes (f64 core → float32 net).</summary>
    public float[] BuildObservation()
    {
        var core = _core.buildObservation();
        var obs = new float[core.Count];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)core[i];
        return obs;
    }

    // ── Scripted baselines (single-sourced in the .pg; tiers + eval bars + env sanity gates) ────────────────

    /// <summary>Uniform-random legal swap, drawn from a caller-owned policy stream (never the refill RNG).</summary>
    public int RandomAction(ulong policySeed, int step)
    {
        // One throwaway stream per decision keeps the call stateless for the caller; the (seed, step) pair
        // makes each move's draw independent and reproducible.
        var rng = new PgCfRng(SeedToInt(policySeed ^ (ulong)(step * 0x9E3779B9L)));
        return _core.randomAction(rng);
    }

    /// <summary>Legal swap with the highest immediate (step-0) points; lowest index wins ties.</summary>
    public int GreedyAction() => _core.greedyAction();

    /// <summary>Immediate step-0 points of a swap (0 if illegal) — the greedy tier's scoring.</summary>
    public int ImmediateScore(int action) => _core.immediateScore(action);

    /// <summary>Legal swap maximizing the deterministic cascade value (gravity-only, refill left empty).</summary>
    public int ExpectimaxAction() => _core.expectimaxAction();

    /// <summary>2-ply deterministic tier (beam-8): the baseline that can plan create→fire (SPECIALS PRD §3.7).</summary>
    public int Expectimax2Action() => _core.expectimax2Action();

    /// <summary>Greedy over the creation-shaped immediate score — prefers building specials.</summary>
    public int SpecialsGreedyAction() => _core.specialsGreedyAction();

    /// <summary>Immediate step-0 points plus the creation-shaping weights (the specials-greedy signal).</summary>
    public int ImmediateScoreShaped(int action) => _core.immediateScoreShaped(action);

    /// <summary>Deterministic cascade value of a swap (consumes no RNG; restores the board).</summary>
    public int DeterministicValue(int action) => _core.deterministicValue(action);

    /// <summary>Deterministic cascade value plus the creation-shaping weights over the whole refill-free
    /// cascade — the shaped observation plane and the dense-regression target (RANKING PRD lever B).</summary>
    public int DeterministicValueShaped(int action) => _core.deterministicValueShaped(action);

    // ── Host-only helpers ────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Overwrite the grid for directed tests (row-major, values 1..6). Score/moves are untouched.</summary>
    public void LoadGrid(ReadOnlySpan<int> grid)
    {
        if (grid.Length != Cells) throw new ArgumentException($"expected {Cells} cells", nameof(grid));
        for (int i = 0; i < Cells; i++) _core.grid[i] = grid[i];
    }

    /// <summary>Serialize the full engine state (grid, RNG, score, counters) for env Save.</summary>
    public void WriteState(BinaryWriter writer)
    {
        writer.Write(_core.rng.state);
        writer.Write(_core.score);
        writer.Write(_core.movesMade);
        writer.Write(_core.lastMoveScore);
        writer.Write(_core.lastCascadeSteps);
        writer.Write(_core.reshuffles);
        for (int i = 0; i < Cells; i++) writer.Write(_core.grid[i]);
    }

    /// <summary>Restore the state written by <see cref="WriteState"/>.</summary>
    public void ReadState(BinaryReader reader)
    {
        _core.rng = new PgCfRng(1) { state = reader.ReadInt32() };
        _core.score = reader.ReadInt32();
        _core.movesMade = reader.ReadInt32();
        _core.lastMoveScore = reader.ReadInt32();
        _core.lastCascadeSteps = reader.ReadInt32();
        _core.reshuffles = reader.ReadInt32();
        _core.grid.Clear();
        for (int i = 0; i < Cells; i++) _core.grid.Add(reader.ReadInt32());
    }
}
