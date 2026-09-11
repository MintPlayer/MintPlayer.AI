namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

// PUBLIC FACADE over the single-source transpiled engine (polyglot/blockdude_solver.pg → PgBlockDudeBoard /
// PgBlockDudeOracle). The rules live ONCE in the .pg, shared with the browser's TypeScript twin; this facade
// adapts the generated internal, camelCase core to the PascalCase API the env, campaign and tests consume.
// See docs/prd/WEBGAMES_RETIREMENT_PRD.md §4.
//
// Never edit the generated blockdude_solver.cs / .ts — change the .pg and rebuild.

/// <summary>Immutable terrain under the mobile pieces.</summary>
public enum BlockDudeTile
{
    /// <summary>Open space.</summary>
    Empty = 0,

    /// <summary>Stone: solid, climbable, and never carryable.</summary>
    Wall = 1,

    /// <summary>The exit. The player may stand in it — that is how a level is won — and it supports a falling
    /// block, but the player falls straight through it.</summary>
    Door = 2,
}

/// <summary>The four moves. Integer values are the action encoding and must not be reordered.</summary>
public enum BlockDudeAction
{
    /// <summary>Face left, then step left if possible.</summary>
    Left = 0,

    /// <summary>Face right, then step right if possible.</summary>
    Right = 1,

    /// <summary>Climb onto the solid cell you are facing.</summary>
    Climb = 2,

    /// <summary>Pick up the block you are facing, or put down the one you are carrying.</summary>
    Grab = 3,
}

/// <summary>
/// A Block Dude position: immutable terrain, a set of carryable blocks, and the player's pose. Immutable —
/// <see cref="Apply"/> returns a new board.
/// </summary>
/// <remarks>
/// <para>Coordinates are row-major with row 0 at the TOP and y increasing downward, so gravity is +y. The
/// original WinForms implementation used 1-based Y-up; the rules are identical, only the axis is flipped, which
/// keeps the engine aligned with the top-first ASCII level assets.</para>
///
/// <para>One entity per cell, with exactly one exception: the player may stand in the door cell. Gravity applies
/// only to the walking player and to a dropped block — never as a global settle pass, which is what lets level
/// 11 ship 14 blocks floating in mid-air.</para>
/// </remarks>
public sealed class BlockDudeBoard
{
    internal readonly PgBlockDudeBoard Inner;

    internal BlockDudeBoard(PgBlockDudeBoard inner) => Inner = inner;

    /// <summary>Total action-space size. Constant: the four moves are always encodable, legal or not.</summary>
    public const int ActionCount = 4;

    /// <summary>Parses an ASCII grid, rows TOP-first: <c>W</c> wall, <c>B</c> block, <c>P</c> player start,
    /// <c>D</c> door, <c>.</c> empty.</summary>
    /// <exception cref="ArgumentException">The grid is empty, ragged, or lacks exactly one player start and one door.</exception>
    public static BlockDudeBoard FromGrid(string[] rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Length == 0) throw new ArgumentException("The grid has no rows.", nameof(rows));

        int width = rows[0].Length;
        if (width == 0) throw new ArgumentException("The grid has no columns.", nameof(rows));
        for (int y = 0; y < rows.Length; y++)
            if (rows[y].Length != width)
                throw new ArgumentException($"Row {y} has {rows[y].Length} cells, expected {width}.", nameof(rows));

        int starts = rows.Sum(r => r.Count(c => c == 'P'));
        if (starts != 1) throw new ArgumentException($"Expected exactly one 'P' player start, found {starts}.", nameof(rows));
        int doors = rows.Sum(r => r.Count(c => c == 'D'));
        if (doors != 1) throw new ArgumentException($"Expected exactly one 'D' door, found {doors}.", nameof(rows));

        return new BlockDudeBoard(PgBlockDudeBoard.fromGrid(new List<string>(rows)));
    }

    public int Width => Inner.width;

    public int Height => Inner.height;

    public int PlayerX => Inner.px;

    public int PlayerY => Inner.py;

    /// <summary>Which way the player faces. The original spawns facing left.</summary>
    public bool FacingRight => Inner.facingRight;

    public bool Carrying => Inner.carrying;

    /// <summary>The player has reached the door.</summary>
    public bool Won => Inner.won;

    /// <summary>Cells holding a carryable block, ascending. Excludes a block currently being carried.</summary>
    public IReadOnlyList<int> BlockCells => Inner.blocks;

    public BlockDudeTile TileAt(int x, int y) => (BlockDudeTile)Inner.tileAt(x, y);

    public bool HasBlock(int x, int y) => Inner.hasBlock(x, y);

    public bool InBounds(int x, int y) => Inner.inBounds(x, y);

    /// <summary>Whether the move would change anything. Turning to face a new direction counts as a change.</summary>
    public bool IsLegal(BlockDudeAction action) => Inner.isLegal((int)action);

    /// <summary>The board after the move, or an equal board when the move is refused.</summary>
    public BlockDudeBoard Apply(BlockDudeAction action) => new(Inner.applyAction((int)action));

    public IEnumerable<BlockDudeAction> LegalActions()
    {
        for (int a = 0; a < ActionCount; a++)
            if (Inner.isLegal(a)) yield return (BlockDudeAction)a;
    }

    /// <summary>32-bit hash of the MOBILE state only — player pose plus the block multiset. Terrain is immutable
    /// and excluded. A hash is not an identity: use <see cref="SameStateAs"/> to compare.</summary>
    public int StateHash => Inner.stateHash();

    /// <summary>Exact state equality: same player pose and same block multiset.</summary>
    public bool SameStateAs(BlockDudeBoard other) => Inner.sameState(other.Inner);

    /// <summary>Renders the board back to the ASCII grid format.</summary>
    public string[] ToGrid()
    {
        var rows = new char[Height][];
        for (int y = 0; y < Height; y++)
        {
            rows[y] = new char[Width];
            for (int x = 0; x < Width; x++)
            {
                rows[y][x] = TileAt(x, y) switch
                {
                    BlockDudeTile.Wall => 'W',
                    BlockDudeTile.Door => 'D',
                    _ => HasBlock(x, y) ? 'B' : '.',
                };
            }
        }
        rows[PlayerY][PlayerX] = 'P';
        return [.. rows.Select(r => new string(r))];
    }
}

/// <summary>
/// Exact supervision for imitation learning: labels every state reachable from a start position with its true
/// distance-to-goal and the full set of optimal actions.
/// </summary>
/// <remarks>
/// <para>Two passes, because gravity is irreversible and the state graph cannot be walked backwards: a forward
/// BFS recording every edge, then a multi-source backward BFS from the won states.</para>
///
/// <para><b>This is viable only for SMALL boards.</b> The shipped level 11 is 42 blocks on 551 cells; nothing
/// exact will ever label it. Training data comes from generated small boards, and <see cref="Truncated"/> must be
/// logged as a rejection so the bias in the accepted set stays visible (PRD §4.5).</para>
/// </remarks>
public sealed class BlockDudeOracle
{
    /// <summary>Matches <c>RushHourImitationCampaign.MaxStatesPerConfig</c>.</summary>
    public const int DefaultMaxStates = 150_000;

    private readonly PgBlockDudeOracle _inner;

    public BlockDudeOracle(BlockDudeBoard start, int maxStates = DefaultMaxStates)
    {
        ArgumentNullException.ThrowIfNull(start);
        if (maxStates <= 0) throw new ArgumentOutOfRangeException(nameof(maxStates));
        _inner = new PgBlockDudeOracle(start.Inner, maxStates);
    }

    /// <summary>The state graph exceeded the cap and NO labels were produced. Log this as a rejection — never
    /// treat it as "this board has no labels".</summary>
    public bool Truncated => _inner.truncated;

    /// <summary>Number of reachable states labelled (0 when <see cref="Truncated"/>).</summary>
    public int StateCount => _inner.stateCount();

    /// <summary>Optimal move count from the start position, or -1 when unsolvable or truncated.</summary>
    public int OptimalFromStart => _inner.optimalFromStart();

    /// <summary>Labelled states in BFS discovery order; index 0 is the start position.</summary>
    public IEnumerable<(BlockDudeBoard Board, int Distance, int OptimalActions)> LabelledStates()
    {
        for (int id = 0; id < _inner.states.Count; id++)
            yield return (new BlockDudeBoard(_inner.states[id]), _inner.distanceOfId(id), _inner.optimalActionsOfId(id));
    }
}
