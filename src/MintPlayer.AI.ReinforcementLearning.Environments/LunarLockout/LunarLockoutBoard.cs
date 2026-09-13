namespace MintPlayer.AI.ReinforcementLearning.Environments.LunarLockout;

// PUBLIC FACADE over the single-source transpiled solver (polyglot/lunarlockout_solver.pg → PgLunarBoard /
// PgLunarOracle). The rules and the exact oracle live ONCE in the .pg, shared with the browser's TypeScript twin;
// this facade adapts the generated internal, camelCase core to the PascalCase API the env, campaign and tests
// consume. See docs/prd/WEBGAMES_RETIREMENT_PRD.md §5.
//
// Never edit the generated lunarlockout_solver.cs / .ts — change the .pg and rebuild.

/// <summary>A slide direction. The integer values are part of the action encoding and must not be reordered.</summary>
public enum LunarDirection
{
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

/// <summary>
/// A Lunar Lockout position: robots on a 5x5 grid, <see cref="TargetCell"/> being the red robot that must reach
/// the centre. Immutable — <see cref="Apply"/> returns a new board.
/// </summary>
/// <remarks>
/// Rules (verified against ThinkFun's published instructions): a robot slides only if another robot lies
/// somewhere in that direction, and travels until it rests against it. The board edge is NOT a backstop, so a
/// slide with nothing ahead is illegal and robots can never leave the grid. Helpers are interchangeable, so
/// <see cref="Key"/> sorts them to collapse symmetric positions.
/// </remarks>
public sealed class LunarLockoutBoard
{
    public const int Size = 5;
    public const int Cells = 25;
    public const int CentreCell = 12;

    internal readonly PgLunarBoard Inner;

    internal LunarLockoutBoard(PgLunarBoard inner) => Inner = inner;

    /// <summary>Parses five rows, top row first: <c>R</c> the target robot, <c>.</c> empty, any other letter a helper.</summary>
    /// <exception cref="ArgumentException">The grid is not 5x5, or does not hold exactly one target robot.</exception>
    public static LunarLockoutBoard FromGrid(string[] rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (rows.Length != Size) throw new ArgumentException($"Expected {Size} rows, got {rows.Length}.", nameof(rows));
        for (int r = 0; r < rows.Length; r++)
            if (rows[r].Length != Size)
                throw new ArgumentException($"Row {r} has {rows[r].Length} cells, expected {Size}.", nameof(rows));

        int targets = rows.Sum(row => row.Count(c => c == 'R'));
        if (targets != 1) throw new ArgumentException($"Expected exactly one 'R' target robot, found {targets}.", nameof(rows));

        return new LunarLockoutBoard(PgLunarBoard.fromGrid(new List<string>(rows)));
    }

    /// <summary>Robot cell indices (<c>row * 5 + col</c>); index 0 is the target robot.</summary>
    public IReadOnlyList<int> Robots => Inner.robots;

    public int RobotCount => Inner.robots.Count;

    public int TargetCell => Inner.robots[0];

    public bool IsSolved => Inner.isSolved();

    /// <summary>Total action-space size for this board: <c>RobotCount * 4</c>. Actions are <c>robot * 4 + direction</c>.</summary>
    public int ActionCount => PgLunarBoard.actionCount(Inner.robots.Count);

    public static int EncodeAction(int robot, LunarDirection direction) => robot * 4 + (int)direction;

    public static (int Robot, LunarDirection Direction) DecodeAction(int action) => (action / 4, (LunarDirection)(action % 4));

    /// <summary>Landing cell for the slide, or -1 when the move is illegal (no blocking robot ahead, or already resting against one).</summary>
    public int Landing(int robot, LunarDirection direction) => Inner.landing(robot, (int)direction);

    public bool IsLegal(int action) => Inner.isLegal(action);

    /// <summary>The board after taking <paramref name="action"/>.</summary>
    /// <exception cref="ArgumentException">The action is illegal on this board.</exception>
    public LunarLockoutBoard Apply(int action)
    {
        if (!IsLegal(action))
        {
            var (robot, direction) = DecodeAction(action);
            throw new ArgumentException($"Robot {robot} cannot slide {direction} from this position.", nameof(action));
        }
        return new LunarLockoutBoard(Inner.applyAction(action));
    }

    public IEnumerable<int> LegalActions()
    {
        int actions = ActionCount;
        for (int a = 0; a < actions; a++)
            if (IsLegal(a)) yield return a;
    }

    /// <summary>Canonical packed state key — target in the low 5 bits, then sorted helper cells, 5 bits each.</summary>
    public int Key => Inner.key();

    /// <summary>Renders the board back to the five-row grid format, using <c>R</c> and then <c>A</c>, <c>B</c>, … for helpers.</summary>
    public string[] ToGrid()
    {
        var rows = new char[Size][];
        for (int r = 0; r < Size; r++)
        {
            rows[r] = new char[Size];
            Array.Fill(rows[r], '.');
        }
        rows[TargetCell / Size][TargetCell % Size] = 'R';
        for (int i = 1; i < Inner.robots.Count; i++)
        {
            int cell = Inner.robots[i];
            rows[cell / Size][cell % Size] = (char)('A' + i - 1);
        }
        return [.. rows.Select(r => new string(r))];
    }
}

/// <summary>
/// Exact supervision for imitation learning: labels EVERY state reachable from a start position with its true
/// distance-to-goal and the full set of optimal actions.
/// </summary>
/// <remarks>
/// Two passes, because a Lunar Lockout slide has no general inverse — unlike a Rush Hour slide, whose
/// reversibility lets <c>RushHourOracle</c> treat the graph as undirected and go straight to a backward BFS.
/// Pass 1 is a forward BFS recording every edge explicitly; pass 2 reverses those edges in memory and runs a
/// multi-source BFS from every solved state.
///
/// <para><see cref="Truncated"/> means the graph exceeded <c>maxStates</c> and NOTHING was labelled. Callers must
/// log that as a rejection — never treat it as "this board has no labels" (PRD §4.5).</para>
/// </remarks>
public sealed class LunarLockoutOracle
{
    /// <summary>Matches <c>RushHourImitationCampaign.MaxStatesPerConfig</c>; a 5x5 board never comes close.</summary>
    public const int DefaultMaxStates = 150_000;

    private readonly PgLunarOracle _inner;

    public LunarLockoutOracle(LunarLockoutBoard start, int maxStates = DefaultMaxStates)
    {
        ArgumentNullException.ThrowIfNull(start);
        if (maxStates <= 0) throw new ArgumentOutOfRangeException(nameof(maxStates));
        _inner = new PgLunarOracle(start.Inner, maxStates);
    }

    /// <summary>The state graph exceeded the cap; no labels were produced. Log this as a rejection.</summary>
    public bool Truncated => _inner.truncated;

    /// <summary>Number of reachable states labelled (0 when <see cref="Truncated"/>).</summary>
    public int StateCount => _inner.stateCount();

    /// <summary>Optimal move count from the start position, or -1 when unsolvable or truncated.</summary>
    public int OptimalFromStart => _inner.optimalFromStart();

    /// <summary>Distance to goal for an arbitrary reachable board, or -1 when unreachable or unknown.</summary>
    public int DistanceOf(LunarLockoutBoard board) => _inner.distanceOf(board.Inner);

    /// <summary>Bitmask of every action that steps exactly one closer to the goal. Supervise against the FULL set:
    /// cross-entropy against one arbitrary representative penalises the other equally-optimal actions and flattens
    /// the policy.</summary>
    public int OptimalActionsOf(LunarLockoutBoard board) => _inner.optimalActionsOf(board.Inner);
}
