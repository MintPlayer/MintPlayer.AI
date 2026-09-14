using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>One state from a human demonstration, with everything a supervised update needs.</summary>
/// <param name="Board">The position.</param>
/// <param name="Action">The move the human played here.</param>
/// <param name="Remaining">Moves from here to the door ALONG THIS PATH. An upper bound on the true optimal
/// distance, never the optimum — a human played it.</param>
/// <param name="Level">Which shipped level this came from.</param>
public readonly record struct BlockDudeDemoState(BlockDudeBoard Board, BlockDudeAction Action, int Remaining, string Level);

/// <summary>
/// Turns the 15 recorded human solutions into training signal.
/// </summary>
/// <remarks>
/// <para><b>Why 3,870 states matter more than the number suggests.</b> They are the ONLY data that exists in the
/// distribution the net is graded on. The generator cannot express 53% of shipped topologies and tops out around
/// 25 optimal moves; the exact oracle cannot label boards this size at all. So every other source of training
/// data is out-of-distribution by construction, and these are not. Measured against that, volume is the wrong
/// axis to judge them on.</para>
///
/// <para><b>Two distinct uses, and the second is the larger one.</b></para>
///
/// <para><see cref="LabelledStates"/> is the direct one: each state on a demonstrated path comes with the action
/// the human played and the true number of moves remaining along that path. That second label lands in the
/// 1-to-909 range, where the value head was measured under-estimating by ~37 moves and has no training data
/// whatsoever (PRD §8.4a). It is free — no search, no oracle.</para>
///
/// <para><see cref="ReverseCurriculumStarts"/> is the one that multiplies. A state 890 moves into Level 11 is a
/// legitimate position on a real shipped board that happens to be 19 moves from the door — something A* solves
/// instantly and the net can learn from, on terrain the generator could never produce. Walk the start backwards
/// along the demonstration and the same level yields an unbroken ladder of tasks from trivial to complete. This
/// is the standard way demonstrations are used to crack long-horizon problems, and it is what converts fifteen
/// trajectories into thousands of distinct training boards.</para>
///
/// <para><b>These are upper bounds, not optima.</b> `Remaining` is the human's move count, so a net that beats it
/// is doing better, not cheating. Never treat it as a target to match exactly, and never report a ratio against
/// it as "× optimal".</para>
/// </remarks>
public static class BlockDudeDemonstrations
{
    /// <summary>
    /// Every state on every demonstrated path, with the human's action and the moves remaining from there.
    /// The terminal (won) state is excluded — there is no action to learn at it.
    /// </summary>
    public static IEnumerable<BlockDudeDemoState> LabelledStates()
    {
        foreach (var solution in BlockDudeSolutions.All)
        {
            var level = Array.Find(BlockDudeLevels.All, l => l.Name == solution.Name);
            if (level is null) continue;

            var board = BlockDudeBoard.FromGrid(level.Grid);
            for (int i = 0; i < solution.Moves.Length; i++)
            {
                yield return new(board, solution.Moves[i], solution.Moves.Length - i, solution.Name);
                board = board.Apply(solution.Moves[i]);
            }
        }
    }

    /// <summary>
    /// Positions from a demonstration that are <paramref name="movesFromEnd"/> moves from the door along the
    /// human's path — the start states for a reverse curriculum.
    /// </summary>
    /// <param name="movesFromEnd">How much of the problem to leave. 1 is a single move from winning; the full
    /// length is the level itself.</param>
    /// <param name="levelName">Restrict to one level, or null for all of them.</param>
    /// <remarks>
    /// Ramp this up as the net succeeds. The point is that the net is never asked to solve more of the level than
    /// it has just shown it can, while every task it sees is on REAL shipped terrain rather than a generated
    /// approximation of it.
    /// </remarks>
    public static IEnumerable<BlockDudeBoard> ReverseCurriculumStarts(int movesFromEnd, string? levelName = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(movesFromEnd);

        foreach (var solution in BlockDudeSolutions.All)
        {
            if (levelName is not null && solution.Name != levelName) continue;

            var level = Array.Find(BlockDudeLevels.All, l => l.Name == solution.Name);
            if (level is null) continue;

            // A level shorter than the requested suffix contributes its start position: the whole level IS the
            // task. Skipping it instead would silently drop the easy levels once the ramp passes their length.
            int prefix = Math.Max(0, solution.Moves.Length - movesFromEnd);

            var board = BlockDudeBoard.FromGrid(level.Grid);
            for (int i = 0; i < prefix; i++) board = board.Apply(solution.Moves[i]);

            yield return board;
        }
    }

    /// <summary>
    /// Every state on one level's demonstrated path, keyed by engine state hash, valued by the moves remaining
    /// from it — the <b>landmarks</b> a search can aim at instead of the door.
    /// </summary>
    /// <remarks>
    /// <para>A search that fails finds nothing and teaches nothing, and on the long levels that is most of them:
    /// asking A* for a 400-move suffix inside an 8-second budget mostly fails, so the levels with the most left
    /// to learn are the ones producing the least data.</para>
    ///
    /// <para>But the door is not the only place worth reaching. Any state on the human's path is a position from
    /// which a winning continuation is already KNOWN, so a search that gets back onto that path further along
    /// has found a genuine solution — its own prefix, then the demonstrated remainder. The part it found is new
    /// data on states the net actually visits, which is precisely the distribution a demonstration alone cannot
    /// supply.</para>
    ///
    /// <para><b>Hashes are 32-bit, so a hit is a candidate and not a proof.</b> Across a 200,000-node search a
    /// collision is a few percent likely, and the cost of believing one is a training sample asserting a win
    /// that does not exist. Callers must confirm a hit by replaying the remainder — see
    /// <see cref="ContinuationFrom"/>.</para>
    /// </remarks>
    public static IReadOnlyDictionary<int, int> PathLandmarks(string levelName)
    {
        var landmarks = new Dictionary<int, int>();
        var solution = BlockDudeSolutions.For(levelName);
        var level = solution is null ? null : Array.Find(BlockDudeLevels.All, l => l.Name == levelName);
        if (solution is null || level is null) return landmarks;

        var board = BlockDudeBoard.FromGrid(level.Grid);
        for (int i = 0; i < solution.Moves.Length; i++)
        {
            // A state visited twice on the path keeps its SMALLEST remaining: reaching it is worth the best
            // continuation known from it, not the first one the human happened to take.
            int remaining = solution.Moves.Length - i;
            if (!landmarks.TryGetValue(board.StateHash, out int known) || remaining < known)
                landmarks[board.StateHash] = remaining;
            board = board.Apply(solution.Moves[i]);
        }
        landmarks[board.StateHash] = 0;   // the won state
        return landmarks;
    }

    /// <summary>
    /// The demonstrated moves that finish the level from <paramref name="board"/>, or null if it is not really on
    /// the path — the confirmation step that makes a 32-bit landmark hit safe to train on.
    /// </summary>
    /// <param name="remaining">The remaining count the landmark table claimed, i.e. how far along to splice in.</param>
    public static IReadOnlyList<BlockDudeAction>? ContinuationFrom(BlockDudeBoard board, string levelName, int remaining)
    {
        var solution = BlockDudeSolutions.For(levelName);
        if (solution is null || remaining < 0 || remaining > solution.Moves.Length) return null;

        var tail = solution.Moves[^remaining..];

        // Replayed, not trusted. If the hash collided, this walks somewhere else and simply does not win.
        var replayed = board;
        foreach (var move in tail) replayed = replayed.Apply(move);
        return replayed.Won ? tail : null;
    }

    /// <summary>The human's move count per level — an upper bound to measure a policy against, and the only
    /// reference point that exists for boards no exact solver can reach.</summary>
    public static IReadOnlyDictionary<string, int> HumanMoveCounts()
        => BlockDudeSolutions.All.ToDictionary(s => s.Name, s => s.Moves.Length);
}
