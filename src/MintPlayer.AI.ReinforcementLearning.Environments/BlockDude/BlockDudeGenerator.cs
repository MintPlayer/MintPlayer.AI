using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>One rung of the size curriculum: the board shapes to sample and the puzzle quality to accept.</summary>
/// <param name="MinWidth">Inclusive board width bound.</param>
/// <param name="MaxWidth">Inclusive board width bound.</param>
/// <param name="MinHeight">Inclusive board height bound.</param>
/// <param name="MaxHeight">Inclusive board height bound.</param>
/// <param name="MinBlocks">Inclusive carryable-block count bound.</param>
/// <param name="MaxBlocks">Inclusive carryable-block count bound.</param>
/// <param name="MinOptimal">Reject boards solvable in fewer moves than this — they teach nothing.</param>
/// <param name="MaxOptimal">Reject boards needing more; they blow the oracle's budget.</param>
/// <param name="MaxFreeCells">Reject boards with more non-wall cells than this. Free cells drive the state
/// count harder than board area does (PRD §4.5.1).</param>
/// <param name="OracleMaxStates">Per-board cap handed to <see cref="BlockDudeOracle"/>.</param>
public sealed record BlockDudeStageSpec(
    int MinWidth, int MaxWidth,
    int MinHeight, int MaxHeight,
    int MinBlocks, int MaxBlocks,
    int MinOptimal, int MaxOptimal,
    int MaxFreeCells,
    int OracleMaxStates);

/// <summary>Why a candidate board was accepted or thrown away. Every rejection is counted: the accept rate is a
/// first-class training metric, because a silently-shrinking accept rate biases the label set toward shallow
/// puzzles (PRD §4.4a).</summary>
public enum BlockDudeGenerationOutcome
{
    Accepted,

    /// <summary>The layout could not be built at all (no surface for the door, player and blocks).</summary>
    Malformed,

    /// <summary>More non-wall cells than the stage allows — it would be labelled, but slowly.</summary>
    TooOpen,

    /// <summary>The oracle exceeded its state cap, so the board yields NO labels.</summary>
    Truncated,

    /// <summary>The door is unreachable however the blocks are used.</summary>
    Unsolvable,

    /// <summary>Solvable in fewer moves than the stage wants.</summary>
    TooEasy,

    /// <summary>Solvable, but needing more moves than the stage wants.</summary>
    TooHard,

    /// <summary>Solvable WITHOUT touching a block, so the blocks are scenery and the board teaches nothing
    /// about the actual mechanic.</summary>
    BlocksAreDecorative,

    /// <summary>Coincides with a shipped level, which must stay a clean hold-out.</summary>
    CollidesWithShippedLevel,
}

/// <summary>
/// Generates training boards the exact oracle can afford to label.
/// </summary>
/// <remarks>
/// <para>Terrain is built as a ground profile with deliberate <b>overhangs and shelves</b>. An earlier design
/// generated a pure <c>height[x]</c> skyline on the belief that Block Dude terrain is a 1D height profile —
/// that was measured on the WebGames synthetic levels and is false for the real ones, every single of which
/// has overhangs (18 to 68 each). A skyline generator would train the net on terrain that does not occur in the
/// content it must eventually play (PRD §4.4a).</para>
///
/// <para>Blocks may be placed <b>floating</b>, because the engine never settles gravity on load and the original
/// level 11 ships 14 blocks in mid-air.</para>
///
/// <para>Generation is a pure function of the RNG, so a run replays exactly given the same seed and counter.</para>
/// </remarks>
public static class BlockDudeGenerator
{
    /// <summary>Builds one candidate and judges it. Returns null unless the outcome is
    /// <see cref="BlockDudeGenerationOutcome.Accepted"/>.</summary>
    public static BlockDudeBoard? TryGenerate(Xoshiro256StarStar rng, BlockDudeStageSpec spec, out BlockDudeGenerationOutcome outcome)
    {
        ArgumentNullException.ThrowIfNull(rng);
        ArgumentNullException.ThrowIfNull(spec);

        string[]? grid = TryBuildLayout(rng, spec);
        if (grid is null)
        {
            outcome = BlockDudeGenerationOutcome.Malformed;
            return null;
        }

        int freeCells = grid.Sum(row => row.Count(c => c != 'W'));
        if (freeCells > spec.MaxFreeCells)
        {
            outcome = BlockDudeGenerationOutcome.TooOpen;
            return null;
        }

        if (MatchesAShippedLevel(grid))
        {
            outcome = BlockDudeGenerationOutcome.CollidesWithShippedLevel;
            return null;
        }

        BlockDudeBoard board;
        try
        {
            board = BlockDudeBoard.FromGrid(grid);
        }
        catch (ArgumentException)
        {
            outcome = BlockDudeGenerationOutcome.Malformed;
            return null;
        }

        var oracle = new BlockDudeOracle(board, spec.OracleMaxStates);
        if (oracle.Truncated)
        {
            outcome = BlockDudeGenerationOutcome.Truncated;
            return null;
        }

        int optimal = oracle.OptimalFromStart;
        if (optimal < 0)
        {
            outcome = BlockDudeGenerationOutcome.Unsolvable;
            return null;
        }
        if (optimal < spec.MinOptimal)
        {
            outcome = BlockDudeGenerationOutcome.TooEasy;
            return null;
        }
        if (optimal > spec.MaxOptimal)
        {
            outcome = BlockDudeGenerationOutcome.TooHard;
            return null;
        }

        // Are the blocks load-bearing? Strip them and re-solve: if the door is still reachable, the puzzle is
        // just a walk and teaches nothing about carrying. Cheaper than it looks — a blockless board has a far
        // smaller state graph.
        var blockless = BlockDudeBoard.FromGrid([.. grid.Select(row => row.Replace('B', '.'))]);
        var blocklessOracle = new BlockDudeOracle(blockless, spec.OracleMaxStates);
        if (!blocklessOracle.Truncated && blocklessOracle.OptimalFromStart >= 0)
        {
            outcome = BlockDudeGenerationOutcome.BlocksAreDecorative;
            return null;
        }

        outcome = BlockDudeGenerationOutcome.Accepted;
        return board;
    }

    /// <summary>Draws candidates until one is accepted or <paramref name="maxAttempts"/> is spent, tallying why
    /// the rejects failed.</summary>
    public static BlockDudeBoard? Generate(
        Xoshiro256StarStar rng, BlockDudeStageSpec spec, int maxAttempts, IDictionary<BlockDudeGenerationOutcome, int>? tally = null)
    {
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            var board = TryGenerate(rng, spec, out var outcome);
            if (tally is not null) tally[outcome] = tally.TryGetValue(outcome, out int n) ? n + 1 : 1;
            if (board is not null) return board;
        }
        return null;
    }

    private static string[]? TryBuildLayout(Xoshiro256StarStar rng, BlockDudeStageSpec spec)
    {
        int width = spec.MinWidth + rng.NextInt(spec.MaxWidth - spec.MinWidth + 1);
        int height = spec.MinHeight + rng.NextInt(spec.MaxHeight - spec.MinHeight + 1);
        if (width < 5 || height < 4) return null;

        var grid = new char[height][];
        for (int y = 0; y < height; y++)
        {
            grid[y] = new char[width];
            Array.Fill(grid[y], '.');
        }

        // A closed box. Without it the player can walk off the edge and the puzzle becomes degenerate.
        for (int x = 0; x < width; x++) { grid[0][x] = 'W'; grid[height - 1][x] = 'W'; }
        for (int y = 0; y < height; y++) { grid[y][0] = 'W'; grid[y][width - 1] = 'W'; }

        // Ground profile: a random walk whose occasional 2-step rises are unclimbable without a block, which is
        // what forces the mechanic to appear.
        int lowest = height - 2;
        int highest = Math.Max(2, height - 2 - (height - 4));
        var ground = new int[width];
        ground[1] = lowest;
        for (int x = 2; x < width - 1; x++)
        {
            int step = rng.NextInt(10) switch
            {
                0 or 1 => -1,
                2 => -2,       // an unclimbable rise
                3 or 4 => 1,
                _ => 0,
            };
            ground[x] = Math.Clamp(ground[x - 1] + step, highest, lowest);
        }
        for (int x = 1; x < width - 1; x++)
            for (int y = ground[x]; y < height - 1; y++)
                grid[y][x] = 'W';

        // Shelves — the overhangs a skyline generator cannot produce.
        int shelves = rng.NextInt(4);
        for (int i = 0; i < shelves; i++)
        {
            int length = 2 + rng.NextInt(3);
            int x0 = 1 + rng.NextInt(Math.Max(1, width - 2 - length));
            int surface = ground[Math.Clamp(x0, 1, width - 2)];
            int y = surface - 2 - rng.NextInt(2);
            if (y < 1) continue;
            for (int x = x0; x < Math.Min(x0 + length, width - 1); x++) grid[y][x] = 'W';
        }

        // Standable surfaces: an empty cell with something solid directly beneath.
        var surfaces = new List<(int X, int Y)>();
        for (int y = 1; y < height - 1; y++)
            for (int x = 1; x < width - 1; x++)
                if (grid[y][x] == '.' && grid[y + 1][x] == 'W')
                    surfaces.Add((x, y));
        if (surfaces.Count < 4) return null;

        // Put the door and the player far apart, so the puzzle is a traverse rather than a step.
        var door = surfaces[rng.NextInt(surfaces.Count)];
        var candidates = surfaces.Where(s => Math.Abs(s.X - door.X) >= Math.Max(3, width / 3)).ToList();
        if (candidates.Count == 0) return null;
        var player = candidates[rng.NextInt(candidates.Count)];

        grid[door.Y][door.X] = 'D';
        grid[player.Y][player.X] = 'P';

        int blocks = spec.MinBlocks + rng.NextInt(spec.MaxBlocks - spec.MinBlocks + 1);
        var free = surfaces.Where(s => grid[s.Y][s.X] == '.').ToList();
        for (int i = 0; i < blocks; i++)
        {
            // ~1 in 6 blocks starts floating: the engine never settles gravity on load, and the original level 11
            // ships 14 blocks in mid-air, so the net must see that shape in training.
            bool floating = rng.NextInt(6) == 0;
            if (floating)
            {
                int fx = 1 + rng.NextInt(width - 2);
                int fy = 1 + rng.NextInt(height - 2);
                if (grid[fy][fx] == '.' && grid[fy + 1][fx] == '.') { grid[fy][fx] = 'B'; continue; }
            }
            if (free.Count == 0) break;
            int pick = rng.NextInt(free.Count);
            var cell = free[pick];
            free.RemoveAt(pick);
            if (grid[cell.Y][cell.X] == '.') grid[cell.Y][cell.X] = 'B';
        }

        return [.. grid.Select(row => new string(row))];
    }

    /// <summary>The shipped levels are a held-out gate set; a generated board must never coincide with one.</summary>
    private static bool MatchesAShippedLevel(string[] grid)
    {
        foreach (var level in BlockDudeLevels.All)
        {
            if (level.Grid.Length != grid.Length) continue;
            if (level.Grid.SequenceEqual(grid)) return true;
        }
        return false;
    }
}
