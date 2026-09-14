using System.Reflection;
using System.Text.Json;

namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>One hand-played solution: the level it solves, and the moves that solve it.</summary>
/// <param name="Name">The level's name in the shipped pack.</param>
/// <param name="Moves">Actions in order. Not optimal — a human played them.</param>
public sealed record BlockDudeSolution(string Name, BlockDudeAction[] Moves);

/// <summary>
/// Human solutions to all 15 shipped levels, read from the embedded <c>levels/blockdude-solutions.json</c>.
/// </summary>
/// <remarks>
/// <para><b>Why these exist.</b> The exact BFS oracle cannot label boards this size — that is precisely why the
/// training curriculum stops well short of them — so a hand-played solution is the only ground truth for the
/// real game content that can exist. They were recorded in the browser by the repo owner on 2026-09-13.</para>
///
/// <para><b>They are not optimal and must never be treated as optimal.</b> Level 11 takes 909 moves here; no one
/// knows the true minimum and proving it is off-scale (PRD §8.5). Anything comparing a policy against these
/// should say "human" rather than "optimal".</para>
///
/// <para>Three uses, in the order they became true: an engine regression suite (3,870 real moves that must stay
/// legal and still reach the door — far harder to satisfy by accident than any hand-written test), training data
/// in the 1-to-900 remaining-distance range where the value head has none, and ground truth for the
/// target-configuration model in <c>BLOCKDUDE_REBUILD_PRD.md</c>.</para>
/// </remarks>
public static class BlockDudeSolutions
{
    private sealed record Document(int Version, Entry[] Solutions);

    private sealed record Entry(string Name, string Moves);

    private static readonly Lazy<BlockDudeSolution[]> Cache = new(Load);

    /// <summary>Every recorded solution, in pack order.</summary>
    public static BlockDudeSolution[] All => Cache.Value;

    /// <summary>The solution for a level, or null when that level has none recorded.</summary>
    public static BlockDudeSolution? For(string levelName)
        => Array.Find(All, s => s.Name == levelName);

    private static BlockDudeSolution[] Load()
    {
        var assembly = Assembly.GetExecutingAssembly();
        string resource = Array.Find(assembly.GetManifestResourceNames(),
                              n => n.EndsWith("blockdude-solutions.json", StringComparison.Ordinal))
                          ?? throw new InvalidOperationException(
                              "Embedded 'blockdude-solutions.json' is missing. Known resources: " +
                              string.Join(", ", assembly.GetManifestResourceNames()));

        using var stream = assembly.GetManifestResourceStream(resource)!;
        var document = JsonSerializer.Deserialize<Document>(stream, new JsonSerializerOptions(JsonSerializerDefaults.Web))
                       ?? throw new InvalidOperationException("blockdude-solutions.json did not parse.");

        return [.. document.Solutions.Select(s => new BlockDudeSolution(s.Name, Parse(s.Name, s.Moves)))];
    }

    /// <summary>One digit per move, matching the engine's action encoding. A stray character is a corrupt asset,
    /// not a move to guess at, so it throws rather than silently dropping part of a solution.</summary>
    private static BlockDudeAction[] Parse(string name, string moves)
    {
        var actions = new BlockDudeAction[moves.Length];
        for (int i = 0; i < moves.Length; i++)
        {
            int digit = moves[i] - '0';
            if (digit < 0 || digit >= BlockDudeBoard.ActionCount)
                throw new InvalidOperationException(
                    $"'{name}' has '{moves[i]}' at index {i}; expected a digit 0..{BlockDudeBoard.ActionCount - 1}.");
            actions[i] = (BlockDudeAction)digit;
        }
        return actions;
    }
}
