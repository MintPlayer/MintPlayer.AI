using MintPlayer.AI.ReinforcementLearning.Environments.LevelPacks;

namespace MintPlayer.AI.ReinforcementLearning.Environments.LunarLockout;

/// <summary>
/// The shipped Lunar Lockout ladder, read from the embedded <c>levels/lunarlockout-levels.json</c> — the single
/// source of truth shared with the Angular app, which fetches the same file from <c>wwwroot/levels</c>.
/// </summary>
/// <remarks>
/// <para>Grid format, rows TOP-first: <c>R</c> the target robot, <c>.</c> empty, any other letter a helper.</para>
///
/// <para>Every <c>OptimalMoves</c> is an exhaustive-BFS result produced by
/// <c>tools/lunarlockout_levels.py</c>, and <c>LunarLockoutOracleTests</c> re-derives all of them through the
/// <c>.pg</c> oracle — two independent implementations, which is how the unusable level pack inherited from the
/// WebGames port was caught (13 of its 15 grids had no legal first move at all).</para>
///
/// <para>Held out of training and used as the gate set (PRD D4).</para>
/// </remarks>
public static class LunarLockoutLevels
{
    private static readonly Lazy<LevelPackEntry[]> Pack =
        new(() => EmbeddedLevelPack.Load("lunarlockout-levels.json"));

    /// <summary>The shipped ladder, easiest first.</summary>
    public static LevelPackEntry[] All => Pack.Value;

    /// <summary>Parses a shipped level into a playable board.</summary>
    public static LunarLockoutBoard Load(int index) => LunarLockoutBoard.FromGrid(All[index].Grid);

    /// <summary>Parses every shipped level into a playable board.</summary>
    public static IEnumerable<LunarLockoutBoard> LoadAll() => All.Select(l => LunarLockoutBoard.FromGrid(l.Grid));
}
