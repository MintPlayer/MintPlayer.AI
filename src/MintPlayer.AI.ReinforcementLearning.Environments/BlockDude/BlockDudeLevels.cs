using MintPlayer.AI.ReinforcementLearning.Environments.LevelPacks;

namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>
/// The shipped Block Dude levels, read from the embedded <c>levels/blockdude-levels.json</c> — the single source
/// of truth shared with the Angular app, which fetches the same file from <c>wwwroot/levels</c>.
/// </summary>
/// <remarks>
/// <para>Grid format, rows TOP-first: <c>W</c> wall (immovable, climbable, never carryable), <c>B</c> carryable
/// block, <c>P</c> player start, <c>D</c> door/exit, <c>.</c> empty. Nothing is implicit — there is no hidden
/// floor or boundary.</para>
///
/// <para>The grids are deliberately NOT gravity-settled. Level 11 ships 14 blocks floating in mid-air, which is
/// authored content: gravity applies only to the walking player and to a dropped block, never as a global settle
/// pass (PRD §4.2).</para>
///
/// <para>Most of these boards are far beyond any exact solver — level 11 is 42 blocks on 551 cells — so they
/// carry no optimal move count. The oracle labels small generated boards for training; these levels are the
/// human content and the search-time benchmark.</para>
/// </remarks>
public static class BlockDudeLevels
{
    private static readonly Lazy<LevelPackEntry[]> Pack =
        new(() => EmbeddedLevelPack.Load("blockdude-levels.json"));

    /// <summary>The shipped levels, in ladder order.</summary>
    public static LevelPackEntry[] All => Pack.Value;

    /// <summary>Parses a shipped level into a playable board.</summary>
    public static BlockDudeBoard Load(int index) => BlockDudeBoard.FromGrid(All[index].Grid);

    /// <summary>Parses every shipped level into a playable board.</summary>
    public static IEnumerable<BlockDudeBoard> LoadAll() => All.Select(l => BlockDudeBoard.FromGrid(l.Grid));
}
