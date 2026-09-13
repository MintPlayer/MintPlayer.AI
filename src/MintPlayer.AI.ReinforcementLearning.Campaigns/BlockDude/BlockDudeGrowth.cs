using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Block Dude's capacity ladder. The mechanism — saturation detection, the single-rung Net2Net step, the
/// persisted rung — is generic and lives in Core (<see cref="SaturationGrowth"/>, <see cref="GrowthLadder"/>);
/// all this supplies is the shape, which is necessarily per-game.
/// </summary>
/// <remarks>
/// Rooted at the net's OWN default trunk via <see cref="GrowthLadder.FromTrunk"/>, so rung 0 is exactly what a
/// non-growing run trains and enabling growth can never start smaller. That is not a stylistic choice: the old
/// shared ladder topped out at <c>[128,128,128]</c>, below this game's <c>[512,512]</c> default, so growth used
/// to SHRINK the net it was meant to enlarge.
/// </remarks>
public static class BlockDudeGrowth
{
    /// <summary>Rung 0 = [512,512], then widen → deepen → widen: [768,768], [768,768,768], [1152,1152,1152].</summary>
    public static readonly GrowthLadder Ladder =
        GrowthLadder.FromTrunk(BlockDudePolicyNet.DefaultTrunk, steps: 3);
}
