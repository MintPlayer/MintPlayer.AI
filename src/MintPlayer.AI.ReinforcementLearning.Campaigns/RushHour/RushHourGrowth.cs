using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Rush Hour's capacity ladder. Rooted at the net's own default trunk, so <c>--grow</c> starts where a plain run
/// starts and climbs from there.
/// </summary>
/// <remarks>
/// Previously this campaign grew along the shared <see cref="DqnGrowth.Stages"/> schedule, which begins at
/// <c>[16]</c> and ends at <c>[128,128,128]</c> — against a default of <c>[384,384]</c>. So <c>--grow</c> started
/// 24× narrower and finished with roughly a quarter of the capacity of leaving the flag off. Any run that used it
/// measured a SMALLER net, and results from before 2026-09-13 should be read with that in mind.
/// </remarks>
public static class RushHourGrowth
{
    public static readonly GrowthLadder Ladder =
        GrowthLadder.FromTrunk(RushHourPolicyNet.DefaultTrunk, steps: 3);
}
