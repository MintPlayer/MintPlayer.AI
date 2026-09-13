using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Every campaign should be watchable in the live network viewer.
/// </summary>
/// <remarks>
/// This exists because forgetting it fails QUIETLY: <c>VizLauncher</c> skips a campaign that exposes no
/// telemetry with a console note and starts nothing, so <c>--viz</c> appears to do nothing at all. It has now
/// happened twice — Block Dude phase 1 went months without it, and phase 2 shipped without it on the very day
/// phase 1 was fixed.
/// </remarks>
public class CampaignTelemetryTests
{
    [Fact]
    public void EveryTrainingCampaignExposesNetworkTelemetry()
    {
        var campaigns = typeof(BlockDudeImitationCampaign).Assembly
            .GetTypes()
            .Where(t => t is { IsClass: true, IsAbstract: false } && t.IsAssignableTo(typeof(ITrainingCampaign)))
            .ToArray();

        Assert.NotEmpty(campaigns);

        var missing = campaigns
            .Where(t => !t.IsAssignableTo(typeof(INetworkTelemetrySource)))
            .Select(t => t.Name)
            .ToArray();

        Assert.True(missing.Length == 0,
            "these campaigns cannot be watched with --viz, which fails silently rather than erroring: " +
            string.Join(", ", missing));
    }
}
