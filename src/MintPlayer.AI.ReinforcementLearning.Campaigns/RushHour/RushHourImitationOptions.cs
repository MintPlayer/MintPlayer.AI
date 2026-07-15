namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>All tunable configuration for a <see cref="RushHourImitationCampaign"/> (defaults = the Lab's flag defaults).</summary>
public sealed record RushHourImitationOptions
{
    public ulong Seed { get; init; } = 1;
    public float LearningRate { get; init; } = 3e-4f;
    /// <summary>Progressively grow the net wider+deeper mid-training (Net2Net, PLAN M37).</summary>
    public bool Grow { get; init; }
    /// <summary>Samples between growth steps (with <see cref="Grow"/>).</summary>
    public int GrowEvery { get; init; } = 2048;
}
