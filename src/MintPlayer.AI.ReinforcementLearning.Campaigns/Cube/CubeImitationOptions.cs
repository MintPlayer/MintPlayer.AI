namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>All tunable configuration for a <see cref="CubeImitationCampaign"/> (defaults = the Lab's flag defaults).</summary>
public sealed record CubeImitationOptions
{
    public ulong Seed { get; init; } = 1;
    public float LearningRate { get; init; } = 3e-4f;
    /// <summary>Trunk width; the width-ladder store ids derive from it (<see cref="CubeIds.ForWidth"/>).</summary>
    public int Width { get; init; } = 512;
    /// <summary>Progressively grow the net wider+deeper mid-training (Net2Net, PLAN M37).</summary>
    public bool Grow { get; init; }
    /// <summary>Samples between growth steps (with <see cref="Grow"/>).</summary>
    public int GrowEvery { get; init; } = 4096;
}

/// <summary>All tunable configuration for a <see cref="CubeEfficientCampaign"/> (defaults = the Lab's flag defaults).</summary>
public sealed record CubeEfficientOptions
{
    public ulong Seed { get; init; } = 1;
    public float LearningRate { get; init; } = 3e-4f;
    /// <summary>Trunk width for the EfficientCube policy net.</summary>
    public int Width { get; init; } = 512;
    /// <summary>Curriculum ceiling: scramble depths sample 1..this.</summary>
    public int MaxScramble { get; init; } = 30;
    /// <summary>Beam width for the eval-time beam search.</summary>
    public int BeamWidth { get; init; } = 2_000;
    /// <summary>Eval episodes per depth.</summary>
    public int EvalEpisodes { get; init; } = 20;
    /// <summary>Progressively grow the net wider+deeper mid-training (Net2Net, PLAN M37).</summary>
    public bool Grow { get; init; }
    /// <summary>Samples between growth steps (with <see cref="Grow"/>).</summary>
    public int GrowEvery { get; init; } = 50_000;
}
