namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Tunables for <see cref="BlockDudeImitationCampaign"/>.</summary>
public sealed record BlockDudeImitationOptions
{
    public ulong Seed { get; init; } = 1;

    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>Progressively grow the net wider and deeper mid-training (Net2Net).</summary>
    public bool Grow { get; init; }

    /// <summary>Samples between growth steps (with <see cref="Grow"/>).</summary>
    public int GrowEvery { get; init; } = 200_000;

    /// <summary>Boards generated and labelled per <c>TrainChunk</c>.</summary>
    public int BoardsPerRound { get; init; } = 4;

    /// <summary>Training samples drawn from each labelled board.</summary>
    public int SamplesPerBoard { get; init; } = 512;

    /// <summary>Samples between curriculum gate evaluations. Counted in SAMPLES, never in wall-clock time —
    /// <c>CampaignRunner</c> fires <c>Evaluate</c> on a clock, and a gate driven by that would make the training
    /// trajectory depend on how fast the machine is.</summary>
    public long GateEverySamples { get; init; } = 100_000;

    /// <summary>Stop after this many samples (0 = run until the runner's own deadline).</summary>
    public long TargetSamples { get; init; }

    /// <summary>Pin training to one curriculum rung, disabling promotion. Diagnostic only.</summary>
    public int? PinStage { get; init; }

    /// <summary>Highest rung this run may reach.</summary>
    public int MaxStage { get; init; } = BlockDudeCurriculum.LastStage;

    /// <summary>Start from a blank slate: delete this phase's net, Adam moments and progress sidecar before
    /// training, instead of resuming them.</summary>
    public bool Fresh { get; init; }

    /// <summary>1 = imitation from the exact oracle; 2 = expert iteration (separate checkpoint ids).</summary>
    public int Phase { get; init; } = 1;

    /// <summary>Give up on a generated board after this many rejected candidates.</summary>
    public int MaxGenerationAttempts { get; init; } = 400;
}
