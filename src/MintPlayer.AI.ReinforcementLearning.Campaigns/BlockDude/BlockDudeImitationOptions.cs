namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Tunables for <see cref="BlockDudeImitationCampaign"/>.</summary>
public sealed record BlockDudeImitationOptions
{
    public ulong Seed { get; init; } = 1;

    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>Progressively grow the net wider and deeper mid-training (Net2Net), one rung of
    /// <see cref="BlockDudeGrowth.Stages"/> at a time, when <see cref="GrowthPlateau"/> reports the gate has
    /// saturated. Rung 0 IS the non-growing shape, so enabling this never starts a run smaller than the default.</summary>
    public bool Grow { get; init; }

    /// <summary>Gate evaluations without a new best gate before the net is called saturated and grown. Counted in
    /// GATES, not samples: the gate is the saturation signal, so patience is naturally measured in observations of
    /// it.</summary>
    public int GrowPatience { get; init; } = 6;

    /// <summary>How much a gate must beat the window's best by to count as progress. Must sit ABOVE the gate's own
    /// jitter — measured at ~10 points on the 64-board hold-out — or upward noise resets patience forever and the
    /// net never grows. The default is deliberately a little under half that swing.</summary>
    public double GrowMinImprovement { get; init; } = 0.04;

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

    /// <summary>Training batch size. A chunk trains nothing until it has collected this many
    /// samples, so lowering it is what lets a test drive a real chunk cheaply (M64.7).</summary>
    public int BatchSize { get; init; } = 256;

    /// <summary>Boards in a rung's fixed greedy hold-out. Same lever as <see cref="BatchSize"/> and for the same
    /// reason (M69/B4): the gate solves every board greedily, so 64 of them is what makes the gate limb too
    /// expensive to reach from a test. Shrinking it takes a PREFIX of the same set — <c>GateBoardsFor</c> draws
    /// from an unchanged RNG — so a small hold-out is a subset of the shipped one, not a different distribution.
    /// The default is the shipped value, so no run changes.</summary>
    public int GateBoards { get; init; } = BlockDudeCurriculum.GateBoards;
}
