namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Tunables for <see cref="BlockDudeExpertIterationCampaign"/>.</summary>
public sealed record BlockDudeExpertIterationOptions
{
    public ulong Seed { get; init; } = 1;

    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>Start from a blank slate rather than resuming this phase's net and frontier.</summary>
    public bool Fresh { get; init; }

    /// <summary>
    /// Warm-start from the phase-1 imitation net instead of random weights. On by default: phase 1 already
    /// learned the local mechanics (climb, carry, place) on small boards, and re-learning them from scratch on
    /// expensive searched data would waste the only signal that is cheap to produce.
    /// </summary>
    public bool WarmStart { get; init; } = true;

    /// <summary>Stop after this many training samples (0 = run to the runner's deadline).</summary>
    public long TargetSamples { get; init; }

    /// <summary>Search attempts per level per round.</summary>
    public int AttemptsPerLevel { get; init; } = 4;

    /// <summary>Node budget for one search. The memory lever — Block Dude states are large.</summary>
    public int Expansions { get; init; } = 40_000;

    /// <summary>Wall-clock ceiling per search, so one hopeless level cannot stall a round.</summary>
    public int SearchSeconds { get; init; } = 6;

    /// <summary>f = g + weight·h. A learned heuristic is not admissible, so above 1 is the practical setting.</summary>
    public float Weight { get; init; } = 2f;

    /// <summary>
    /// How much the policy head's prior steers the search that generates training data — one nat of surprise
    /// costs this many moves. 0 falls back to value-only search.
    /// </summary>
    /// <remarks>
    /// This is a self-improvement loop, so the search is not only the measuring instrument but the data source:
    /// a search that reaches further solves longer suffixes, which moves the frontier, which is the run's actual
    /// progress. Using the policy head here feeds the better-trained head back into producing its own next
    /// batch of training data.
    /// </remarks>
    public float PolicyWeight { get; init; } = 1f;

    /// <summary>Where a level's frontier starts: moves from the door it must solve before moving outward.
    /// Defaults to the depth the current net was measured to manage unaided (PRD §6a).</summary>
    public int InitialFrontier { get; init; } = 20;

    /// <summary>Multiplier applied to a level's frontier when it clears <see cref="AdvanceRate"/>. A ratio rather
    /// than a step: the far end of a 909-move level is not reached by adding 10 at a time.</summary>
    public double FrontierGrowth { get; init; } = 1.5;

    /// <summary>Fraction of a round's attempts a level must solve before its frontier moves outward.</summary>
    public double AdvanceRate { get; init; } = 0.75;

    /// <summary>
    /// Share of each training batch drawn from the HUMAN demonstrations rather than searched solutions. Keeps a
    /// permanent anchor in true long-horizon data: searched solutions cluster near whatever the frontier
    /// currently is, so without this the far-distance labels would fade out of the mix as the frontier moves.
    /// </summary>
    public double DemoShare { get; init; } = 0.25;
}
