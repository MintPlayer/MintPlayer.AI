namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// All tunable configuration for a <see cref="DqnScoreCampaign"/>, as one record with sensible defaults —
/// the same shape as <see cref="SelfPlayOptions"/>: adding a knob is a single field, not another constructor
/// parameter threaded through every call site. The environment is NOT here: it's a dependency (constructor),
/// not a knob — the caller builds/injects the env, so a test can hand the campaign a stub.
/// </summary>
public record DqnScoreOptions
{
    public ulong Seed { get; init; } = 1;
    /// <summary>Trainer steps per <c>TrainChunk</c> (a "step" is game-specific: grid moves, drops, …).</summary>
    public int ChunkSteps { get; init; } = 5_000;
    /// <summary>Absolute step cap across resumes (0 = run by wall-clock only).</summary>
    public long TargetSteps { get; init; } = 0;
    /// <summary>Fixed-seed greedy episodes per eval.</summary>
    public int EvalEpisodes { get; init; } = 20;
    public float LearningRate { get; init; } = 5e-4f;
    /// <summary>ε-start; pass a low value (e.g. 0.2) to refine a warm-started net.</summary>
    public float EpsilonStart { get; init; } = 1.0f;
    /// <summary>Trunk widths for the Dueling Q-net.</summary>
    public int[] Hidden { get; init; } = [128, 128];
    public double Gamma { get; init; } = 0.99;
    /// <summary>Progressively grow the net wider+deeper mid-training (Net2Net, PLAN M37).</summary>
    public bool Grow { get; init; }
    /// <summary>Steps between growth steps (with <see cref="Grow"/>).</summary>
    public int GrowEvery { get; init; } = 5_000;
}

/// <summary>
/// <see cref="FruitCakeDqnCampaign"/> knobs on top of the shared spine's. Reward shaping is NOT here — it lives
/// on the injected training env (<c>FruitCakeEnv.ShapeRewards</c>/<c>ShapingGamma</c>), because it changes what
/// the env emits, not how the campaign trains.
/// </summary>
public sealed record FruitCakeDqnOptions : DqnScoreOptions
{
    /// <summary>NoisyNets exploration (learned σ) instead of ε-greedy; trains as a separate resume-state line.</summary>
    public bool Noisy { get; init; }
    /// <summary>n-step return horizon (1 = single-step DQN).</summary>
    public int NStep { get; init; } = 1;
}
