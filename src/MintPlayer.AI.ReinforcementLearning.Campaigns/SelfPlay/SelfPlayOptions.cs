using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// All tunable configuration for a <see cref="SelfPlayCampaign{TState}"/>, as one record with sensible defaults.
/// Bundling the knobs here keeps the campaign constructor small and makes adding a new knob a single field — not
/// another constructor parameter threaded through every call site (the telescoping-constructor boilerplate this
/// replaces). Defaults match the pre-refactor constructor defaults exactly, so a caller that sets nothing behaves
/// identically. The MCTS knobs (simulations, cpuct, Dirichlet α, root-noise) live in <see cref="Search"/>, which is
/// already the options bag for search — no need to re-list them here.
/// </summary>
public sealed record SelfPlayOptions
{
    public ulong Seed { get; init; } = 1;
    public float LearningRate { get; init; } = 1e-3f;
    public int Hidden { get; init; } = 256;                 // flat-MLP trunk width (used as [Hidden, Hidden]); ignored by conv
    public Mcts.Config Search { get; init; } = new();       // MCTS: Simulations / Cpuct / DirichletAlpha / RootNoiseFrac
    public int GamesPerChunk { get; init; } = 32;
    public int TempMoves { get; init; } = 8;                // plies of temperature-1 sampling before switching to argmax
    public int EvalGames { get; init; } = 20;
    public int WindowCapacity { get; init; } = 40_000;      // replay-window size (AlphaZero-scale runs want ~500k+)
    public int BatchSize { get; init; } = 128;
    public int EpochsPerChunk { get; init; } = 1;           // shuffled passes over the window per chunk
    public int MaxPlies { get; init; } = 512;               // ply cap; capped games are material-adjudicated, not drawn
    public long TargetGames { get; init; } = 0;             // absolute self-play game cap (0 = run by wall-clock only)
    public double OpponentRandomFrac { get; init; } = 0;    // fraction of self-play games the learner plays vs a random opponent
    public LadderOptions? Ladder { get; init; }             // auto-difficulty ladder (null = off)
    public float MaterialWeight { get; init; } = 0f;        // α: blend of game outcome z with dense material target
    public float ValueWeight { get; init; } = 1f;           // value-loss weight relative to policy loss
    public float GradClipNorm { get; init; } = 5f;
    public bool Parallel { get; init; }                     // fan self-play generation across cores (DOP-invariant)
    public int? MaxDop { get; init; }                       // cap on that parallelism (null = runtime default)
    public int LeafBatch { get; init; } = 1;                // MCTS leaf-inference batch (>1 → Mcts.SearchBatched)
}
