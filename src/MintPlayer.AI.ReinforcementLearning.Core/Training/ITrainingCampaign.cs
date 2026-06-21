using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>One named metric from a campaign evaluation — a stable CSV column and a console value.</summary>
public readonly record struct CampaignMetric(string Name, double Value);

/// <summary>
/// The result of one campaign evaluation: an ordered metric list (stable CSV columns) plus a preformatted
/// one-line console summary. Deliberately MINIMAL — campaigns that need richer or multiple logs (e.g. a second
/// per-depth CSV) own those themselves. Folding every campaign's eval shape into this type would make it a
/// god-struct and re-leak per-game complexity into <see cref="CampaignRunner"/>.
/// </summary>
public sealed record CampaignEval(IReadOnlyList<CampaignMetric> Metrics, string Summary);

/// <summary>
/// A long-running, resumable training campaign driven by <see cref="CampaignRunner"/>. The runner owns the
/// wall-clock budget, the eval/checkpoint cadence, and result reporting; the campaign owns everything
/// game-specific — the trainer, the net, the data source, the eval metric, and its own checkpoint format.
/// <para>
/// Lifecycle per run: <see cref="Resume"/> once at the start (load prior state, or begin fresh), then repeated
/// <see cref="TrainChunk"/> calls interleaved with <see cref="Evaluate"/> + <see cref="Checkpoint"/> on the
/// runner's cadence, until the time budget elapses or <see cref="IsComplete"/> turns true. Extends
/// <see cref="IDisposable"/> so a campaign holding a device-resident GPU stack is torn down by the runner.
/// </para>
/// </summary>
public interface ITrainingCampaign : IDisposable
{
    /// <summary>The model-store environment id this campaign trains under (e.g. "snake", "cube").</summary>
    string Environment { get; }

    /// <summary>
    /// Load prior training state from <paramref name="store"/> and continue from it if present; otherwise start
    /// fresh. Returns true when a checkpoint was resumed. Called exactly once, before training begins.
    /// </summary>
    bool Resume(IModelStore store);

    /// <summary>
    /// Advance training by one chunk (the campaign picks the chunk size). Returns cumulative progress
    /// (steps / samples / iterations) — used for reporting and as the basis for <see cref="IsComplete"/>.
    /// </summary>
    long TrainChunk();

    /// <summary>Evaluate the current model; returns its metrics and a console summary line.</summary>
    CampaignEval Evaluate();

    /// <summary>Persist the net and full resume state to <paramref name="store"/>.</summary>
    void Checkpoint(IModelStore store);

    /// <summary>
    /// Whether the campaign has reached a hard stop independent of the time budget (e.g. a target sample count).
    /// Defaults to false — most campaigns simply run until the runner's wall-clock deadline.
    /// </summary>
    bool IsComplete => false;

    /// <summary>
    /// Optional escape hatch for an eval-only invocation whose output does not fit <see cref="CampaignEval"/>
    /// (e.g. a value-curve probe or a head-to-head report). When it returns true the runner treats the run as
    /// handled and trains nothing. Defaults to false (no standalone mode), so the runner falls back to a plain
    /// <see cref="Evaluate"/>.
    /// </summary>
    bool TryRunStandaloneEval(IModelStore store) => false;
}
