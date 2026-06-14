namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// A fully device-resident DAVI training step, decoupled from any GPU backend (PLAN M20 Stage 3). When
/// injected into <see cref="ValueIterationTrainer{TState}"/>, it replaces the autograd
/// forward+backward+Adam of the online net with one on-device step — the online weights are mastered on
/// the device and only synced back to the CPU net (for eval / checkpoint / target-net copy) via
/// <see cref="SyncToHost"/>. This removes the CPU-bound train step that otherwise dominates once the
/// successor evaluation is resident (Stage 2). The default (null) keeps the autograd train path.
/// </summary>
public interface IResidentTrainStep
{
    /// <summary>
    /// One DAVI train step on <paramref name="rows"/> feature rows against the given regression targets
    /// (both row-major, in the trainer's scaled units): resident forward + backward + grad clip + Adam.
    /// Returns the batch-mean loss.
    /// </summary>
    float Step(float[] features, float[] targets, int rows);

    /// <summary>Write the resident (online) weights back into the CPU value net the trainer holds.</summary>
    void SyncToHost();
}
