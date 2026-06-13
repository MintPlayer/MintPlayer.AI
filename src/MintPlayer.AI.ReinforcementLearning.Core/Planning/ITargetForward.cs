using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// No-grad batch forward for DAVI's bootstrapping successor evaluation — the dominant per-step
/// cost (ActionCount× forwards over the frozen target net). Decoupled from any GPU backend so the
/// Core layer stays device-agnostic: <see cref="ValueIterationTrainer{TState}"/> calls
/// <see cref="OnTargetSynced"/> exactly when it refreshes the target net (once at start, then every
/// <see cref="ValueIterationOptions.TargetUpdateInterval"/> steps), and <see cref="Forward"/> for
/// each successor batch. That lets a device-resident implementation (the Ilgpu <c>DeviceMlp</c>)
/// upload weights ONCE per sync rather than per step — removing the per-call weight transfer that
/// otherwise dominates a wide-net forward (PLAN M20). The default CPU implementation just retains
/// the net reference and runs the backend-agnostic autograd forward.
/// </summary>
public interface ITargetForward
{
    /// <summary>
    /// Adopt <paramref name="target"/>'s current weights as the net to evaluate. Called by the
    /// trainer whenever the target net is (re)synced — including once before training begins.
    /// </summary>
    void OnTargetSynced(Mlp target);

    /// <summary>Raw scalar outputs for <paramref name="rows"/> feature rows (row-major), no autograd.</summary>
    float[] Forward(float[] features, int rows);
}

/// <summary>
/// Default <see cref="ITargetForward"/>: holds the synced target net and runs the standard autograd
/// forward through <c>Backend.Current</c>. No device residency — used when no GPU path is injected.
/// </summary>
public sealed class AutogradTargetForward(int featureSize) : ITargetForward
{
    private Mlp? _net;

    public void OnTargetSynced(Mlp target) => _net = target;

    public float[] Forward(float[] features, int rows)
    {
        if (_net is null) throw new InvalidOperationException("OnTargetSynced must be called before Forward.");
        using (GradMode.NoGrad())
            return _net.Forward(new Tensor(features, rows, featureSize)).Data.ToArray();
    }
}
