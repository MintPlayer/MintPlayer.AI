using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// One AlphaZero training step for a two-headed policy/value net: forward → soft-CE policy loss + MSE value loss →
/// backward → grad-norm clip → Adam, with a weight-sync lifecycle. This is the <b>training</b> dual of
/// <see cref="IPolicyValueForward"/>: it is the seam that lets a GPU-<b>resident</b> trainer (weights + gradients +
/// Adam moments + activations kept on-device for the whole step) stand in for the plain autograd train step, without
/// <see cref="IPolicyValueNet"/> or the Core layer taking a GPU dependency. It is the two-headed analogue of
/// <see cref="Planning.IResidentTrainStep"/> (which is single-scalar-head, for the cube's cost-to-go net).
/// <para>
/// The per-step config (observation size, action count, value-loss weight, grad clip, learning rate) is baked in at
/// construction — it never changes between steps — so <see cref="Step"/> takes only the batch. A resident implementation
/// masters the trained weights on the device; <see cref="SyncToHost"/> writes them back into the CPU net that eval,
/// arena, the ladder and checkpointing all read. The autograd default masters the CPU net directly, so its
/// <see cref="SyncToHost"/> is a no-op.
/// </para>
/// </summary>
public interface IPolicyValueTrainStep
{
    /// <summary>One batch: forward + CE/MSE backward + grad-norm clip + Adam. <paramref name="obs"/> is row-major
    /// (length <c>batch·observationSize</c>); <paramref name="policyTargets"/> is row-major π (length
    /// <c>batch·actions</c>, each row summing to 1); <paramref name="valueTargets"/> is z ∈ [-1,1] (length
    /// <c>batch</c>). Returns the batch's policy (CE) and value (MSE) losses.</summary>
    (double PolicyLoss, double ValueLoss) Step(float[] obs, float[] policyTargets, float[] valueTargets, int batch);

    /// <summary>Write the trained weights back into the CPU net (a resident trainer re-downloads them; the autograd
    /// default is a no-op since it trains the CPU net in place). Called on the owner thread before eval/checkpoint.</summary>
    void SyncToHost();
}

/// <summary>
/// The default, GPU-free <see cref="IPolicyValueTrainStep"/>: the AlphaZero loss (soft-CE policy + <c>tanh</c>-MSE value)
/// over an <see cref="IPolicyValueNet"/>'s own autograd graph, stepped by the shared <see cref="Adam"/>. This is the
/// non-GPU fallback AND the guarantee that routing training through this seam changes nothing on the CPU path: the op
/// sequence is byte-for-byte the one the self-play campaign ran inline before the seam existed (verified by the
/// DOP-invariance checkpoint-hash test). Holds the net + optimizer by reference, so <see cref="SyncToHost"/> is a no-op.
/// </summary>
public sealed class AutogradPolicyValueTrainStep(
    IPolicyValueNet net, Adam adam, int observationSize, int actions, float valueWeight = 1f, float gradClipNorm = 5f)
    : IPolicyValueTrainStep
{
    public (double PolicyLoss, double ValueLoss) Step(float[] obs, float[] policyTargets, float[] valueTargets, int batch)
    {
        var (logits, value) = net.Forward(new Tensor(obs, batch, observationSize));
        var logProbs = logits.LogSoftmax();
        var ce = logProbs.Mul(new Tensor(policyTargets, batch, actions)).Sum().MulScalar(-1f / batch);
        var predicted = value.Reshape(batch).Tanh();           // bound the value head to [-1,1] for a WDL outcome
        var valueLoss = predicted.MseLoss(new Tensor(valueTargets, batch));
        var loss = valueWeight == 1f ? ce.Add(valueLoss) : ce.Add(valueLoss.MulScalar(valueWeight));

        adam.ZeroGrad();
        loss.Backward();
        adam.ClipGradNorm(gradClipNorm);
        adam.Step();
        return (ce.Data[0], valueLoss.Data[0]);
    }

    public void SyncToHost() { } // the CPU net is the master; nothing to copy back
}
