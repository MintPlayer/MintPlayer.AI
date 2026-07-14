using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// A batched, <b>inference-only</b> forward for a two-headed policy/value net, with a weight-sync lifecycle. This is the
/// seam that lets a GPU-<b>resident</b> implementation (weights uploaded to the device once and kept there, re-uploaded
/// only on <see cref="OnWeightsSynced"/>) stand in for the plain autograd forward — without <see cref="IPolicyValueNet"/>
/// or the Core layer taking a GPU dependency. It is the two-headed analogue of <see cref="Planning.ITargetForward"/>
/// (which is single-scalar-head, for the cube's cost-to-go net).
/// <para>
/// It returns <b>raw</b> head outputs — policy logits and the linear (pre-<c>tanh</c>) value — because the caller applies
/// masked-softmax over the legal moves and <c>tanh</c> itself (it has each state's legal-move set; the device forward
/// does not). Weights are re-synced on the training thread's own cadence (e.g. once per self-play chunk), never per
/// forward — that is the whole point of residency.
/// </para>
/// </summary>
public interface IPolicyValueForward
{
    /// <summary>Adopt the current weights from <paramref name="net"/> (a device implementation re-uploads them). Called
    /// on the owner thread when the net has changed — NOT inside the hot forward path.</summary>
    void OnWeightsSynced(IPolicyValueNet net);

    /// <summary>Forward over <paramref name="rows"/> observations packed row-major into <paramref name="observations"/>
    /// (length <c>rows · observationSize</c>). Returns raw <c>Logits</c> (length <c>rows · actions</c>, row-major) and
    /// the linear <c>Value</c> (length <c>rows</c>) — no softmax, no tanh.</summary>
    (float[] Logits, float[] Value) Forward(float[] observations, int rows);
}

/// <summary>
/// The default, GPU-free <see cref="IPolicyValueForward"/>: runs the net's own autograd <see cref="IPolicyValueNet.Forward"/>
/// under <c>NoGrad</c>. Behaviour-identical to calling <c>net.Forward</c> directly, so it is both the non-GPU fallback and
/// a guarantee that routing inference through this seam changes nothing on the CPU path. Holds the net by reference, so
/// in-place weight updates (training mutates the same net) are seen without an explicit sync.
/// </summary>
public sealed class AutogradPolicyValueForward(IPolicyValueNet net, int observationSize) : IPolicyValueForward
{
    private IPolicyValueNet _net = net;

    public void OnWeightsSynced(IPolicyValueNet updated) => _net = updated;

    public (float[] Logits, float[] Value) Forward(float[] observations, int rows)
    {
        using (GradMode.NoGrad())
        {
            var (logits, value) = _net.Forward(new Tensor(observations, rows, observationSize));
            return (logits.Data, value.Data); // logits [rows*actions], value [rows] — raw, caller does softmax/tanh
        }
    }
}
