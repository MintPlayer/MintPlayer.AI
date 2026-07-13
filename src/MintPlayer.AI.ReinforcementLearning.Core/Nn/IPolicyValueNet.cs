using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// A two-headed policy/value network: an observation batch → raw policy logits [B, actions] + a scalar value [B, 1]
/// (the value is left linear; the trainer bounds it via <c>tanh</c>). This is the seam the AlphaZero self-play stack
/// (<c>SelfPlayCampaign</c> / <c>PolicyValueTraining</c> / the MCTS leaf evaluator) depends on, so alternative
/// architectures — the flat <see cref="PolicyValueNet"/> and the convolutional <see cref="ConvResidualPolicyValueNet"/>
/// — are interchangeable without the campaign knowing which it holds. Construction and checkpoint reload stay
/// architecture-specific (a net factory owns those); this interface is only what a trained net must offer at runtime.
/// </summary>
public interface IPolicyValueNet
{
    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B, actions] + value [B, 1].</summary>
    (Tensor Logits, Tensor Value) Forward(Tensor observations);

    /// <summary>Every trainable tensor, in a stable order (drives the optimizer and checkpoint round-trip).</summary>
    IEnumerable<Tensor> Parameters();

    /// <summary>Per-layer activations for one input row (the live-network viewer seam).</summary>
    float[][] LayerActivations(Tensor observation);

    /// <summary>Serializes the net (the <paramref name="kind"/> tag is supplied by the owner/factory).</summary>
    void Save(Stream destination, string kind);

    /// <summary>A short human-readable shape summary for logs (e.g. "trunk [256,256]" or "conv 64f×6b").</summary>
    string Describe();
}
