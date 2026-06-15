using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Invalid-action masking for a <see cref="Categorical"/> policy: illegal actions are pushed to a large
/// negative logit so softmax assigns them ~0 probability (never sampled, ~0 entropy contribution, ~0
/// gradient), while the legal actions keep an exactly-normalized distribution. Implemented as an
/// <b>additive logit bias</b> (0 for legal, <see cref="MaskedLogitBias"/> for illegal) so it composes with
/// autograd — the bias is constant, so gradients flow only to the legal logits. This is the single shared
/// definition used by both inference (<c>PolicyAgent</c>) and training (PPO), so a masked rollout and
/// its update use identical semantics.
/// </summary>
public static class ActionMasking
{
    /// <summary>
    /// Finite (not −∞) so <c>exp</c> underflows cleanly to 0 in log-softmax/entropy — −∞ would make the
    /// 0·log0 entropy term NaN. exp(−1e9) = 0 in float, so the masked action contributes nothing.
    /// </summary>
    public const float MaskedLogitBias = -1e9f;

    /// <summary>
    /// Additive logit bias for a [<paramref name="rows"/>, <paramref name="cols"/>] batch from a row-major
    /// legality mask of the same shape: 0 where legal, <see cref="MaskedLogitBias"/> where illegal.
    /// </summary>
    public static float[] Bias(ReadOnlySpan<bool> mask, int rows, int cols)
    {
        var bias = new float[rows * cols];
        for (int i = 0; i < bias.Length; i++)
            if (!mask[i]) bias[i] = MaskedLogitBias;
        return bias;
    }

    /// <summary>Adds the legality mask's bias to <paramref name="logits"/> (autograd-recorded).</summary>
    public static Tensor Apply(Tensor logits, ReadOnlySpan<bool> mask)
        => logits.Add(new Tensor(Bias(mask, logits.Rows, logits.Cols), logits.Rows, logits.Cols));
}
