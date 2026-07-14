using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

/// <summary>
/// One supervised batch for a <see cref="PolicyValueNet"/> trained on <b>soft</b> targets — the AlphaZero loss:
/// cross-entropy of a target policy distribution π against the policy head, plus mean-squared error of a game
/// outcome z ∈ [-1,1] against the <c>tanh</c> of the value head. It is the same loss shape the imitation campaigns
/// use (<see cref="CubePolicyTraining.TrainStep"/>), generalized to raw arrays + a soft π + a bounded value, so any
/// <see cref="Core.Planning.IZeroSumGame{TState}"/> can drive it with no environment dependency.
/// </summary>
internal static class PolicyValueTraining
{
    /// <param name="obs">Row-major observations, length <paramref name="batch"/>×<paramref name="obsSize"/>.</param>
    /// <param name="policyTargets">Row-major target distributions π, length <paramref name="batch"/>×<paramref name="actions"/> (each row sums to 1).</param>
    /// <param name="valueTargets">Outcome z per row in [-1,1], length <paramref name="batch"/>.</param>
    /// <returns>The batch's policy (CE) and value (MSE) losses.</returns>
    /// <param name="valueWeight">Weight on the value (MSE) term relative to the policy (CE) term. 1 = the original
    /// equal sum. Down-weighting (e.g. 0.25) is the standard fix for value-head overfitting → strength regression at
    /// small scale (Leela Zero cut it 1.0→0.25). Weight 1 keeps the exact original graph (bitwise back-compat).</param>
    public static (double PolicyLoss, double ValueLoss) TrainStep(
        IPolicyValueNet net, Adam adam, float[] obs, float[] policyTargets, float[] valueTargets,
        int batch, int obsSize, int actions, float valueWeight = 1f)
    {
        var (logits, value) = net.Forward(new Tensor(obs, batch, obsSize));
        var logProbs = logits.LogSoftmax();
        var ce = logProbs.Mul(new Tensor(policyTargets, batch, actions)).Sum().MulScalar(-1f / batch);
        var predicted = value.Reshape(batch).Tanh();           // bound the value head to [-1,1] for a WDL outcome
        var valueLoss = predicted.MseLoss(new Tensor(valueTargets, batch));
        var loss = valueWeight == 1f ? ce.Add(valueLoss) : ce.Add(valueLoss.MulScalar(valueWeight));

        adam.ZeroGrad();
        loss.Backward();
        adam.ClipGradNorm(5f);
        adam.Step();
        return (ce.Data[0], valueLoss.Data[0]);
    }
}
