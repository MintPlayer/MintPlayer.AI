using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Diagonal Gaussian over continuous actions parameterized by per-action mean and log-σ [B,A], with the
/// tanh squash SAC uses to bound actions to (−1,1). Reparameterized samples and their log-probs are built
/// from autograd ops so the actor loss differentiates through the sample; the N(0,1) noise is a detached
/// constant (a bare <see cref="Tensor"/> has no graph parents, so gradient never leaks into it). The mirror
/// of <see cref="Categorical"/> for continuous control.
/// </summary>
public readonly struct Normal
{
    /// <summary>log-σ clamp range (Haarnoja SAC convention) — keeps σ = exp(log-σ) from over/underflowing.</summary>
    public const float LogStdMin = -20f;
    public const float LogStdMax = 2f;

    private static readonly float HalfLog2Pi = 0.5f * MathF.Log(2f * MathF.PI);

    private readonly Tensor _mean;    // [B,A]
    private readonly Tensor _logStd;  // [B,A], already clamped
    private readonly Tensor _std;     // exp(log-σ), cached

    public Normal(Tensor mean, Tensor logStd)
    {
        _mean = mean;
        _logStd = logStd;
        _std = logStd.Exp();
    }

    /// <summary>
    /// Builds a squashed Gaussian from a policy net's [B,2·A] output: the first A columns are the mean, the
    /// last A the log-σ (clamped to <see cref="LogStdMin"/>..<see cref="LogStdMax"/>).
    /// </summary>
    public static Normal FromNetOutput(Tensor netOutput, int actionDim)
    {
        var mean = netOutput.SliceCols(0, actionDim);
        var logStd = netOutput.SliceCols(actionDim, actionDim).Clamp(LogStdMin, LogStdMax);
        return new Normal(mean, logStd);
    }

    /// <summary>
    /// Reparameterized squashed sample: a = tanh(mean + σ·ε), ε ~ N(0,1) detached. Returns the action [B,A]
    /// in (−1,1) and its log-prob [B] (the Gaussian log-prob of the pre-squash sample minus the tanh
    /// change-of-variables correction, summed over action dims).
    /// </summary>
    public (Tensor Action, Tensor LogProb) RSample(Xoshiro256StarStar rng)
    {
        var eps = Tensor.RandomNormal(rng, 0f, 1f, _mean.Shape); // constant leaf — stop-gradient
        var preSquash = _mean.Add(_std.Mul(eps));                // mean + σ·ε
        var action = preSquash.Tanh();                           // squashed to (−1,1)
        return (action, LogProb(eps, action));
    }

    /// <summary>Deterministic action for evaluation/serving: tanh(mean) → [B,A].</summary>
    public Tensor Mode() => _mean.Tanh();

    // log N(a;μ,σ) per dim, with (a−μ)/σ ≡ ε, minus the tanh correction −log(1−tanh²+ε); summed over dims.
    private Tensor LogProb(Tensor eps, Tensor action)
    {
        var gaussian = _logStd.MulScalar(-1f)
            .Sub(eps.Square().MulScalar(0.5f))
            .Sub(Tensor.Full(HalfLog2Pi, _mean.Shape));
        var correction = Tensor.Full(1f + 1e-6f, action.Shape).Sub(action.Square()).Log();
        return gaussian.Sub(correction).SumRows();
    }
}
