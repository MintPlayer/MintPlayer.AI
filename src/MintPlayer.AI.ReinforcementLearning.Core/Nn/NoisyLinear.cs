using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// NoisyNets linear layer (Fortunato et al. 2017): a drop-in <see cref="Linear"/> replacement whose
/// weight and bias carry a learnable noise SCALE alongside their mean —
/// <c>W = μ_w + σ_w ⊙ ε_w</c>, <c>b = μ_b + σ_b ⊙ ε_b</c>. The network explores by perturbing its
/// own parameters with freshly-sampled noise ε; because σ is trained by backprop, exploration
/// self-anneals where the policy is confident and persists where it isn't — a learned,
/// state-dependent replacement for ε-greedy.
///
/// The factorized variant is used (Fortunato §3.2): only <c>in + out</c> unit-Gaussian samples are
/// drawn per step and combined as an outer product — <c>ε_w = f(ε_out) ⊗ f(ε_in)</c>,
/// <c>ε_b = f(ε_out)</c>, with <c>f(x) = sgn(x)·√|x|</c>. The ε tensors are sampled as ordinary
/// CONSTANTS (RequiresGrad = false) in <see cref="ResampleNoise"/>, so the autograd graph only ever
/// multiplies a constant into the σ parameter: gradients reach μ and σ, never the noise — exactly as
/// NoisyNets requires, and with no new autograd op (this rides on Add/Mul/MatMul/AddBias).
///
/// <see cref="NoiseEnabled"/> defaults to FALSE — a freshly loaded layer is deterministic (forward
/// uses the means), so serving/eval is reproducible and unchanged; the trainer turns noise on.
///
/// Used by <see cref="DuelingQNet"/>'s heads when constructed noisy; see also the NoisyNets PRD.
/// </summary>
public sealed class NoisyLinear : IModule
{
    /// <summary>σ₀ in the factorized init σ = σ₀/√in (Fortunato et al. 2017).</summary>
    public const float Sigma0 = 0.5f;

    private readonly int _in;
    private readonly int _out;
    private readonly Tensor _epsilonWeight; // [in,out] constant, overwritten each ResampleNoise
    private readonly Tensor _epsilonBias;   // [out]    constant, overwritten each ResampleNoise

    public NoisyLinear(int inputSize, int outputSize, Xoshiro256StarStar rng)
    {
        _in = inputSize;
        _out = outputSize;

        float limit = 1f / MathF.Sqrt(inputSize);
        float sigma = Sigma0 / MathF.Sqrt(inputSize);

        MeanWeight = new Tensor(new float[inputSize * outputSize], inputSize, outputSize) { RequiresGrad = true };
        SigmaWeight = new Tensor(new float[inputSize * outputSize], inputSize, outputSize) { RequiresGrad = true };
        MeanBias = new Tensor(new float[outputSize], outputSize) { RequiresGrad = true };
        SigmaBias = new Tensor(new float[outputSize], outputSize) { RequiresGrad = true };

        Init.Uniform(MeanWeight, limit, rng);
        Init.Uniform(MeanBias, limit, rng);
        Array.Fill(SigmaWeight.Data, sigma);
        Array.Fill(SigmaBias.Data, sigma);

        _epsilonWeight = new Tensor(new float[inputSize * outputSize], inputSize, outputSize); // RequiresGrad = false
        _epsilonBias = new Tensor(new float[outputSize], outputSize);                          // RequiresGrad = false
    }

    /// <summary>Mean weight μ_w [in,out] — equals a plain <see cref="Linear"/>'s weight with noise off.</summary>
    public Tensor MeanWeight { get; }

    /// <summary>Learnable noise scale σ_w [in,out].</summary>
    public Tensor SigmaWeight { get; }

    /// <summary>Mean bias μ_b [out] — equals a plain <see cref="Linear"/>'s bias with noise off.</summary>
    public Tensor MeanBias { get; }

    /// <summary>Learnable noise scale σ_b [out].</summary>
    public Tensor SigmaBias { get; }

    /// <summary>When false (default), forward uses the means only → deterministic; the trainer sets it true.</summary>
    public bool NoiseEnabled { get; set; }

    public Tensor Forward(Tensor input)
    {
        if (!NoiseEnabled)
            return input.MatMul(MeanWeight).AddBias(MeanBias);

        var weight = MeanWeight.Add(SigmaWeight.Mul(_epsilonWeight));
        var bias = MeanBias.Add(SigmaBias.Mul(_epsilonBias));
        return input.MatMul(weight).AddBias(bias);
    }

    // Only the learnable means and scales are parameters; ε is sampled noise — not learned, not serialized.
    public IEnumerable<Tensor> Parameters()
    {
        yield return MeanWeight;
        yield return SigmaWeight;
        yield return MeanBias;
        yield return SigmaBias;
    }

    /// <summary>
    /// Draws fresh factorized noise: one ε_in ∈ ℝ^in and one ε_out ∈ ℝ^out, each ~ N(0,1), passed
    /// through f(x)=sgn(x)·√|x| and combined as ε_w = f(ε_out) ⊗ f(ε_in), ε_b = f(ε_out).
    /// </summary>
    public void ResampleNoise(Xoshiro256StarStar rng)
    {
        var fIn = Tensor.RandomNormal(rng, 0f, 1f, _in).Data;
        var fOut = Tensor.RandomNormal(rng, 0f, 1f, _out).Data;
        for (int i = 0; i < _in; i++) fIn[i] = SignedSqrt(fIn[i]);
        for (int j = 0; j < _out; j++) fOut[j] = SignedSqrt(fOut[j]);

        var ew = _epsilonWeight.Data;
        for (int i = 0; i < _in; i++)
        {
            float fi = fIn[i];
            int rowBase = i * _out;
            for (int j = 0; j < _out; j++)
                ew[rowBase + j] = fi * fOut[j];
        }
        fOut.CopyTo(_epsilonBias.Data.AsSpan());
    }

    private static float SignedSqrt(float x) => MathF.Sign(x) * MathF.Sqrt(MathF.Abs(x));
}
