using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// A deep residual value network (PLAN M21): an input projection into a fixed width, then a stack of
/// pre-activation residual blocks, then a scalar head. Each block is
/// <c>x → x + W₂·ReLU(LayerNorm(W₁·x))</c>; the skip connection lets gradients reach early layers, and
/// <see cref="Tensor.LayerNorm"/> (not BatchNorm) keeps depth stable under DAVI's frozen-target
/// bootstrap. Depth-with-residuals is the untried lever after M17 showed width alone is diminishing —
/// it raises the cost-to-go function's representational ceiling so the learned value stays accurate at
/// the scramble depths a plain MLP plateaus on.
/// <para>
/// Implements <see cref="IValueNet"/>, so it drops into the same DAVI trainer / batched search /
/// checkpoint paths as <see cref="Mlp"/> with no special-casing — only the device-resident GPU forward
/// (<c>DeviceMlp</c>) is MLP-specific; a residual net trains via the autograd path (whose GEMMs still
/// route to the GPU through the adaptive backend).
/// </para>
/// </summary>
public sealed class ResidualMlp : IValueNet
{
    private readonly int _inputSize;
    private readonly int _width;
    private readonly Linear _inProj;
    private readonly Tensor _inGamma, _inBeta;            // LayerNorm after the input projection
    private readonly Linear[] _block1, _block2;           // the two Linears of each residual block
    private readonly Tensor[] _blockGamma, _blockBeta;    // LayerNorm inside each block
    private readonly Linear _head;                        // width → 1 (scalar cost-to-go)

    /// <param name="inputSize">Observation width.</param>
    /// <param name="width">Hidden/residual width (the trunk dimension every block preserves).</param>
    /// <param name="blocks">Number of residual blocks (depth lever; 3–4 is the M21 default).</param>
    public ResidualMlp(int inputSize, int width, int blocks, Xoshiro256StarStar rng)
    {
        if (blocks < 1) throw new ArgumentException("A ResidualMlp needs at least one residual block.");
        _inputSize = inputSize;
        _width = width;

        _inProj = new Linear(inputSize, width, rng, Activation.Relu);
        (_inGamma, _inBeta) = NewNormParams(width);

        _block1 = new Linear[blocks];
        _block2 = new Linear[blocks];
        _blockGamma = new Tensor[blocks];
        _blockBeta = new Tensor[blocks];
        for (int i = 0; i < blocks; i++)
        {
            _block1[i] = new Linear(width, width, rng, Activation.Relu);
            (_blockGamma[i], _blockBeta[i]) = NewNormParams(width);
            _block2[i] = new Linear(width, width, rng, Activation.Relu);
        }

        _head = new Linear(width, 1, rng, Activation.None);
    }

    /// <summary>LayerNorm scale (γ=1) and shift (β=0), both learnable [width].</summary>
    private static (Tensor Gamma, Tensor Beta) NewNormParams(int width)
    {
        var gamma = new Tensor(new float[width], width) { RequiresGrad = true };
        Array.Fill(gamma.Data, 1f);
        var beta = new Tensor(new float[width], width) { RequiresGrad = true };
        return (gamma, beta);
    }

    public int InputSize => _inputSize;
    public int Width => _width;
    public int Blocks => _block1.Length;

    public Tensor Forward(Tensor input)
    {
        var x = _inProj.Forward(input).LayerNorm(_inGamma, _inBeta).Relu();
        for (int i = 0; i < _block1.Length; i++)
        {
            var h = _block1[i].Forward(x).LayerNorm(_blockGamma[i], _blockBeta[i]).Relu();
            h = _block2[i].Forward(h);
            x = x.Add(h); // residual skip
        }
        return _head.Forward(x);
    }

    public IEnumerable<Tensor> Parameters()
    {
        foreach (var p in _inProj.Parameters()) yield return p;
        yield return _inGamma; yield return _inBeta;
        for (int i = 0; i < _block1.Length; i++)
        {
            foreach (var p in _block1[i].Parameters()) yield return p;
            yield return _blockGamma[i]; yield return _blockBeta[i];
            foreach (var p in _block2[i].Parameters()) yield return p;
        }
        foreach (var p in _head.Parameters()) yield return p;
    }

    public IValueNet CloneStructure() => new ResidualMlp(_inputSize, _width, _block1.Length, new Xoshiro256StarStar(0));

    public void CopyFrom(IValueNet source)
    {
        using var mine = Parameters().GetEnumerator();
        using var theirs = source.Parameters().GetEnumerator();
        while (mine.MoveNext() && theirs.MoveNext())
            theirs.Current.Data.CopyTo(mine.Current.Data.AsSpan());
    }
}
