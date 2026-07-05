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

    /// <inheritdoc/>
    public IValueNet GrowInput(int newInputSize)
    {
        // Only the input projection consumes the observation; blocks and head are width-bound and copy unchanged.
        var grown = new ResidualMlp(newInputSize, _width, _block1.Length, new Xoshiro256StarStar(0));
        NetTransfer.TransferGrownInput(grown, this);
        return grown;
    }

    /// <summary>
    /// Net2WiderNet (Chen et al. 2016) growth: a NEW residual net of width <paramref name="newWidth"/> whose
    /// learned function starts (near-)identical to this one, so a campaign can train cheap at a small width and
    /// add capacity on demand instead of paying the wide GEMM from the start. Trunk units are replicated
    /// round-robin (<c>src = j % width</c>); a unit-producing weight/bias/γ/β copies its source column, a
    /// unit-consuming weight divides by the source's replication count so the next matmul's sum is unchanged.
    /// <para>
    /// Exactness vs LayerNorm: LN normalizes across the trunk, so duplicating units only leaves its mean/variance
    /// untouched when replication is <b>uniform</b> — i.e. when <paramref name="newWidth"/> is an integer multiple
    /// of the current width (e.g. 512→1024). Then the transfer is function-preserving to fp error. For a
    /// non-multiple, replication is off by one on some units and the function is perturbed slightly (a warm
    /// start, not exact). <paramref name="symmetryNoise"/> adds a tiny relative jitter to the duplicated units so
    /// they receive distinct gradients and can diverge — without it the copies stay tied and add no real capacity.
    /// Pass 0 to verify exact preservation.
    /// </para>
    /// </summary>
    public ResidualMlp WidenTo(int newWidth, Xoshiro256StarStar rng, float symmetryNoise = 1e-3f)
    {
        if (newWidth <= _width) throw new ArgumentException($"WidenTo expects a larger width than the current {_width}, got {newWidth}.");
        int w = _width, w2 = newWidth;

        int[] src = new int[w2];
        int[] rep = new int[w];                       // replication count per source unit
        for (int j = 0; j < w2; j++) { src[j] = j % w; rep[src[j]] = rep[src[j]] + 1; }
        bool isCopy(int j) => j >= w;                 // round-robin ⇒ [0,w) are the originals, [w,w2) the duplicates

        var grown = new ResidualMlp(_inputSize, w2, _block1.Length, rng);

        // Produce the widened dimension: copy source column j%w (+ jitter on duplicates).
        void WidenOut(Linear oldL, Linear newL)
        {
            int inDim = oldL.Weight.Rows;
            for (int c2 = 0; c2 < w2; c2++)
            {
                int s = src[c2];
                for (int r = 0; r < inDim; r++)
                {
                    float v = oldL.Weight.Data[r * w + s];
                    newL.Weight.Data[r * w2 + c2] = isCopy(c2) ? v * (1f + symmetryNoise * (2f * (float)rng.NextDouble() - 1f)) : v;
                }
                newL.Bias.Data[c2] = oldL.Bias.Data[s];
            }
        }
        // Consume the widened dimension: copy source row j%w, divided by its replication count (sum-preserving).
        void WidenIn(Linear oldL, Linear newL)
        {
            int outDim = oldL.Weight.Cols;
            for (int r2 = 0; r2 < w2; r2++)
            {
                int s = src[r2];
                float scale = 1f / rep[s];
                for (int c = 0; c < outDim; c++) newL.Weight.Data[r2 * outDim + c] = oldL.Weight.Data[s * outDim + c] * scale;
            }
            oldL.Bias.Data.CopyTo(newL.Bias.Data.AsSpan()); // bias is on the (unchanged) output dim
        }
        // γ/β live on the trunk: copy source unit (no scaling — LayerNorm is invariant to uniform duplication).
        void WidenNorm(Tensor oldG, Tensor oldB, Tensor newG, Tensor newB)
        {
            for (int j = 0; j < w2; j++) { newG.Data[j] = oldG.Data[src[j]]; newB.Data[j] = oldB.Data[src[j]]; }
        }

        WidenOut(_inProj, grown._inProj);                       // inputSize → w produces the trunk
        WidenNorm(_inGamma, _inBeta, grown._inGamma, grown._inBeta);
        for (int i = 0; i < _block1.Length; i++)
        {
            // block1: consumes the trunk (in) AND produces the block's hidden (out) — widen both axes.
            WidenBoth(_block1[i], grown._block1[i]);
            WidenNorm(_blockGamma[i], _blockBeta[i], grown._blockGamma[i], grown._blockBeta[i]);
            WidenBoth(_block2[i], grown._block2[i]);            // block2: hidden (in) → trunk (out)
        }
        WidenIn(_head, grown._head);                            // trunk → scalar consumes only

        return grown;

        // Both axes widened (a w×w trunk weight): rows consume (÷rep), columns produce (+jitter on dup cols).
        void WidenBoth(Linear oldL, Linear newL)
        {
            for (int r2 = 0; r2 < w2; r2++)
            {
                int sr = src[r2];
                float rowScale = 1f / rep[sr];
                for (int c2 = 0; c2 < w2; c2++)
                {
                    int sc = src[c2];
                    float v = oldL.Weight.Data[sr * w + sc] * rowScale;
                    newL.Weight.Data[r2 * w2 + c2] = isCopy(c2) ? v * (1f + symmetryNoise * (2f * (float)rng.NextDouble() - 1f)) : v;
                }
            }
            for (int c2 = 0; c2 < w2; c2++) newL.Bias.Data[c2] = oldL.Bias.Data[src[c2]];
        }
    }

    public void CopyFrom(IValueNet source) => NetTransfer.CopyParameters(this, source);
}
