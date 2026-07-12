using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Dueling Q-network (Wang et al. 2016): a shared ReLU trunk splits into a scalar <b>state-value</b> stream
/// V(s) and a per-action <b>advantage</b> stream A(s,a), recombined as
/// <c>Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))</c>. Subtracting the advantage mean fixes the identifiability
/// of the V/A decomposition (otherwise a constant could shift freely between the two streams) and is what
/// makes dueling train stably. The benefit: the value of a state is learned once, shared across all actions,
/// so states where the action choice barely matters are evaluated more sample-efficiently than a plain MLP
/// that must learn each Q(s,a) independently.
/// <para>
/// Implements <see cref="IValueNet"/> — the same [B,in]→[B,out] forward / parameter-sync / structural-clone
/// contract as <see cref="Mlp"/> — so it drops into the DQN trainer (and its target-net sync) unchanged.
/// </para>
/// </summary>
public sealed class DuelingQNet : IValueNet
{
    private readonly Linear[] _trunk;
    private readonly IModule _valueHead;     // → [B,1]       (Linear, or NoisyLinear when noisy)
    private readonly IModule _advantageHead; // → [B,actions] (Linear, or NoisyLinear when noisy)
    private readonly int _actions;
    private readonly int[] _hidden;
    private readonly int _inputSize;
    private readonly bool _noisy;
    private readonly Tensor _ones;           // constant [1,actions], broadcasts (V − meanA) across actions

    public DuelingQNet(int inputSize, int[] hidden, int actions, Xoshiro256StarStar rng, bool noisy = false)
    {
        if (hidden.Length < 1)
            throw new ArgumentException("A dueling Q-net needs at least one shared hidden layer for the two heads.");
        _inputSize = inputSize;
        _actions = actions;
        _hidden = [.. hidden];
        _noisy = noisy;

        _trunk = new Linear[hidden.Length];
        int prev = inputSize;
        for (int i = 0; i < hidden.Length; i++)
        {
            _trunk[i] = new Linear(prev, hidden[i], rng, Activation.Relu);
            prev = hidden[i];
        }
        // NoisyNets exploration lives on the heads (trunk stays plain — heads-only is standard and sufficient).
        _valueHead = noisy ? new NoisyLinear(prev, 1, rng) : new Linear(prev, 1, rng, Activation.None);
        _advantageHead = noisy ? new NoisyLinear(prev, actions, rng) : new Linear(prev, actions, rng, Activation.None);

        var ones = new float[actions];
        Array.Fill(ones, 1f);
        _ones = new Tensor(ones, 1, actions); // RequiresGrad = false (a constant)
    }

    public int InputSize => _inputSize;

    /// <summary>True when the heads are <see cref="NoisyLinear"/> (learned exploration instead of ε-greedy).</summary>
    public bool Noisy => _noisy;
    public int Actions => _actions;
    public int[] HiddenSizes => [.. _hidden];

    public Tensor Forward(Tensor input)
    {
        var x = input;
        foreach (var layer in _trunk)
            x = layer.Forward(x).Relu();

        var value = _valueHead.Forward(x);        // [B,1]
        var advantage = _advantageHead.Forward(x); // [B,actions]
        // meanA over actions, per row → [B,1].
        var meanAdvantage = advantage.SumRows().MulScalar(1f / _actions).Reshape(input.Rows, 1);
        // Q = A + (V − meanA) broadcast across the action columns (outer product with the ones row).
        return advantage.Add(value.Sub(meanAdvantage).MatMul(_ones));
    }

    public IEnumerable<Tensor> Parameters()
    {
        foreach (var layer in _trunk)
            foreach (var p in layer.Parameters()) yield return p;
        foreach (var p in _valueHead.Parameters()) yield return p;
        foreach (var p in _advantageHead.Parameters()) yield return p;
    }

    /// <summary>
    /// Per-layer activations for a single input row, in <see cref="Parameters()"/> layer order: each trunk layer's
    /// post-ReLU output, then the value head V(s) and the advantage head A(s,·). Lets a telemetry viewer show every
    /// hidden neuron's current value (not just input/output). Read-only — allocates throwaway intermediates and
    /// never touches parameters, so it's safe to call while training runs.
    /// </summary>
    public float[][] LayerActivations(Tensor input)
    {
        var acts = new List<float[]>(_trunk.Length + 2);
        var x = input;
        foreach (var layer in _trunk)
        {
            x = layer.Forward(x).Relu();
            acts.Add([.. x.Data]);
        }
        acts.Add([.. _valueHead.Forward(x).Data]);
        acts.Add([.. _advantageHead.Forward(x).Data]);
        return [.. acts];
    }

    public IValueNet CloneStructure() => new DuelingQNet(_inputSize, _hidden, _actions, new Xoshiro256StarStar(0), _noisy);

    /// <summary>The shared-trunk hidden widths — used to drive progressive growth schedules.</summary>
    public int[] Trunk => [.. _hidden];

    /// <summary>
    /// Net2WiderNet (Chen et al. 2016): a wider net that computes the <b>exact same function</b>. Each widened
    /// trunk layer's new units duplicate a randomly-chosen existing unit; the next layer's incoming weights from a
    /// duplicated unit are split evenly across its copies so every downstream sum is unchanged. Lets training add
    /// capacity mid-run without the loss spike a cold-init would cause. Plain (non-noisy) nets only.
    /// </summary>
    public DuelingQNet WidenTo(int[] newHidden, Xoshiro256StarStar rng)
    {
        if (_noisy) throw new NotSupportedException("WidenTo is only supported for non-noisy DuelingQNet.");
        if (newHidden.Length != _hidden.Length) throw new ArgumentException("WidenTo preserves depth; use Deepen to add a layer.");
        for (int i = 0; i < _hidden.Length; i++)
            if (newHidden[i] < _hidden[i]) throw new ArgumentException("WidenTo cannot shrink a layer.");

        // Per trunk layer: map each new unit → an old unit (identity for originals, random for the extras) and
        // count how many new units map to each old unit (the function-preserving outgoing-weight split factor).
        var map = new int[_hidden.Length][];
        var count = new int[_hidden.Length][];
        for (int i = 0; i < _hidden.Length; i++)
        {
            int oldW = _hidden[i], newW = newHidden[i];
            var g = new int[newW];
            var cnt = new int[oldW];
            for (int j = 0; j < oldW; j++) { g[j] = j; cnt[j] = 1; }
            for (int j = oldW; j < newW; j++) { int s = rng.NextInt(oldW); g[j] = s; cnt[s]++; }
            map[i] = g; count[i] = cnt;
        }

        var grown = new DuelingQNet(_inputSize, newHidden, _actions, rng, noisy: false);
        for (int i = 0; i < _hidden.Length; i++)
        {
            Linear oldL = _trunk[i], newL = grown._trunk[i];
            int oldOut = _hidden[i], newOut = newHidden[i];
            int newIn = i == 0 ? _inputSize : newHidden[i - 1];
            for (int r = 0; r < newIn; r++)
            {
                int sr = i == 0 ? r : map[i - 1][r];
                float scale = i == 0 ? 1f : 1f / count[i - 1][sr];
                for (int o = 0; o < newOut; o++)
                    newL.Weight.Data[r * newOut + o] = oldL.Weight.Data[sr * oldOut + map[i][o]] * scale;
            }
            for (int o = 0; o < newOut; o++) newL.Bias.Data[o] = oldL.Bias.Data[map[i][o]];
        }
        WidenHeadInput((Linear)_valueHead, (Linear)grown._valueHead, map[^1], count[^1], 1);
        WidenHeadInput((Linear)_advantageHead, (Linear)grown._advantageHead, map[^1], count[^1], _actions);
        return grown;
    }

    // A head reads the last trunk layer's (now widened) output: split each duplicated unit's incoming weight
    // across its copies so the head's pre-activation is unchanged. Output width (and bias) are untouched.
    private static void WidenHeadInput(Linear oldHead, Linear newHead, int[] mapLast, int[] countLast, int outDim)
    {
        for (int r = 0; r < mapLast.Length; r++)
        {
            int sr = mapLast[r];
            float scale = 1f / countLast[sr];
            for (int o = 0; o < outDim; o++)
                newHead.Weight.Data[r * outDim + o] = oldHead.Weight.Data[sr * outDim + o] * scale;
        }
        oldHead.Bias.Data.CopyTo(newHead.Bias.Data.AsSpan());
    }

    /// <summary>
    /// Net2DeeperNet: a deeper net (one extra trunk layer, same width as the last) computing the <b>same
    /// function</b>. The inserted layer is initialized to identity (W = I, b = 0); since it follows a ReLU its
    /// input is already ≥ 0, so ReLU(I·x) = x. Plain (non-noisy) nets only.
    /// </summary>
    public DuelingQNet Deepen(Xoshiro256StarStar rng)
    {
        if (_noisy) throw new NotSupportedException("Deepen is only supported for non-noisy DuelingQNet.");
        int w = _hidden[^1];
        var newHidden = new int[_hidden.Length + 1];
        Array.Copy(_hidden, newHidden, _hidden.Length);
        newHidden[^1] = w;

        var grown = new DuelingQNet(_inputSize, newHidden, _actions, rng, noisy: false);
        for (int i = 0; i < _trunk.Length; i++)
        {
            _trunk[i].Weight.Data.CopyTo(grown._trunk[i].Weight.Data.AsSpan());
            _trunk[i].Bias.Data.CopyTo(grown._trunk[i].Bias.Data.AsSpan());
        }
        var identity = grown._trunk[^1]; // new last trunk layer → identity (function-preserving after a ReLU)
        Array.Clear(identity.Weight.Data);
        for (int k = 0; k < w; k++) identity.Weight.Data[k * w + k] = 1f;
        Array.Clear(identity.Bias.Data);
        CopyLinear((Linear)_valueHead, (Linear)grown._valueHead);
        CopyLinear((Linear)_advantageHead, (Linear)grown._advantageHead);
        return grown;
    }

    private static void CopyLinear(Linear src, Linear dst)
    {
        src.Weight.Data.CopyTo(dst.Weight.Data.AsSpan());
        src.Bias.Data.CopyTo(dst.Bias.Data.AsSpan());
    }

    /// <inheritdoc/>
    public IValueNet GrowInput(int newInputSize)
    {
        // Only the trunk's first (plain) Linear consumes the input; the heads (noisy or plain) are untouched.
        var grown = new DuelingQNet(newInputSize, _hidden, _actions, new Xoshiro256StarStar(0), _noisy);
        NetTransfer.TransferGrownInput(grown, this);
        return grown;
    }

    /// <summary>Draws fresh exploration noise on the noisy heads (no-op for a plain net).</summary>
    public void ResampleNoise(Xoshiro256StarStar rng)
    {
        if (_valueHead is NoisyLinear v) v.ResampleNoise(rng);
        if (_advantageHead is NoisyLinear a) a.ResampleNoise(rng);
    }

    /// <summary>Turns the heads' noise on (training) or off (deterministic eval/serving); no-op for a plain net.</summary>
    public void SetNoiseEnabled(bool enabled)
    {
        if (_valueHead is NoisyLinear v) v.NoiseEnabled = enabled;
        if (_advantageHead is NoisyLinear a) a.NoiseEnabled = enabled;
    }

    /// <summary>
    /// Builds a noisy copy of this (plain) net: trunk + head weights are copied into the noisy net's MEANS
    /// and σ is freshly initialized. With noise off the result is behaviorally identical, so a continued run
    /// merely ADDS learnable exploration to the trained policy instead of cold-starting — the PRD's
    /// "promote-plain→noisy" warm-start. Throws if this net is already noisy.
    /// </summary>
    public DuelingQNet ToNoisy(Xoshiro256StarStar rng)
    {
        if (_noisy)
            throw new InvalidOperationException("This DuelingQNet is already noisy.");

        var noisy = new DuelingQNet(_inputSize, _hidden, _actions, rng, noisy: true);
        for (int i = 0; i < _trunk.Length; i++)
        {
            _trunk[i].Weight.Data.CopyTo(noisy._trunk[i].Weight.Data.AsSpan());
            _trunk[i].Bias.Data.CopyTo(noisy._trunk[i].Bias.Data.AsSpan());
        }
        CopyMeans((Linear)_valueHead, (NoisyLinear)noisy._valueHead);
        CopyMeans((Linear)_advantageHead, (NoisyLinear)noisy._advantageHead);
        return noisy;

        static void CopyMeans(Linear plain, NoisyLinear noisyHead)
        {
            plain.Weight.Data.CopyTo(noisyHead.MeanWeight.Data.AsSpan());
            plain.Bias.Data.CopyTo(noisyHead.MeanBias.Data.AsSpan());
        }
    }

    /// <summary>Copies every parameter from a structurally identical net (target-network sync).</summary>
    public void CopyFrom(IValueNet source) => NetTransfer.CopyParameters(this, source);
}
