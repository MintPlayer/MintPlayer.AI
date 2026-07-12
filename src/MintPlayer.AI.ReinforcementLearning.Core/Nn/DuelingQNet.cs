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
