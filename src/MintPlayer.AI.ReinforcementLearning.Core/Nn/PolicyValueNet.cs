using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// A two-headed policy/value network: a shared <b>variable-depth</b> ReLU trunk over an observation, a policy head
/// (one logit per action) and a scalar value head. It is the shared core of the imitation policy nets
/// (<c>CubePolicyNet</c>, <c>RushHourPolicyNet</c>), which wrap it to add their environment sizes and checkpoint
/// kind. The trunk is a <see cref="Linear"/><c>[]</c> (not a fixed pair) so the net can grow deeper as well as
/// wider mid-training via the function-preserving <see cref="Net2Net"/> transforms.
/// </summary>
public sealed class PolicyValueNet
{
    private readonly Linear[] _trunk;
    private readonly Linear _policyHead, _valueHead;
    private readonly int _inputSize, _actions;

    public PolicyValueNet(int inputSize, int[] hidden, int actions, Xoshiro256StarStar rng)
    {
        if (hidden.Length < 1) throw new ArgumentException("A policy/value net needs at least one trunk layer.");
        _inputSize = inputSize;
        _actions = actions;
        _trunk = new Linear[hidden.Length];
        int prev = inputSize;
        for (int i = 0; i < hidden.Length; i++)
        {
            _trunk[i] = new Linear(prev, hidden[i], rng, Activation.Relu);
            prev = hidden[i];
        }
        _policyHead = new Linear(prev, actions, rng, Activation.None);
        _valueHead = new Linear(prev, 1, rng, Activation.None);
    }

    public int InputSize => _inputSize;
    public int Actions => _actions;

    /// <summary>The shared-trunk hidden widths (drives growth schedules).</summary>
    public int[] Trunk => [.. _trunk.Select(l => l.Weight.Cols)];

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B, actions] + value [B, 1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations)
    {
        var x = observations;
        foreach (var layer in _trunk) x = layer.Forward(x).Relu();
        return (_policyHead.Forward(x), _valueHead.Forward(x));
    }

    public IEnumerable<Tensor> Parameters()
    {
        foreach (var layer in _trunk)
            foreach (var p in layer.Parameters()) yield return p;
        foreach (var p in _policyHead.Parameters()) yield return p;
        foreach (var p in _valueHead.Parameters()) yield return p;
    }

    /// <summary>Per-layer activations for one input row, in <see cref="Parameters()"/> order (each trunk layer's
    /// post-ReLU output, then the policy head, then the value head) — for a live-network viewer.</summary>
    public float[][] LayerActivations(Tensor observation)
    {
        var acts = new List<float[]>(_trunk.Length + 2);
        var x = observation;
        foreach (var layer in _trunk) { x = layer.Forward(x).Relu(); acts.Add([.. x.Data]); }
        acts.Add([.. _policyHead.Forward(x).Data]);
        acts.Add([.. _valueHead.Forward(x).Data]);
        return [.. acts];
    }

    /// <summary>Net2WiderNet — a wider net computing the same function (see <see cref="Net2Net"/>).</summary>
    public PolicyValueNet WidenTo(int[] newHidden, Xoshiro256StarStar rng)
    {
        if (newHidden.Length != _trunk.Length) throw new ArgumentException("WidenTo preserves depth; use Deepen to add a layer.");
        for (int i = 0; i < _trunk.Length; i++)
            if (newHidden[i] < _trunk[i].Weight.Cols) throw new ArgumentException("WidenTo cannot shrink a layer.");

        var grown = new PolicyValueNet(_inputSize, newHidden, _actions, rng);
        Net2Net.WidenTrunk(_inputSize, _trunk, Trunk, grown._trunk, newHidden,
            [(_policyHead, grown._policyHead, _actions), (_valueHead, grown._valueHead, 1)], rng);
        return grown;
    }

    /// <summary>Net2DeeperNet — one extra trunk layer (identity-init), same function (see <see cref="Net2Net"/>).</summary>
    public PolicyValueNet Deepen(Xoshiro256StarStar rng)
    {
        int w = _trunk[^1].Weight.Cols;
        var newHidden = new int[_trunk.Length + 1];
        for (int i = 0; i < _trunk.Length; i++) newHidden[i] = _trunk[i].Weight.Cols;
        newHidden[^1] = w;

        var grown = new PolicyValueNet(_inputSize, newHidden, _actions, rng);
        for (int i = 0; i < _trunk.Length; i++) Net2Net.CopyLinear(_trunk[i], grown._trunk[i]);
        Net2Net.SetIdentity(grown._trunk[^1]);
        Net2Net.CopyLinear(_policyHead, grown._policyHead);
        Net2Net.CopyLinear(_valueHead, grown._valueHead);
        return grown;
    }

    /// <summary>The policy path (trunk → policy head) as a standalone <see cref="Mlp"/> — for a GPU-resident forward
    /// over the policy logits (the value head is irrelevant to search). Weights are COPIED (a frozen snapshot).</summary>
    public Mlp PolicyAsMlp()
    {
        int[] sizes = [_inputSize, .. _trunk.Select(l => l.Weight.Cols), _actions];
        var mlp = new Mlp(sizes, new Xoshiro256StarStar(0), Activation.Relu);
        for (int i = 0; i < _trunk.Length; i++) Net2Net.CopyLinear(_trunk[i], mlp.Layers[i]);
        Net2Net.CopyLinear(_policyHead, mlp.Layers[^1]);
        return mlp;
    }

    private IEnumerable<Linear> AllLayers() => [.. _trunk, _policyHead, _valueHead];

    // ── Checkpoint (kind supplied by the wrapper) ─────────────────────────────────────────────────────────────
    // v2: trunk widths (int[]) + every layer's floats. v1 (shipped): a single hidden int → a two-layer trunk.
    private const int Version = 2;

    public void Save(Stream destination, string kind)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, kind, Version);
        CheckpointFormat.WriteInts(writer, Trunk);
        foreach (var layer in AllLayers())
        {
            CheckpointFormat.WriteFloats(writer, layer.Weight.Data);
            CheckpointFormat.WriteFloats(writer, layer.Bias.Data);
        }
    }

    /// <summary>Loads a policy/value net. <paramref name="inputSize"/>/<paramref name="actions"/> come from the
    /// environment (they were never stored in the file); the trunk shape comes from the file (v1: one hidden width →
    /// two layers; v2: an explicit widths array), so grown checkpoints round-trip and shipped v1 files still load.</summary>
    public static PolicyValueNet Load(Stream source, string kind, int inputSize, int actions)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        int version = CheckpointFormat.ReadHeader(reader, kind, Version);
        int[] hidden = version >= 2 ? CheckpointFormat.ReadInts(reader) : [reader.ReadInt32(), 0];
        if (version < 2) hidden[1] = hidden[0]; // v1 stored one hidden width for a fixed two-layer trunk

        var net = new PolicyValueNet(inputSize, hidden, actions, new Xoshiro256StarStar(0));
        foreach (var layer in net.AllLayers())
        {
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Weight.Data.AsSpan());
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Bias.Data.AsSpan());
        }
        return net;
    }
}
