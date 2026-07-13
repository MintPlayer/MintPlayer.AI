using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// An AlphaZero-style two-headed policy/value network over a spatial board (PLAN M42): the observation is a stack of
/// <c>planes</c> feature maps of size <c>boardH×boardW</c>, carried as a rank-2 [B, planes·H·W] tensor. A 3×3 conv
/// stem lifts it to <c>filters</c> channels, a tower of <c>blocks</c> residual blocks
/// (<c>x → relu(x + LN(conv(relu(LN(conv(x))))))</c>) processes it at constant H×W, then a 1×1-conv policy head → a
/// linear to the flat action space, and a 1×1-conv value head → two linears to a scalar. Every spatial layer keeps
/// H×W (stride 1, pad 1 for 3×3, pad 0 for 1×1), so the residual skip is always shape-matched.
/// <para>
/// Unlike the flat <see cref="PolicyValueNet"/>, this preserves the board's 2-D structure, so it can learn local
/// piece-interaction patterns (the lever the flat MLP lacked — see PLAN M40.4 plateau). Normalization is
/// <see cref="Tensor.LayerNorm"/> over each sample's whole feature map (not BatchNorm), matching the repo's
/// deliberate choice for stability under a moving target (<see cref="ResidualMlp"/>). The value head stays linear;
/// the trainer applies <c>tanh</c>. Implements <see cref="IPolicyValueNet"/>, so it drops into the self-play stack in
/// place of the flat net.
/// </para>
/// </summary>
public sealed class ConvResidualPolicyValueNet : IPolicyValueNet
{
    private readonly int _planes, _h, _w, _actions, _filters, _blocks;

    private readonly Conv _stem;
    private readonly Norm _stemNorm;
    private readonly Conv[] _conv1, _conv2;      // the two convs of each residual block
    private readonly Norm[] _norm1, _norm2;
    private readonly Conv _policyConv;           // 1×1 filters→2
    private readonly Norm _policyNorm;
    private readonly Linear _policyHead;         // 2·H·W → actions
    private readonly Conv _valueConv;            // 1×1 filters→1
    private readonly Norm _valueNorm;
    private readonly Linear _valueHidden, _valueHead; // H·W → filters → 1

    private const int PolicyHeadChannels = 2;

    public ConvResidualPolicyValueNet(int planes, int boardH, int boardW, int actions, int filters, int blocks,
        Xoshiro256StarStar rng)
    {
        if (blocks < 1) throw new ArgumentException("A residual tower needs at least one block.");
        if (filters < 1) throw new ArgumentException("A conv net needs at least one filter.");
        _planes = planes; _h = boardH; _w = boardW; _actions = actions; _filters = filters; _blocks = blocks;
        int hw = boardH * boardW;

        _stem = new Conv(planes, filters, k: 3, pad: 1, rng);
        _stemNorm = new Norm(filters * hw);

        _conv1 = new Conv[blocks]; _conv2 = new Conv[blocks];
        _norm1 = new Norm[blocks]; _norm2 = new Norm[blocks];
        for (int i = 0; i < blocks; i++)
        {
            _conv1[i] = new Conv(filters, filters, k: 3, pad: 1, rng);
            _norm1[i] = new Norm(filters * hw);
            _conv2[i] = new Conv(filters, filters, k: 3, pad: 1, rng);
            _norm2[i] = new Norm(filters * hw);
        }

        _policyConv = new Conv(filters, PolicyHeadChannels, k: 1, pad: 0, rng);
        _policyNorm = new Norm(PolicyHeadChannels * hw);
        _policyHead = new Linear(PolicyHeadChannels * hw, actions, rng, Activation.None);

        _valueConv = new Conv(filters, 1, k: 1, pad: 0, rng);
        _valueNorm = new Norm(hw);
        _valueHidden = new Linear(hw, filters, rng, Activation.Relu);
        _valueHead = new Linear(filters, 1, rng, Activation.None);
    }

    public int InputSize => _planes * _h * _w;
    public int Actions => _actions;
    public string Describe() => $"conv {_filters}f×{_blocks}b ({_planes}×{_h}×{_w})";

    public (Tensor Logits, Tensor Value) Forward(Tensor observations)
    {
        var x = _stem.Forward(observations, _h, _w).LayerNorm(_stemNorm.Gamma, _stemNorm.Beta).Relu();
        for (int i = 0; i < _blocks; i++)
        {
            var h = _conv1[i].Forward(x, _h, _w).LayerNorm(_norm1[i].Gamma, _norm1[i].Beta).Relu();
            h = _conv2[i].Forward(h, _h, _w).LayerNorm(_norm2[i].Gamma, _norm2[i].Beta);
            x = x.Add(h).Relu();
        }

        var p = _policyConv.Forward(x, _h, _w).LayerNorm(_policyNorm.Gamma, _policyNorm.Beta).Relu();
        var logits = _policyHead.Forward(p);

        var v = _valueConv.Forward(x, _h, _w).LayerNorm(_valueNorm.Gamma, _valueNorm.Beta).Relu();
        var value = _valueHead.Forward(_valueHidden.Forward(v).Relu());
        return (logits, value);
    }

    public float[][] LayerActivations(Tensor observation)
    {
        var acts = new List<float[]>(2 + _blocks);
        var x = _stem.Forward(observation, _h, _w).LayerNorm(_stemNorm.Gamma, _stemNorm.Beta).Relu();
        acts.Add([.. x.Data]);
        for (int i = 0; i < _blocks; i++)
        {
            var h = _conv1[i].Forward(x, _h, _w).LayerNorm(_norm1[i].Gamma, _norm1[i].Beta).Relu();
            h = _conv2[i].Forward(h, _h, _w).LayerNorm(_norm2[i].Gamma, _norm2[i].Beta);
            x = x.Add(h).Relu();
            acts.Add([.. x.Data]);
        }
        var (logits, value) = Forward(observation);
        acts.Add([.. logits.Data]);
        acts.Add([.. value.Data]);
        return [.. acts];
    }

    // Enumerated in a fixed order so the optimizer and the checkpoint round-trip agree.
    public IEnumerable<Tensor> Parameters()
    {
        foreach (var p in _stem.Parameters()) yield return p;
        yield return _stemNorm.Gamma; yield return _stemNorm.Beta;
        for (int i = 0; i < _blocks; i++)
        {
            foreach (var p in _conv1[i].Parameters()) yield return p;
            yield return _norm1[i].Gamma; yield return _norm1[i].Beta;
            foreach (var p in _conv2[i].Parameters()) yield return p;
            yield return _norm2[i].Gamma; yield return _norm2[i].Beta;
        }
        foreach (var p in _policyConv.Parameters()) yield return p;
        yield return _policyNorm.Gamma; yield return _policyNorm.Beta;
        foreach (var p in _policyHead.Parameters()) yield return p;
        foreach (var p in _valueConv.Parameters()) yield return p;
        yield return _valueNorm.Gamma; yield return _valueNorm.Beta;
        foreach (var p in _valueHidden.Parameters()) yield return p;
        foreach (var p in _valueHead.Parameters()) yield return p;
    }

    // ── Checkpoint ────────────────────────────────────────────────────────────────────────────────────────────────
    // v1: [planes, boardH, boardW, filters, blocks] then every parameter's floats in Parameters() order.
    private const int Version = 1;

    public void Save(Stream destination, string kind)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, kind, Version);
        CheckpointFormat.WriteInts(writer, [_planes, _h, _w, _filters, _blocks]);
        foreach (var p in Parameters()) CheckpointFormat.WriteFloats(writer, p.Data);
    }

    /// <summary>Loads a conv policy/value net. <paramref name="actions"/> comes from the environment (never stored);
    /// the spatial + tower shape comes from the file, so a shipped checkpoint reconstructs exactly.</summary>
    public static ConvResidualPolicyValueNet Load(Stream source, string kind, int actions)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, kind, Version);
        int[] dims = CheckpointFormat.ReadInts(reader); // planes, H, W, filters, blocks
        var net = new ConvResidualPolicyValueNet(dims[0], dims[1], dims[2], actions, dims[3], dims[4],
            new Xoshiro256StarStar(0));
        foreach (var p in net.Parameters())
            CheckpointFormat.ReadFloats(reader).CopyTo(p.Data.AsSpan());
        return net;
    }

    // ── Building blocks ───────────────────────────────────────────────────────────────────────────────────────────
    // A single conv layer (weight [inC·k·k, outC] + bias [outC]) that runs through the Conv2D autograd op. Spatial
    // size is preserved (pad = (k-1)/2 for odd k, stride 1), so shapes chain and the residual skip stays aligned.
    private sealed class Conv
    {
        private readonly Tensor _weight, _bias;
        private readonly int _inC, _outC, _k, _pad;

        public Conv(int inC, int outC, int k, int pad, Xoshiro256StarStar rng)
        {
            float std = MathF.Sqrt(2f / (inC * k * k)); // He init (ReLU)
            _weight = new Tensor(Tensor.RandomNormal(rng, 0f, std, inC * k * k, outC).Data, inC * k * k, outC) { RequiresGrad = true };
            _bias = new Tensor(new float[outC], outC) { RequiresGrad = true };
            _inC = inC; _outC = outC; _k = k; _pad = pad;
        }

        public Tensor Forward(Tensor x, int h, int w) => x.Conv2D(_weight, _bias, _inC, h, w, _outC, _k, _k, 1, _pad);

        public IEnumerable<Tensor> Parameters() { yield return _weight; yield return _bias; }
    }

    // LayerNorm scale/shift over a whole flattened feature map (γ=1, β=0), both learnable [size].
    private sealed class Norm
    {
        public Tensor Gamma { get; }
        public Tensor Beta { get; }

        public Norm(int size)
        {
            Gamma = new Tensor(new float[size], size) { RequiresGrad = true };
            Array.Fill(Gamma.Data, 1f);
            Beta = new Tensor(new float[size], size) { RequiresGrad = true };
        }
    }
}
