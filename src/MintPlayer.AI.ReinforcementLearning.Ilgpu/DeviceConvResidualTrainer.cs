using ILGPU;
using ILGPU.Runtime;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// A fully device-resident AlphaZero training step for a <see cref="ConvResidualPolicyValueNet"/> (PLAN M44): the net's
/// weights, Adam moments, gradients and all forward activations live on the GPU, so one train step — resident forward
/// (caching x̂/σ per LayerNorm + the post-ReLU maps + the im2col columns per conv), the full two-headed backward
/// (soft-CE policy + tanh-MSE value), a global-norm gradient clip, and the Adam update — runs with no host↔device
/// transfer beyond uploading the batch (observations + π + z) and downloading the two heads (for the loss). It is the
/// training-side sibling of <see cref="DeviceConvPolicyValueNet"/> (the M43 inference forward) and the two-headed conv
/// analogue of <see cref="DeviceResidualTrainer"/>.
/// <para>
/// Weights are mastered here; <see cref="SyncToHost()"/> writes them back into the CPU
/// <see cref="ConvResidualPolicyValueNet"/> the campaign holds (for eval, arena, the ladder, and checkpointing). Adam +
/// clip mirror the managed <see cref="Adam"/> exactly (same β, ε, bias correction, <c>maxNorm/(norm+1e-6)</c> clip).
/// GPU reductions aren't bitwise-reproducible, so this path is opt-in and the CPU autograd step stays the deterministic
/// reference; correctness is pinned by a gradient-parity test vs autograd (see the tests), not by bit-equality.
/// </para>
/// </summary>
public sealed class DeviceConvResidualTrainer : IPolicyValueTrainStep, IDisposable
{
    private const int PolicyChannels = 2; // must match ConvResidualPolicyValueNet.PolicyHeadChannels

    private sealed class Param : IDisposable
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense> W, G, M, V;
        public Param(Accelerator acc, long len)
        {
            W = acc.Allocate1D<float>(len); G = acc.Allocate1D<float>(len);
            M = acc.Allocate1D<float>(len); V = acc.Allocate1D<float>(len);
            M.MemSetToZero(); V.MemSetToZero();
        }
        public void Dispose() { W.Dispose(); G.Dispose(); M.Dispose(); V.Dispose(); }
    }

    private readonly IlgpuBackend _backend;
    private readonly ConvResidualPolicyValueNet _hostNet;
    private readonly int _planes, _h, _w, _hw, _filters, _blocks, _actions, _batch, _m; // _m = batch·hw (conv GEMM rows)
    private readonly int _fhw, _phw;                                                     // filters·hw, policyChannels·hw
    private readonly float _lr, _beta1, _beta2, _eps, _clipNorm, _valueScale;            // _valueScale = valueWeight·2/batch
    private int _step;

    // Parameters in ConvResidualPolicyValueNet.Parameters() order.
    private readonly Param _stemW, _stemB, _stemNG, _stemNB;
    private readonly Param[] _b1W, _b1B, _n1G, _n1B, _b2W, _b2B, _n2G, _n2B;
    private readonly Param _pConvW, _pConvB, _pNG, _pNB, _pHeadW, _pHeadB;
    private readonly Param _vConvW, _vConvB, _vNG, _vNB, _vHidW, _vHidB, _vHeadW, _vHeadB;
    private readonly List<Param> _all = [];

    // Forward caches (per-conv im2col columns; per-LN x̂/1σ; post-ReLU maps) — inputs the backward needs.
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _colsStem, _colsP, _colsV;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _cols1, _cols2;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _xhatStem, _invStem, _xhatP, _invP, _xhatV, _invV;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _xhat1, _inv1, _xhat2, _inv2;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _x0, _pRelu, _vRelu, _vHidRelu;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _xOut, _h1;

    // Working buffers (transient within a step).
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _input, _stream, _hbuf, _h2buf, _pbuf, _vbuf, _vhid, _value, _logits;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dLogits, _dValue, _dP, _dV, _dVHid, _dConvTmp;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _matScratch, _dMatScratch, _dColsScratch, _dx, _dxAdd, _dH, _normAccum;

    private const int SumSqPartitions = 256;

    internal DeviceConvResidualTrainer(IlgpuBackend backend, ConvResidualPolicyValueNet net, int batch,
        float learningRate, float clipNorm, int actions, float valueWeight, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        _backend = backend; _hostNet = net;
        _planes = net.Planes; _h = net.BoardH; _w = net.BoardW; _hw = _h * _w;
        _filters = net.Filters; _blocks = net.Blocks; _actions = actions; _batch = batch;
        _m = batch * _hw; _fhw = _filters * _hw; _phw = PolicyChannels * _hw;
        _lr = learningRate; _clipNorm = clipNorm; _beta1 = beta1; _beta2 = beta2; _eps = eps;
        _valueScale = valueWeight * 2f / batch;

        var acc = backend.Accelerator;
        int F = _filters, hw = _hw, B = batch, M = _m;
        MemoryBuffer1D<float, Stride1D.Dense> Buf(long len) => acc.Allocate1D<float>(len);
        MemoryBuffer1D<float, Stride1D.Dense>[] Arr() => new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];

        lock (backend.DeviceLock)
        {
            // Parameters, in Parameters() order (weight then bias per conv; γ then β per norm).
            Param Pm(long len) { var p = new Param(acc, len); _all.Add(p); return p; }
            _stemW = Pm((long)_planes * 9 * F); _stemB = Pm(F); _stemNG = Pm(_fhw); _stemNB = Pm(_fhw);
            _b1W = new Param[_blocks]; _b1B = new Param[_blocks]; _n1G = new Param[_blocks]; _n1B = new Param[_blocks];
            _b2W = new Param[_blocks]; _b2B = new Param[_blocks]; _n2G = new Param[_blocks]; _n2B = new Param[_blocks];
            for (int i = 0; i < _blocks; i++)
            {
                _b1W[i] = Pm((long)F * 9 * F); _b1B[i] = Pm(F); _n1G[i] = Pm(_fhw); _n1B[i] = Pm(_fhw);
                _b2W[i] = Pm((long)F * 9 * F); _b2B[i] = Pm(F); _n2G[i] = Pm(_fhw); _n2B[i] = Pm(_fhw);
            }
            _pConvW = Pm((long)F * PolicyChannels); _pConvB = Pm(PolicyChannels); _pNG = Pm(_phw); _pNB = Pm(_phw);
            _pHeadW = Pm((long)_phw * _actions); _pHeadB = Pm(_actions);
            _vConvW = Pm(F); _vConvB = Pm(1); _vNG = Pm(hw); _vNB = Pm(hw);
            _vHidW = Pm((long)hw * F); _vHidB = Pm(F); _vHeadW = Pm(F); _vHeadB = Pm(1);

            UploadWeights(net);

            // Forward caches.
            _colsStem = Buf((long)M * _planes * 9); _cols1 = Arr(); _cols2 = Arr();
            _colsP = Buf((long)M * F); _colsV = Buf((long)M * F);
            _xhatStem = Buf((long)B * _fhw); _invStem = Buf(B);
            _xhat1 = Arr(); _inv1 = Arr(); _xhat2 = Arr(); _inv2 = Arr();
            _xhatP = Buf((long)B * _phw); _invP = Buf(B); _xhatV = Buf((long)B * hw); _invV = Buf(B);
            _x0 = Buf((long)B * _fhw); _xOut = Arr(); _h1 = Arr();
            _pRelu = Buf((long)B * _phw); _vRelu = Buf((long)B * hw); _vHidRelu = Buf((long)B * F);
            for (int i = 0; i < _blocks; i++)
            {
                _cols1[i] = Buf((long)M * F * 9); _cols2[i] = Buf((long)M * F * 9);
                _xhat1[i] = Buf((long)B * _fhw); _inv1[i] = Buf(B); _xhat2[i] = Buf((long)B * _fhw); _inv2[i] = Buf(B);
                _xOut[i] = Buf((long)B * _fhw); _h1[i] = Buf((long)B * _fhw);
            }

            // Working buffers. Scratch is sized to the largest shape and sub-viewed for smaller convs/heads.
            int maxInCk2 = Math.Max(_planes, F) * 9;
            _input = Buf((long)B * _planes * hw); _stream = Buf((long)B * _fhw);
            _hbuf = Buf((long)B * _fhw); _h2buf = Buf((long)B * _fhw);
            _pbuf = Buf((long)B * _phw); _vbuf = Buf((long)B * hw); _vhid = Buf((long)B * F);
            _value = Buf(B); _logits = Buf((long)B * _actions);
            _dLogits = Buf((long)B * _actions); _dValue = Buf(B);
            _dP = Buf((long)B * _phw); _dV = Buf((long)B * hw); _dVHid = Buf((long)B * F); _dConvTmp = Buf((long)B * _fhw);
            _matScratch = Buf((long)M * F); _dMatScratch = Buf((long)M * F); _dColsScratch = Buf((long)M * maxInCk2);
            _dx = Buf((long)B * _fhw); _dxAdd = Buf((long)B * _fhw); _dH = Buf((long)B * _fhw); _normAccum = Buf(1);
        }
    }

    private void UploadWeights(ConvResidualPolicyValueNet net)
    {
        using var p = net.Parameters().GetEnumerator();
        foreach (var param in _all) { p.MoveNext(); param.W.CopyFromCPU(p.Current.Data); }
    }

    public void SyncToHost() => SyncToHost(_hostNet);

    /// <summary>Write the resident (trained) weights back into the CPU net — for eval, arena, the ladder, checkpointing.</summary>
    public void SyncToHost(ConvResidualPolicyValueNet net)
    {
        lock (_backend.DeviceLock)
        {
            _backend.Accelerator.Synchronize();
            using var p = net.Parameters().GetEnumerator();
            foreach (var param in _all) { p.MoveNext(); param.W.CopyToCPU(p.Current.Data); }
        }
    }

    public (double PolicyLoss, double ValueLoss) Step(float[] obs, float[] policyTargets, float[] valueTargets, int batch)
    {
        if (batch != _batch) throw new ArgumentException($"{nameof(DeviceConvResidualTrainer)} is fixed to batch {_batch}, got {batch}.");
        lock (_backend.DeviceLock)
        {
            var loss = ForwardBackward(obs, policyTargets, valueTargets);
            ClipAndUpdate();
            return loss;
        }
    }

    /// <summary>Test hook: forward+backward only, then download gradients in Parameters() order (for the parity test).</summary>
    internal float[][] DebugGradients(float[] obs, float[] policyTargets, float[] valueTargets)
    {
        lock (_backend.DeviceLock)
        {
            ForwardBackward(obs, policyTargets, valueTargets);
            _backend.Accelerator.Synchronize();
            var grads = new float[_all.Count][];
            for (int i = 0; i < _all.Count; i++) { grads[i] = new float[_all[i].G.Length]; _all[i].G.CopyToCPU(grads[i]); }
            return grads;
        }
    }

    // Resident forward (caching activations) + host-computed head-loss grads + full two-headed backward, leaving
    // gradients in each param's G. Returns the batch's (policy CE, value MSE) losses. Caller holds the lock.
    private (double PolicyLoss, double ValueLoss) ForwardBackward(float[] obs, float[] pi, float[] z)
    {
        var b = _backend;
        int F = _filters, hw = _hw, M = _m, rows = _batch, act = _actions, ph = _phw, fhw = _fhw;
        _input.CopyFromCPU(obs);

        // ── FORWARD ──
        // Stem: 3×3 conv (planes→F) → LN(train) → ReLU → x0.
        ConvFwd(_input.View, _stemW.W.View, _stemB.W.View, _stream.View, _colsStem.View, _planes, F, 3, 1);
        b.LaunchLayerNormTrain(_stream.View, _stemNG.W.View, _stemNB.W.View, _xhatStem.View, _invStem.View, rows, fhw);
        b.LaunchRelu(_stream.View);
        b.LaunchCopy(_x0.View, _stream.View);

        // Residual tower: x = ReLU(x + LN(conv2(ReLU(LN(conv1 x))))).
        for (int i = 0; i < _blocks; i++)
        {
            ConvFwd(_stream.View, _b1W[i].W.View, _b1B[i].W.View, _hbuf.View, _cols1[i].View, F, F, 3, 1);
            b.LaunchLayerNormTrain(_hbuf.View, _n1G[i].W.View, _n1B[i].W.View, _xhat1[i].View, _inv1[i].View, rows, fhw);
            b.LaunchRelu(_hbuf.View);
            b.LaunchCopy(_h1[i].View, _hbuf.View);
            ConvFwd(_hbuf.View, _b2W[i].W.View, _b2B[i].W.View, _h2buf.View, _cols2[i].View, F, F, 3, 1);
            b.LaunchLayerNormTrain(_h2buf.View, _n2G[i].W.View, _n2B[i].W.View, _xhat2[i].View, _inv2[i].View, rows, fhw);
            b.LaunchAddInto(_stream.View, _h2buf.View);
            b.LaunchRelu(_stream.View);
            b.LaunchCopy(_xOut[i].View, _stream.View);
        }

        // Policy head: 1×1 conv (F→2) → LN → ReLU → flatten → Linear(2·hw → actions).
        ConvFwd(_stream.View, _pConvW.W.View, _pConvB.W.View, _pbuf.View, _colsP.View, F, PolicyChannels, 1, 0);
        b.LaunchLayerNormTrain(_pbuf.View, _pNG.W.View, _pNB.W.View, _xhatP.View, _invP.View, rows, ph);
        b.LaunchRelu(_pbuf.View);
        b.LaunchCopy(_pRelu.View, _pbuf.View);
        b.LaunchGemmTiled(_pbuf.View, _pHeadW.W.View, _logits.View, GemmDims.AB(rows, ph, act, 0));
        b.LaunchBiasActivation(_logits.View, _pHeadB.W.View, act, 0);

        // Value head: 1×1 conv (F→1) → LN → ReLU → Linear(hw → F) → ReLU → Linear(F → 1).
        ConvFwd(_stream.View, _vConvW.W.View, _vConvB.W.View, _vbuf.View, _colsV.View, F, 1, 1, 0);
        b.LaunchLayerNormTrain(_vbuf.View, _vNG.W.View, _vNB.W.View, _xhatV.View, _invV.View, rows, hw);
        b.LaunchRelu(_vbuf.View);
        b.LaunchCopy(_vRelu.View, _vbuf.View);
        b.LaunchGemmTiled(_vbuf.View, _vHidW.W.View, _vhid.View, GemmDims.AB(rows, hw, F, 0));
        b.LaunchBiasActivation(_vhid.View, _vHidB.W.View, F, 1); // ReLU
        b.LaunchCopy(_vHidRelu.View, _vhid.View);
        b.LaunchGemmTiled(_vhid.View, _vHeadW.W.View, _value.View, GemmDims.AB(rows, F, 1, 0));
        b.LaunchBiasActivation(_value.View, _vHeadB.W.View, 1, 0); // linear

        // ── LOSS GRADIENTS (host — softmax/tanh stay off the device, matching the forward; heads are tiny) ──
        var loss = ComputeHeadGrads(pi, z);

        // ── BACKWARD: policy head → dx (policy branch) ──
        b.LaunchGemmTiled(_pRelu.View, _dLogits.View, _pHeadW.G.View, GemmDims.AtB(rows, ph, act, 0));
        b.LaunchBiasGrad(_dLogits.View, _pHeadB.G.View, rows, act);
        b.LaunchGemmTiled(_dLogits.View, _pHeadW.W.View, _dP.View, GemmDims.ABt(rows, ph, act, 0)); // dP[rows,ph]=dLogits·pHeadWᵀ (shared=act)
        b.LaunchReluBackward(_dP.View, _pRelu.View);
        b.LaunchLayerNormParamGrad(_dP.View, _xhatP.View, _pNG.G.View, _pNB.G.View, rows, ph);
        var dConvP = _dConvTmp.View.SubView(0, (long)rows * ph);
        b.LaunchLayerNormInputGrad(_dP.View, _xhatP.View, _invP.View, _pNG.W.View, dConvP, rows, ph);
        ConvBwd(dConvP, _colsP.View, _pConvW, _pConvB, _dx.View, F, PolicyChannels, 1, 0);

        // ── BACKWARD: value head → dxAdd (value branch), then dx += dxAdd ──
        b.LaunchGemmTiled(_vHidRelu.View, _dValue.View, _vHeadW.G.View, GemmDims.AtB(rows, F, 1, 0));
        b.LaunchBiasGrad(_dValue.View, _vHeadB.G.View, rows, 1);
        b.LaunchGemmTiled(_dValue.View, _vHeadW.W.View, _dVHid.View, GemmDims.ABt(rows, F, 1, 0)); // dVHid[rows,F]=dValue·vHeadWᵀ (shared=1)
        b.LaunchReluBackward(_dVHid.View, _vHidRelu.View);
        b.LaunchGemmTiled(_vRelu.View, _dVHid.View, _vHidW.G.View, GemmDims.AtB(rows, hw, F, 0));
        b.LaunchBiasGrad(_dVHid.View, _vHidB.G.View, rows, F);
        b.LaunchGemmTiled(_dVHid.View, _vHidW.W.View, _dV.View, GemmDims.ABt(rows, hw, F, 0)); // dV[rows,hw]=dVHid·vHidWᵀ (shared=F)
        b.LaunchReluBackward(_dV.View, _vRelu.View);
        b.LaunchLayerNormParamGrad(_dV.View, _xhatV.View, _vNG.G.View, _vNB.G.View, rows, hw);
        var dConvV = _dConvTmp.View.SubView(0, (long)rows * hw);
        b.LaunchLayerNormInputGrad(_dV.View, _xhatV.View, _invV.View, _vNG.W.View, dConvV, rows, hw);
        ConvBwd(dConvV, _colsV.View, _vConvW, _vConvB, _dxAdd.View, F, 1, 1, 0);
        b.LaunchAddInto(_dx.View, _dxAdd.View); // dx = grad wrt tower output (policy + value branches)

        // ── BACKWARD: residual tower ──
        for (int i = _blocks - 1; i >= 0; i--)
        {
            b.LaunchReluBackward(_dx.View, _xOut[i].View);      // through ReLU(x_in + h2); dx now = grad wrt (x_in + h2) = dh2 + skip
            // conv2 path (dh2 = dx)
            b.LaunchLayerNormParamGrad(_dx.View, _xhat2[i].View, _n2G[i].G.View, _n2B[i].G.View, rows, fhw);
            var dConv2 = _dConvTmp.View.SubView(0, (long)rows * fhw);
            b.LaunchLayerNormInputGrad(_dx.View, _xhat2[i].View, _inv2[i].View, _n2G[i].W.View, dConv2, rows, fhw);
            ConvBwd(dConv2, _cols2[i].View, _b2W[i], _b2B[i], _dH.View, F, F, 3, 1); // dH = grad wrt post-ReLU h
            // conv1 path
            b.LaunchReluBackward(_dH.View, _h1[i].View);
            b.LaunchLayerNormParamGrad(_dH.View, _xhat1[i].View, _n1G[i].G.View, _n1B[i].G.View, rows, fhw);
            var dConv1 = _dConvTmp.View.SubView(0, (long)rows * fhw);
            b.LaunchLayerNormInputGrad(_dH.View, _xhat1[i].View, _inv1[i].View, _n1G[i].W.View, dConv1, rows, fhw);
            ConvBwd(dConv1, _cols1[i].View, _b1W[i], _b1B[i], _dxAdd.View, F, F, 3, 1); // dxAdd = conv1-path grad into block input
            b.LaunchAddInto(_dx.View, _dxAdd.View); // dx = skip + conv path = grad wrt block input
        }

        // ── BACKWARD: stem ──
        b.LaunchReluBackward(_dx.View, _x0.View);
        b.LaunchLayerNormParamGrad(_dx.View, _xhatStem.View, _stemNG.G.View, _stemNB.G.View, rows, fhw);
        var dConvStem = _dConvTmp.View.SubView(0, (long)rows * fhw);
        b.LaunchLayerNormInputGrad(_dx.View, _xhatStem.View, _invStem.View, _stemNG.W.View, dConvStem, rows, fhw);
        // stem conv weight/bias grad (no dInput — the input is the observation).
        ConvWeightBiasGrad(dConvStem, _colsStem.View, _stemW, _stemB, _planes, F);
        return loss;
    }

    // Download the two heads, compute their loss gradients on the host (softmax−π for policy, valueWeight·2/B·(tanh−z)
    // (1−tanh²) for value), upload dLogits/dValue for the device backward, and return the (mean-CE, mean-MSE) losses.
    // Mirrors AutogradPolicyValueTrainStep's loss exactly; keeping softmax/tanh on the host matches DeviceConvPolicyValueNet.
    private (double PolicyLoss, double ValueLoss) ComputeHeadGrads(float[] pi, float[] z)
    {
        _backend.Accelerator.Synchronize();
        var logits = new float[(long)_batch * _actions];
        var value = new float[_batch];
        _logits.CopyToCPU(logits);
        _value.CopyToCPU(value);

        var dLogits = new float[(long)_batch * _actions];
        var dValue = new float[_batch];
        float invB = 1f / _batch;
        double ce = 0, mse = 0;
        for (int r = 0; r < _batch; r++)
        {
            int off = r * _actions;
            float max = logits[off];
            for (int j = 1; j < _actions; j++) if (logits[off + j] > max) max = logits[off + j];
            float sum = 0f;
            for (int j = 0; j < _actions; j++) sum += MathF.Exp(logits[off + j] - max);
            float logSum = max + MathF.Log(sum), invSum = 1f / sum;
            for (int j = 0; j < _actions; j++)
            {
                float sm = MathF.Exp(logits[off + j] - max) * invSum;
                dLogits[off + j] = (sm - pi[off + j]) * invB;   // (softmax − π)/B
                ce += -pi[off + j] * (logits[off + j] - logSum);
            }
            float t = MathF.Tanh(value[r]), d = t - z[r];
            dValue[r] = _valueScale * d * (1f - t * t);           // valueWeight·(2/B)·(tanh−z)(1−tanh²)
            mse += d * d;
        }
        _dLogits.CopyFromCPU(dLogits);
        _dValue.CopyFromCPU(dValue);
        return (ce / _batch, mse / _batch);
    }

    // One conv layer forward: im2col (cache cols) → tiled GEMM → scatter+bias → NCHW out. Mirrors DeviceConvPolicyValueNet.Conv.
    private void ConvFwd(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> outp,
        ArrayView1D<float, Stride1D.Dense> cols, int inC, int outC, int k, int pad)
    {
        int inCk2 = inC * k * k;
        var mat = _matScratch.View.SubView(0, (long)_m * outC);
        _backend.LaunchIm2Col(input, cols, inC, k, pad, _h, _w);
        _backend.LaunchGemmTiled(cols, weight, mat, GemmDims.AB(_m, inCk2, outC, 0));
        _backend.LaunchScatterBias(mat, bias, outp, outC, _hw);
    }

    // Conv backward given the grad wrt this conv's NCHW output: gather → dW/dBias → dCols → col2im into dInput.
    private void ConvBwd(ArrayView1D<float, Stride1D.Dense> dOutNCHW, ArrayView1D<float, Stride1D.Dense> cols,
        Param weight, Param bias, ArrayView1D<float, Stride1D.Dense> dInput, int inC, int outC, int k, int pad)
    {
        int inCk2 = inC * k * k;
        var dMat = _dMatScratch.View.SubView(0, (long)_m * outC);
        var dCols = _dColsScratch.View.SubView(0, (long)_m * inCk2);
        _backend.LaunchGatherNCHWToMOutC(dOutNCHW, dMat, outC, _hw);
        _backend.LaunchGemmTiled(cols, dMat, weight.G.View, GemmDims.AtB(_m, inCk2, outC, 0)); // dW [inCk2, outC]
        _backend.LaunchBiasGrad(dMat, bias.G.View, _m, outC);                                   // dBias [outC]
        _backend.LaunchGemmTiled(dMat, weight.W.View, dCols, GemmDims.ABt(_m, inCk2, outC, 0)); // dCols[M,inCk2]=dMat·Wᵀ (shared=outC)
        _backend.LaunchCol2Im(dCols, dInput, inC, k, pad, _h, _w);
    }

    // Conv backward that only needs weight/bias grads (the stem: no dInput to propagate).
    private void ConvWeightBiasGrad(ArrayView1D<float, Stride1D.Dense> dOutNCHW, ArrayView1D<float, Stride1D.Dense> cols,
        Param weight, Param bias, int inC, int outC)
    {
        int inCk2 = inC * 9; // stem k=3
        var dMat = _dMatScratch.View.SubView(0, (long)_m * outC);
        _backend.LaunchGatherNCHWToMOutC(dOutNCHW, dMat, outC, _hw);
        _backend.LaunchGemmTiled(cols, dMat, weight.G.View, GemmDims.AtB(_m, inCk2, outC, 0));
        _backend.LaunchBiasGrad(dMat, bias.G.View, _m, outC);
    }

    // Global-norm gradient clip (mirrors Adam.ClipGradNorm) + Adam update. Caller holds the lock.
    private void ClipAndUpdate()
    {
        var b = _backend;
        if (_clipNorm > 0f)
        {
            _normAccum.MemSetToZero();
            foreach (var p in _all) b.LaunchSumSq(p.G.View, _normAccum.View, SumSqPartitions);
            var accum = new float[1];
            _backend.Accelerator.Synchronize();
            _normAccum.CopyToCPU(accum);
            float norm = MathF.Sqrt(accum[0]);
            if (norm > _clipNorm)
            {
                float scale = _clipNorm / (norm + 1e-6f);
                foreach (var p in _all) b.LaunchScaleInPlace(p.G.View, scale);
            }
        }

        _step++;
        var adam = new AdamParams(_lr, _beta1, _beta2, _eps,
            1f - MathF.Pow(_beta1, _step), 1f - MathF.Pow(_beta2, _step));
        foreach (var p in _all) b.LaunchAdamUpdate(p.W.View, p.G.View, p.M.View, p.V.View, adam);
    }

    public void Dispose()
    {
        lock (_backend.DeviceLock)
        {
            foreach (var p in _all) p.Dispose();
            _colsStem.Dispose(); _colsP.Dispose(); _colsV.Dispose();
            _xhatStem.Dispose(); _invStem.Dispose(); _xhatP.Dispose(); _invP.Dispose(); _xhatV.Dispose(); _invV.Dispose();
            _x0.Dispose(); _pRelu.Dispose(); _vRelu.Dispose(); _vHidRelu.Dispose();
            for (int i = 0; i < _blocks; i++)
            {
                _cols1[i].Dispose(); _cols2[i].Dispose(); _xhat1[i].Dispose(); _inv1[i].Dispose();
                _xhat2[i].Dispose(); _inv2[i].Dispose(); _xOut[i].Dispose(); _h1[i].Dispose();
            }
            _input.Dispose(); _stream.Dispose(); _hbuf.Dispose(); _h2buf.Dispose(); _pbuf.Dispose(); _vbuf.Dispose();
            _vhid.Dispose(); _value.Dispose(); _logits.Dispose();
            _dLogits.Dispose(); _dValue.Dispose(); _dP.Dispose(); _dV.Dispose(); _dVHid.Dispose(); _dConvTmp.Dispose();
            _matScratch.Dispose(); _dMatScratch.Dispose(); _dColsScratch.Dispose();
            _dx.Dispose(); _dxAdd.Dispose(); _dH.Dispose(); _normAccum.Dispose();
        }
    }
}
