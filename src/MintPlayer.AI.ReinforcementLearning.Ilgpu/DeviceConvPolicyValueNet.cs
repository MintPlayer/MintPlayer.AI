using ILGPU;
using ILGPU.Runtime;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// A <see cref="ConvResidualPolicyValueNet"/> whose weights live <b>resident on the device</b>, with the whole forward
/// — stem, residual tower, and both heads — chained entirely on-device (M43). Each conv runs as device im2col →
/// tiled GEMM → scatter+bias; LayerNorm(+ReLU), the residual skip-add, and the dense heads reuse the existing resident
/// kernels. It implements <see cref="IPolicyValueForward"/> (the two-headed analogue of <c>ITargetForward</c>): weights
/// upload once and re-upload only on <see cref="OnWeightsSynced"/>, so a self-play forward transfers only the
/// observation batch up and the two heads down — removing the host↔device round-trip per conv that makes the
/// autograd/host-span conv forward transfer-bound.
/// <para>
/// Weights are read from <see cref="ConvResidualPolicyValueNet.Parameters"/> in its defined order (the same contract
/// the checkpoint relies on), so no extra structural accessors are needed. Returns raw logits + linear (pre-tanh)
/// value; the caller applies masked-softmax + tanh. Every device touch is under the backend lock.
/// </para>
/// </summary>
public sealed class DeviceConvPolicyValueNet : IPolicyValueForward, IDisposable
{
    private const int PolicyChannels = 2; // must match ConvResidualPolicyValueNet.PolicyHeadChannels

    private readonly IlgpuBackend _backend;
    private readonly int _planes, _h, _w, _hw, _filters, _blocks, _actions;

    // Resident parameter buffers, mirroring ConvResidualPolicyValueNet.Parameters() order.
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _stemW, _stemB, _stemNG, _stemNB;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _b1W, _b1B, _n1G, _n1B, _b2W, _b2B, _n2G, _n2B;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _pConvW, _pConvB, _pNG, _pNB, _pHeadW, _pHeadB;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _vConvW, _vConvB, _vNG, _vNB, _vHidW, _vHidB, _vHeadW, _vHeadB;

    internal DeviceConvPolicyValueNet(IlgpuBackend backend, ConvResidualPolicyValueNet net)
    {
        _backend = backend;
        _planes = net.Planes; _h = net.BoardH; _w = net.BoardW; _hw = _h * _w;
        _filters = net.Filters; _blocks = net.Blocks; _actions = net.Actions;
        int F = _filters, hw = _hw;

        var acc = backend.Accelerator;
        MemoryBuffer1D<float, Stride1D.Dense> A(long len) => acc.Allocate1D<float>(len);
        MemoryBuffer1D<float, Stride1D.Dense>[] Arr() => new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
        lock (backend.DeviceLock)
        {
            _stemW = A((long)_planes * 9 * F); _stemB = A(F); _stemNG = A((long)F * hw); _stemNB = A((long)F * hw);
            _b1W = Arr(); _b1B = Arr(); _n1G = Arr(); _n1B = Arr(); _b2W = Arr(); _b2B = Arr(); _n2G = Arr(); _n2B = Arr();
            for (int i = 0; i < _blocks; i++)
            {
                _b1W[i] = A((long)F * 9 * F); _b1B[i] = A(F); _n1G[i] = A((long)F * hw); _n1B[i] = A((long)F * hw);
                _b2W[i] = A((long)F * 9 * F); _b2B[i] = A(F); _n2G[i] = A((long)F * hw); _n2B[i] = A((long)F * hw);
            }
            _pConvW = A((long)F * PolicyChannels); _pConvB = A(PolicyChannels);
            _pNG = A((long)PolicyChannels * hw); _pNB = A((long)PolicyChannels * hw);
            _pHeadW = A((long)PolicyChannels * hw * _actions); _pHeadB = A(_actions);
            _vConvW = A(F); _vConvB = A(1); _vNG = A(hw); _vNB = A(hw);
            _vHidW = A((long)hw * F); _vHidB = A(F); _vHeadW = A(F); _vHeadB = A(1);
            Upload(net);
        }
    }

    public void OnWeightsSynced(IPolicyValueNet net)
    {
        if (net is not ConvResidualPolicyValueNet conv)
            throw new NotSupportedException($"{nameof(DeviceConvPolicyValueNet)} supports {nameof(ConvResidualPolicyValueNet)} only, not {net.GetType().Name}.");
        lock (_backend.DeviceLock) Upload(conv);
    }

    private void Upload(ConvResidualPolicyValueNet net)
    {
        using var p = net.Parameters().GetEnumerator();
        void Next(MemoryBuffer1D<float, Stride1D.Dense> dst) { p.MoveNext(); dst.CopyFromCPU(p.Current.Data); }
        Next(_stemW); Next(_stemB); Next(_stemNG); Next(_stemNB);
        for (int i = 0; i < _blocks; i++)
        {
            Next(_b1W[i]); Next(_b1B[i]); Next(_n1G[i]); Next(_n1B[i]);
            Next(_b2W[i]); Next(_b2B[i]); Next(_n2G[i]); Next(_n2B[i]);
        }
        Next(_pConvW); Next(_pConvB); Next(_pNG); Next(_pNB); Next(_pHeadW); Next(_pHeadB);
        Next(_vConvW); Next(_vConvB); Next(_vNG); Next(_vNB); Next(_vHidW); Next(_vHidB); Next(_vHeadW); Next(_vHeadB);
    }

    /// <summary>
    /// Forward <paramref name="rows"/> observations (row-major NCHW <c>[rows, planes·h·w]</c>) against the resident
    /// weights. Returns raw logits <c>[rows·actions]</c> + linear value <c>[rows]</c>. Only the batch and the two heads
    /// cross the bus.
    /// </summary>
    public (float[] Logits, float[] Value) Forward(float[] observations, int rows)
    {
        var acc = _backend.Accelerator;
        int F = _filters, hw = _hw;
        lock (_backend.DeviceLock)
        {
            var temps = new List<IDisposable>();
            MemoryBuffer1D<float, Stride1D.Dense> T(long len) { var b = acc.Allocate1D<float>(len); temps.Add(b); return b; }
            try
            {
                var input = T((long)rows * _planes * hw);
                input.CopyFromCPU(observations);

                // Stem: 3×3 conv (planes→F) → LN → ReLU.
                var x = T((long)rows * F * hw);
                Conv(input.View, _stemW.View, _stemB.View, x.View, _planes, F, 3, 1, rows, T);
                _backend.LaunchLayerNorm(x.View, _stemNG.View, _stemNB.View, rows, F * hw, relu: true);

                // Residual tower: x = ReLU(x + LN(conv2(ReLU(LN(conv1 x))))).
                var h = T((long)rows * F * hw);
                var h2 = T((long)rows * F * hw);
                for (int i = 0; i < _blocks; i++)
                {
                    Conv(x.View, _b1W[i].View, _b1B[i].View, h.View, F, F, 3, 1, rows, T);
                    _backend.LaunchLayerNorm(h.View, _n1G[i].View, _n1B[i].View, rows, F * hw, relu: true);
                    Conv(h.View, _b2W[i].View, _b2B[i].View, h2.View, F, F, 3, 1, rows, T);
                    _backend.LaunchLayerNorm(h2.View, _n2G[i].View, _n2B[i].View, rows, F * hw, relu: false);
                    _backend.LaunchAddInto(x.View, h2.View);
                    _backend.LaunchRelu(x.View);
                }

                // Policy head: 1×1 conv (F→2) → LN → ReLU → flatten → Linear(2·hw → actions).
                var p = T((long)rows * PolicyChannels * hw);
                Conv(x.View, _pConvW.View, _pConvB.View, p.View, F, PolicyChannels, 1, 0, rows, T);
                _backend.LaunchLayerNorm(p.View, _pNG.View, _pNB.View, rows, PolicyChannels * hw, relu: true);
                var logits = T((long)rows * _actions);
                _backend.LaunchGemmTiled(p.View, _pHeadW.View, logits.View, GemmDims.AB(rows, PolicyChannels * hw, _actions, 0));
                _backend.LaunchBiasActivation(logits.View, _pHeadB.View, _actions, 0);

                // Value head: 1×1 conv (F→1) → LN → ReLU → Linear(hw → F) → ReLU → Linear(F → 1).
                var v = T((long)rows * hw);
                Conv(x.View, _vConvW.View, _vConvB.View, v.View, F, 1, 1, 0, rows, T);
                _backend.LaunchLayerNorm(v.View, _vNG.View, _vNB.View, rows, hw, relu: true);
                var vHid = T((long)rows * F);
                _backend.LaunchGemmTiled(v.View, _vHidW.View, vHid.View, GemmDims.AB(rows, hw, F, 0));
                _backend.LaunchBiasActivation(vHid.View, _vHidB.View, F, 1);
                var value = T(rows);
                _backend.LaunchGemmTiled(vHid.View, _vHeadW.View, value.View, GemmDims.AB(rows, F, 1, 0));
                _backend.LaunchBiasActivation(value.View, _vHeadB.View, 1, 0);

                acc.Synchronize();
                var logitsHost = new float[(long)rows * _actions];
                var valueHost = new float[rows];
                logits.CopyToCPU(logitsHost);
                value.CopyToCPU(valueHost);
                return (logitsHost, valueHost);
            }
            finally { foreach (var t in temps) t.Dispose(); }
        }
    }

    // One conv layer on-device: im2col → tiled GEMM (cols[M,inC·k²]·weight[inC·k²,outC]) → scatter+bias → NCHW out.
    private void Conv(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> outp,
        int inC, int outC, int k, int pad, int rows, Func<long, MemoryBuffer1D<float, Stride1D.Dense>> T)
    {
        int m = rows * _hw, inCk2 = inC * k * k;
        var cols = T((long)m * inCk2);
        var mat = T((long)m * outC);
        _backend.LaunchIm2Col(input, cols.View, inC, k, pad, _h, _w);
        _backend.LaunchGemmTiled(cols.View, weight, mat.View, GemmDims.AB(m, inCk2, outC, 0));
        _backend.LaunchScatterBias(mat.View, bias, outp, outC, _hw);
    }

    public void Dispose()
    {
        lock (_backend.DeviceLock)
        {
            _stemW.Dispose(); _stemB.Dispose(); _stemNG.Dispose(); _stemNB.Dispose();
            for (int i = 0; i < _blocks; i++)
            {
                _b1W[i].Dispose(); _b1B[i].Dispose(); _n1G[i].Dispose(); _n1B[i].Dispose();
                _b2W[i].Dispose(); _b2B[i].Dispose(); _n2G[i].Dispose(); _n2B[i].Dispose();
            }
            _pConvW.Dispose(); _pConvB.Dispose(); _pNG.Dispose(); _pNB.Dispose(); _pHeadW.Dispose(); _pHeadB.Dispose();
            _vConvW.Dispose(); _vConvB.Dispose(); _vNG.Dispose(); _vNB.Dispose();
            _vHidW.Dispose(); _vHidB.Dispose(); _vHeadW.Dispose(); _vHeadB.Dispose();
        }
    }
}
