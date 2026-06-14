using ILGPU;
using ILGPU.Runtime;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// A <see cref="ResidualMlp"/> whose weights live <b>resident on the device</b>, with the full forward
/// (input projection + LayerNorm/ReLU, then each residual block's GEMM→bias→LayerNorm→ReLU→GEMM→bias
/// →skip-add, then the scalar head) chained entirely on-device (PLAN M20 Stage 2). It implements
/// <see cref="ITargetForward"/> for DAVI's successor evaluation: weights upload once and re-upload only
/// on <see cref="OnTargetSynced"/>, so the per-step forward transfers only the input batch up and the
/// scalar outputs down — removing the host↔device round-trip per GEMM that makes a residual net's
/// host-span (autograd) successor eval transfer-bound, the dominant cost of the residual campaign.
/// <para>
/// Weights are read from <see cref="ResidualMlp.Parameters"/> in its defined order (the same contract
/// the checkpoint relies on), so no extra structural accessors are needed. Every device touch is under
/// the backend lock — a sync racing a forward would read half-updated weights.
/// </para>
/// </summary>
public sealed class DeviceResidualMlp : ITargetForward, IDisposable
{
    private readonly IlgpuBackend _backend;
    private readonly int _inputSize, _width, _blocks;

    // Resident parameter buffers, mirroring ResidualMlp.Parameters() order.
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _inW, _inB, _inG, _inBeta;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _w1, _b1, _g, _beta, _w2, _b2;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _headW, _headB;

    internal DeviceResidualMlp(IlgpuBackend backend, ResidualMlp net)
    {
        _backend = backend;
        _inputSize = net.InputSize;
        _width = net.Width;
        _blocks = net.Blocks;

        var acc = backend.Accelerator;
        MemoryBuffer1D<float, Stride1D.Dense> Alloc(long len) => acc.Allocate1D<float>(len);
        lock (backend.DeviceLock)
        {
            _inW = Alloc((long)_inputSize * _width); _inB = Alloc(_width); _inG = Alloc(_width); _inBeta = Alloc(_width);
            _w1 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _b1 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _g = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _beta = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _w2 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _b2 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            for (int i = 0; i < _blocks; i++)
            {
                _w1[i] = Alloc((long)_width * _width); _b1[i] = Alloc(_width);
                _g[i] = Alloc(_width); _beta[i] = Alloc(_width);
                _w2[i] = Alloc((long)_width * _width); _b2[i] = Alloc(_width);
            }
            _headW = Alloc(_width); _headB = Alloc(1);
            Upload(net);
        }
    }

    /// <summary>Re-upload the (just-synced) target net's weights into the resident buffers.</summary>
    public void OnTargetSynced(IValueNet target)
    {
        if (target is not ResidualMlp res)
            throw new NotSupportedException($"DeviceResidualMlp supports {nameof(ResidualMlp)} only, not {target.GetType().Name}.");
        lock (_backend.DeviceLock) Upload(res);
    }

    private void Upload(ResidualMlp net)
    {
        // Walk Parameters() in order: inW, inB, inG, inBeta, (w1,b1,g,beta,w2,b2)×blocks, headW, headB.
        using var p = net.Parameters().GetEnumerator();
        void Next(MemoryBuffer1D<float, Stride1D.Dense> dst) { p.MoveNext(); dst.CopyFromCPU(p.Current.Data); }
        Next(_inW); Next(_inB); Next(_inG); Next(_inBeta);
        for (int i = 0; i < _blocks; i++) { Next(_w1[i]); Next(_b1[i]); Next(_g[i]); Next(_beta[i]); Next(_w2[i]); Next(_b2[i]); }
        Next(_headW); Next(_headB);
    }

    /// <summary>
    /// Forward <paramref name="rows"/> feature rows against the resident weights, returning the raw
    /// scalar output per row. Only the input batch and the final scalars cross the bus.
    /// </summary>
    public float[] Forward(float[] features, int rows)
    {
        var acc = _backend.Accelerator;
        var temps = new List<IDisposable>();
        MemoryBuffer1D<float, Stride1D.Dense> Temp(long len) { var b = acc.Allocate1D<float>(len); temps.Add(b); return b; }

        lock (_backend.DeviceLock)
        {
            try
            {
                var input = Temp((long)rows * _inputSize);
                input.CopyFromCPU(features);

                // Input projection → LayerNorm → ReLU.
                var x = Temp((long)rows * _width);
                _backend.LaunchGemmTiled(input.View, _inW.View, x.View, new GemmDims(rows, _width, _inputSize, _inputSize, 1, _width, 1, accumulate: 0));
                _backend.LaunchBiasActivation(x.View, _inB.View, _width, activation: 0);
                _backend.LaunchLayerNorm(x.View, _inG.View, _inBeta.View, rows, _width, relu: true);

                var h = Temp((long)rows * _width);
                var h2 = Temp((long)rows * _width);
                var square = new GemmDims(rows, _width, _width, _width, 1, _width, 1, accumulate: 0);
                for (int i = 0; i < _blocks; i++)
                {
                    _backend.LaunchGemmTiled(x.View, _w1[i].View, h.View, square);
                    _backend.LaunchBiasActivation(h.View, _b1[i].View, _width, activation: 0);
                    _backend.LaunchLayerNorm(h.View, _g[i].View, _beta[i].View, rows, _width, relu: true);

                    _backend.LaunchGemmTiled(h.View, _w2[i].View, h2.View, square);
                    _backend.LaunchBiasActivation(h2.View, _b2[i].View, _width, activation: 0);

                    _backend.LaunchAddInto(x.View, h2.View); // residual skip: x += block(x)
                }

                var output = Temp(rows);
                _backend.LaunchGemmTiled(x.View, _headW.View, output.View, new GemmDims(rows, 1, _width, _width, 1, 1, 1, accumulate: 0));
                _backend.LaunchBiasActivation(output.View, _headB.View, 1, activation: 0);

                acc.Synchronize();
                var result = new float[rows];
                output.CopyToCPU(result);
                return result;
            }
            finally
            {
                foreach (var t in temps) t.Dispose();
            }
        }
    }

    public void Dispose()
    {
        lock (_backend.DeviceLock)
        {
            _inW.Dispose(); _inB.Dispose(); _inG.Dispose(); _inBeta.Dispose();
            for (int i = 0; i < _blocks; i++) { _w1[i].Dispose(); _b1[i].Dispose(); _g[i].Dispose(); _beta[i].Dispose(); _w2[i].Dispose(); _b2[i].Dispose(); }
            _headW.Dispose(); _headB.Dispose();
        }
    }
}
