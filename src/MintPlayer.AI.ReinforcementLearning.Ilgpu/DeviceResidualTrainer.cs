using ILGPU;
using ILGPU.Runtime;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// A fully device-resident training step for a <see cref="ResidualMlp"/> (PLAN M20 Stage 3): the online
/// net's weights, Adam moments, gradients and forward activations all live on the GPU, so one DAVI
/// train step — forward (caching x̂/σ and post-ReLU activations), full backward through the residual
/// chain, global-norm gradient clip, and the Adam update — runs without a single host↔device transfer
/// beyond uploading the input batch + targets and (for logging) downloading the row outputs. This
/// removes the CPU-bound autograd train step that remained the residual campaign's bottleneck after
/// Stage 2 made the successor eval resident.
/// <para>
/// The weights are mastered here; <see cref="SyncToHost"/> writes them back into the CPU
/// <see cref="ResidualMlp"/> the campaign holds (for eval, checkpointing, and target-net sync). Adam +
/// clip mirror the managed <see cref="Adam"/> exactly (same β, ε, bias correction, and
/// <c>maxNorm/(norm+1e-6)</c> clip) so the resident path is the same optimizer, just on-device.
/// </para>
/// </summary>
public sealed class DeviceResidualTrainer : IResidentTrainStep, IDisposable
{
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
    private readonly ResidualMlp _hostNet; // the CPU net these resident weights are synced back into
    private readonly int _inputSize, _width, _blocks, _batch;
    private readonly float _lr, _beta1, _beta2, _eps, _clipNorm, _huberDelta;
    private int _step;

    // Parameters in ResidualMlp.Parameters() order.
    private readonly Param _inW, _inB, _inG, _inBeta;
    private readonly Param[] _w1, _b1, _g, _beta, _w2, _b2;
    private readonly Param _headW, _headB;
    private readonly List<Param> _all = [];

    // Forward activation caches (sized to the train batch).
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _xhatIn, _invStdIn;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _xIn, _xhat1, _invStd1, _h;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _input, _target, _x, _tmpA, _tmpB, _out, _dOut, _dx, _normAccum;

    private const int SumSqPartitions = 256;

    internal DeviceResidualTrainer(IlgpuBackend backend, ResidualMlp net, int batch,
        float learningRate, float clipNorm, float huberDelta = 1f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        _backend = backend;
        _hostNet = net;
        _inputSize = net.InputSize; _width = net.Width; _blocks = net.Blocks; _batch = batch;
        _lr = learningRate; _clipNorm = clipNorm; _huberDelta = huberDelta;
        _beta1 = beta1; _beta2 = beta2; _eps = eps;

        var acc = backend.Accelerator;
        MemoryBuffer1D<float, Stride1D.Dense> Buf(long len) => acc.Allocate1D<float>(len);

        lock (backend.DeviceLock)
        {
            Param Pm(long len) { var p = new Param(acc, len); _all.Add(p); return p; }
            _inW = Pm((long)_inputSize * _width); _inB = Pm(_width); _inG = Pm(_width); _inBeta = Pm(_width);
            _w1 = new Param[_blocks]; _b1 = new Param[_blocks]; _g = new Param[_blocks];
            _beta = new Param[_blocks]; _w2 = new Param[_blocks]; _b2 = new Param[_blocks];
            for (int i = 0; i < _blocks; i++)
            {
                _w1[i] = Pm((long)_width * _width); _b1[i] = Pm(_width); _g[i] = Pm(_width);
                _beta[i] = Pm(_width); _w2[i] = Pm((long)_width * _width); _b2[i] = Pm(_width);
            }
            _headW = Pm(_width); _headB = Pm(1);

            UploadWeights(net);

            _xhatIn = Buf((long)batch * _width); _invStdIn = Buf(batch);
            _xIn = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _xhat1 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _invStd1 = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            _h = new MemoryBuffer1D<float, Stride1D.Dense>[_blocks];
            for (int i = 0; i < _blocks; i++)
            {
                _xIn[i] = Buf((long)batch * _width); _xhat1[i] = Buf((long)batch * _width);
                _invStd1[i] = Buf(batch); _h[i] = Buf((long)batch * _width);
            }
            _input = Buf((long)batch * _inputSize); _target = Buf(batch); _x = Buf((long)batch * _width);
            _tmpA = Buf((long)batch * _width); _tmpB = Buf((long)batch * _width);
            _out = Buf(batch); _dOut = Buf(batch); _dx = Buf((long)batch * _width); _normAccum = Buf(1);
        }
    }

    private void UploadWeights(ResidualMlp net)
    {
        using var p = net.Parameters().GetEnumerator();
        foreach (var param in _all) { p.MoveNext(); param.W.CopyFromCPU(p.Current.Data); }
    }

    /// <summary>Write the resident (online) weights back into the captured host net (the <see cref="IResidentTrainStep"/> hook).</summary>
    public void SyncToHost() => SyncToHost(_hostNet);

    /// <summary>Write the resident (online) weights back into the CPU net — for eval, checkpoint, target sync.</summary>
    public void SyncToHost(ResidualMlp net)
    {
        lock (_backend.DeviceLock)
        {
            using var p = net.Parameters().GetEnumerator();
            foreach (var param in _all) { p.MoveNext(); param.W.CopyToCPU(p.Current.Data); }
        }
    }

    /// <summary>
    /// One DAVI train step on <paramref name="rows"/> (== the configured batch): resident forward
    /// (caching activations), backward, gradient-norm clip, Adam update. Returns the mean Huber loss.
    /// </summary>
    public float Step(float[] features, float[] targets, int rows)
    {
        if (rows != _batch) throw new ArgumentException($"DeviceResidualTrainer is fixed to batch {_batch}, got {rows}.");
        lock (_backend.DeviceLock)
        {
            ForwardBackward(features, targets, rows);
            ClipAndUpdate(rows);

            // ── loss (host-side, from the row outputs) ──
            _backend.Accelerator.Synchronize();
            var outHost = new float[rows];
            _out.CopyToCPU(outHost);
            float total = 0f;
            for (int i = 0; i < rows; i++)
            {
                float diff = outHost[i] - targets[i], abs = MathF.Abs(diff);
                total += abs <= _huberDelta ? 0.5f * diff * diff : _huberDelta * (abs - 0.5f * _huberDelta);
            }
            return total / rows;
        }
    }

    /// <summary>Test hook: run forward+backward only and download the gradients in Parameters() order.</summary>
    internal float[][] DebugGradients(float[] features, float[] targets, int rows)
    {
        lock (_backend.DeviceLock)
        {
            ForwardBackward(features, targets, rows);
            _backend.Accelerator.Synchronize();
            var grads = new float[_all.Count][];
            for (int i = 0; i < _all.Count; i++) { grads[i] = new float[_all[i].G.Length]; _all[i].G.CopyToCPU(grads[i]); }
            return grads;
        }
    }

    /// <summary>Resident forward (caching activations) + full backward, leaving gradients in each param's G buffer. Caller holds the lock.</summary>
    private void ForwardBackward(float[] features, float[] targets, int rows)
    {
        var b = _backend;
        {
            _input.CopyFromCPU(features);
            _target.CopyFromCPU(targets);

            // ── forward (cache x̂/σ per LayerNorm, post-ReLU per ReLU, block inputs) ──
            b.LaunchGemmTiled(_input.View, _inW.W.View, _x.View, GemmDims.AB(rows, _inputSize, _width, 0));
            b.LaunchBiasActivation(_x.View, _inB.W.View, _width, 0);
            b.LaunchLayerNormTrain(_x.View, _inG.W.View, _inBeta.W.View, _xhatIn.View, _invStdIn.View, rows, _width);
            b.LaunchRelu(_x.View); // _x is now x0 (post-ReLU residual stream)

            for (int i = 0; i < _blocks; i++)
            {
                b.LaunchCopy(_xIn[i].View, _x.View); // snapshot block input
                b.LaunchGemmTiled(_x.View, _w1[i].W.View, _tmpA.View, GemmDims.AB(rows, _width, _width, 0));
                b.LaunchBiasActivation(_tmpA.View, _b1[i].W.View, _width, 0);
                b.LaunchLayerNormTrain(_tmpA.View, _g[i].W.View, _beta[i].W.View, _xhat1[i].View, _invStd1[i].View, rows, _width);
                b.LaunchRelu(_tmpA.View);
                b.LaunchCopy(_h[i].View, _tmpA.View); // cache post-ReLU h
                b.LaunchGemmTiled(_tmpA.View, _w2[i].W.View, _tmpB.View, GemmDims.AB(rows, _width, _width, 0));
                b.LaunchBiasActivation(_tmpB.View, _b2[i].W.View, _width, 0);
                b.LaunchAddInto(_x.View, _tmpB.View); // residual: x += block(x)
            }

            b.LaunchGemmTiled(_x.View, _headW.W.View, _out.View, GemmDims.AB(rows, _width, 1, 0));
            b.LaunchBiasActivation(_out.View, _headB.W.View, 1, 0);

            // ── loss + output gradient ──
            b.LaunchHuberGrad(_out.View, _target.View, _dOut.View, 1f / rows);

            // ── backward ── head
            b.LaunchGemmTiled(_x.View, _dOut.View, _headW.G.View, GemmDims.AtB(rows, _width, 1, 0));
            b.LaunchBiasGrad(_dOut.View, _headB.G.View, rows, 1);
            b.LaunchGemmTiled(_dOut.View, _headW.W.View, _dx.View, GemmDims.ABt(rows, _width, 1, 0)); // dx = grad wrt xFinal

            for (int i = _blocks - 1; i >= 0; i--)
            {
                // da2 = dx; dW2 = hᵀ·da2; db2 = colsum(da2); dh = da2·W2ᵀ
                b.LaunchGemmTiled(_h[i].View, _dx.View, _w2[i].G.View, GemmDims.AtB(rows, _width, _width, 0));
                b.LaunchBiasGrad(_dx.View, _b2[i].G.View, rows, _width);
                b.LaunchGemmTiled(_dx.View, _w2[i].W.View, _tmpA.View, GemmDims.ABt(rows, _width, _width, 0)); // tmpA = dh
                b.LaunchReluBackward(_tmpA.View, _h[i].View); // dn1 = dh·(h>0)
                b.LaunchLayerNormParamGrad(_tmpA.View, _xhat1[i].View, _g[i].G.View, _beta[i].G.View, rows, _width);
                b.LaunchLayerNormInputGrad(_tmpA.View, _xhat1[i].View, _invStd1[i].View, _g[i].W.View, _tmpB.View, rows, _width); // tmpB = da1
                b.LaunchGemmTiled(_xIn[i].View, _tmpB.View, _w1[i].G.View, GemmDims.AtB(rows, _width, _width, 0));
                b.LaunchBiasGrad(_tmpB.View, _b1[i].G.View, rows, _width);
                b.LaunchGemmTiled(_tmpB.View, _w1[i].W.View, _tmpA.View, GemmDims.ABt(rows, _width, _width, 0)); // tmpA = dx_h
                b.LaunchAddInto(_dx.View, _tmpA.View); // dx (skip) += dx_h
            }

            // inProj backward: dx is grad wrt x0 (post-ReLU); _xIn[0] is that post-ReLU snapshot.
            b.LaunchReluBackward(_dx.View, _xIn[0].View);
            b.LaunchLayerNormParamGrad(_dx.View, _xhatIn.View, _inG.G.View, _inBeta.G.View, rows, _width);
            b.LaunchLayerNormInputGrad(_dx.View, _xhatIn.View, _invStdIn.View, _inG.W.View, _tmpB.View, rows, _width); // tmpB = da_in
            b.LaunchGemmTiled(_input.View, _tmpB.View, _inW.G.View, GemmDims.AtB(rows, _inputSize, _width, 0));
            b.LaunchBiasGrad(_tmpB.View, _inB.G.View, rows, _width);
        }
    }

    /// <summary>Global-norm gradient clip (mirrors Adam.ClipGradNorm) + the Adam update. Caller holds the lock.</summary>
    private void ClipAndUpdate(int rows)
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
            _xhatIn.Dispose(); _invStdIn.Dispose();
            for (int i = 0; i < _blocks; i++) { _xIn[i].Dispose(); _xhat1[i].Dispose(); _invStd1[i].Dispose(); _h[i].Dispose(); }
            _input.Dispose(); _target.Dispose(); _x.Dispose(); _tmpA.Dispose(); _tmpB.Dispose();
            _out.Dispose(); _dOut.Dispose(); _dx.Dispose(); _normAccum.Dispose();
        }
    }
}
