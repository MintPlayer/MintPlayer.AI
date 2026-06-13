using ILGPU;
using ILGPU.Runtime;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// A scalar-output MLP whose weights live <b>resident on the device</b> (PLAN M20, bottleneck #2).
/// It implements <see cref="ITargetForward"/> for DAVI's successor evaluation: weights upload once on
/// construction and re-upload only on <see cref="OnTargetSynced"/> (the trainer's ~200-step target
/// sync), so the per-step <see cref="Forward"/> transfers only the input batch up and the scalar
/// outputs down — eliminating the per-call weight re-upload that
/// <see cref="IlgpuBackend.MlpForwardScalar"/> still pays (~67 MB/layer at 8192-wide).
/// <para>
/// Reuses the backend's tiled GEMM and bias+activation kernels and its device lock — a
/// <see cref="OnTargetSynced"/> racing a <see cref="Forward"/> would read half-updated weights, so
/// every device touch is serialized under the same lock as the host-span GEMMs.
/// </para>
/// </summary>
public sealed class DeviceMlp : ITargetForward, IDisposable
{
    private readonly IlgpuBackend _backend;
    private readonly int[] _sizes;          // [in, hidden…, 1]
    private readonly int _activation;       // 0 none, 1 ReLU
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _weights;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _biases;

    internal DeviceMlp(IlgpuBackend backend, Mlp net)
    {
        _backend = backend;
        _activation = IlgpuBackend.ResolveActivation(net);
        _sizes = net.Sizes;
        var layers = net.Layers;
        _weights = new MemoryBuffer1D<float, Stride1D.Dense>[layers.Count];
        _biases = new MemoryBuffer1D<float, Stride1D.Dense>[layers.Count];
        lock (_backend.DeviceLock)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                _weights[i] = _backend.Accelerator.Allocate1D<float>(layers[i].Weight.Data.Length);
                _biases[i] = _backend.Accelerator.Allocate1D<float>(layers[i].Bias.Data.Length);
            }
            Upload(net);
        }
    }

    /// <summary>Re-upload the (just-synced) target net's weights into the resident buffers.</summary>
    public void OnTargetSynced(Mlp target)
    {
        lock (_backend.DeviceLock) Upload(target);
    }

    private void Upload(Mlp net)
    {
        var layers = net.Layers;
        for (int i = 0; i < layers.Count; i++)
        {
            _weights[i].CopyFromCPU(layers[i].Weight.Data);
            _biases[i].CopyFromCPU(layers[i].Bias.Data);
        }
    }

    /// <summary>
    /// Forward <paramref name="rows"/> feature rows against the resident weights, returning the raw
    /// scalar output per row. Only the input batch and the final scalars cross the bus.
    /// </summary>
    public float[] Forward(float[] features, int rows)
    {
        var acc = _backend.Accelerator;
        var temps = new List<IDisposable>();
        lock (_backend.DeviceLock)
        {
            try
            {
                var input = acc.Allocate1D<float>(features.Length);
                input.CopyFromCPU(features);
                temps.Add(input);

                ArrayView1D<float, Stride1D.Dense> activations = input.View;
                int inDim = _sizes[0];
                MemoryBuffer1D<float, Stride1D.Dense> output = input;
                for (int i = 0; i < _weights.Length; i++)
                {
                    int outDim = _sizes[i + 1];
                    output = acc.Allocate1D<float>((long)rows * outDim);
                    temps.Add(output);

                    _backend.LaunchGemmTiled(activations, _weights[i].View, output.View,
                        new GemmDims(rows, outDim, inDim, inDim, 1, outDim, 1, accumulate: 0));
                    bool isOutputLayer = i == _weights.Length - 1;
                    _backend.LaunchBiasActivation(output.View, _biases[i].View, outDim, isOutputLayer ? 0 : _activation);

                    activations = output.View;
                    inDim = outDim;
                }

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
            foreach (var w in _weights) w.Dispose();
            foreach (var b in _biases) b.Dispose();
        }
    }
}
