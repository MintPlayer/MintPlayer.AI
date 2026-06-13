using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// GPU compute backend (PLAN M12c) implemented with ILGPU — C# kernels JIT-compiled to the
/// device (CUDA on an NVIDIA GPU, OpenCL, or ILGPU's CPU accelerator when no GPU is present,
/// which keeps CI and GPU-less machines green). It implements the same
/// <see cref="IComputeBackend"/> seam as <see cref="ManagedBackend"/>, so swapping it in
/// (<c>Backend.Current = new IlgpuBackend()</c>) needs no change to the autograd or algorithm
/// code.
/// <para>
/// <b>v1 is a host-span backend:</b> each GEMM uploads its operands, launches a kernel
/// (one output element per thread, accumulating into the destination to honor the
/// <see cref="IComputeBackend"/> "c += a·b" contract), synchronizes and downloads the result.
/// At small classic-control sizes the host↔device transfer dominates and this LOSES to the
/// CPU (PRD §10); it wins only for large GEMMs (wide nets, big batches). The transfer-free
/// optimization — device-resident tensors so operands live on the GPU across the training
/// step — is the next milestone and must be driven by the M12b crossover measurements on real
/// hardware. Correctness here is validated against <see cref="ManagedBackend"/> via ILGPU's
/// CPU accelerator; bitwise equality across backends is NOT expected (the GPU may fuse
/// multiply-add), only close agreement.
/// </para>
/// <para>Device work is serialized under a lock: the single default stream is not safe for
/// concurrent launches, and the training hot loop calls GEMM from one thread anyway.</para>
/// </summary>
public sealed class IlgpuBackend : IComputeBackend, IDisposable
{
    private delegate void GemmKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c,
        int m, int k, int n);

    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int> _gemm;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int> _gemmTransposeA;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int> _gemmTransposeB;
    // Device-resident inference path (M12c-perf): forward GEMM that WRITES (not accumulates) +
    // a fused bias+activation kernel, so a whole MLP forward chains on-device without per-layer transfer.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int> _gemmWrite;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _biasActivation;
    private readonly object _lock = new();

    /// <param name="preferCpu">
    /// When true, selects ILGPU's CPU accelerator even if a GPU is present (used by tests so
    /// they run identically on any machine). Default false: pick the best device — the CUDA
    /// GPU when available, otherwise CPU.
    /// </param>
    public IlgpuBackend(bool preferCpu = false)
    {
        _context = Context.CreateDefault();
        _accelerator = SelectDevice(_context, preferCpu).CreateAccelerator(_context);

        _gemm = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Gemm_Kernel);
        _gemmTransposeA = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(GemmTransposeA_Kernel);
        _gemmTransposeB = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(GemmTransposeB_Kernel);
        _gemmWrite = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(GemmWrite_Kernel);
        _biasActivation = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(BiasActivation_Kernel);
    }

    /// <summary>
    /// Picks the device. Default: prefer a discrete CUDA GPU specifically — on laptops with
    /// both an integrated Intel iGPU (OpenCL) and a discrete NVIDIA card, ILGPU's
    /// <c>GetPreferredDevice</c> picks the weaker iGPU, so we select CUDA first, then any
    /// other non-CPU device, then the CPU accelerator. <paramref name="preferCpu"/> forces
    /// the CPU accelerator (tests, GPU-less machines).
    /// </summary>
    private static Device SelectDevice(Context context, bool preferCpu)
    {
        if (preferCpu)
            return context.GetPreferredDevice(preferCPU: true);
        return context.Devices.OfType<CudaDevice>().Cast<Device>().FirstOrDefault()
            ?? context.Devices.FirstOrDefault(d => d.AcceleratorType != AcceleratorType.CPU)
            ?? context.GetPreferredDevice(preferCPU: true);
    }

    /// <summary>The selected device's name (e.g. "NVIDIA GeForce RTX 3060 Laptop GPU" or a CPU).</summary>
    public string AcceleratorName => _accelerator.Name;

    /// <summary>True when a real GPU (CUDA/OpenCL) was selected, false for the CPU accelerator.</summary>
    public bool IsGpu => _accelerator.AcceleratorType != AcceleratorType.CPU;

    /// <summary>Human-readable list of every ILGPU device available on this machine.</summary>
    public static string DescribeDevices()
    {
        using var context = Context.CreateDefault();
        return string.Join("; ", context.Devices.Select(d => $"{d.AcceleratorType}:{d.Name}"));
    }

    public void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Launch(_gemm, a, b, c, outputLength: m * n, m, k, n);

    public void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Launch(_gemmTransposeA, a, b, c, outputLength: k * n, m, k, n);

    public void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Launch(_gemmTransposeB, a, b, c, outputLength: m * k, m, k, n);

    private void Launch(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int> kernel,
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int outputLength, int m, int k, int n)
    {
        // Spans can't cross the (potential) async boundary; copy to arrays for upload/download.
        float[] aHost = a.ToArray(), bHost = b.ToArray(), cHost = c.ToArray();
        lock (_lock)
        {
            using var bufA = _accelerator.Allocate1D<float>(aHost.Length);
            using var bufB = _accelerator.Allocate1D<float>(bHost.Length);
            using var bufC = _accelerator.Allocate1D<float>(cHost.Length);
            bufA.CopyFromCPU(aHost);
            bufB.CopyFromCPU(bHost);
            bufC.CopyFromCPU(cHost); // upload existing destination — the kernel ACCUMULATES into it
            kernel(new Index1D(outputLength), bufA.View, bufB.View, bufC.View, m, k, n);
            _accelerator.Synchronize();
            bufC.CopyToCPU(cHost);
        }
        cHost.CopyTo(c);
    }

    // ── kernels: one thread per output element, sequential reduction over the shared dim ──

    private static void Gemm_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, int m, int k, int n)
    {
        int gid = index;
        if (gid >= m * n) return;
        int row = gid / n, col = gid % n;
        float acc = 0f;
        for (int p = 0; p < k; p++) acc += a[row * k + p] * b[p * n + col];
        c[gid] += acc;
    }

    private static void GemmTransposeA_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, int m, int k, int n)
    {
        // c[k,n] += aᵀ·b  for a[m,k], b[m,n]
        int gid = index;
        if (gid >= k * n) return;
        int p = gid / n, j = gid % n;
        float acc = 0f;
        for (int i = 0; i < m; i++) acc += a[i * k + p] * b[i * n + j];
        c[gid] += acc;
    }

    private static void GemmTransposeB_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, int m, int k, int n)
    {
        // c[m,k] += a·bᵀ  for a[m,n], b[k,n]
        int gid = index;
        if (gid >= m * k) return;
        int i = gid / k, p = gid % k;
        float acc = 0f;
        for (int j = 0; j < n; j++) acc += a[i * n + j] * b[p * n + j];
        c[gid] += acc;
    }

    /// <summary>
    /// Device-resident forward pass of a scalar-output MLP (M12c-perf): uploads the input batch
    /// ONCE, then chains GEMM → (bias + activation) layer by layer with activations staying on the
    /// device, and downloads only the final <paramref name="batch"/> scalars. This eliminates the
    /// per-layer host↔device round-trips that make the per-GEMM host-span path transfer-bound — the
    /// win that lets the GPU pay off for DAVI's successor evaluation. Weights are uploaded per call
    /// (they change during training); the saving is the resident intermediate activations.
    /// Supports None/ReLU hidden activations (the value nets use ReLU); Tanh would need XMath.
    /// </summary>
    public float[] MlpForwardScalar(Mlp net, ReadOnlySpan<float> input, int batch)
    {
        var sizes = net.Sizes;
        if (sizes[^1] != 1)
            throw new ArgumentException($"MlpForwardScalar expects a scalar-output net, got output size {sizes[^1]}.");
        int activation = net.HiddenActivation switch
        {
            Activation.None => 0,
            Activation.Relu => 1,
            _ => throw new NotSupportedException($"Resident forward supports None/ReLU, not {net.HiddenActivation}."),
        };

        var layers = net.Layers;
        float[] inputHost = input.ToArray();
        var buffers = new List<IDisposable>();
        lock (_lock)
        {
            try
            {
                var current = _accelerator.Allocate1D<float>(inputHost.Length);
                current.CopyFromCPU(inputHost);
                buffers.Add(current);
                ArrayView1D<float, Stride1D.Dense> activations = current.View;
                int inDim = sizes[0];

                MemoryBuffer1D<float, Stride1D.Dense> output = current;
                for (int i = 0; i < layers.Count; i++)
                {
                    int outDim = layers[i].Weight.Cols;
                    var w = _accelerator.Allocate1D<float>(layers[i].Weight.Data.Length); w.CopyFromCPU(layers[i].Weight.Data); buffers.Add(w);
                    var b = _accelerator.Allocate1D<float>(layers[i].Bias.Data.Length); b.CopyFromCPU(layers[i].Bias.Data); buffers.Add(b);
                    output = _accelerator.Allocate1D<float>(batch * outDim); buffers.Add(output);

                    _gemmWrite(new Index1D(batch * outDim), activations, w.View, output.View, batch, inDim, outDim);
                    bool isOutputLayer = i == layers.Count - 1;
                    _biasActivation(new Index1D(batch * outDim), output.View, b.View, outDim, isOutputLayer ? 0 : activation);

                    activations = output.View;
                    inDim = outDim;
                }

                _accelerator.Synchronize();
                var result = new float[batch];
                output.CopyToCPU(result);
                return result;
            }
            finally
            {
                foreach (var buffer in buffers) buffer.Dispose();
            }
        }
    }

    /// <summary>Forward GEMM that WRITES c = a·b (no accumulate) — for the resident inference chain.</summary>
    private static void GemmWrite_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, int m, int k, int n)
    {
        int gid = index;
        if (gid >= m * n) return;
        int row = gid / n, col = gid % n;
        float acc = 0f;
        for (int p = 0; p < k; p++) acc += a[row * k + p] * b[p * n + col];
        c[gid] = acc;
    }

    /// <summary>In-place bias add + activation (0 = none, 1 = ReLU) over a [rows, dim] buffer.</summary>
    private static void BiasActivation_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> bias, int dim, int activation)
    {
        int gid = index;
        float v = data[gid] + bias[gid % dim];
        if (activation == 1 && v < 0f) v = 0f;
        data[gid] = v;
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
