using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// Shape of a single tiled GEMM: the destination's <see cref="Rows"/>×<see cref="Cols"/>, the
/// shared reduction length <see cref="Red"/>, and the row/column strides that let ONE kernel
/// serve all three layouts (A·B, Aᵀ·B, A·Bᵀ) — the operand element at logical (r, t)/(t, c) is
/// read with these strides, so the kernel's inner multiply loop is layout-agnostic.
/// <see cref="Accumulate"/> selects <c>c += …</c> (1, the <see cref="IComputeBackend"/> contract)
/// vs. <c>c = …</c> (0, the resident-inference write).
/// </summary>
public readonly struct GemmDims(
    int rows, int cols, int red,
    int aRowStride, int aColStride, int bRowStride, int bColStride, int accumulate)
{
    public readonly int Rows = rows, Cols = cols, Red = red;
    public readonly int ARowStride = aRowStride, AColStride = aColStride;
    public readonly int BRowStride = bRowStride, BColStride = bColStride;
    public readonly int Accumulate = accumulate;
}

/// <summary>
/// GPU compute backend (PLAN M12c/M19) implemented with ILGPU — C# kernels JIT-compiled to the
/// device (CUDA on an NVIDIA GPU, OpenCL, or ILGPU's CPU accelerator when no GPU is present,
/// which keeps CI and GPU-less machines green). It implements the same
/// <see cref="IComputeBackend"/> seam as <see cref="ManagedBackend"/>, so swapping it in
/// (<c>Backend.Current = new IlgpuBackend()</c>) needs no change to the autograd or algorithm
/// code.
/// <para>
/// <b>The GEMM kernel is shared-memory tiled</b> (M19): each 16×16 thread group cooperatively
/// stages a tile of each operand into shared memory, then multiply-accumulates from there, so each
/// loaded value is reused across the tile instead of re-read from global memory per output element.
/// That lifts the kernel off the memory wall the original one-thread-per-output kernel hit.
/// </para>
/// <para>
/// <b>Host-span vs. resident:</b> the three <see cref="IComputeBackend"/> GEMMs are host-span — each
/// uploads its operands, launches, synchronizes and downloads — so per-call transfer dominates at
/// small sizes and they win only for large GEMMs (PRD §10), routed there by <see cref="AdaptiveBackend"/>.
/// The transfer-free path is <see cref="DeviceMlp"/> (M20): weights stay resident on the device across
/// the training step and re-upload only on a target-net sync. Correctness is validated against
/// <see cref="ManagedBackend"/> via ILGPU's CPU accelerator; bitwise equality across backends is NOT
/// expected (the GPU may fuse multiply-add), only close agreement.
/// </para>
/// <para>Device work is serialized under a lock: the single default stream is not safe for
/// concurrent launches, and the training hot loop calls GEMM from one thread anyway.</para>
/// </summary>
public sealed class IlgpuBackend : IComputeBackend, IDisposable
{
    /// <summary>
    /// Max tile edge — the compile-time size of the shared-memory staging tiles (MaxTile² floats).
    /// The ACTUAL tile used per launch is <see cref="_tile"/> ≤ this, chosen to fit the device's
    /// group-size limit (a GPU allows 16; ILGPU's CPU accelerator caps a group at the logical-core
    /// count, so it runs a smaller tile). 16×16 = 256 threads/group is well within a GPU's 1024.
    /// </summary>
    internal const int MaxTile = 16;

    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _tile; // ≤ MaxTile; tile² ≤ device max threads/group
    private readonly Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int> _gemmTiled;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _biasActivation;
    // Resident residual-net forward (M20 Stage 2): row-wise LayerNorm(+optional ReLU) and an
    // elementwise add (the residual skip), so a ResidualMlp forward chains fully on-device.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float> _layerNorm;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _addInto;
    // The original one-thread-per-output kernel, retained ONLY for the M19 naive-vs-tiled bench
    // comparison (see BenchGemmGflops); never on the production routing path.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims> _gemmNaive;
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

        // Largest square tile whose thread group (tile²) fits the device limit, capped at MaxTile.
        // GPU: 1024 limit → tile 16. CPU accelerator: limit = #logical cores → a smaller tile.
        int t = MaxTile;
        while (t > 1 && t * t > _accelerator.MaxNumThreadsPerGroup) t--;
        _tile = t;

        _gemmTiled = _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int>(GemmTiled_Kernel);
        _biasActivation = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(BiasActivation_Kernel);
        _gemmNaive = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims>(GemmNaive_Kernel);
        _layerNorm = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float>(LayerNorm_Kernel);
        _addInto = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(AddInto_Kernel);
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

    // ── IComputeBackend: host-span GEMMs (operands up, result down, every call) ──
    // Strides map each logical layout onto the single tiled kernel (see GemmDims).

    public void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => LaunchHostSpan(a, b, c, new GemmDims(m, n, k, k, 1, n, 1, accumulate: 1));

    public void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        // c[k,n] += aᵀ·b: rows=k, cols=n, reduce over m. A read as Aᵀ (row=p stride 1, t=i stride k).
        => LaunchHostSpan(a, b, c, new GemmDims(k, n, m, 1, k, n, 1, accumulate: 1));

    public void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        // c[m,k] += a·bᵀ: rows=m, cols=k, reduce over n. B read as Bᵀ (t=j stride 1, col=p stride n).
        => LaunchHostSpan(a, b, c, new GemmDims(m, k, n, n, 1, 1, n, accumulate: 1));

    private void LaunchHostSpan(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, GemmDims dims)
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
            bufC.CopyFromCPU(cHost); // upload existing destination — accumulate==1 kernels add into it
            LaunchGemmTiled(bufA.View, bufB.View, bufC.View, dims);
            _accelerator.Synchronize();
            bufC.CopyToCPU(cHost);
        }
        cHost.CopyTo(c);
    }

    // ── internal device-launch helpers: operate on already-resident buffers; the caller holds _lock.
    //    Shared by the host-span GEMMs above and the resident DeviceMlp forward. ──

    internal Accelerator Accelerator => _accelerator;
    internal object DeviceLock => _lock;

    /// <summary>Launch the tiled GEMM over resident views with a grid sized to cover Rows×Cols.</summary>
    internal void LaunchGemmTiled(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, GemmDims d)
    {
        var grid = new Index2D((d.Cols + _tile - 1) / _tile, (d.Rows + _tile - 1) / _tile); // X = columns, Y = rows
        var group = new Index2D(_tile, _tile);
        _gemmTiled(new KernelConfig(grid, group), a, b, c, d, _tile);
    }

    /// <summary>Launch the in-place bias+activation over a resident [rows·dim] buffer.</summary>
    internal void LaunchBiasActivation(ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> bias, int dim, int activation)
        => _biasActivation(new Index1D((int)data.Length), data, bias, dim, activation);

    /// <summary>In-place row-wise LayerNorm with scale γ/shift β and optional ReLU, over a [rows·dim] buffer (one thread per row).</summary>
    internal void LaunchLayerNorm(ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, int rows, int dim, bool relu)
        => _layerNorm(new Index1D(rows), data, gamma, beta, dim, relu ? 1 : 0, 1e-5f);

    /// <summary>Elementwise <c>a += b</c> over two equal-length resident buffers (the residual skip).</summary>
    internal void LaunchAddInto(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
        => _addInto(new Index1D((int)a.Length), a, b);

    // ── tiled GEMM kernel: one 16×16 group computes a 16×16 output tile, staging operand tiles
    //    through shared memory and accumulating across the reduction in tile-sized steps. ──

    private static void GemmTiled_Kernel(
        ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, GemmDims d, int tile)
    {
        // Shared staging is sized to the compile-time max; only a tile×tile sub-region is used.
        var tileA = SharedMemory.Allocate<float>(MaxTile * MaxTile);
        var tileB = SharedMemory.Allocate<float>(MaxTile * MaxTile);

        int ty = Group.IdxY, tx = Group.IdxX;
        int row = Grid.IdxY * tile + ty; // output row this thread writes
        int col = Grid.IdxX * tile + tx; // output column this thread writes

        float acc = 0f;
        int numTiles = (d.Red + tile - 1) / tile;
        for (int t = 0; t < numTiles; t++)
        {
            int aT = t * tile + tx; // reduction coord loaded by this thread for A
            int bT = t * tile + ty; // reduction coord loaded by this thread for B
            // Out-of-range loads write 0 so the inner loop stays uniform (no divergence on the tail).
            tileA[ty * tile + tx] = (row < d.Rows && aT < d.Red) ? a[row * d.ARowStride + aT * d.AColStride] : 0f;
            tileB[ty * tile + tx] = (bT < d.Red && col < d.Cols) ? b[bT * d.BRowStride + col * d.BColStride] : 0f;
            Group.Barrier();

            for (int p = 0; p < tile; p++)
                acc += tileA[ty * tile + p] * tileB[p * tile + tx];
            Group.Barrier();
        }

        if (row < d.Rows && col < d.Cols)
        {
            int idx = row * d.Cols + col;
            if (d.Accumulate == 1) c[idx] += acc; else c[idx] = acc;
        }
    }

    /// <summary>In-place bias add + activation (0 = none, 1 = ReLU) over a [rows, dim] buffer.</summary>
    private static void BiasActivation_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> bias, int dim, int activation)
    {
        int gid = index;
        float v = data[gid] + bias[gid % dim];
        if (activation == 1 && v < 0f) v = 0f;
        data[gid] = v;
    }

    /// <summary>
    /// In-place row-wise LayerNorm with learned scale/shift and optional ReLU: one thread owns a row,
    /// computes the row mean and variance, then writes γ·(x−μ)/√(σ²+ε)+β (then ReLU if requested).
    /// Inference-only (the successor eval is no-grad) — the autograd LayerNorm carries the backward.
    /// </summary>
    private static void LayerNorm_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, int dim, int relu, float eps)
    {
        int row = index;
        int baseIdx = row * dim;
        float mean = 0f;
        for (int c = 0; c < dim; c++) mean += data[baseIdx + c];
        mean /= dim;
        float var = 0f;
        for (int c = 0; c < dim; c++) { float dd = data[baseIdx + c] - mean; var += dd * dd; }
        var /= dim;
        float inv = 1f / MathF.Sqrt(var + eps);
        for (int c = 0; c < dim; c++)
        {
            float v = gamma[c] * (data[baseIdx + c] - mean) * inv + beta[c];
            if (relu == 1 && v < 0f) v = 0f;
            data[baseIdx + c] = v;
        }
    }

    /// <summary>Elementwise <c>a[i] += b[i]</c> — the residual skip connection.</summary>
    private static void AddInto_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
        => a[index] += b[index];

    /// <summary>The original naive kernel (one thread per output, no reuse) — bench baseline only.</summary>
    private static void GemmNaive_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, GemmDims d)
    {
        int gid = index;
        if (gid >= d.Rows * d.Cols) return;
        int row = gid / d.Cols, col = gid % d.Cols;
        float acc = 0f;
        for (int t = 0; t < d.Red; t++) acc += a[row * d.ARowStride + t * d.AColStride] * b[t * d.BRowStride + col * d.BColStride];
        if (d.Accumulate == 1) c[gid] += acc; else c[gid] = acc;
    }

    /// <summary>
    /// Device-resident forward pass of a scalar-output MLP: uploads the input batch and the weights
    /// ONCE per call, then chains GEMM → (bias + activation) layer by layer with activations staying
    /// on the device, and downloads only the final <paramref name="batch"/> scalars. Weights re-upload
    /// every call here — for the transfer-free path that keeps weights resident across calls, use
    /// <see cref="CreateResidentForward"/> / <see cref="DeviceMlp"/>. Supports None/ReLU hidden
    /// activations (the value nets use ReLU); Tanh would need XMath.
    /// </summary>
    public float[] MlpForwardScalar(Mlp net, ReadOnlySpan<float> input, int batch)
    {
        int activation = ResolveActivation(net);
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
                int inDim = net.Sizes[0];

                MemoryBuffer1D<float, Stride1D.Dense> output = current;
                for (int i = 0; i < layers.Count; i++)
                {
                    int outDim = layers[i].Weight.Cols;
                    var w = _accelerator.Allocate1D<float>(layers[i].Weight.Data.Length); w.CopyFromCPU(layers[i].Weight.Data); buffers.Add(w);
                    var b = _accelerator.Allocate1D<float>(layers[i].Bias.Data.Length); b.CopyFromCPU(layers[i].Bias.Data); buffers.Add(b);
                    output = _accelerator.Allocate1D<float>(batch * outDim); buffers.Add(output);

                    LaunchGemmTiled(activations, w.View, output.View, new GemmDims(batch, outDim, inDim, inDim, 1, outDim, 1, accumulate: 0));
                    bool isOutputLayer = i == layers.Count - 1;
                    LaunchBiasActivation(output.View, b.View, outDim, isOutputLayer ? 0 : activation);

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

    /// <summary>
    /// Creates a <see cref="DeviceMlp"/> bound to this backend: a scalar-output MLP whose weights
    /// live resident on the device, re-uploaded only on <see cref="DeviceMlp.OnTargetSynced"/>.
    /// This is the M20 transfer-free successor-eval path for DAVI — the dominant cost — without the
    /// per-call weight upload that <see cref="MlpForwardScalar"/> still pays.
    /// </summary>
    public DeviceMlp CreateResidentForward(Mlp net) => new(this, net);

    /// <summary>
    /// Creates a <see cref="DeviceResidualMlp"/> bound to this backend: a deep residual value net whose
    /// weights live resident on the device, with the full forward (GEMM + bias + LayerNorm/ReLU +
    /// residual add + head) chained on-device (M20 Stage 2). This removes the per-call host↔device
    /// transfer that makes a residual net's host-span successor evaluation transfer-bound.
    /// </summary>
    public DeviceResidualMlp CreateResidentForward(ResidualMlp net) => new(this, net);

    /// <summary>Maps an MLP's hidden activation to the kernel's activation code (0 none, 1 ReLU).</summary>
    internal static int ResolveActivation(Mlp net)
    {
        if (net.Sizes[^1] != 1)
            throw new ArgumentException($"Resident forward expects a scalar-output net, got output size {net.Sizes[^1]}.");
        return net.HiddenActivation switch
        {
            Activation.None => 0,
            Activation.Relu => 1,
            _ => throw new NotSupportedException($"Resident forward supports None/ReLU, not {net.HiddenActivation}."),
        };
    }

    /// <summary>
    /// Bench-only (PLAN M19): GFLOP/s of a square GEMM with operands held resident (no per-iteration
    /// transfer), so it isolates kernel throughput. <paramref name="tiled"/> false selects the legacy
    /// naive kernel — the baseline for the committed naive-vs-tiled table.
    /// </summary>
    public double BenchGemmGflops(int m, int k, int n, int iterations, bool tiled)
    {
        var dims = new GemmDims(m, n, k, k, 1, n, 1, accumulate: 0);
        lock (_lock)
        {
            using var bufA = _accelerator.Allocate1D<float>((long)m * k);
            using var bufB = _accelerator.Allocate1D<float>((long)k * n);
            using var bufC = _accelerator.Allocate1D<float>((long)m * n);

            void Run()
            {
                if (tiled) LaunchGemmTiled(bufA.View, bufB.View, bufC.View, dims);
                else _gemmNaive(new Index1D(m * n), bufA.View, bufB.View, bufC.View, dims);
            }

            for (int i = 0; i < 10; i++) Run();
            _accelerator.Synchronize();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) Run();
            _accelerator.Synchronize();
            sw.Stop();

            return 2.0 * m * k * n * iterations / sw.Elapsed.TotalSeconds / 1e9;
        }
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
