using ILGPU;
using ILGPU.Runtime;
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
    private readonly object _lock = new();

    /// <param name="preferCpu">
    /// When true, selects ILGPU's CPU accelerator even if a GPU is present (used by tests so
    /// they run identically on any machine). Default false: pick the best device — the CUDA
    /// GPU when available, otherwise CPU.
    /// </param>
    public IlgpuBackend(bool preferCpu = false)
    {
        _context = Context.CreateDefault();
        var device = _context.GetPreferredDevice(preferCPU: preferCpu);
        _accelerator = device.CreateAccelerator(_context);

        _gemm = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Gemm_Kernel);
        _gemmTransposeA = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(GemmTransposeA_Kernel);
        _gemmTransposeB = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(GemmTransposeB_Kernel);
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

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
