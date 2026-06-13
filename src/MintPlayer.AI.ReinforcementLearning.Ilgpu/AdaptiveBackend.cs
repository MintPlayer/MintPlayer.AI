using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// Routes each GEMM to whichever device wins at that size: the multithreaded CPU
/// (<see cref="ManagedBackend"/>) for small/medium products, the discrete CUDA GPU
/// (<see cref="IlgpuBackend"/>) for large ones, by a multiply-accumulate-count threshold.
/// With no GPU present it is pure CPU. Callers just set
/// <c>Backend.Current = new AdaptiveBackend()</c> and get the best of both with no knobs —
/// the complexity of "which device?" is absorbed here (PRD §4, pull complexity downwards).
/// <para>
/// The default threshold is calibrated to the measured host-span crossover (PLAN §M12b): on
/// the dev RTX 3060 the multithreaded CPU still wins below ~256 M MACs because per-call
/// host↔device transfer dominates, and the GPU pulls clear only on large square GEMMs. That
/// threshold is machine-specific and should drop sharply once device-resident tensors remove
/// the per-call transfer (M12c-perf) — override it per machine via the constructor.
/// </para>
/// </summary>
public sealed class AdaptiveBackend : IComputeBackend, IDisposable
{
    /// <summary>MAC count (m·k·n) at/above which a GEMM is sent to the GPU. See class remarks.</summary>
    public const long DefaultGpuMacThreshold = 256_000_000;

    private readonly ManagedBackend _cpu;
    private readonly IlgpuBackend? _gpu;
    private readonly long _gpuMacThreshold;

    /// <param name="gpuMacThreshold">m·k·n at/above which GEMMs route to the GPU (if one exists).</param>
    /// <param name="cpuMaxDegreeOfParallelism">Forwarded to <see cref="ManagedBackend"/>; null = core count.</param>
    public AdaptiveBackend(long gpuMacThreshold = DefaultGpuMacThreshold, int? cpuMaxDegreeOfParallelism = null)
    {
        _cpu = new ManagedBackend(cpuMaxDegreeOfParallelism);
        _gpuMacThreshold = gpuMacThreshold;

        // Spin up a GPU backend only if a real discrete/accelerator GPU is present; otherwise
        // dispose it and stay CPU-only (the CPU accelerator would just re-measure the CPU).
        var gpu = new IlgpuBackend();
        if (gpu.IsGpu) _gpu = gpu;
        else gpu.Dispose();
    }

    /// <summary>True when a GPU was found and large GEMMs will be offloaded to it.</summary>
    public bool GpuAvailable => _gpu is not null;

    /// <summary>
    /// The GPU backend, or null when CPU-only. Exposed so a caller can reuse this single GPU
    /// context for device-resident inference (e.g. <see cref="IlgpuBackend.MlpForwardScalar"/>)
    /// instead of spinning up a second one.
    /// </summary>
    public IlgpuBackend? Gpu => _gpu;

    /// <summary>One-line description of the routing for diagnostics/logging.</summary>
    public string Describe() => _gpu is null
        ? "CPU only (no GPU found)"
        : $"CPU + GPU ({_gpu.AcceleratorName}); GEMMs ≥ {_gpuMacThreshold:N0} MACs → GPU";

    private IComputeBackend Route(int m, int k, int n)
        => _gpu is not null && (long)m * k * n >= _gpuMacThreshold ? _gpu : _cpu;

    public void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Route(m, k, n).Gemm(a, b, c, m, k, n);

    public void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Route(m, k, n).GemmTransposeA(a, b, c, m, k, n);

    public void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
        => Route(m, k, n).GemmTransposeB(a, b, c, m, k, n);

    public void Dispose() => _gpu?.Dispose(); // ManagedBackend holds no unmanaged state
}
