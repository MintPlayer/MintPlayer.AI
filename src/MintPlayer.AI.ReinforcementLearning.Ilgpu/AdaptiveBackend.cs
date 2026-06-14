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

    // Elementwise ops always stay on the CPU: at autograd granularity they're transfer-bound on the GPU.
    public void Map(UnaryOp op, ReadOnlySpan<float> x, Span<float> y) => _cpu.Map(op, x, y);
    public void MapBackward(UnaryOp op, ReadOnlySpan<float> x, ReadOnlySpan<float> y, ReadOnlySpan<float> dy, Span<float> dx) => _cpu.MapBackward(op, x, y, dy, dx);
    public void Zip(BinaryOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) => _cpu.Zip(op, a, b, result);
    public void Scale(ReadOnlySpan<float> x, float s, Span<float> y) => _cpu.Scale(x, s, y);
    public void Clamp(ReadOnlySpan<float> x, float min, float max, Span<float> y) => _cpu.Clamp(x, min, max, y);
    public void AddBias(ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int rows, int cols, Span<float> y) => _cpu.AddBias(x, bias, rows, cols, y);
    public void AddInto(Span<float> dst, ReadOnlySpan<float> src) => _cpu.AddInto(dst, src);
    public void SubInto(Span<float> dst, ReadOnlySpan<float> src) => _cpu.SubInto(dst, src);
    public void MulAddInto(Span<float> dst, ReadOnlySpan<float> x, ReadOnlySpan<float> y) => _cpu.MulAddInto(dst, x, y);
    public void AxpyInto(Span<float> dst, float a, ReadOnlySpan<float> x) => _cpu.AxpyInto(dst, a, x);
    public void ClampBackwardInto(ReadOnlySpan<float> x, float min, float max, ReadOnlySpan<float> dy, Span<float> dx) => _cpu.ClampBackwardInto(x, min, max, dy, dx);
    public void MinBackwardInto(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> dy, Span<float> dst, bool forA) => _cpu.MinBackwardInto(a, b, dy, dst, forA);
    public void BiasGradInto(ReadOnlySpan<float> dy, Span<float> dbias, int rows, int cols) => _cpu.BiasGradInto(dy, dbias, rows, cols);
    public float Sum(ReadOnlySpan<float> x) => _cpu.Sum(x);
    public void AddScalarInto(Span<float> dst, float s) => _cpu.AddScalarInto(dst, s);
    public void SumRows(ReadOnlySpan<float> x, int rows, int cols, Span<float> outp) => _cpu.SumRows(x, rows, cols, outp);
    public void SumRowsBackwardInto(ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpu.SumRowsBackwardInto(dy, rows, cols, dx);
    public void LogSoftmax(ReadOnlySpan<float> x, int rows, int cols, Span<float> y) => _cpu.LogSoftmax(x, rows, cols, y);
    public void LogSoftmaxBackwardInto(ReadOnlySpan<float> y, ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpu.LogSoftmaxBackwardInto(y, dy, rows, cols, dx);
    public void Gather(ReadOnlySpan<float> x, ReadOnlySpan<int> indices, int rows, int cols, Span<float> outp) => _cpu.Gather(x, indices, rows, cols, outp);
    public void GatherBackwardInto(ReadOnlySpan<int> indices, ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpu.GatherBackwardInto(indices, dy, rows, cols, dx);
    public float HuberLoss(ReadOnlySpan<float> pred, ReadOnlySpan<float> target, float delta) => _cpu.HuberLoss(pred, target, delta);
    public void HuberGradInto(ReadOnlySpan<float> pred, ReadOnlySpan<float> target, float delta, float scale, Span<float> dst, bool negate) => _cpu.HuberGradInto(pred, target, delta, scale, dst, negate);
    public void LayerNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, int rows, int cols, float eps, Span<float> y, Span<float> xhat, Span<float> invStd) => _cpu.LayerNorm(x, gamma, beta, rows, cols, eps, y, xhat, invStd);
    public void LayerNormParamGradInto(ReadOnlySpan<float> dy, ReadOnlySpan<float> xhat, int rows, int cols, Span<float> dGamma, Span<float> dBeta) => _cpu.LayerNormParamGradInto(dy, xhat, rows, cols, dGamma, dBeta);
    public void LayerNormInputGradInto(ReadOnlySpan<float> dy, ReadOnlySpan<float> xhat, ReadOnlySpan<float> invStd, ReadOnlySpan<float> gamma, int rows, int cols, Span<float> dx) => _cpu.LayerNormInputGradInto(dy, xhat, invStd, gamma, rows, cols, dx);

    public void Dispose() => _gpu?.Dispose(); // ManagedBackend holds no unmanaged state
}
