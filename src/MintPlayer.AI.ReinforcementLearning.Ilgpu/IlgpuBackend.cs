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

    // Factories for the three layouts of a [m,k]·[k,n] layer, sharing one (m, k, n) convention so a
    // Linear's forward / weight-grad / input-grad are constructed without hand-rolling strides:
    //   forward  C[m,n] = A·B          (A[m,k], B[k,n])
    //   weightΔ  C[k,n] = Aᵀ·B         (A[m,k], B[m,n])   — dW = Xᵀ·dC
    //   inputΔ   C[m,k] = A·Bᵀ         (A[m,n], B[k,n])   — dX = dC·Wᵀ

    /// <summary>C[m,n] = A·B for A[m,k]·B[k,n].</summary>
    public static GemmDims AB(int m, int k, int n, int accumulate) => new(m, n, k, k, 1, n, 1, accumulate);

    /// <summary>C[k,n] = Aᵀ·B for A[m,k], B[m,n] (weight gradient).</summary>
    public static GemmDims AtB(int m, int k, int n, int accumulate) => new(k, n, m, 1, k, n, 1, accumulate);

    /// <summary>C[m,k] = A·Bᵀ for A[m,n], B[k,n] (input gradient).</summary>
    public static GemmDims ABt(int m, int k, int n, int accumulate) => new(m, k, n, n, 1, 1, n, accumulate);
}

/// <summary>Which GEMM kernel to benchmark (see <see cref="IlgpuBackend.BenchGemmGflops"/>).</summary>
public enum GemmKind { Naive, Tiled, RegBlocked }

/// <summary>
/// Adam hyper-parameters + per-step bias-correction for the on-device update kernel (M20 Stage 3):
/// learning rate, the two decay rates, ε, and the precomputed bias-correction denominators
/// (<c>1−β₁ᵗ</c>, <c>1−β₂ᵗ</c>) for the current step <c>t</c> (computed host-side so the kernel stays
/// branch-free). Blittable so it passes by value to the kernel.
/// </summary>
public readonly struct AdamParams(float lr, float beta1, float beta2, float eps, float biasCorr1, float biasCorr2)
{
    public readonly float Lr = lr, Beta1 = beta1, Beta2 = beta2, Eps = eps, BiasCorr1 = biasCorr1, BiasCorr2 = biasCorr2;
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

    // Register-blocked GEMM (P.1/M19b): each thread computes a RegTM×RegTN micro-tile in registers,
    // reading RegBK-deep shared-memory tiles — far higher arithmetic intensity than one-thread-per-output.
    // A group is groupEdge×groupEdge threads computing a (groupEdge·RegTM)×(groupEdge·RegTN) output block;
    // groupEdge adapts to the device's group-size limit (16²=256 threads on a GPU, smaller on the CPU
    // accelerator). Shared tiles are sized to the compile-time max block (RegBMax) and a sub-region used.
    private const int RegTM = 4, RegTN = 4, RegBK = 8, RegBMax = 64;

    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _tile;       // ≤ MaxTile; tile² ≤ device max threads/group (simple tiled kernel)
    private readonly int _regEdge;    // groupEdge for the register-blocked kernel; edge² ≤ device limit
    private readonly ManagedBackend _cpuOps = new(); // elementwise/reduction fallback (see Map)
    private readonly Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int> _gemmTiled;
    private readonly Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int> _gemmReg;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _biasActivation;
    // Resident residual-net forward (M20 Stage 2): row-wise LayerNorm(+optional ReLU) and an
    // elementwise add (the residual skip), so a ResidualMlp forward chains fully on-device.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float> _layerNorm;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _addInto;
    // Resident training (M20 Stage 3): backward + Adam kernels, so the whole DAVI step runs on-device.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _biasGrad;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _reluBackward;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> _lnInputGrad;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _lnParamGrad;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _huberGrad;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AdamParams> _adamUpdate;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int> _sumSq;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float> _scaleInPlace;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float> _layerNormTrain;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>> _relu;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _copy;
    // Conv residency (M43): the two data-movement kernels that bracket each conv's GEMM.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int> _im2col;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _scatterBias;
    // Conv TRAINING residency (M44): the transposes of im2col/scatter for the conv backward. The two-headed loss grads
    // (softmax−π, tanh-MSE) are computed on the HOST — the repo deliberately keeps softmax/tanh off the device (no
    // ILGPU.Algorithms/XMath), and the heads are tiny (rows·actions / rows), so the transfer is negligible.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int> _col2im;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> _gatherNCHW;
    // The original one-thread-per-output kernel, retained ONLY for the M19 naive-vs-tiled bench
    // comparison (see BenchGemmGflops); never on the production routing path.
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims> _gemmNaive;
    private readonly object _lock = new();
    private readonly bool _ownsContext; // dispose _context in Dispose (true only for the standalone/own-context path)

    /// <param name="preferCpu">
    /// When true, selects ILGPU's CPU accelerator even if a GPU is present (used by tests so
    /// they run identically on any machine). Default false: pick the best device — the first
    /// CUDA GPU when available, otherwise CPU.
    /// </param>
    public IlgpuBackend(bool preferCpu = false) : this(Context.CreateDefault(), preferCpu) { }

    // Owns a freshly-created context; pins to the first device SelectDevices returns (CUDA-first, else CPU).
    private IlgpuBackend(Context ownedContext, bool preferCpu)
        : this(ownedContext, SelectDevices(ownedContext, preferCpu)[0], ownsContext: true) { }

    /// <summary>Pin a backend to a specific <paramref name="device"/> on a SHARED <paramref name="context"/> (multi-GPU,
    /// M45): the caller owns and disposes the context, so this backend disposes only its accelerator. Enumerate devices
    /// with <see cref="SelectDevices"/>.</summary>
    internal IlgpuBackend(Context context, Device device) : this(context, device, ownsContext: false) { }

    // Core ctor: build the accelerator on `device` and JIT every kernel onto it.
    private IlgpuBackend(Context context, Device device, bool ownsContext)
    {
        _context = context;
        _ownsContext = ownsContext;
        _accelerator = device.CreateAccelerator(_context);

        // Largest square tile whose thread group (tile²) fits the device limit, capped at MaxTile.
        // GPU: 1024 limit → tile 16. CPU accelerator: limit = #logical cores → a smaller tile.
        int t = MaxTile;
        while (t > 1 && t * t > _accelerator.MaxNumThreadsPerGroup) t--;
        _tile = t;

        // Register-blocked group edge: largest g with g² ≤ device limit and g ≤ RegBMax/RegTM (so the
        // block fits the shared tiles). GPU: 16 (256 threads, 64×64 block). CPU accelerator: ~4.
        int e = RegBMax / RegTM;
        while (e > 1 && e * e > _accelerator.MaxNumThreadsPerGroup) e--;
        _regEdge = e;

        _gemmTiled = _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int>(GemmTiled_Kernel);
        _gemmReg = _accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims, int>(GemmRegBlocked_Kernel);
        _biasActivation = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(BiasActivation_Kernel);
        _gemmNaive = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, GemmDims>(GemmNaive_Kernel);
        _layerNorm = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float>(LayerNorm_Kernel);
        _addInto = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(AddInto_Kernel);
        _biasGrad = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(BiasGrad_Kernel);
        _reluBackward = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ReluBackward_Kernel);
        _lnInputGrad = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(LayerNormInputGrad_Kernel);
        _lnParamGrad = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(LayerNormParamGrad_Kernel);
        _huberGrad = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(HuberGrad_Kernel);
        _adamUpdate = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, AdamParams>(AdamUpdate_Kernel);
        _sumSq = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(SumSq_Kernel);
        _scaleInPlace = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(ScaleInPlace_Kernel);
        _layerNormTrain = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>(LayerNormTrain_Kernel);
        _relu = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(Relu_Kernel);
        _copy = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Copy_Kernel);
        _im2col = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>(Im2Col_Kernel);
        _scatterBias = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(ScatterBias_Kernel);
        _col2im = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>(Col2Im_Kernel);
        _gatherNCHW = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(GatherNCHWToMOutC_Kernel);
    }

    /// <summary>
    /// Enumerates the devices to build accelerators on. Default: <b>every</b> discrete CUDA GPU (in ILGPU's device
    /// order) — on a multi-GPU box all NVIDIA cards are returned so the self-play dataflow can be sharded across them
    /// (M45); on a laptop with an Intel iGPU (OpenCL) + one NVIDIA card, only the CUDA card is returned (never the
    /// weaker iGPU). If no CUDA device exists, falls back to a single other non-CPU device, else the CPU accelerator.
    /// <paramref name="preferCpu"/> forces the single CPU accelerator (tests, GPU-less machines).
    /// </summary>
    internal static IReadOnlyList<Device> SelectDevices(Context context, bool preferCpu)
    {
        if (preferCpu)
            return [context.GetPreferredDevice(preferCPU: true)];
        var cuda = context.Devices.OfType<CudaDevice>().Cast<Device>().ToList();
        if (cuda.Count > 0) return cuda;
        return [context.Devices.FirstOrDefault(d => d.AcceleratorType != AcceleratorType.CPU)
            ?? context.GetPreferredDevice(preferCPU: true)];
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

    // Elementwise ops run on the CPU helper, not host-span GPU kernels: at autograd granularity these
    // are tiny and transfer-bound, so a GPU round-trip would only lose (and transcendentals would need
    // ILGPU.Algorithms). The GPU's job through IComputeBackend is the compute-heavy GEMM; real on-device
    // elementwise lives in the resident paths (DeviceMlp / DeviceResidualTrainer kernels).
    public void Map(UnaryOp op, ReadOnlySpan<float> x, Span<float> y) => _cpuOps.Map(op, x, y);
    public void MapBackward(UnaryOp op, ReadOnlySpan<float> x, ReadOnlySpan<float> y, ReadOnlySpan<float> dy, Span<float> dx) => _cpuOps.MapBackward(op, x, y, dy, dx);
    public void Zip(BinaryOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) => _cpuOps.Zip(op, a, b, result);
    public void Scale(ReadOnlySpan<float> x, float s, Span<float> y) => _cpuOps.Scale(x, s, y);
    public void Clamp(ReadOnlySpan<float> x, float min, float max, Span<float> y) => _cpuOps.Clamp(x, min, max, y);
    public void AddBias(ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int rows, int cols, Span<float> y) => _cpuOps.AddBias(x, bias, rows, cols, y);
    public void AddInto(Span<float> dst, ReadOnlySpan<float> src) => _cpuOps.AddInto(dst, src);
    public void SubInto(Span<float> dst, ReadOnlySpan<float> src) => _cpuOps.SubInto(dst, src);
    public void MulAddInto(Span<float> dst, ReadOnlySpan<float> x, ReadOnlySpan<float> y) => _cpuOps.MulAddInto(dst, x, y);
    public void AxpyInto(Span<float> dst, float a, ReadOnlySpan<float> x) => _cpuOps.AxpyInto(dst, a, x);
    public void ClampBackwardInto(ReadOnlySpan<float> x, float min, float max, ReadOnlySpan<float> dy, Span<float> dx) => _cpuOps.ClampBackwardInto(x, min, max, dy, dx);
    public void MinBackwardInto(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> dy, Span<float> dst, bool forA) => _cpuOps.MinBackwardInto(a, b, dy, dst, forA);
    public void BiasGradInto(ReadOnlySpan<float> dy, Span<float> dbias, int rows, int cols) => _cpuOps.BiasGradInto(dy, dbias, rows, cols);
    public float Sum(ReadOnlySpan<float> x) => _cpuOps.Sum(x);
    public void AddScalarInto(Span<float> dst, float s) => _cpuOps.AddScalarInto(dst, s);
    public void SumRows(ReadOnlySpan<float> x, int rows, int cols, Span<float> outp) => _cpuOps.SumRows(x, rows, cols, outp);
    public void SumRowsBackwardInto(ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpuOps.SumRowsBackwardInto(dy, rows, cols, dx);
    public void LogSoftmax(ReadOnlySpan<float> x, int rows, int cols, Span<float> y) => _cpuOps.LogSoftmax(x, rows, cols, y);
    public void LogSoftmaxBackwardInto(ReadOnlySpan<float> y, ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpuOps.LogSoftmaxBackwardInto(y, dy, rows, cols, dx);
    public void Gather(ReadOnlySpan<float> x, ReadOnlySpan<int> indices, int rows, int cols, Span<float> outp) => _cpuOps.Gather(x, indices, rows, cols, outp);
    public void GatherBackwardInto(ReadOnlySpan<int> indices, ReadOnlySpan<float> dy, int rows, int cols, Span<float> dx) => _cpuOps.GatherBackwardInto(indices, dy, rows, cols, dx);
    public float HuberLoss(ReadOnlySpan<float> pred, ReadOnlySpan<float> target, float delta) => _cpuOps.HuberLoss(pred, target, delta);
    public void HuberGradInto(ReadOnlySpan<float> pred, ReadOnlySpan<float> target, float delta, float scale, Span<float> dst, bool negate) => _cpuOps.HuberGradInto(pred, target, delta, scale, dst, negate);
    public void LayerNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, int rows, int cols, float eps, Span<float> y, Span<float> xhat, Span<float> invStd) => _cpuOps.LayerNorm(x, gamma, beta, rows, cols, eps, y, xhat, invStd);
    public void LayerNormParamGradInto(ReadOnlySpan<float> dy, ReadOnlySpan<float> xhat, int rows, int cols, Span<float> dGamma, Span<float> dBeta) => _cpuOps.LayerNormParamGradInto(dy, xhat, rows, cols, dGamma, dBeta);
    public void LayerNormInputGradInto(ReadOnlySpan<float> dy, ReadOnlySpan<float> xhat, ReadOnlySpan<float> invStd, ReadOnlySpan<float> gamma, int rows, int cols, Span<float> dx) => _cpuOps.LayerNormInputGradInto(dy, xhat, invStd, gamma, rows, cols, dx);

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

    /// <summary>
    /// Launch the production GEMM over resident views (the register-blocked kernel, P.1) with a grid
    /// sized to cover Rows×Cols in (groupEdge·RegTM)×(groupEdge·RegTN) blocks. This is the path every
    /// caller uses (host-span GEMMs, DeviceMlp, DeviceResidualMlp/Trainer).
    /// </summary>
    internal void LaunchGemmTiled(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, GemmDims d)
    {
        int blockM = _regEdge * RegTM, blockN = _regEdge * RegTN;
        var grid = new Index2D((d.Cols + blockN - 1) / blockN, (d.Rows + blockM - 1) / blockM); // X = columns, Y = rows
        var group = new Index2D(_regEdge, _regEdge);
        _gemmReg(new KernelConfig(grid, group), a, b, c, d, _regEdge);
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

    /// <summary>Device im2col: gather NCHW input <c>[B, inC·h·w]</c> into <c>cols[M, inC·k²]</c> (M=B·h·w),
    /// zero-filling padding. One thread per cols element. Caller holds the lock.</summary>
    internal void LaunchIm2Col(ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> cols, int inC, int k, int pad, int h, int w)
        => _im2col(new Index1D((int)cols.Length), x, cols, inC, k, pad, h, w);

    /// <summary>Scatter a conv GEMM result <c>mat[M, outC]</c> to NCHW <c>out[B, outC·hw]</c>, adding the
    /// per-channel bias (broadcast over the hw spatial positions). One thread per output element. Caller holds the lock.</summary>
    internal void LaunchScatterBias(ArrayView1D<float, Stride1D.Dense> mat, ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> outp, int outC, int hw)
        => _scatterBias(new Index1D((int)outp.Length), mat, bias, outp, outC, hw);

    // ── conv training launch helpers (M44; caller holds the lock) ──

    /// <summary>Device col2im (the im2col transpose): scatter-add the column gradient <c>dCols[M, inC·k²]</c> back to
    /// NCHW <c>dInput[B, inC·h·w]</c>. One thread per <b>input</b> element gathers every kernel tap that read it, so it
    /// is atomics-free. Caller holds the lock.</summary>
    internal void LaunchCol2Im(ArrayView1D<float, Stride1D.Dense> dCols, ArrayView1D<float, Stride1D.Dense> dInput, int inC, int k, int pad, int h, int w)
        => _col2im(new Index1D((int)dInput.Length), dCols, dInput, inC, k, pad, h, w);

    /// <summary>Gather NCHW <c>dOut[B, outC·hw]</c> → <c>dMat[M, outC]</c> (M=B·hw) — the transpose of ScatterBias (no
    /// bias). One thread per dMat element. Caller holds the lock.</summary>
    internal void LaunchGatherNCHWToMOutC(ArrayView1D<float, Stride1D.Dense> dOut, ArrayView1D<float, Stride1D.Dense> dMat, int outC, int hw)
        => _gatherNCHW(new Index1D((int)dMat.Length), dOut, dMat, outC, hw);

    // ── resident training launch helpers (caller holds the lock) ──

    /// <summary>Bias gradient: <c>dBias[c] = Σ_r dy[r,c]</c> (write), one thread per column.</summary>
    internal void LaunchBiasGrad(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> dBias, int rows, int dim)
        => _biasGrad(new Index1D(dim), dy, dBias, rows, dim);

    /// <summary>ReLU backward in place: <c>grad[i] = 0 where post[i] ≤ 0</c> (post = the ReLU output).</summary>
    internal void LaunchReluBackward(ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> post)
        => _reluBackward(new Index1D((int)grad.Length), grad, post);

    /// <summary>LayerNorm input gradient (write to dx), one thread per row, using cached x̂ and 1/σ.</summary>
    internal void LaunchLayerNormInputGrad(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> invStd, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> dx, int rows, int dim)
        => _lnInputGrad(new Index1D(rows), dy, xhat, invStd, gamma, dx, dim);

    /// <summary>LayerNorm γ/β gradients (write), one thread per column: dγ[c]=Σ dy·x̂, dβ[c]=Σ dy.</summary>
    internal void LaunchLayerNormParamGrad(ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> dGamma, ArrayView1D<float, Stride1D.Dense> dBeta, int rows, int dim)
        => _lnParamGrad(new Index1D(dim), dy, xhat, dGamma, dBeta, rows, dim);

    /// <summary>Huber gradient into dy: <c>clamp(pred−target, ±δ)·scale</c> ([rows] vectors, scale=1/rows).</summary>
    internal void LaunchHuberGrad(ArrayView1D<float, Stride1D.Dense> pred, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> dOut, float scale)
        => _huberGrad(new Index1D((int)pred.Length), pred, target, dOut, scale);

    /// <summary>In-place Adam update of one parameter buffer against its gradient + moment buffers.</summary>
    internal void LaunchAdamUpdate(ArrayView1D<float, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> g, ArrayView1D<float, Stride1D.Dense> m, ArrayView1D<float, Stride1D.Dense> v, AdamParams p)
        => _adamUpdate(new Index1D((int)w.Length), w, g, m, v, p);

    /// <summary>Accumulate Σ data[i]² into <paramref name="accum"/>[0] (atomic); pre-zero accum once.</summary>
    internal void LaunchSumSq(ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> accum, int partitions)
        => _sumSq(new Index1D(partitions), data, accum, partitions);

    /// <summary>In-place scale: <c>buf[i] *= scalar</c> (gradient-norm clipping).</summary>
    internal void LaunchScaleInPlace(ArrayView1D<float, Stride1D.Dense> buf, float scalar)
        => _scaleInPlace(new Index1D((int)buf.Length), buf, scalar);

    /// <summary>LayerNorm forward that also caches x̂ and 1/σ for the backward (training path), no activation.</summary>
    internal void LaunchLayerNormTrain(ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> invStd, int rows, int dim)
        => _layerNormTrain(new Index1D(rows), data, gamma, beta, xhat, invStd, dim, 1e-5f);

    /// <summary>In-place ReLU over a resident buffer.</summary>
    internal void LaunchRelu(ArrayView1D<float, Stride1D.Dense> data)
        => _relu(new Index1D((int)data.Length), data);

    /// <summary>Device-to-device copy <c>dst[i] = src[i]</c> (activation snapshot).</summary>
    internal void LaunchCopy(ArrayView1D<float, Stride1D.Dense> dst, ArrayView1D<float, Stride1D.Dense> src)
        => _copy(new Index1D((int)dst.Length), dst, src);

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

    /// <summary>
    /// Register-blocked GEMM (P.1): a groupEdge×groupEdge group computes a (groupEdge·RegTM)×(groupEdge·RegTN)
    /// output block; each thread holds a RegTM×RegTN accumulator micro-tile in registers and streams
    /// RegBK-deep tiles of A and B through shared memory. Each shared value is reused RegTM/RegTN times from
    /// registers, so far fewer shared-memory reads per output than the one-thread-per-output tiled kernel.
    /// Generic over layout via <see cref="GemmDims"/> strides; boundary guards zero-fill loads and guard stores.
    /// </summary>
    private static void GemmRegBlocked_Kernel(
        ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c, GemmDims d, int groupEdge)
    {
        var As = SharedMemory.Allocate<float>(RegBMax * RegBK); // [blockM, RegBK]
        var Bs = SharedMemory.Allocate<float>(RegBK * RegBMax); // [RegBK, blockN]

        int blockM = groupEdge * RegTM, blockN = groupEdge * RegTN;
        int tx = Group.IdxX, ty = Group.IdxY;
        int threadId = ty * groupEdge + tx;
        int nThreads = groupEdge * groupEdge;

        int blockRow = Grid.IdxY * blockM;
        int blockCol = Grid.IdxX * blockN;
        int rowStart = blockRow + ty * RegTM;
        int colStart = blockCol + tx * RegTN;

        // 4×4 accumulators as explicit registers (an array lands in slow local memory under ILGPU, killing
        // the point of register-blocking — named scalars stay in registers).
        float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0,
              c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = (d.Red + RegBK - 1) / RegBK;
        for (int kt = 0; kt < numKTiles; kt++)
        {
            int kBase = kt * RegBK;
            // Cooperatively stage A's blockM×RegBK tile and B's RegBK×blockN tile (strided over threads).
            for (int idx = threadId; idx < blockM * RegBK; idx += nThreads)
            {
                int r = idx / RegBK, k = idx % RegBK;
                int gRow = blockRow + r, gK = kBase + k;
                As[r * RegBK + k] = (gRow < d.Rows && gK < d.Red) ? a[gRow * d.ARowStride + gK * d.AColStride] : 0f;
            }
            for (int idx = threadId; idx < RegBK * blockN; idx += nThreads)
            {
                int k = idx / blockN, col = idx % blockN;
                int gK = kBase + k, gCol = blockCol + col;
                Bs[k * blockN + col] = (gK < d.Red && gCol < d.Cols) ? b[gK * d.BRowStride + gCol * d.BColStride] : 0f;
            }
            Group.Barrier();

            int aBase = ty * RegTM, bBase = tx * RegTN;
            for (int k = 0; k < RegBK; k++)
            {
                float a0 = As[(aBase + 0) * RegBK + k], a1 = As[(aBase + 1) * RegBK + k],
                      a2 = As[(aBase + 2) * RegBK + k], a3 = As[(aBase + 3) * RegBK + k];
                int bRow = k * blockN + bBase;
                float b0 = Bs[bRow + 0], b1 = Bs[bRow + 1], b2 = Bs[bRow + 2], b3 = Bs[bRow + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            Group.Barrier();
        }

        StoreReg(c, d, rowStart + 0, colStart + 0, c00); StoreReg(c, d, rowStart + 0, colStart + 1, c01);
        StoreReg(c, d, rowStart + 0, colStart + 2, c02); StoreReg(c, d, rowStart + 0, colStart + 3, c03);
        StoreReg(c, d, rowStart + 1, colStart + 0, c10); StoreReg(c, d, rowStart + 1, colStart + 1, c11);
        StoreReg(c, d, rowStart + 1, colStart + 2, c12); StoreReg(c, d, rowStart + 1, colStart + 3, c13);
        StoreReg(c, d, rowStart + 2, colStart + 0, c20); StoreReg(c, d, rowStart + 2, colStart + 1, c21);
        StoreReg(c, d, rowStart + 2, colStart + 2, c22); StoreReg(c, d, rowStart + 2, colStart + 3, c23);
        StoreReg(c, d, rowStart + 3, colStart + 0, c30); StoreReg(c, d, rowStart + 3, colStart + 1, c31);
        StoreReg(c, d, rowStart + 3, colStart + 2, c32); StoreReg(c, d, rowStart + 3, colStart + 3, c33);
    }

    /// <summary>One guarded micro-tile store for the register-blocked kernel (accumulate or write).</summary>
    private static void StoreReg(ArrayView1D<float, Stride1D.Dense> c, GemmDims d, int gRow, int gCol, float val)
    {
        if (gRow >= d.Rows || gCol >= d.Cols) return;
        int idx = gRow * d.Cols + gCol;
        if (d.Accumulate == 1) c[idx] += val; else c[idx] = val;
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

    /// <summary>
    /// Device im2col (M43): gather each output position's receptive field into <c>cols[M, inC·k²]</c>, zero-filling
    /// padding. One thread per cols element. Input <c>x</c> is NCHW <c>[B, inC·h·w]</c>; cols row m = b·hw + oh·w + ow,
    /// cols col = ((c·k+kh)·k+kw) — the exact weight index order Conv2D's im2col uses, so the reused GEMM stays correct.
    /// </summary>
    private static void Im2Col_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> cols, int inC, int k, int pad, int h, int w)
    {
        int gid = index;
        int hw = h * w;
        int inCk2 = inC * k * k;
        int m = gid / inCk2, kcol = gid % inCk2;
        int b = m / hw, sp = m % hw;
        int oh = sp / w, ow = sp % w;
        int kw = kcol % k, t = kcol / k;
        int kh = t % k, c = t / k;
        int ih = oh - pad + kh, iw = ow - pad + kw;
        cols[gid] = (ih >= 0 && ih < h && iw >= 0 && iw < w) ? x[b * inC * hw + c * hw + ih * w + iw] : 0f;
    }

    /// <summary>
    /// Device scatter + per-channel bias (M43): a conv GEMM result <c>mat[M, outC]</c> → NCHW <c>out[B, outC·hw]</c>,
    /// adding <c>bias[oc]</c> (broadcast over the hw spatial positions — distinct from BiasActivation's per-column
    /// bias). One thread per output element; m = b·hw + sp.
    /// </summary>
    private static void ScatterBias_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> mat, ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> outp, int outC, int hw)
    {
        int gid = index;
        int sp = gid % hw, rem = gid / hw;
        int oc = rem % outC, b = rem / outC;
        outp[gid] = mat[(b * hw + sp) * outC + oc] + bias[oc];
    }

    // ── conv training kernels (M44): the transposes of im2col/scatter + the two-headed loss grads ──

    /// <summary>
    /// Device col2im — the exact transpose of <see cref="Im2Col_Kernel"/> (M44). Scatter-adds the column-gradient
    /// <c>dCols[M, inC·k²]</c> back into NCHW <c>dInput[B, inC·h·w]</c>. One thread per <b>input</b> element sums over
    /// every kernel tap (kh,kw) that read it — an input (b,c,ih,iw) was read by output (oh=ih+pad−kh, ow=iw+pad−kw) at
    /// column ((c·k+kh)·k+kw), matching Im2Col's index order — so no atomics are needed.
    /// </summary>
    private static void Col2Im_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dCols, ArrayView1D<float, Stride1D.Dense> dInput, int inC, int k, int pad, int h, int w)
    {
        int gid = index;
        int hw = h * w;
        int inCk2 = inC * k * k;
        int b = gid / (inC * hw), rem = gid % (inC * hw);
        int c = rem / hw, sp = rem % hw;
        int ih = sp / w, iw = sp % w;
        float s = 0f;
        for (int kh = 0; kh < k; kh++)
        {
            int oh = ih + pad - kh;
            if (oh < 0 || oh >= h) continue;
            for (int kw = 0; kw < k; kw++)
            {
                int ow = iw + pad - kw;
                if (ow < 0 || ow >= w) continue;
                int m = b * hw + oh * w + ow;
                int kcol = (c * k + kh) * k + kw;
                s += dCols[(long)m * inCk2 + kcol];
            }
        }
        dInput[gid] = s;
    }

    /// <summary>
    /// Gather NCHW <c>dOut[B, outC·hw]</c> → conv-GEMM-shaped <c>dMat[M, outC]</c> (M=B·hw) — the transpose of
    /// <see cref="ScatterBias_Kernel"/> without the bias (M44). One thread per dMat element; the same index decode as
    /// scatter, run in reverse.
    /// </summary>
    private static void GatherNCHWToMOutC_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dOut, ArrayView1D<float, Stride1D.Dense> dMat, int outC, int hw)
    {
        int gid = index;
        int oc = gid % outC, rem = gid / outC; // rem = m = b·hw + sp
        int sp = rem % hw, b = rem / hw;
        dMat[gid] = dOut[(long)(b * outC + oc) * hw + sp];
    }

    // ── resident training kernels (M20 Stage 3) ──

    /// <summary>Bias gradient: one thread per column, <c>dBias[c] = Σ_r dy[r·dim+c]</c>.</summary>
    private static void BiasGrad_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> dBias, int rows, int dim)
    {
        int c = index;
        float s = 0f;
        for (int r = 0; r < rows; r++) s += dy[r * dim + c];
        dBias[c] = s;
    }

    /// <summary>ReLU backward in place: zero the gradient where the (cached) ReLU output was ≤ 0.</summary>
    private static void ReluBackward_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> post)
    {
        if (post[index] <= 0f) grad[index] = 0f;
    }

    /// <summary>LayerNorm input gradient (one thread per row): dx = (1/σ)(dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂)), dx̂ = dy·γ.</summary>
    private static void LayerNormInputGrad_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> invStd, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> dx, int dim)
    {
        int r = index;
        int b = r * dim;
        float sum1 = 0f, sum2 = 0f;
        for (int c = 0; c < dim; c++) { float dxh = dy[b + c] * gamma[c]; sum1 += dxh; sum2 += dxh * xhat[b + c]; }
        float invN = 1f / dim, inv = invStd[r];
        for (int c = 0; c < dim; c++)
        {
            float dxh = dy[b + c] * gamma[c];
            dx[b + c] = inv * (dxh - sum1 * invN - xhat[b + c] * sum2 * invN);
        }
    }

    /// <summary>LayerNorm γ/β gradients (one thread per column): dγ[c]=Σ_r dy·x̂, dβ[c]=Σ_r dy.</summary>
    private static void LayerNormParamGrad_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dy, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> dGamma, ArrayView1D<float, Stride1D.Dense> dBeta, int rows, int dim)
    {
        int c = index;
        float dg = 0f, db = 0f;
        for (int r = 0; r < rows; r++) { float d = dy[r * dim + c]; dg += d * xhat[r * dim + c]; db += d; }
        dGamma[c] = dg; dBeta[c] = db;
    }

    /// <summary>Mean-Huber gradient (δ=1) into dOut: <c>clamp(pred−target, ±1)·scale</c>, scale = 1/rows.</summary>
    private static void HuberGrad_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> pred, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> dOut, float scale)
    {
        float diff = pred[index] - target[index];
        float d = diff < -1f ? -1f : (diff > 1f ? 1f : diff);
        dOut[index] = d * scale;
    }

    /// <summary>In-place Adam update of one parameter buffer (one thread per element).</summary>
    private static void AdamUpdate_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> w, ArrayView1D<float, Stride1D.Dense> g, ArrayView1D<float, Stride1D.Dense> m, ArrayView1D<float, Stride1D.Dense> v, AdamParams p)
    {
        int i = index;
        float gi = g[i];
        float mi = p.Beta1 * m[i] + (1f - p.Beta1) * gi;
        float vi = p.Beta2 * v[i] + (1f - p.Beta2) * gi * gi;
        m[i] = mi; v[i] = vi;
        float mhat = mi / p.BiasCorr1, vhat = vi / p.BiasCorr2;
        w[i] -= p.Lr * mhat / (MathF.Sqrt(vhat) + p.Eps);
    }

    /// <summary>Σ data[i]² accumulated atomically into accum[0]; <paramref name="partitions"/> strided threads.</summary>
    private static void SumSq_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> accum, int partitions)
    {
        int i = index;
        float local = 0f;
        for (long j = i; j < data.Length; j += partitions) { float x = data[j]; local += x * x; }
        Atomic.Add(ref accum[0], local);
    }

    /// <summary>In-place scale (gradient-norm clipping): <c>buf[i] *= scalar</c>.</summary>
    private static void ScaleInPlace_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> buf, float scalar)
        => buf[index] *= scalar;

    /// <summary>LayerNorm forward caching x̂ and 1/σ for backward, no scale-shift-relu fusion beyond γ/β.</summary>
    private static void LayerNormTrain_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView1D<float, Stride1D.Dense> xhat, ArrayView1D<float, Stride1D.Dense> invStd, int dim, float eps)
    {
        int r = index;
        int b = r * dim;
        float mean = 0f;
        for (int c = 0; c < dim; c++) mean += data[b + c];
        mean /= dim;
        float var = 0f;
        for (int c = 0; c < dim; c++) { float dd = data[b + c] - mean; var += dd * dd; }
        var /= dim;
        float inv = 1f / MathF.Sqrt(var + eps);
        invStd[r] = inv;
        for (int c = 0; c < dim; c++)
        {
            float xh = (data[b + c] - mean) * inv;
            xhat[b + c] = xh;
            data[b + c] = gamma[c] * xh + beta[c];
        }
    }

    /// <summary>In-place ReLU.</summary>
    private static void Relu_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> data)
    {
        if (data[index] < 0f) data[index] = 0f;
    }

    /// <summary>Device-to-device copy.</summary>
    private static void Copy_Kernel(Index1D index, ArrayView1D<float, Stride1D.Dense> dst, ArrayView1D<float, Stride1D.Dense> src)
        => dst[index] = src[index];

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

    /// <summary>
    /// Creates a <see cref="DeviceConvPolicyValueNet"/> bound to this backend: a two-headed AlphaZero conv-residual
    /// net whose weights live resident on-device, with the whole tower + both heads chained on-device via im2col +
    /// the tiled GEMM + scatter/bias + LayerNorm/ReLU/add (M43). This is the conv analogue of
    /// <see cref="CreateResidentForward(ResidualMlp)"/> — it removes the per-conv host↔device round-trip that makes a
    /// conv net's self-play inference transfer-bound, so `--gpu --leaf-batch N` actually keeps the device busy.
    /// </summary>
    public DeviceConvPolicyValueNet CreateResidentForward(ConvResidualPolicyValueNet net) => new(this, net);

    /// <summary>
    /// Creates a <see cref="DeviceConvResidualTrainer"/> bound to this backend: the two-headed AlphaZero conv-net
    /// training step (resident forward caching x̂/σ + full backward + grad-norm clip + Adam), the training-side sibling
    /// of <see cref="CreateResidentForward(ConvResidualPolicyValueNet)"/> and the conv analogue of
    /// <see cref="CreateResidentTrainer(ResidualMlp,int,float,float,float,float)"/> (M44). Weights are mastered
    /// on-device; call <see cref="DeviceConvResidualTrainer.SyncToHost()"/> to read them back for eval/checkpoint.
    /// </summary>
    public DeviceConvResidualTrainer CreateResidentTrainer(ConvResidualPolicyValueNet net, int batch, float learningRate,
        float clipNorm, int actions, float valueWeight, float beta2 = 0.999f)
        => new(this, net, batch, learningRate, clipNorm, actions, valueWeight, beta2: beta2);

    /// <summary>
    /// Creates a <see cref="DeviceResidualTrainer"/> bound to this backend: a fully device-resident DAVI
    /// train step (forward + backward + clip + Adam) for a residual value net (M20 Stage 3). The online
    /// weights are mastered on-device; call <see cref="DeviceResidualTrainer.SyncToHost"/> to read them
    /// back for eval/checkpoint/target sync.
    /// </summary>
    public DeviceResidualTrainer CreateResidentTrainer(ResidualMlp net, int batch, float learningRate, float clipNorm, float huberDelta = 1f, float beta2 = 0.999f)
        => new(this, net, batch, learningRate, clipNorm, huberDelta, beta2: beta2);

    /// <summary>
    /// Maps an MLP's hidden activation to the kernel's activation code (0 none, 1 ReLU). Output width is
    /// unconstrained: <see cref="DeviceMlp"/> sizes its result by the final layer (1 for a scalar value
    /// net, e.g. 12 for the EfficientCube policy head).
    /// </summary>
    internal static int ResolveActivation(Mlp net)
    {
        return net.HiddenActivation switch
        {
            Activation.None => 0,
            Activation.Relu => 1,
            _ => throw new NotSupportedException($"Resident forward supports None/ReLU, not {net.HiddenActivation}."),
        };
    }

    /// <summary>
    /// Bench-only (PLAN M19/M19b): GFLOP/s of a square GEMM with operands held resident (no per-iteration
    /// transfer), so it isolates kernel throughput. Compares the naive, simple-tiled, and register-blocked
    /// kernels — the evidence for the optimization table.
    /// </summary>
    public double BenchGemmGflops(int m, int k, int n, int iterations, GemmKind kind)
    {
        var dims = new GemmDims(m, n, k, k, 1, n, 1, accumulate: 0);
        lock (_lock)
        {
            using var bufA = _accelerator.Allocate1D<float>((long)m * k);
            using var bufB = _accelerator.Allocate1D<float>((long)k * n);
            using var bufC = _accelerator.Allocate1D<float>((long)m * n);

            void Run()
            {
                switch (kind)
                {
                    case GemmKind.Naive: _gemmNaive(new Index1D(m * n), bufA.View, bufB.View, bufC.View, dims); break;
                    case GemmKind.Tiled:
                        _gemmTiled(new KernelConfig(new Index2D((n + _tile - 1) / _tile, (m + _tile - 1) / _tile), new Index2D(_tile, _tile)),
                            bufA.View, bufB.View, bufC.View, dims, _tile);
                        break;
                    default: LaunchGemmTiled(bufA.View, bufB.View, bufC.View, dims); break; // register-blocked (production)
                }
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
        if (_ownsContext) _context.Dispose(); // a shared context (multi-GPU) is owned + disposed by the caller
    }
}
