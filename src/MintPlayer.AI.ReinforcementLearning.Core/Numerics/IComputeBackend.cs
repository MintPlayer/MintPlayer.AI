using System.Numerics.Tensors;

namespace MintPlayer.AI.ReinforcementLearning.Core.Numerics;

/// <summary>Elementwise unary nonlinearity, selected by code so one <see cref="IComputeBackend.Map"/> /
/// <see cref="IComputeBackend.MapBackward"/> pair covers them all (keeps the seam's surface small).</summary>
public enum UnaryOp { Relu, Tanh, Exp, Log, Square }

/// <summary>Elementwise binary op for <see cref="IComputeBackend.Zip"/> (forward). Backward is op-specific
/// and composed from the accumulate primitives (AddInto/SubInto/MulAddInto/MinBackwardInto).</summary>
public enum BinaryOp { Add, Sub, Mul, Min }

/// <summary>
/// The seam between the autograd layer and raw compute kernels: every op the tape records routes its
/// raw math through here, so an alternative backend (TorchSharp/ILGPU) can run the whole graph without
/// touching the algorithm code (PRD §4, "scale-up seam"). v1 shipped only GEMM here; the elementwise
/// and reduction ops are being migrated behind the seam (PLAN M20 Stage 3 / general port, Phase 1) so a
/// device backend can keep tensors resident.
/// <para>
/// Convention: forward ops WRITE their destination; <c>*Backward</c> ops ACCUMULATE (+=) into the
/// gradient destination (a tensor used by several ops sums its contributions), so callers zero grads as
/// needed. The default <see cref="ManagedBackend"/> is bitwise-deterministic regardless of thread count.
/// </para>
/// </summary>
public interface IComputeBackend
{
    /// <summary>c += a·b for row-major a[m,k], b[k,n], c[m,n].</summary>
    void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);

    /// <summary>c += aᵀ·b for row-major a[m,k], b[m,n], c[k,n].</summary>
    void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);

    /// <summary>c += a·bᵀ for row-major a[m,n], b[k,n], c[m,k].</summary>
    void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);

    /// <summary>Elementwise unary forward: <c>y[i] = op(x[i])</c> (writes y).</summary>
    void Map(UnaryOp op, ReadOnlySpan<float> x, Span<float> y);

    /// <summary>
    /// Elementwise unary backward, accumulating into <paramref name="dx"/>: <c>dx[i] += op'(·)·dy[i]</c>.
    /// Both the input <paramref name="x"/> and the forward output <paramref name="y"/> are supplied so
    /// each op uses whichever its derivative needs (ReLU/Square/Log use x; Tanh/Exp use y).
    /// </summary>
    void MapBackward(UnaryOp op, ReadOnlySpan<float> x, ReadOnlySpan<float> y, ReadOnlySpan<float> dy, Span<float> dx);

    /// <summary>Elementwise binary forward: <c>result[i] = op(a[i], b[i])</c> (writes result).</summary>
    void Zip(BinaryOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result);

    /// <summary><c>x[i] = s·x[i]</c> written to <paramref name="y"/> (MulScalar forward).</summary>
    void Scale(ReadOnlySpan<float> x, float s, Span<float> y);

    /// <summary>Clamp forward: <c>y[i] = clamp(x[i], min, max)</c>.</summary>
    void Clamp(ReadOnlySpan<float> x, float min, float max, Span<float> y);

    /// <summary>x[B,N] + bias[N] broadcast over rows → y[B,N].</summary>
    void AddBias(ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int rows, int cols, Span<float> y);

    // ── gradient-accumulate primitives (all do dst += …), the building blocks of op backwards ──

    /// <summary><c>dst[i] += src[i]</c>.</summary>
    void AddInto(Span<float> dst, ReadOnlySpan<float> src);
    /// <summary><c>dst[i] -= src[i]</c>.</summary>
    void SubInto(Span<float> dst, ReadOnlySpan<float> src);
    /// <summary><c>dst[i] += x[i]·y[i]</c> (Mul backward).</summary>
    void MulAddInto(Span<float> dst, ReadOnlySpan<float> x, ReadOnlySpan<float> y);
    /// <summary><c>dst[i] += a·x[i]</c> (MulScalar backward).</summary>
    void AxpyInto(Span<float> dst, float a, ReadOnlySpan<float> x);
    /// <summary>Clamp backward: <c>dx[i] += dy[i]</c> where <c>min &lt; x[i] &lt; max</c>.</summary>
    void ClampBackwardInto(ReadOnlySpan<float> x, float min, float max, ReadOnlySpan<float> dy, Span<float> dx);
    /// <summary>Min backward: <c>dst[i] += dy[i]</c> on the side selected by <paramref name="forA"/> (ties → a).</summary>
    void MinBackwardInto(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> dy, Span<float> dst, bool forA);
    /// <summary>Bias gradient: <c>dbias[c] += Σ_r dy[r,c]</c> over a [rows,cols] grad.</summary>
    void BiasGradInto(ReadOnlySpan<float> dy, Span<float> dbias, int rows, int cols);
}

public static class Backend
{
    public static IComputeBackend Current { get; set; } = new ManagedBackend();
}

/// <summary>
/// Pure managed SIMD backend. The BCL ships no matmul, so GEMM is hand-rolled here:
/// i-k-j saxpy ordering keeps all inner loops on contiguous rows, vectorized via
/// <see cref="TensorPrimitives"/>. Large GEMMs (wide imitation nets, big batches) are
/// parallelized across cores by partitioning DISJOINT OUTPUT ROWS — never a reduction —
/// so results stay <b>bitwise-identical</b> to the sequential path regardless of worker
/// count, preserving the SDK's determinism guarantee. Small RL nets (≤ a threshold of
/// multiply-accumulates) stay on the thin sequential path, where they are latency-bound
/// and thread dispatch would only cost. The pointer pinning is still pure managed — no
/// P/Invoke, no native dependency.
/// </summary>
public sealed unsafe class ManagedBackend : IComputeBackend
{
    /// <summary>
    /// Below this many multiply-accumulates a GEMM is latency-bound; thread dispatch costs
    /// more than it saves, so the sequential path runs. ~1 M MACs ≈ a 256×64×64 step.
    /// </summary>
    private const long ParallelMacThreshold = 1L << 20;

    private readonly int _maxDop;

    /// <param name="maxDegreeOfParallelism">
    /// Worker cap for large GEMMs; defaults to the core count. Pass 1 to force the
    /// sequential path. Results are bitwise-identical regardless of this value (the
    /// parallelism partitions disjoint output rows, never a reduction), so seeded-curve
    /// and determinism tests are unaffected by the choice.
    /// </param>
    public ManagedBackend(int? maxDegreeOfParallelism = null)
        => _maxDop = Math.Max(1, maxDegreeOfParallelism ?? Environment.ProcessorCount);

    public void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        if (!Parallelize(m, (long)m * k * n)) { GemmRows(a, b, c, k, n, 0, m); return; }
        int la = a.Length, lb = b.Length, lc = c.Length;
        fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = c)
        {
            nint ia = (nint)pa, ib = (nint)pb, ic = (nint)pc;
            ForRowRanges(m, (s, e) => GemmRows(
                new ReadOnlySpan<float>((float*)ia, la), new ReadOnlySpan<float>((float*)ib, lb),
                new Span<float>((float*)ic, lc), k, n, s, e));
        }
    }

    public void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        // Output is c[k,n]; parallelize over its k rows (p), so each worker owns disjoint
        // rows. This is the i/p loops swapped vs. the saxpy form, with identical per-row
        // accumulation order — hence bitwise-identical.
        if (!Parallelize(k, (long)m * k * n)) { GemmTransposeARows(a, b, c, m, k, n, 0, k); return; }
        int la = a.Length, lb = b.Length, lc = c.Length;
        fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = c)
        {
            nint ia = (nint)pa, ib = (nint)pb, ic = (nint)pc;
            ForRowRanges(k, (s, e) => GemmTransposeARows(
                new ReadOnlySpan<float>((float*)ia, la), new ReadOnlySpan<float>((float*)ib, lb),
                new Span<float>((float*)ic, lc), m, k, n, s, e));
        }
    }

    public void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        if (!Parallelize(m, (long)m * k * n)) { GemmTransposeBRows(a, b, c, k, n, 0, m); return; }
        int la = a.Length, lb = b.Length, lc = c.Length;
        fixed (float* pa = a) fixed (float* pb = b) fixed (float* pc = c)
        {
            nint ia = (nint)pa, ib = (nint)pb, ic = (nint)pc;
            ForRowRanges(m, (s, e) => GemmTransposeBRows(
                new ReadOnlySpan<float>((float*)ia, la), new ReadOnlySpan<float>((float*)ib, lb),
                new Span<float>((float*)ic, lc), k, n, s, e));
        }
    }

    // ── sequential row-range kernels (one output-row band; the unit of parallel work) ──

    private static void GemmRows(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int k, int n, int rowStart, int rowEnd)
    {
        for (int i = rowStart; i < rowEnd; i++)
        {
            var cRow = c.Slice(i * n, n);
            for (int p = 0; p < k; p++)
            {
                float aip = a[i * k + p];
                if (aip != 0f)
                    TensorPrimitives.MultiplyAdd(b.Slice(p * n, n), aip, cRow, cRow);
            }
        }
    }

    private static void GemmTransposeARows(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n, int rowStart, int rowEnd)
    {
        for (int p = rowStart; p < rowEnd; p++)
        {
            var cRow = c.Slice(p * n, n);
            for (int i = 0; i < m; i++)
            {
                float aip = a[i * k + p];
                if (aip != 0f)
                    TensorPrimitives.MultiplyAdd(b.Slice(i * n, n), aip, cRow, cRow);
            }
        }
    }

    private static void GemmTransposeBRows(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int k, int n, int rowStart, int rowEnd)
    {
        for (int i = rowStart; i < rowEnd; i++)
        {
            var aRow = a.Slice(i * n, n);
            for (int p = 0; p < k; p++)
                c[i * k + p] += TensorPrimitives.Dot(aRow, b.Slice(p * n, n));
        }
    }

    public void Map(UnaryOp op, ReadOnlySpan<float> x, Span<float> y)
    {
        switch (op)
        {
            case UnaryOp.Relu: for (int i = 0; i < x.Length; i++) y[i] = Math.Max(0f, x[i]); break;
            case UnaryOp.Tanh: TensorPrimitives.Tanh(x, y); break;
            case UnaryOp.Exp: TensorPrimitives.Exp(x, y); break;
            case UnaryOp.Log: TensorPrimitives.Log(x, y); break;
            case UnaryOp.Square: TensorPrimitives.Multiply(x, x, y); break;
        }
    }

    public void MapBackward(UnaryOp op, ReadOnlySpan<float> x, ReadOnlySpan<float> y, ReadOnlySpan<float> dy, Span<float> dx)
    {
        switch (op)
        {
            case UnaryOp.Relu: for (int i = 0; i < x.Length; i++) { if (x[i] > 0f) dx[i] += dy[i]; } break;
            case UnaryOp.Tanh: for (int i = 0; i < x.Length; i++) dx[i] += (1f - y[i] * y[i]) * dy[i]; break;
            case UnaryOp.Exp: for (int i = 0; i < x.Length; i++) dx[i] += y[i] * dy[i]; break;
            case UnaryOp.Log: for (int i = 0; i < x.Length; i++) dx[i] += dy[i] / x[i]; break;
            case UnaryOp.Square: for (int i = 0; i < x.Length; i++) dx[i] += 2f * x[i] * dy[i]; break;
        }
    }

    public void Zip(BinaryOp op, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        switch (op)
        {
            case BinaryOp.Add: TensorPrimitives.Add(a, b, result); break;
            case BinaryOp.Sub: TensorPrimitives.Subtract(a, b, result); break;
            case BinaryOp.Mul: TensorPrimitives.Multiply(a, b, result); break;
            case BinaryOp.Min: for (int i = 0; i < a.Length; i++) result[i] = Math.Min(a[i], b[i]); break;
        }
    }

    public void Scale(ReadOnlySpan<float> x, float s, Span<float> y) => TensorPrimitives.Multiply(x, s, y);

    public void Clamp(ReadOnlySpan<float> x, float min, float max, Span<float> y)
    {
        for (int i = 0; i < x.Length; i++) y[i] = Math.Clamp(x[i], min, max);
    }

    public void AddBias(ReadOnlySpan<float> x, ReadOnlySpan<float> bias, int rows, int cols, Span<float> y)
    {
        for (int r = 0; r < rows; r++)
            TensorPrimitives.Add(x.Slice(r * cols, cols), bias, y.Slice(r * cols, cols));
    }

    public void AddInto(Span<float> dst, ReadOnlySpan<float> src) => TensorPrimitives.Add(dst, src, dst);
    public void SubInto(Span<float> dst, ReadOnlySpan<float> src) => TensorPrimitives.Subtract(dst, src, dst);
    public void MulAddInto(Span<float> dst, ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] += x[i] * y[i];
    }
    public void AxpyInto(Span<float> dst, float a, ReadOnlySpan<float> x)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] += a * x[i];
    }
    public void ClampBackwardInto(ReadOnlySpan<float> x, float min, float max, ReadOnlySpan<float> dy, Span<float> dx)
    {
        for (int i = 0; i < x.Length; i++) if (x[i] > min && x[i] < max) dx[i] += dy[i];
    }
    public void MinBackwardInto(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> dy, Span<float> dst, bool forA)
    {
        for (int i = 0; i < a.Length; i++) if (forA ? a[i] <= b[i] : a[i] > b[i]) dst[i] += dy[i];
    }
    public void BiasGradInto(ReadOnlySpan<float> dy, Span<float> dbias, int rows, int cols)
    {
        for (int r = 0; r < rows; r++) TensorPrimitives.Add(dbias, dy.Slice(r * cols, cols), dbias);
    }

    private bool Parallelize(int rows, long macs) => _maxDop > 1 && rows >= 2 && macs >= ParallelMacThreshold;

    /// <summary>Split <paramref name="rows"/> into ≤ _maxDop contiguous bands run concurrently.</summary>
    private void ForRowRanges(int rows, Action<int, int> body)
    {
        int dop = Math.Min(_maxDop, rows);
        int chunk = (rows + dop - 1) / dop;
        Parallel.For(0, dop, new ParallelOptions { MaxDegreeOfParallelism = dop }, t =>
        {
            int s = t * chunk, e = Math.Min(s + chunk, rows);
            if (s < e) body(s, e);
        });
    }
}
