using System.Numerics.Tensors;

namespace MintPlayer.AI.ReinforcementLearning.Core.Numerics;

/// <summary>
/// The seam between the autograd layer and raw compute kernels. v1 ships only the
/// managed CPU backend; a TorchSharp/ILGPU backend can be slotted in later without
/// touching the algorithm code (PRD §4, "scale-up seam").
/// All GEMM kernels ACCUMULATE (+=) into the destination, which callers zero as needed —
/// backward passes accumulate gradients, so this is the common case.
/// </summary>
public interface IComputeBackend
{
    /// <summary>c += a·b for row-major a[m,k], b[k,n], c[m,n].</summary>
    void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);

    /// <summary>c += aᵀ·b for row-major a[m,k], b[m,n], c[k,n].</summary>
    void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);

    /// <summary>c += a·bᵀ for row-major a[m,n], b[k,n], c[m,k].</summary>
    void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n);
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
