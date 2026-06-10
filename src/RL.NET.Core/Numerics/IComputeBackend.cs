using System.Numerics.Tensors;

namespace RLNet.Core.Numerics;

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
/// <see cref="TensorPrimitives"/>. At RL net sizes (≤ a few hundred wide) this is
/// latency-bound, where thin managed code beats native-interop round trips.
/// </summary>
public sealed class ManagedBackend : IComputeBackend
{
    public void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
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

    public void GemmTransposeA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            var bRow = b.Slice(i * n, n);
            for (int p = 0; p < k; p++)
            {
                float aip = a[i * k + p];
                if (aip != 0f)
                {
                    var cRow = c.Slice(p * n, n);
                    TensorPrimitives.MultiplyAdd(bRow, aip, cRow, cRow);
                }
            }
        }
    }

    public void GemmTransposeB(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            var aRow = a.Slice(i * n, n);
            for (int p = 0; p < k; p++)
                c[i * k + p] += TensorPrimitives.Dot(aRow, b.Slice(p * n, n));
        }
    }
}
