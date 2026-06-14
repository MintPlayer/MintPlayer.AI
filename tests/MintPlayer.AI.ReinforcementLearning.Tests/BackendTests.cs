using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The managed GEMM parallelizes large products across cores. These lock in the two
/// promises that make that safe: it stays correct, and it is BITWISE-identical to the
/// sequential path (parallelism partitions disjoint output rows, never a reduction), so
/// the SDK's determinism guarantee survives.
/// </summary>
public class BackendTests
{
    private static float[] Random(int count, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    // Big enough to cross ParallelMacThreshold and span several worker bands.
    private const int M = 257, K = 193, N = 129;

    [Fact]
    public void Gemm_ParallelMatchesSequential_Bitwise()
    {
        var a = Random(M * K, 1); var b = Random(K * N, 2);
        var seq = new float[M * N]; var par = new float[M * N];

        new ManagedBackend(1).Gemm(a, b, seq, M, K, N);
        new ManagedBackend(8).Gemm(a, b, par, M, K, N);

        Assert.Equal(seq, par); // exact byte-for-byte
    }

    [Fact]
    public void GemmTransposeA_ParallelMatchesSequential_Bitwise()
    {
        // c[k,n] += aᵀ·b for a[m,k], b[m,n]
        var a = Random(M * K, 3); var b = Random(M * N, 4);
        var seq = new float[K * N]; var par = new float[K * N];

        new ManagedBackend(1).GemmTransposeA(a, b, seq, M, K, N);
        new ManagedBackend(8).GemmTransposeA(a, b, par, M, K, N);

        Assert.Equal(seq, par);
    }

    [Fact]
    public void GemmTransposeB_ParallelMatchesSequential_Bitwise()
    {
        // c[m,k] += a·bᵀ for a[m,n], b[k,n]
        var a = Random(M * N, 5); var b = Random(K * N, 6);
        var seq = new float[M * K]; var par = new float[M * K];

        new ManagedBackend(1).GemmTransposeB(a, b, seq, M, K, N);
        new ManagedBackend(8).GemmTransposeB(a, b, par, M, K, N);

        Assert.Equal(seq, par);
    }

    [Fact]
    public void Gemm_MatchesNaiveReference()
    {
        var a = Random(M * K, 7); var b = Random(K * N, 8);
        var got = new float[M * N];
        new ManagedBackend(8).Gemm(a, b, got, M, K, N);

        var want = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int p = 0; p < K; p++) acc += a[i * K + p] * b[p * N + j];
                want[i * N + j] = acc;
            }

        for (int idx = 0; idx < want.Length; idx++)
            Assert.Equal(want[idx], got[idx], 3); // 3 decimals: SIMD vs scalar summation order differs
    }
}
