using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The adaptive backend must compute correct GEMMs whichever device it routes to. Forcing the
/// threshold to its extremes exercises both routes deterministically — all-CPU (exact match to
/// ManagedBackend) and all-GPU-if-present (approximate, FMA vs separate mul+add). On a GPU-less
/// machine both routes are the CPU, so everything is exact.
/// </summary>
public class AdaptiveBackendTests
{
    private static float[] Random(int count, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    private const int M = 96, K = 80, N = 72;

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 1e-3f * (1f + MathF.Abs(expected[i]));
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"index {i}: expected {expected[i]}, got {actual[i]} (tol {tol})");
        }
    }

    [Fact]
    public void AllCpuRoute_MatchesManagedBackend()
    {
        var a = Random(M * K, 1); var b = Random(K * N, 2);
        var reference = new float[M * N]; var got = new float[M * N];
        new ManagedBackend(1).Gemm(a, b, reference, M, K, N);

        using var adaptive = new AdaptiveBackend(gpuMacThreshold: long.MaxValue); // never route to GPU
        adaptive.Gemm(a, b, got, M, K, N);

        AssertClose(reference, got);
    }

    [Fact]
    public void AllGpuRoute_MatchesManagedBackend()
    {
        var a = Random(M * K, 3); var b = Random(K * N, 4);
        var reference = new float[M * N]; var got = new float[M * N];
        new ManagedBackend(1).Gemm(a, b, reference, M, K, N);

        using var adaptive = new AdaptiveBackend(gpuMacThreshold: 0); // route everything to GPU if present
        adaptive.Gemm(a, b, got, M, K, N);

        AssertClose(reference, got);
    }

    [Fact]
    public void AllRoutes_CorrectForEveryKernel()
    {
        // Whichever way each kernel routes, the result must match the CPU reference.
        using var adaptive = new AdaptiveBackend(gpuMacThreshold: 0);

        var a1 = Random(M * K, 5); var b1 = Random(K * N, 6);
        var rGemm = new float[M * N]; var gGemm = new float[M * N];
        new ManagedBackend(1).Gemm(a1, b1, rGemm, M, K, N);
        adaptive.Gemm(a1, b1, gGemm, M, K, N);
        AssertClose(rGemm, gGemm);

        var a2 = Random(M * K, 7); var b2 = Random(M * N, 8);
        var rTA = new float[K * N]; var gTA = new float[K * N];
        new ManagedBackend(1).GemmTransposeA(a2, b2, rTA, M, K, N);
        adaptive.GemmTransposeA(a2, b2, gTA, M, K, N);
        AssertClose(rTA, gTA);

        var a3 = Random(M * N, 9); var b3 = Random(K * N, 10);
        var rTB = new float[M * K]; var gTB = new float[M * K];
        new ManagedBackend(1).GemmTransposeB(a3, b3, rTB, M, K, N);
        adaptive.GemmTransposeB(a3, b3, gTB, M, K, N);
        AssertClose(rTB, gTB);
    }
}
