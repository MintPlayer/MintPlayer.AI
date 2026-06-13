using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The ILGPU backend (M12c) must compute the same three GEMMs as <see cref="ManagedBackend"/>.
/// These run on ILGPU's CPU accelerator (<c>preferCpu: true</c>) so they pass on any machine,
/// GPU or not — the CUDA path runs the identical kernels. Cross-backend equality is
/// approximate, not bitwise: a GPU/accelerator may fuse multiply-add, rounding differently
/// than the CPU's separate TensorPrimitives ops.
/// </summary>
public class IlgpuBackendTests
{
    private static float[] Random(int count, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    private const int M = 64, K = 48, N = 40;

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        // Cross-backend agreement is approximate: the CPU's SIMD-tree reduction
        // (TensorPrimitives) and the kernel's sequential/fused reduction round differently.
        // A relative+absolute tolerance is the right test; a wrong kernel is off by O(1).
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 1e-3f * (1f + MathF.Abs(expected[i]));
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"index {i}: expected {expected[i]}, got {actual[i]} (tol {tol})");
        }
    }

    [Fact]
    public void Gemm_MatchesManagedBackend()
    {
        var a = Random(M * K, 1); var b = Random(K * N, 2);
        var cpu = new float[M * N]; var gpu = new float[M * N];

        new ManagedBackend(1).Gemm(a, b, cpu, M, K, N);
        using var backend = new IlgpuBackend(preferCpu: true);
        backend.Gemm(a, b, gpu, M, K, N);

        AssertClose(cpu, gpu);
    }

    [Fact]
    public void GemmTransposeA_MatchesManagedBackend()
    {
        var a = Random(M * K, 3); var b = Random(M * N, 4);
        var cpu = new float[K * N]; var gpu = new float[K * N];

        new ManagedBackend(1).GemmTransposeA(a, b, cpu, M, K, N);
        using var backend = new IlgpuBackend(preferCpu: true);
        backend.GemmTransposeA(a, b, gpu, M, K, N);

        AssertClose(cpu, gpu);
    }

    [Fact]
    public void GemmTransposeB_MatchesManagedBackend()
    {
        var a = Random(M * N, 5); var b = Random(K * N, 6);
        var cpu = new float[M * K]; var gpu = new float[M * K];

        new ManagedBackend(1).GemmTransposeB(a, b, cpu, M, K, N);
        using var backend = new IlgpuBackend(preferCpu: true);
        backend.GemmTransposeB(a, b, gpu, M, K, N);

        AssertClose(cpu, gpu);
    }

    [Fact]
    public void MlpForwardScalar_MatchesAutogradForward()
    {
        // The device-resident forward (GEMM→bias→ReLU chained on-device) must match the
        // backend-agnostic autograd Mlp.Forward for a scalar-output ReLU net.
        var rng = new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(11);
        var net = new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp(
            [12, 32, 16, 1], rng, MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu);

        const int batch = 20, inDim = 12;
        var input = Random(batch * inDim, 22);

        // Reference: autograd forward on the default ManagedBackend.
        var reference = net.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor(input, batch, inDim)).Data;

        using var backend = new IlgpuBackend(preferCpu: true);
        var got = backend.MlpForwardScalar(net, input, batch);

        AssertClose(reference, got);
    }

    [Fact]
    public void Gemm_Accumulates_LikeTheContract()
    {
        // The interface contract is c += a·b; running twice must double the result.
        var a = Random(M * K, 7); var b = Random(K * N, 8);
        var once = new float[M * N]; var twice = new float[M * N];

        using var backend = new IlgpuBackend(preferCpu: true);
        backend.Gemm(a, b, once, M, K, N);
        backend.Gemm(a, b, twice, M, K, N);
        backend.Gemm(a, b, twice, M, K, N);

        for (int i = 0; i < once.Length; i++)
            Assert.Equal(2f * once[i], twice[i], 3);
    }
}
