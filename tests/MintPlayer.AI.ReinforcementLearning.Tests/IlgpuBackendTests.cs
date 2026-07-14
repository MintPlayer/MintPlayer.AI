using MintPlayer.AI.ReinforcementLearning.Core.Nn;
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

    [Theory]
    // The tiled kernel (16×16) must be correct for exact-tile multiples AND ragged tails on every
    // dimension (the cube's k=324 is not a multiple of 16), plus skinny/rectangular shapes.
    [InlineData(16, 16, 16)]    // single exact tile
    [InlineData(32, 32, 32)]    // multiple exact tiles
    [InlineData(17, 16, 16)]    // row tail
    [InlineData(16, 16, 17)]    // col tail
    [InlineData(16, 17, 16)]    // reduction (k) tail
    [InlineData(31, 47, 19)]    // tails on all three dims
    [InlineData(8, 324, 12)]    // the cube successor-eval shape (k=324, ragged)
    [InlineData(128, 256, 64)]  // larger rectangular
    public void Gemm_Tiled_MatchesManagedBackend_AcrossShapes(int m, int k, int n)
    {
        var a = Random(m * k, (ulong)(100 + m)); var b = Random(k * n, (ulong)(200 + n));
        var cpu = new float[m * n]; var gpu = new float[m * n];

        new ManagedBackend(1).Gemm(a, b, cpu, m, k, n);
        using var backend = new IlgpuBackend(preferCpu: true);
        backend.Gemm(a, b, gpu, m, k, n);

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
    public void DeviceMlp_MatchesAutogradForward()
    {
        // The resident-weight forward (M20) must match the autograd Mlp.Forward for a scalar-output
        // ReLU net — same chain as MlpForwardScalar, but with weights uploaded once and held resident.
        var rng = new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(33);
        var net = new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp(
            [12, 32, 16, 1], rng, MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu);

        const int batch = 20, inDim = 12;
        var input = Random(batch * inDim, 44);
        var reference = net.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor(input, batch, inDim)).Data;

        using var backend = new IlgpuBackend(preferCpu: true);
        using var resident = backend.CreateResidentForward(net);
        var got = resident.Forward(input, batch);

        AssertClose(reference, got);
    }

    [Fact]
    public void DeviceMlp_OnTargetSynced_AdoptsNewWeights()
    {
        // OnTargetSynced must re-upload weights: after syncing to a different net, Forward returns
        // that net's outputs — this is what lets the trainer refresh resident weights per target sync.
        var rng = new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(55);
        var first = new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp(
            [12, 24, 1], rng, MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu);
        var second = new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp(
            [12, 24, 1], new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(56),
            MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu);

        const int batch = 16, inDim = 12;
        var input = Random(batch * inDim, 66);
        var secondRef = second.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor(input, batch, inDim)).Data;

        using var backend = new IlgpuBackend(preferCpu: true);
        using var resident = backend.CreateResidentForward(first); // resident on `first`
        resident.OnTargetSynced(second);                           // …then re-synced to `second`
        var got = resident.Forward(input, batch);

        AssertClose(secondRef, got);
    }

    [Fact]
    public void DeviceResidualMlp_MatchesAutogradForward()
    {
        // The resident residual forward (M20 Stage 2 — GEMM + bias + LayerNorm/ReLU + skip-add + head,
        // all on-device) must match the autograd ResidualMlp.Forward within tolerance.
        var rng = new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(77);
        var net = new MintPlayer.AI.ReinforcementLearning.Core.Nn.ResidualMlp(inputSize: 12, width: 16, blocks: 3, rng);

        const int batch = 8, inDim = 12;
        var input = Random(batch * inDim, 88);
        var reference = net.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor(input, batch, inDim)).Data;

        using var backend = new IlgpuBackend(preferCpu: true);
        using var resident = backend.CreateResidentForward(net);
        var got = resident.Forward(input, batch);

        // Deeper net (LayerNorm + 3 blocks) accumulates more cross-backend rounding → a touch looser.
        Assert.Equal(reference.Length, got.Length);
        for (int i = 0; i < reference.Length; i++)
        {
            float tol = 3e-3f * (1f + MathF.Abs(reference[i]));
            Assert.True(MathF.Abs(reference[i] - got[i]) <= tol, $"index {i}: expected {reference[i]}, got {got[i]} (tol {tol})");
        }
    }

    [Fact]
    public void DeviceResidualMlp_OnTargetSynced_AdoptsNewWeights()
    {
        var first = new MintPlayer.AI.ReinforcementLearning.Core.Nn.ResidualMlp(12, 16, 2, new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(91));
        var second = new MintPlayer.AI.ReinforcementLearning.Core.Nn.ResidualMlp(12, 16, 2, new MintPlayer.AI.ReinforcementLearning.Core.Random.Xoshiro256StarStar(92));

        const int batch = 6, inDim = 12;
        var input = Random(batch * inDim, 93);
        var secondRef = second.Forward(new MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor(input, batch, inDim)).Data;

        using var backend = new IlgpuBackend(preferCpu: true);
        using var resident = backend.CreateResidentForward(first);
        ((MintPlayer.AI.ReinforcementLearning.Core.Planning.ITargetForward)resident).OnTargetSynced(second);
        var got = resident.Forward(input, batch);

        for (int i = 0; i < secondRef.Length; i++)
        {
            float tol = 3e-3f * (1f + MathF.Abs(secondRef[i]));
            Assert.True(MathF.Abs(secondRef[i] - got[i]) <= tol, $"index {i}: expected {secondRef[i]}, got {got[i]} (tol {tol})");
        }
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

    /// <summary>
    /// M43.2: the GPU-resident conv forward (<see cref="DeviceConvPolicyValueNet"/>) must match the autograd conv net's
    /// forward. Runs on the ILGPU CPU accelerator (no discrete GPU needed); cross-backend agreement is approximate
    /// (the resident tower's fused/sequential GEMMs + LayerNorm round differently than the CPU autograd path), so a
    /// relative+absolute tolerance, looser than the single-GEMM test to absorb accumulation over the ~14-conv tower.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(6)]
    public void DeviceConvForward_matches_autograd_conv_net(int rows)
    {
        const int planes = 18, board = 8, actions = 4672;
        int obsSize = planes * board * board;
        var net = new ConvResidualPolicyValueNet(planes, board, board, actions, filters: 8, blocks: 2,
            new Xoshiro256StarStar(1));
        var obs = Random(rows * obsSize, 7);

        var (logitsT, valueT) = net.Forward(new Core.Numerics.Tensor(obs, rows, obsSize)); // CPU autograd reference

        using var backend = new IlgpuBackend(preferCpu: true);
        using var device = backend.CreateResidentForward(net);
        var (logits, value) = device.Forward(obs, rows);

        AssertCloseTol(logitsT.Data, logits, 3e-3f);
        AssertCloseTol(valueT.Data, value, 3e-3f);
    }

    /// <summary>
    /// M44.3: the resident conv TRAINER's backward (<see cref="DeviceConvResidualTrainer"/>) must produce the same
    /// gradients as the autograd path for one AlphaZero train step — validating every new/reused backward kernel at
    /// once (col2im, the NCHW gather, the two loss grads, plus the reused GEMM-transposes / LN grads / ReLU grad / bias
    /// grad over conv maps). Gradients (not post-Adam weights) are compared, since step-1 Adam ≈ lr·sign(grad) would
    /// mask magnitude errors. Runs on the ILGPU CPU accelerator; the CUDA path runs the identical kernels.
    /// </summary>
    [Fact]
    public void DeviceConvResidualTrainer_GradientsMatchAutograd()
    {
        const int planes = 3, board = 4, actions = 8, filters = 4, blocks = 2, batch = 4;
        const float valueWeight = 0.5f;
        int obsSize = planes * board * board;
        var net = new ConvResidualPolicyValueNet(planes, board, board, actions, filters, blocks, new Xoshiro256StarStar(101));
        var obs = Random(batch * obsSize, 202);
        var pi = RandomDistributions(batch, actions, 203);          // each row sums to 1 (a valid π target)
        var z = Random(batch, 204);                                 // outcomes in [-1,1]

        // Autograd reference: the EXACT AutogradPolicyValueTrainStep loss path → backward → read grads.
        var adam = new Adam(net.Parameters(), 1e-3f);
        adam.ZeroGrad();
        var (logits, value) = net.Forward(new Tensor(obs, batch, obsSize));
        var ce = logits.LogSoftmax().Mul(new Tensor(pi, batch, actions)).Sum().MulScalar(-1f / batch);
        var valueLoss = value.Reshape(batch).Tanh().MseLoss(new Tensor(z, batch));
        ce.Add(valueLoss.MulScalar(valueWeight)).Backward();
        var cpuGrads = net.Parameters().Select(p => p.Grad!.ToArray()).ToArray();

        using var backend = new IlgpuBackend(preferCpu: true);
        using var trainer = backend.CreateResidentTrainer(net, batch, learningRate: 1e-3f, clipNorm: 1e9f, actions, valueWeight);
        var gpuGrads = trainer.DebugGradients(obs, pi, z);

        Assert.Equal(cpuGrads.Length, gpuGrads.Length);
        for (int p = 0; p < cpuGrads.Length; p++)
        {
            Assert.Equal(cpuGrads[p].Length, gpuGrads[p].Length);
            for (int i = 0; i < cpuGrads[p].Length; i++)
            {
                float tol = 5e-3f * (1f + MathF.Abs(cpuGrads[p][i]));
                Assert.True(MathF.Abs(cpuGrads[p][i] - gpuGrads[p][i]) <= tol,
                    $"param {p} idx {i}: cpu {cpuGrads[p][i]}, gpu {gpuGrads[p][i]} (tol {tol})");
            }
        }
    }

    /// <summary>M44.3: after a step, SyncToHost writes the resident (trained) weights back into the CPU conv net so
    /// eval/arena/checkpoint reflect them — and it actually changed the weights.</summary>
    [Fact]
    public void DeviceConvResidualTrainer_SyncToHost_RoundTrips()
    {
        const int planes = 3, board = 4, actions = 8, filters = 4, blocks = 1, batch = 4;
        int obsSize = planes * board * board;
        var net = new ConvResidualPolicyValueNet(planes, board, board, actions, filters, blocks, new Xoshiro256StarStar(111));
        var before = net.Parameters().First().Data.ToArray();
        var obs = Random(batch * obsSize, 222);
        var pi = RandomDistributions(batch, actions, 223);
        var z = Random(batch, 224);

        using var backend = new IlgpuBackend(preferCpu: true);
        using var trainer = backend.CreateResidentTrainer(net, batch, learningRate: 1e-2f, clipNorm: 5f, actions, valueWeight: 1f);
        trainer.Step(obs, pi, z, batch);
        trainer.SyncToHost();

        var after = net.Parameters().First().Data.ToArray();
        Assert.Contains(Enumerable.Range(0, before.Length), i => MathF.Abs(before[i] - after[i]) > 1e-6f);
        _ = net.Forward(new Tensor(obs, batch, obsSize)); // the synced-back net still runs
    }

    // Random row-major [rows, cols] where each row is a probability distribution (positive, sums to 1).
    private static float[] RandomDistributions(int rows, int cols, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            float sum = 0f;
            for (int c = 0; c < cols; c++) { float v = (float)rng.NextDouble() + 1e-3f; data[r * cols + c] = v; sum += v; }
            for (int c = 0; c < cols; c++) data[r * cols + c] /= sum;
        }
        return data;
    }

    private static void AssertCloseTol(float[] expected, float[] actual, float rel)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = rel * (1f + MathF.Abs(expected[i]));
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"index {i}: expected {expected[i]}, got {actual[i]} (tol {tol})");
        }
    }
}
