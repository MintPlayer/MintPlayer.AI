using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The deep residual value net (PLAN M21) and its LayerNorm autograd op. LayerNorm's backward is the
/// only nontrivial new gradient, so it gets a finite-difference check; ResidualMlp gets shape,
/// checkpoint-round-trip and trainer-integration coverage (it must plug into the DAVI trainer exactly
/// like a plain MLP via <see cref="IValueNet"/>).
/// </summary>
public class ResidualMlpTests
{
    private static float[] Random(int count, ulong seed)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    [Fact]
    public void LayerNorm_NormalizesEachRow()
    {
        // With γ=1, β=0 each row of the output has ~zero mean and ~unit variance.
        const int rows = 4, cols = 8;
        var x = new Tensor(Random(rows * cols, 1), rows, cols);
        var gamma = Tensor.Full(1f, cols);
        var beta = Tensor.Zeros(cols);

        var y = x.LayerNorm(gamma, beta);

        for (int r = 0; r < rows; r++)
        {
            double mean = 0; for (int c = 0; c < cols; c++) mean += y.Data[r * cols + c]; mean /= cols;
            double var = 0; for (int c = 0; c < cols; c++) { double d = y.Data[r * cols + c] - mean; var += d * d; } var /= cols;
            Assert.True(Math.Abs(mean) < 1e-4, $"row {r} mean {mean}");
            Assert.True(Math.Abs(var - 1.0) < 1e-2, $"row {r} var {var}");
        }
    }

    [Fact]
    public void LayerNorm_GradientMatchesNumeric()
    {
        // Finite-difference check of the LayerNorm backward for x, γ and β through a scalar loss.
        const int rows = 3, cols = 5;
        float[] xd = Random(rows * cols, 2), gd = Random(cols, 3), bd = Random(cols, 4);
        float[] w = Random(rows * cols, 5); // arbitrary weights so the loss has a nontrivial gradient

        float Loss(float[] xx, float[] gg, float[] bb)
        {
            using (GradMode.NoGrad())
                return new Tensor((float[])xx.Clone(), rows, cols)
                    .LayerNorm(new Tensor((float[])gg.Clone(), cols), new Tensor((float[])bb.Clone(), cols))
                    .Mul(new Tensor(w, rows, cols)).Sum().Data[0];
        }

        var x = new Tensor((float[])xd.Clone(), rows, cols) { RequiresGrad = true };
        var g = new Tensor((float[])gd.Clone(), cols) { RequiresGrad = true };
        var b = new Tensor((float[])bd.Clone(), cols) { RequiresGrad = true };
        x.LayerNorm(g, b).Mul(new Tensor(w, rows, cols)).Sum().Backward();

        const float eps = 1e-3f;
        void Check(float[] analytic, float[] baseVals, Func<float[], float> withPerturbed, string name)
        {
            for (int i = 0; i < baseVals.Length; i++)
            {
                float orig = baseVals[i];
                baseVals[i] = orig + eps; float plus = withPerturbed(baseVals);
                baseVals[i] = orig - eps; float minus = withPerturbed(baseVals);
                baseVals[i] = orig;
                float numeric = (plus - minus) / (2 * eps);
                float tol = 1e-2f * (1f + Math.Abs(numeric));
                Assert.True(Math.Abs(numeric - analytic[i]) <= tol, $"{name}[{i}]: numeric {numeric}, analytic {analytic[i]}");
            }
        }

        Check(x.Grad!, (float[])xd.Clone(), p => Loss(p, gd, bd), "dx");
        Check(g.Grad!, (float[])gd.Clone(), p => Loss(xd, p, bd), "dg");
        Check(b.Grad!, (float[])bd.Clone(), p => Loss(xd, gd, p), "db");
    }

    [Fact]
    public void ResidualMlp_ForwardHasScalarOutput()
    {
        var net = new ResidualMlp(inputSize: 12, width: 16, blocks: 3, new Xoshiro256StarStar(7));
        const int batch = 5;
        var y = net.Forward(new Tensor(Random(batch * 12, 8), batch, 12));
        Assert.Equal(batch, y.Rows);
        Assert.Equal(1, y.Cols);
        Assert.Equal(12, net.InputSize);
        foreach (float v in y.Data) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void ResidualMlp_CheckpointRoundTrips()
    {
        var net = new ResidualMlp(inputSize: 12, width: 16, blocks: 2, new Xoshiro256StarStar(9));
        var input = new Tensor(Random(3 * 12, 10), 3, 12);
        var before = net.Forward(input).Data.ToArray();

        using var ms = new MemoryStream();
        ResidualMlpCheckpoint.Save(net, ms);
        ms.Position = 0;
        var loaded = ResidualMlpCheckpoint.Load(ms);

        Assert.Equal(net.Width, loaded.Width);
        Assert.Equal(net.Blocks, loaded.Blocks);
        var after = loaded.Forward(input).Data;
        for (int i = 0; i < before.Length; i++) Assert.Equal(before[i], after[i], 5);
    }

    [Fact]
    public void ResidualMlp_CloneStructure_IndependentButCopyable()
    {
        IValueNet net = new ResidualMlp(12, 16, 2, new Xoshiro256StarStar(11));
        var clone = net.CloneStructure();
        clone.CopyFrom(net);

        var input = new Tensor(Random(4 * 12, 12), 4, 12);
        var a = net.Forward(input).Data;
        var b = clone.Forward(input).Data;
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 5); // synced ⇒ identical outputs
    }

    [Fact]
    public void ResidualMlp_TrainsInDaviTrainer_NoThrow()
    {
        // The residual net must plug into the DAVI trainer (IValueNet) and take steps without error,
        // producing a finite value — the integration smoke (a full learning gate would be [Slow]).
        var model = new CubeModel();
        var net = new ResidualMlp(RubiksCubeEnv.ObservationSize, width: 32, blocks: 2, new Xoshiro256StarStar(13));
        var trainer = new ValueIterationTrainer<FaceletCube>(
            model, Featurize, net, new Adam(net.Parameters(), 1e-3f),
            new ValueIterationOptions { BatchSize = 16, DistanceScale = 1f, TargetUpdateInterval = 20 });

        var rng = new Xoshiro256StarStar(14);
        FaceletCube Sample()
        {
            var c = new FaceletCube();
            c.Apply(FaceletCube.ScrambleMoves(rng, 1 + rng.NextInt(2), quarterTurnsOnly: true));
            return c;
        }

        trainer.Train(Sample, iterations: 40);
        Assert.True(float.IsFinite(trainer.Value(new FaceletCube())));
    }

    private static float[] Featurize(FaceletCube cube)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);
        return obs;
    }

    [Theory]
    [InlineData(16, 32)]  // 2× — uniform replication
    [InlineData(16, 48)]  // 3× — uniform replication
    public void WidenTo_IntegerMultiple_PreservesFunction(int width, int newWidth)
    {
        // Net2WiderNet growth: at an integer-multiple width with NO symmetry noise, LayerNorm's mean/variance
        // are unchanged by uniform duplication, so the grown net computes the same function (to fp error).
        var net = new ResidualMlp(inputSize: 12, width, blocks: 3, new Xoshiro256StarStar(7));
        var input = new Tensor(Random(4 * 12, 8), 4, 12);
        var before = net.Forward(input).Data.ToArray();

        var wide = net.WidenTo(newWidth, new Xoshiro256StarStar(99), symmetryNoise: 0f);

        Assert.Equal(newWidth, wide.Width);
        Assert.Equal(net.Blocks, wide.Blocks);
        var after = wide.Forward(input).Data;
        for (int i = 0; i < before.Length; i++) Assert.Equal(before[i], after[i], 3); // function-preserving
    }

    [Fact]
    public void WidenTo_WithNoise_StartsNearOriginal_NotRandom()
    {
        // With symmetry-breaking noise the grown net is a WARM START — close to the original (so capability
        // carries over), not a random re-init. The jitter only lets the duplicated units diverge in training.
        var net = new ResidualMlp(inputSize: 12, width: 16, blocks: 2, new Xoshiro256StarStar(5));
        var input = new Tensor(Random(6 * 12, 6), 6, 12);
        var before = net.Forward(input).Data.ToArray();

        var wide = net.WidenTo(32, new Xoshiro256StarStar(77), symmetryNoise: 1e-3f);
        var after = wide.Forward(input).Data;

        for (int i = 0; i < before.Length; i++)
            Assert.True(MathF.Abs(before[i] - after[i]) < 0.05f * (1f + MathF.Abs(before[i])),
                $"widen-with-noise drifted too far at {i}: {before[i]} vs {after[i]}");
    }

    [Fact]
    public void DeviceResidualTrainer_GradientsMatchAutograd()
    {
        // The resident backward (M20 Stage 3) must produce the same gradients as the autograd path for
        // one DAVI train step — this validates every backward kernel (GEMM-transpose, LayerNorm grad,
        // ReLU grad, bias grad, Huber grad) at once. Gradients (not post-Adam weights) are compared,
        // because step-1 Adam ≈ lr·sign(grad) would mask magnitude errors.
        const int inDim = 12, width = 16, blocks = 2, batch = 8;
        var net = new ResidualMlp(inDim, width, blocks, new Xoshiro256StarStar(101));
        var features = Random(batch * inDim, 202);
        var targets = Random(batch, 203);

        // Autograd reference: forward → Reshape → mean-Huber → backward (the trainer's exact loss path).
        var adam = new Adam(net.Parameters(), 1e-3f);
        adam.ZeroGrad();
        var pred = net.Forward(new Tensor(features, batch, inDim)).Reshape(batch);
        pred.HuberLoss(new Tensor(targets, batch), 1f).Backward();
        var cpuGrads = net.Parameters().Select(p => p.Grad!.ToArray()).ToArray();

        // Resident path: forward+backward only, download gradients in the same Parameters() order.
        var ilgpu = new MintPlayer.AI.ReinforcementLearning.Ilgpu.IlgpuBackend(preferCpu: true);
        try
        {
            using var trainer = ilgpu.CreateResidentTrainer(net, batch, learningRate: 1e-3f, clipNorm: 1e9f);
            var gpuGrads = trainer.DebugGradients(features, targets, batch);

            Assert.Equal(cpuGrads.Length, gpuGrads.Length);
            for (int p = 0; p < cpuGrads.Length; p++)
            {
                Assert.Equal(cpuGrads[p].Length, gpuGrads[p].Length);
                for (int i = 0; i < cpuGrads[p].Length; i++)
                {
                    float tol = 2e-3f * (1f + MathF.Abs(cpuGrads[p][i]));
                    Assert.True(MathF.Abs(cpuGrads[p][i] - gpuGrads[p][i]) <= tol,
                        $"param {p} idx {i}: cpu {cpuGrads[p][i]}, gpu {gpuGrads[p][i]} (tol {tol})");
                }
            }
        }
        finally { ilgpu.Dispose(); }
    }

    [Fact]
    public void DeviceResidualTrainer_SyncToHost_RoundTrips()
    {
        // After taking steps, SyncToHost must write the resident weights back so the CPU net (used for
        // eval/checkpoint) reflects them — and a forward then runs without error.
        const int inDim = 12, width = 16, blocks = 2, batch = 8;
        var net = new ResidualMlp(inDim, width, blocks, new Xoshiro256StarStar(111));
        var before = net.Forward(new Tensor(Random(batch * inDim, 222), batch, inDim)).Data.ToArray();
        var features = Random(batch * inDim, 333);
        var targets = Random(batch, 444);

        var ilgpu = new MintPlayer.AI.ReinforcementLearning.Ilgpu.IlgpuBackend(preferCpu: true);
        try
        {
            using var trainer = ilgpu.CreateResidentTrainer(net, batch, 1e-2f, 5f);
            for (int s = 0; s < 5; s++) trainer.Step(features, targets, batch);
            trainer.SyncToHost(net);

            // Weights moved (training changed them) and the net still produces finite output.
            var after = net.Forward(new Tensor(Random(batch * inDim, 222), batch, inDim)).Data;
            bool changed = false;
            for (int i = 0; i < before.Length; i++) if (MathF.Abs(before[i] - after[i]) > 1e-6f) changed = true;
            Assert.True(changed, "SyncToHost should reflect the resident weight updates");
            foreach (float v in after) Assert.True(float.IsFinite(v));
        }
        finally { ilgpu.Dispose(); }
    }
}
