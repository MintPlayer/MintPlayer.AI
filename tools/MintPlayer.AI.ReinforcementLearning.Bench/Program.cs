using System.Diagnostics;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

// M2 spike benchmark (PLAN.md): validate the managed-GEMM assumption before building on it.
// Targets (PRD §7): ≥ 1,000 Adam steps/sec at batch 64 on a 4→64→64→2 net, single thread.

Console.WriteLine("MintPlayer.AI.ReinforcementLearning numerics spike benchmark");
Console.WriteLine($"  hardware-accelerated SIMD: {System.Numerics.Vector.IsHardwareAccelerated}, Vector<float>.Count = {System.Numerics.Vector<float>.Count}");
Console.WriteLine();

BenchGemm(64, 64, 64);
BenchGemm(64, 4, 64);
BenchGemm(256, 128, 128);
Console.WriteLine();
BenchTrainingStep(batch: 64, hidden: 64, targetStepsPerSec: 1000);   // the PRD §7 acceptance config
BenchTrainingStep(batch: 256, hidden: 128, targetStepsPerSec: null); // larger stress config, informational

static void BenchGemm(int m, int k, int n)
{
    var rng = new Xoshiro256StarStar(1);
    var a = Tensor.RandomNormal(rng, 0f, 1f, m, k);
    var b = Tensor.RandomNormal(rng, 0f, 1f, k, n);
    var c = new float[m * n];

    // Warm up the JIT, then measure.
    for (int i = 0; i < 1000; i++) Backend.Current.Gemm(a.Data, b.Data, c, m, k, n);

    const int iterations = 20_000;
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        Array.Clear(c);
        Backend.Current.Gemm(a.Data, b.Data, c, m, k, n);
    }
    sw.Stop();

    double flops = 2.0 * m * k * n * iterations;
    double gflops = flops / sw.Elapsed.TotalSeconds / 1e9;
    Console.WriteLine($"GEMM [{m},{k}]x[{k},{n}]: {iterations / sw.Elapsed.TotalSeconds,10:N0} ops/sec  ({gflops:F2} GFLOP/s)");
}

static void BenchTrainingStep(int batch, int hidden, int? targetStepsPerSec)
{
    var rng = new Xoshiro256StarStar(2);
    var net = new Mlp([4, hidden, hidden, 2], rng, Activation.Relu);
    var adam = new Adam(net.Parameters(), 1e-3f);
    var input = Tensor.RandomNormal(rng, 0f, 1f, batch, 4);
    var target = Tensor.RandomNormal(rng, 0f, 1f, batch, 2);

    for (int i = 0; i < 500; i++) TrainStep(net, adam, input, target);

    const int iterations = 5_000;
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++) TrainStep(net, adam, input, target);
    sw.Stop();

    double stepsPerSec = iterations / sw.Elapsed.TotalSeconds;
    string verdict = targetStepsPerSec is null ? "(informational)"
        : stepsPerSec >= targetStepsPerSec ? $"PASS (target >= {targetStepsPerSec:N0})" : $"FAIL (target >= {targetStepsPerSec:N0})";
    Console.WriteLine($"full Adam step (fwd+bwd+clip+step), batch {batch}, 4->{hidden}->{hidden}->2: " +
                      $"{stepsPerSec,8:N0} steps/sec  ({stepsPerSec * batch:N0} samples/sec)  {verdict}");

    static void TrainStep(Mlp net, Adam adam, Tensor input, Tensor target)
    {
        adam.ZeroGrad();
        var loss = net.Forward(input).MseLoss(target);
        loss.Backward();
        adam.ClipGradNorm(10f);
        adam.Step();
    }
}
