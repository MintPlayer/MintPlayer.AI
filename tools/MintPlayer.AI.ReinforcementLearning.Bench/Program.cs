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

// ── M12b: CPU thread-scaling sweep ───────────────────────────────────────────────
// The committed evidence for M12a's gate: GEMM throughput vs. worker count, and the
// realistic end-to-end training-step gain (Amdahl-limited — only the GEMM parallelizes,
// not the elementwise ReLU/loss/Adam steps). Run on an OTHERWISE-IDLE machine; a
// competing training campaign saturates the cores and flattens the curve.
Console.WriteLine();
Console.WriteLine($"CPU thread-scaling sweep ({Environment.ProcessorCount} logical cores)");
Console.WriteLine("  NOTE: numbers are only meaningful on an idle machine — close other CPU jobs first.");
Console.WriteLine();

// Real M17 wide-net training shapes (cube imitation, batch 256, 324→1024→1024→12).
GemmScaling(256, 324, 1024, "trunk1 fwd  (B·obs→H)");
GemmScaling(256, 1024, 1024, "trunk2 fwd  (B·H→H)");
Console.WriteLine();
TrainStepScaling(batch: 256, inDim: 324, hidden: 1024, outDim: 12, label: "cube 1024-wide step");

// ── GPU column (M12c): ILGPU backend, host-span (includes host↔device transfer) ──
Console.WriteLine();
BenchGpu();

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

// ILGPU backend on the same wide-net GEMM shapes. The host-span backend includes
// host↔device transfer every call, so this is the HONEST end-to-end GPU number for the
// current (pre-device-tensor) design — small shapes lose to the CPU, large ones win.
// On a GPU-less machine ILGPU selects its CPU accelerator and this is skipped.
static void BenchGpu()
{
    Console.WriteLine("GPU (ILGPU) backend");
    Console.WriteLine($"  devices: {MintPlayer.AI.ReinforcementLearning.Ilgpu.IlgpuBackend.DescribeDevices()}");

    using var gpu = new MintPlayer.AI.ReinforcementLearning.Ilgpu.IlgpuBackend();
    Console.WriteLine($"  selected: {gpu.AcceleratorName} ({(gpu.IsGpu ? "GPU" : "CPU accelerator")})");
    if (!gpu.IsGpu)
    {
        Console.WriteLine("  no GPU present — skipping the GPU GEMM sweep (CPU accelerator would just re-measure the CPU).");
        return;
    }

    Console.WriteLine("  host-span (incl. transfer) vs. resident-operand kernel throughput (M19 naive→tiled):");
    foreach (var (m, k, n) in new[] { (256, 324, 1024), (256, 1024, 1024), (1024, 1024, 1024), (2048, 2048, 2048) })
    {
        var rng = new Xoshiro256StarStar(1);
        var a = Tensor.RandomNormal(rng, 0f, 1f, m, k);
        var b = Tensor.RandomNormal(rng, 0f, 1f, k, n);
        var c = new float[m * n];
        double flopsPerIter = 2.0 * m * k * n;

        for (int i = 0; i < 50; i++) { Array.Clear(c); gpu.Gemm(a.Data, b.Data, c, m, k, n); }
        const int iterations = 500;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) { Array.Clear(c); gpu.Gemm(a.Data, b.Data, c, m, k, n); }
        sw.Stop();
        double hostSpanGflops = flopsPerIter * iterations / sw.Elapsed.TotalSeconds / 1e9;

        // Operands resident (no per-iter transfer) — isolates kernel compute. naive vs. tiled.
        double naive = gpu.BenchGemmGflops(m, k, n, iterations, tiled: false);
        double tiled = gpu.BenchGemmGflops(m, k, n, iterations, tiled: true);
        Console.WriteLine($"  GEMM [{m},{k}]x[{k},{n}]: host-span {hostSpanGflops,7:F1} | resident naive {naive,7:F1} → tiled {tiled,7:F1} GFLOP/s  ({tiled / naive:F1}×)");
    }
}

// Logical worker counts to sweep: 1, 2, 4, … up to the core count, plus the core count.
static int[] DopLadder()
{
    var rungs = new List<int>();
    for (int d = 1; d < Environment.ProcessorCount; d *= 2) rungs.Add(d);
    rungs.Add(Environment.ProcessorCount);
    return rungs.Distinct().ToArray();
}

static void GemmScaling(int m, int k, int n, string label)
{
    var rng = new Xoshiro256StarStar(1);
    var a = Tensor.RandomNormal(rng, 0f, 1f, m, k);
    var b = Tensor.RandomNormal(rng, 0f, 1f, k, n);
    var c = new float[m * n];
    double flopsPerIter = 2.0 * m * k * n;

    Console.WriteLine($"GEMM [{m},{k}]x[{k},{n}]  {label}");
    double baseGflops = 0;
    foreach (int dop in DopLadder())
    {
        var backend = new ManagedBackend(dop);
        for (int i = 0; i < 200; i++) { Array.Clear(c); backend.Gemm(a.Data, b.Data, c, m, k, n); }

        const int iterations = 2_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) { Array.Clear(c); backend.Gemm(a.Data, b.Data, c, m, k, n); }
        sw.Stop();

        double gflops = flopsPerIter * iterations / sw.Elapsed.TotalSeconds / 1e9;
        if (dop == 1) baseGflops = gflops;
        Console.WriteLine($"    dop {dop,3}: {gflops,7:F1} GFLOP/s   {gflops / baseGflops,5:F2}x");
    }
}

static void TrainStepScaling(int batch, int inDim, int hidden, int outDim, string label)
{
    Console.WriteLine($"full Adam step, batch {batch}, {inDim}->{hidden}->{hidden}->{outDim}  {label}");
    double baseSps = 0;
    foreach (int dop in DopLadder())
    {
        Backend.Current = new ManagedBackend(dop);
        var rng = new Xoshiro256StarStar(2);
        var net = new Mlp([inDim, hidden, hidden, outDim], rng, Activation.Relu);
        var adam = new Adam(net.Parameters(), 1e-3f);
        var input = Tensor.RandomNormal(rng, 0f, 1f, batch, inDim);
        var target = Tensor.RandomNormal(rng, 0f, 1f, batch, outDim);

        for (int i = 0; i < 50; i++) Step(net, adam, input, target);

        const int iterations = 400;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) Step(net, adam, input, target);
        sw.Stop();

        double sps = iterations / sw.Elapsed.TotalSeconds;
        if (dop == 1) baseSps = sps;
        Console.WriteLine($"    dop {dop,3}: {sps,8:N0} steps/s  ({sps * batch,12:N0} samples/s)  {sps / baseSps,5:F2}x");
    }
    Backend.Current = new ManagedBackend(); // restore default

    static void Step(Mlp net, Adam adam, Tensor input, Tensor target)
    {
        adam.ZeroGrad();
        var loss = net.Forward(input).MseLoss(target);
        loss.Backward();
        adam.ClipGradNorm(10f);
        adam.Step();
    }
}
