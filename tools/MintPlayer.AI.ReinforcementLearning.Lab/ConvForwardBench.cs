using System.Diagnostics;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// Micro-benchmark for the M43 GPU-resident conv forward: times the resident <see cref="DeviceConvPolicyValueNet"/>
/// against the autograd forward (through the <see cref="AdaptiveBackend"/> — the current non-resident path) at a
/// realistic self-play leaf batch, on the selected device. Also checks on-GPU parity (the M43.2 test verifies this on
/// the CPU accelerator; this exercises the real CUDA kernels). Dev tool — `--game chess --bench-forward`.
/// </summary>
internal static class ConvForwardBench
{
    public static void Run(int filters, int blocks, int leafBatch, int iters)
    {
        const int planes = 18, board = 8, actions = 4672, obsSize = planes * board * board;
        Console.WriteLine($"conv-forward bench: {filters}f×{blocks}b, leaf-batch {leafBatch}, {iters} timed iters");
        Console.WriteLine("  devices: " + IlgpuBackend.DescribeDevices());

        var adaptive = new AdaptiveBackend();
        Console.WriteLine($"  selected: {adaptive.Gpu?.AcceleratorName ?? "CPU accelerator only"}");

        var rng = new Xoshiro256StarStar(1);
        var net = new ConvResidualPolicyValueNet(planes, board, board, actions, filters, blocks, rng);
        var obs = new float[(long)leafBatch * obsSize];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)(rng.NextDouble() * 2 - 1);

        // Non-resident path: autograd forward routed through the AdaptiveBackend (large conv GEMMs → GPU host-span,
        // re-uploading weights per GEMM; the elementwise/LayerNorm run on CPU). This is what batched self-play uses today.
        Backend.Current = adaptive;
        var autograd = new AutogradPolicyValueForward(net, obsSize);
        var (aL, aV) = autograd.Forward(obs, leafBatch); // warm up (JIT + first-touch allocs)
        double autoMs = TimeMs(() => autograd.Forward(obs, leafBatch), iters);
        Console.WriteLine($"  autograd (AdaptiveBackend): {autoMs:F2} ms/forward  ({leafBatch / (autoMs / 1000):N0} leaves/s)");

        if (adaptive.Gpu is { } gpu)
        {
            using var device = gpu.CreateResidentForward(net);
            var (dL, dV) = device.Forward(obs, leafBatch); // warm up (weight upload + JIT)
            double resMs = TimeMs(() => device.Forward(obs, leafBatch), iters);
            Console.WriteLine($"  resident (DeviceConvPolicyValueNet): {resMs:F2} ms/forward  ({leafBatch / (resMs / 1000):N0} leaves/s)  → {autoMs / resMs:F1}× speedup");

            double maxL = 0, maxV = 0;
            for (int i = 0; i < aL.Length; i++) maxL = Math.Max(maxL, Math.Abs(aL[i] - dL[i]));
            for (int i = 0; i < aV.Length; i++) maxV = Math.Max(maxV, Math.Abs(aV[i] - dV[i]));
            Console.WriteLine($"  on-GPU parity: max |Δlogit| {maxL:E2}, max |Δvalue| {maxV:E2}");
        }
        else Console.WriteLine("  no discrete GPU selected → resident path skipped (autograd is the only path here)");

        (adaptive as IDisposable)?.Dispose();
    }

    private static double TimeMs(Action f, int iters)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) f();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }
}
