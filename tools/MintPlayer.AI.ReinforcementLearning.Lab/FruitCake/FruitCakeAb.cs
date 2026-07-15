using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// FruitCake A/B: judge two saved <see cref="DuelingQNet"/>s head-to-head over many PAIRED-seed greedy
/// episodes, so the verdict averages out the high single-eval variance (a 10-episode eval bounced 750–971
/// on the SAME net, so one such number proves nothing). Both nets play the same seed set (paired → the gap
/// estimate has lower variance), and the difference is reported with a standard error so "better" means
/// statistically better, not a lucky draw.
///
/// Episodes are independent, so they run concurrently with the compute backend forced single-threaded —
/// the N episodes ARE the parallelism (see docs/OPTIMIZATIONS.md: don't stack a second layer on the
/// already-parallel backend). Serving is deterministic (noise off), so a noisy candidate is judged on its
/// means — exactly the policy that would ship.
/// </summary>
internal static class FruitCakeAb
{
    public static void Run(string baselineDir, string candidateDir, int episodes, ulong seedBase)
    {
        var baseline = Load(baselineDir);
        var candidate = Load(candidateDir);

        Console.WriteLine($"FruitCake A/B — {episodes} paired greedy episodes (seeds {seedBase}..{seedBase + (ulong)episodes - 1}):");
        Console.WriteLine($"  baseline : {baselineDir}  (noisy={baseline.Noisy})");
        Console.WriteLine($"  candidate: {candidateDir}  (noisy={candidate.Noisy})");

        var baseScore = new double[episodes];
        var candScore = new double[episodes];
        var baseTier = new int[episodes];
        var candTier = new int[episodes];

        // The N episodes are the parallelism; single-thread the backend so we don't oversubscribe.
        Backend.Current = new ManagedBackend(maxDegreeOfParallelism: 1);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Parallel.For(0, episodes, i =>
        {
            ulong seed = seedBase + (ulong)i;
            (baseScore[i], baseTier[i]) = Play(baseline, seed);
            (candScore[i], candTier[i]) = Play(candidate, seed);
        });
        sw.Stop();

        Report("baseline ", baseScore, baseTier);
        Report("candidate", candScore, candTier);

        // Paired difference (same seed per row) → a lower-variance estimate of the true gap.
        var diff = new double[episodes];
        int candWins = 0;
        for (int i = 0; i < episodes; i++)
        {
            diff[i] = candScore[i] - baseScore[i];
            if (candScore[i] > baseScore[i]) candWins++;
        }
        double meanDiff = diff.Average();
        double se = Std(diff, meanDiff) / Math.Sqrt(episodes);
        string verdict = meanDiff > 2 * se ? "candidate is SIGNIFICANTLY BETTER → ship it"
            : meanDiff < -2 * se ? "candidate is SIGNIFICANTLY WORSE → keep the baseline"
            : "NO significant difference → keep the baseline (don't ship a tie)";

        Console.WriteLine($"  paired Δ (cand − base): {meanDiff:+0.0;-0.0} ± {se:0.0} (SE) | candidate wins {candWins}/{episodes} ({candWins * 100.0 / episodes:0}%)");
        Console.WriteLine($"  VERDICT: {verdict}   ({sw.Elapsed.TotalSeconds:F0}s)");
    }

    private static DuelingQNet Load(string dir)
    {
        var store = new FileModelStore(dir);
        using var stream = store.TryOpenRead("fruitcake", "dqn")
            ?? throw new FileNotFoundException($"No fruitcake.dqn.ckpt under '{dir}'.");
        return DuelingQNetCheckpoint.Load(stream);
    }

    private static (double Score, int MaxTier) Play(DuelingQNet net, ulong seed)
    {
        var env = new FruitCakeEnv();
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount); // noise off (loaded default) → deterministic
        var (obs, _) = env.Reset(seed);
        int maxTier = 0;
        while (true)
        {
            var step = env.Step(agent.Act(obs, greedy: true));
            obs = step.Observation;
            foreach (var b in env.World.Bodies)
                if (b.Tier > maxTier) maxTier = b.Tier;
            if (step.Done) break;
        }
        return (env.Score, maxTier);
    }

    private static void Report(string label, double[] score, int[] tier)
    {
        double mean = score.Average();
        double sd = Std(score, mean);
        var sorted = (double[])score.Clone();
        Array.Sort(sorted);
        double median = sorted[sorted.Length / 2];
        var hist = tier.GroupBy(t => t).OrderByDescending(g => g.Key).Select(g => $"t{g.Key}:{g.Count()}");
        Console.WriteLine($"  {label}: mean {mean,7:F1} ± {sd,5:F0} (SD)  median {median,6:F0}  meanTier {tier.Average():F2}  [{string.Join(" ", hist)}]");
    }

    private static double Std(double[] xs, double mean)
    {
        double s = 0;
        foreach (var x in xs) s += (x - mean) * (x - mean);
        return Math.Sqrt(s / xs.Length);
    }
}
