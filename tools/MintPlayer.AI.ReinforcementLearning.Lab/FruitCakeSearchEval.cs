using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// Measures the F1 serving-side <see cref="FruitCakeSearch"/> against the plain net policy on the SAME net and
/// PAIRED seeds — the honest test of "does forward-model search break the tier ceiling the reactive net can't?"
/// Both arms use the shipped net; arm A acts greedily on its Q-values (what we ship today), arm B runs the
/// depth-1/2 search using the net's max-Q as the leaf board value. Reports each arm's score + max-tier
/// distribution (incl. the watermelon count) and the paired score Δ.
///
/// Games are independent → run concurrently with the compute backend forced single-threaded (the games are the
/// parallelism; see docs/OPTIMIZATIONS.md). Search is far costlier per drop than greedy, so prefer fewer games
/// while iterating on depth/top-K, then a larger run for the verdict.
/// </summary>
internal static class FruitCakeSearchEval
{
    public static void Run(string netDir, int episodes, int depth, int topK, ulong seedBase, string leaf = "net")
    {
        var net = Load(netDir);

        Console.WriteLine($"FruitCake search eval — {episodes} paired games (seeds {seedBase}..{seedBase + (ulong)episodes - 1}), depth={depth}, topK={topK}, leaf={leaf}:");
        Console.WriteLine($"  net: {netDir}");

        var greedyScore = new double[episodes];
        var searchScore = new double[episodes];
        var greedyTier = new int[episodes];
        var searchTier = new int[episodes];

        Backend.Current = new ManagedBackend(maxDegreeOfParallelism: 1);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Parallel.For(0, episodes, i =>
        {
            ulong seed = seedBase + (ulong)i;
            (greedyScore[i], greedyTier[i]) = PlayGreedy(net, seed);
            (searchScore[i], searchTier[i]) = PlaySearch(net, seed, depth, topK, leaf);
        });
        sw.Stop();

        Report("greedy(net)", greedyScore, greedyTier);
        Report("search(net)", searchScore, searchTier);

        var diff = new double[episodes];
        int searchWins = 0;
        for (int i = 0; i < episodes; i++)
        {
            diff[i] = searchScore[i] - greedyScore[i];
            if (searchScore[i] > greedyScore[i]) searchWins++;
        }
        double meanDiff = diff.Average();
        double se = Std(diff, meanDiff) / Math.Sqrt(episodes);
        int greedyMelon = greedyTier.Count(t => t >= FruitCatalog.TopTier);
        int searchMelon = searchTier.Count(t => t >= FruitCatalog.TopTier);
        Console.WriteLine($"  paired Δ (search − greedy): {meanDiff:+0.0;-0.0} ± {se:0.0} (SE) | search wins {searchWins}/{episodes} ({searchWins * 100.0 / episodes:0}%)");
        Console.WriteLine($"  watermelons: greedy {greedyMelon}/{episodes}, search {searchMelon}/{episodes}   ({sw.Elapsed.TotalSeconds:F0}s)");
    }

    private static DuelingQNet Load(string dir)
    {
        var store = new FileModelStore(dir);
        using var stream = store.TryOpenRead("fruitcake", "dqn")
            ?? throw new FileNotFoundException($"No fruitcake.dqn.ckpt under '{dir}'.");
        return DuelingQNetCheckpoint.Load(stream);
    }

    private static (double Score, int MaxTier) PlayGreedy(DuelingQNet net, ulong seed)
    {
        var env = new FruitCakeEnv();
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);
        var (obs, _) = env.Reset(seed);
        int maxTier = 0;
        while (true)
        {
            var step = env.Step(agent.Act(obs, greedy: true));
            obs = step.Observation;
            maxTier = Math.Max(maxTier, BoardMaxTier(env.World));
            if (step.Done) break;
        }
        return (env.Score, maxTier);
    }

    private static (double Score, int MaxTier) PlaySearch(DuelingQNet net, ulong seed, int depth, int topK, string leaf)
    {
        var env = new FruitCakeEnv();
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);

        // Net leaf: the net's sense of the board, marginalized over the (unknown) upcoming fruit by averaging
        // max-Q across the droppable tiers — board features dominate, the exact next is second-order.
        double NetValue(FruitCakeWorld w)
        {
            double sum = 0;
            foreach (var d in FruitCatalog.Droppable)
                sum += Max(agent.QValues(FruitCakeEnv.BuildObservation(w, d.Tier, d.Tier)));
            return sum / FruitCatalog.Droppable.Count;
        }
        // Tier potential: reward having big fruit on the board (geometric in tier) minus a pile-height penalty —
        // a watermelon-SEEKING leaf, unlike the pineapple-capped net. Height term keeps it from hoarding into a loss.
        static double TierPot(FruitCakeWorld w)
        {
            double v = 0;
            foreach (var b in w.Bodies) v += Math.Pow(2, b.Tier);
            return v - 8 * w.PileHeight();
        }

        Func<FruitCakeWorld, double> boardValue = leaf switch
        {
            "height" => FruitCakeSearch.HeuristicBoardValue,
            "tierpot" => TierPot,
            "blend" => w => NetValue(w) + 0.25 * TierPot(w),
            _ => NetValue,
        };
        var search = new FruitCakeSearch(boardValue) { MaxDepth = depth, TopK = topK };

        env.Reset(seed);
        int maxTier = 0;
        while (true)
        {
            int col = search.ChooseColumn(env.World, env.CurrentTier, env.NextTier);
            var step = env.Step(col);
            maxTier = Math.Max(maxTier, BoardMaxTier(env.World));
            if (step.Done) break;
        }
        return (env.Score, maxTier);
    }

    private static int BoardMaxTier(FruitCakeWorld world)
    {
        int max = 0;
        foreach (var b in world.Bodies) if (b.Tier > max) max = b.Tier;
        return max;
    }

    private static float Max(float[] xs)
    {
        float m = float.NegativeInfinity;
        foreach (var x in xs) if (x > m) m = x;
        return m;
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
