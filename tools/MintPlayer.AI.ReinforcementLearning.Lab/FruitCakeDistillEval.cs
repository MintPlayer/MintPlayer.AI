using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// The honest verdict for planner distillation (F6): on the SAME paired seeds, compares three serving arms on the
/// shipped DQN + the distilled <see cref="FruitCakePolicyNet"/> (both loaded from --data):
/// <list type="bullet">
/// <item><b>dqn+search</b> — the current shipped system (DQN max-Q leaf, depth-N search). The baseline to beat.</item>
/// <item><b>policy</b> — the distilled policy head alone, one forward pass, NO search (the cheap-serving win).</item>
/// <item><b>policy+search</b> — the distilled net's value head as the search leaf (does the planning-aware leaf
/// push search past the DQN-leaf ceiling?).</item>
/// </list>
/// Reports each arm's score + max-tier distribution incl. the watermelon count. Games are independent → run
/// concurrently with the backend forced single-threaded (the games are the parallelism).
/// </summary>
internal static class FruitCakeDistillEval
{
    public static void Run(string netDir, int episodes, int depth, int topK, int topK2, ulong seedBase)
    {
        var dqn = LoadDqn(netDir);
        var policy = LoadPolicy(netDir);

        Console.WriteLine($"FruitCake distillation eval — {episodes} paired games (seeds {seedBase}..{seedBase + (ulong)episodes - 1}), search depth={depth}, topK={topK}, topK2={topK2}:");
        Console.WriteLine($"  net dir: {netDir}");

        var dqnSearchScore = new double[episodes];
        var policyScore = new double[episodes];
        var policySearchScore = new double[episodes];
        var dqnSearchTier = new int[episodes];
        var policyTier = new int[episodes];
        var policySearchTier = new int[episodes];

        Backend.Current = new ManagedBackend(maxDegreeOfParallelism: 1);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Parallel.For(0, episodes, i =>
        {
            ulong seed = seedBase + (ulong)i;
            (dqnSearchScore[i], dqnSearchTier[i]) = PlayDqnSearch(dqn, seed, depth, topK, topK2);
            (policyScore[i], policyTier[i]) = PlayPolicy(policy, seed);
            (policySearchScore[i], policySearchTier[i]) = PlayPolicySearch(policy, seed, depth, topK, topK2);
        });
        sw.Stop();

        Report("dqn+search   ", dqnSearchScore, dqnSearchTier, episodes);
        Report("policy       ", policyScore, policyTier, episodes);
        Report("policy+search", policySearchScore, policySearchTier, episodes);
        Console.WriteLine($"  ({sw.Elapsed.TotalSeconds:F0}s)");
    }

    private static DuelingQNet LoadDqn(string dir)
    {
        var store = new FileModelStore(dir);
        using var stream = store.TryOpenRead("fruitcake", "dqn")
            ?? throw new FileNotFoundException($"No fruitcake.dqn.ckpt under '{dir}'.");
        return DuelingQNetCheckpoint.Load(stream);
    }

    private static FruitCakePolicyNet LoadPolicy(string dir)
    {
        var store = new FileModelStore(dir);
        using var stream = store.TryOpenRead("fruitcake", "policy")
            ?? throw new FileNotFoundException($"No fruitcake.policy.ckpt under '{dir}' (train with --distill first).");
        return FruitCakePolicyNet.Load(stream);
    }

    private static (double Score, int MaxTier) PlayDqnSearch(DuelingQNet net, ulong seed, int depth, int topK, int topK2)
    {
        var agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);
        double Leaf(FruitCakeWorld w)
        {
            double sum = 0;
            foreach (var d in FruitCatalog.Droppable)
                sum += Max(agent.QValues(FruitCakeEnv.BuildObservation(w, d.Tier, d.Tier)));
            return sum / FruitCatalog.Droppable.Count;
        }
        return PlaySearch(new FruitCakeSearch(Leaf) { MaxDepth = depth, TopK = topK, TopK2 = topK2 }, seed);
    }

    private static (double Score, int MaxTier) PlayPolicySearch(FruitCakePolicyNet net, ulong seed, int depth, int topK, int topK2)
        => PlaySearch(new FruitCakeSearch(net.BoardValue) { MaxDepth = depth, TopK = topK, TopK2 = topK2 }, seed);

    private static (double Score, int MaxTier) PlaySearch(FruitCakeSearch search, ulong seed)
    {
        var env = new FruitCakeEnv();
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

    private static (double Score, int MaxTier) PlayPolicy(FruitCakePolicyNet net, ulong seed)
    {
        var env = new FruitCakeEnv();
        env.Reset(seed);
        int maxTier = 0;
        while (true)
        {
            int col = net.ChooseColumn(env.World, env.CurrentTier, env.NextTier);
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

    private static void Report(string label, double[] score, int[] tier, int episodes)
    {
        double mean = score.Average();
        double sd = Std(score, mean);
        int melon = tier.Count(t => t >= FruitCatalog.TopTier);
        var hist = tier.GroupBy(t => t).OrderByDescending(g => g.Key).Select(g => $"t{g.Key}:{g.Count()}");
        Console.WriteLine($"  {label}: mean {mean,7:F1} ± {sd,5:F0} (SD)  meanTier {tier.Average():F2}  watermelon {melon}/{episodes}  [{string.Join(" ", hist)}]");
    }

    private static double Std(double[] xs, double mean)
    {
        double s = 0;
        foreach (var x in xs) s += (x - mean) * (x - mean);
        return Math.Sqrt(s / xs.Length);
    }
}
