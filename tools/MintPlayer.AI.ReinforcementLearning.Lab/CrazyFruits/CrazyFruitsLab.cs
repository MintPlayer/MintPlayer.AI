using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

/// <summary>
/// `--game crazyfruits` entry point (PLAN M49): runs the score-maximizing <see cref="CrazyFruitsDqnCampaign"/>
/// on the shared <see cref="CampaignRunner"/>. CPU-only (a 448→256→256 MLP is far below the GPU threshold).
/// `--baselines N` skips training and prints the scripted-policy table (random / greedy / expectimax-1, plus
/// the trained net when `--net` exists) over N seeded episodes with 95% CIs — the M49.2/M49.3 gate evidence.
/// </summary>
internal static class CrazyFruitsLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        int moveBudget = a.Int("--move-budget", 30);
        int chunkSteps = a.Int("--chunk-steps", 5_000);
        long targetSteps = a.Long("--steps", 150_000);
        int evalEpisodes = a.Int("--episodes", 20);
        float learningRate = a.Flt("--lr", 5e-4f);
        float explore = a.Flt("--explore", 1.0f);   // ε-start; low (e.g. 0.2) to refine a warm-started net
        int[] hidden = a.Ints("--hidden", [256, 256]);
        double gamma = a.Dbl("--gamma", 0.99);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");
        int growEvery = a.Int("--grow-every", 5000);
        int baselines = a.Int("--baselines", 0);
        string netPath = a.Str("--net", Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "crazyfruits.dqn.ckpt"));

        if (baselines > 0)
        {
            RunBaselines(baselines, moveBudget, seed, netPath);
            return;
        }

        var options = new DqnScoreOptions
        {
            Seed = seed, ChunkSteps = chunkSteps, TargetSteps = targetSteps, EvalEpisodes = evalEpisodes,
            LearningRate = learningRate, EpsilonStart = explore, Hidden = hidden, Gamma = gamma,
            Grow = grow, GrowEvery = growEvery,
        };
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddCrazyFruitsDqnCampaign(
                trainEnv: new CrazyFruitsEnv(moveBudget),
                evalEnv: new CrazyFruitsEnv(moveBudget),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "crazyfruits-dqn.csv")));
    }

    /// <summary>
    /// The falsifiable eval protocol (PRD §4): every policy plays the SAME seeded boards (seed 5000+e — the
    /// campaign's held-out eval line) for the move budget; report mean ± 95% CI. Gates: greedy beats random
    /// with non-overlapping CIs (M49.2); the net beats random by ≥ +30% with non-overlapping CIs (M49.3).
    /// </summary>
    private static void RunBaselines(int episodes, int moveBudget, ulong seed, string netPath)
    {
        Console.WriteLine($"Crazy Fruits baselines: {episodes} episodes × {moveBudget} moves (eval seeds 5000+e)");

        var results = new List<(string Name, double Mean, double Ci)>
        {
            RunPolicy("random", episodes, moveBudget, (board, move) => board.RandomAction(seed ^ 0xC0FFEE, move)),
            RunPolicy("greedy", episodes, moveBudget, (board, _) => board.GreedyAction()),
            RunPolicy("expectimax-1", episodes, moveBudget, (board, _) => board.ExpectimaxAction()),
        };

        if (File.Exists(netPath))
        {
            using var stream = File.OpenRead(netPath);
            var net = DuelingQNetCheckpoint.Load(stream);
            var agent = new GreedyQAgent(net, CrazyFruitsEnv.ActionCount);
            var env = new CrazyFruitsEnv(moveBudget);
            double sum = 0, sumSq = 0;
            for (int e = 0; e < episodes; e++)
            {
                var (obs, _) = env.Reset((ulong)(5_000 + e));
                while (true)
                {
                    var step = env.Step(agent.Act(obs, env.CurrentActionMask(), greedy: true));
                    obs = step.Observation;
                    if (step.Done) break;
                }
                sum += env.Score;
                sumSq += (double)env.Score * env.Score;
            }
            results.Add(Summarize($"net ({Path.GetFileName(netPath)})", episodes, sum, sumSq));
        }
        else
        {
            Console.WriteLine($"  (no net at {netPath} — scripted baselines only)");
        }

        foreach (var (name, mean, ci) in results)
            Console.WriteLine($"  {name,-28} mean {mean,8:F1} ± {ci:F1} (95% CI)");

        var random = results[0];
        var greedy = results[1];
        Console.WriteLine($"greedy vs random: {(greedy.Mean - greedy.Ci > random.Mean + random.Ci ? "CI-SEPARATED" : "OVERLAPPING")} " +
                          $"(+{100 * (greedy.Mean - random.Mean) / random.Mean:F0}%)");
        if (results.Count == 4)
        {
            var net = results[3];
            Console.WriteLine($"net vs random: +{100 * (net.Mean - random.Mean) / random.Mean:F1}% " +
                              $"({(net.Mean - net.Ci > random.Mean + random.Ci ? "CI-SEPARATED" : "OVERLAPPING")}; gate ≥ +30%, separated)");
            Console.WriteLine($"net vs greedy: {100 * (net.Mean - greedy.Mean) / greedy.Mean:+0.0;-0.0}% (reported, not gated)");
        }
    }

    private static (string, double, double) RunPolicy(string name, int episodes, int moveBudget, Func<CrazyFruitsBoard, int, int> policy)
    {
        double sum = 0, sumSq = 0;
        for (int e = 0; e < episodes; e++)
        {
            // Reset through the env seed path so every policy sees the same boards as the net eval.
            var env = new CrazyFruitsEnv(moveBudget);
            env.Reset((ulong)(5_000 + e));
            var b = env.Board;
            for (int move = 0; move < moveBudget; move++)
                b.ApplySwap(policy(b, move));
            sum += b.Score;
            sumSq += (double)b.Score * b.Score;
        }
        return Summarize(name, episodes, sum, sumSq);
    }

    private static (string, double, double) Summarize(string name, int n, double sum, double sumSq)
    {
        double mean = sum / n;
        double variance = Math.Max(0, sumSq / n - mean * mean);
        double ci = 1.96 * Math.Sqrt(variance / n);
        return (name, mean, ci);
    }
}
