using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
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
        // Creation shaping on the TRAIN env only (SPECIALS PRD §3.5: fire-only game score means γ=0 needs a
        // reward-side creation signal); the eval env scores the bare game, so gates stay honest.
        bool shape = !a.Has("--no-shape");
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddCrazyFruitsDqnCampaign(
                trainEnv: new CrazyFruitsEnv(moveBudget) { ShapeCreationRewards = shape },
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
            RunPolicy("specials-greedy", episodes, moveBudget, (board, _) => board.SpecialsGreedyAction()),
            RunPolicy("expectimax-1", episodes, moveBudget, (board, _) => board.ExpectimaxAction()),
            RunPolicy("expectimax-2", episodes, moveBudget, (board, _) => board.Expectimax2Action()),
        };

        DuelingQNet? net = null;
        if (File.Exists(netPath))
        {
            using var stream = File.OpenRead(netPath);
            net = DuelingQNetCheckpoint.Load(stream);
            if (net.InputSize != CrazyFruitsEnv.ObservationSize)
            {
                Console.WriteLine($"  (net at {netPath} has input width {net.InputSize} ≠ {CrazyFruitsEnv.ObservationSize} — stale pre-specials checkpoint, skipped)");
                net = null;
            }
        }
        if (net is not null)
        {
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
        else if (!File.Exists(netPath))
        {
            Console.WriteLine($"  (no net at {netPath} — scripted baselines only)");
        }

        foreach (var (name, mean, ci) in results)
            Console.WriteLine($"  {name,-28} mean {mean,8:F1} ± {ci:F1} (95% CI)");

        var random = results[0];
        var greedy = results[1];
        var e1 = results[3];
        var e2 = results[4];
        Console.WriteLine($"greedy vs random: {(greedy.Mean - greedy.Ci > random.Mean + random.Ci ? "CI-SEPARATED" : "OVERLAPPING")} " +
                          $"(+{100 * (greedy.Mean - random.Mean) / random.Mean:F0}%)");
        // SPECIALS PRD M50.2 gates: tier ordering + the pre-training env validation (specials must not be
        // so self-firing that random flattens the skill landscape) + the M50.3 escalation trigger input.
        Console.WriteLine($"expectimax-2 vs expectimax-1: {100 * (e2.Mean - e1.Mean) / e1.Mean:+0.0;-0.0}% (escalation trigger fires above +10%)");
        Console.WriteLine($"env validation: random = {random.Mean / e2.Mean:P0} of expectimax-2 " +
                          $"({(random.Mean < 0.70 * e2.Mean ? "OK (< 70%)" : "TOO SELF-FIRING (≥ 70%) — fix scoring before training")})");
        if (results.Count == 6)
        {
            var netRow = results[5];
            double gapShare = (netRow.Mean - random.Mean) / (e1.Mean - random.Mean);
            Console.WriteLine($"net vs random: +{100 * (netRow.Mean - random.Mean) / random.Mean:F1}% " +
                              $"({(netRow.Mean - netRow.Ci > random.Mean + random.Ci ? "CI-SEPARATED" : "OVERLAPPING")}; gate ≥ +30%, separated)");
            Console.WriteLine($"net gap share (random→expectimax-1): {gapShare:P0} (gate ≥ 64% — the M49 ratio)");
            Console.WriteLine($"net vs greedy: {100 * (netRow.Mean - greedy.Mean) / greedy.Mean:+0.0;-0.0}% (reported, not gated)");
        }
    }

    private static (string, double, double) RunPolicy(string name, int episodes, int moveBudget, Func<CrazyFruitsBoard, int, int> policy)
    {
        double sum = 0, sumSq = 0;
        long created = 0, fired = 0;
        for (int e = 0; e < episodes; e++)
        {
            // Reset through the env seed path so every policy sees the same boards as the net eval.
            var env = new CrazyFruitsEnv(moveBudget);
            env.Reset((ulong)(5_000 + e));
            var b = env.Board;
            for (int move = 0; move < moveBudget; move++)
            {
                b.ApplySwap(policy(b, move));
                created += b.MoveCreatedStriped + b.MoveCreatedWrapped + b.MoveCreatedBombs;
                fired += b.MoveSpecialsFired;
            }
            sum += b.Score;
            sumSq += (double)b.Score * b.Score;
        }
        Console.WriteLine($"  {name,-28} specials/episode: created {(double)created / episodes:F2}, fired {(double)fired / episodes:F2}");
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
