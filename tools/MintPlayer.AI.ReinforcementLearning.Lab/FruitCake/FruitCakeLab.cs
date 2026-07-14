using MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// `--game fruitcake` entry point: parses the campaign flags and runs the score-maximizing
/// <see cref="FruitCakeDqnCampaign"/> on the shared <see cref="CampaignRunner"/>. CPU-only (the small 41→14
/// Dueling net is far below the GPU routing threshold — the cost is the physics-in-the-loop env, which is CPU).
/// Loop, resume, eval cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
internal static class FruitCakeLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        int chunkSteps = a.Int("--chunk-steps", 2_000); // drops per chunk (each drop = simulate-to-rest; far costlier than a grid step)
        long targetSteps = a.Long("--steps", 0);        // 0 = time-bounded only (score-maximizing); a hard drop cap otherwise
        int evalEpisodes = a.Int("--episodes", 10);
        float learningRate = a.Flt("--lr", 5e-4f);
        float explore = a.Flt("--explore", 1.0f);       // ε-start; pass a low value (e.g. 0.2) to refine a warm-started net
        int[] hidden = a.Ints("--hidden", [256, 256]);  // trunk widths for the Dueling Q-net
        double gamma = a.Dbl("--gamma", 0.99);          // discount; high for the long drop horizon (PRD bundle uses 0.997)
        int nStep = a.Int("--nstep", 1);                // n-step return horizon (1 = single-step DQN)
        bool shape = a.Has("--shape");                  // enable reward shaping (tier-reached bonus + potential-based adjacency/height)
        bool evalOnly = a.Has("--eval-only");
        bool noisy = a.Has("--noisy");                  // NoisyNets exploration (learned σ) instead of ε-greedy
        bool ab = a.Has("--ab");                        // head-to-head eval of --data's net vs --baseline's net (no training)
        string baselineDir = a.Str("--baseline", "");   // the net to compare --data's net against
        int abEpisodes = a.Int("--ab-episodes", 200);   // paired greedy games per net (averages out eval noise)
        bool searchEval = a.Has("--search-eval");       // F1 forward-model search vs plain net greedy, on --data's net
        int depth = a.Int("--depth", 2);                // search lookahead (1 or 2)
        int topK = a.Int("--topk", 5);                  // depth-2 first-ply expansion width
        int topK2 = a.Int("--topk2", 3);                // deeper-ply expansion width (depth 3)
        string leaf = a.Str("--leaf", "net");           // search leaf value (net | height | tierpot | blend)
        bool grow = a.Has("--grow");                    // progressively grow the net wider+deeper mid-training (Net2Net demo)
        int growEvery = a.Int("--grow-every", 2000);    // drops between growth steps (with --grow)

        // A growing run starts from the tiny first stage and adds capacity mid-training (Net2Wider/DeeperNet).
        if (grow) hidden = DqnGrowth.Start;

        if (searchEval)
        {
            // F1: does forward-model search beat the plain net on max-tier? No training/host needed.
            FruitCakeSearchEval.Run(dataDir, abEpisodes, depth, topK, seedBase: 20_000, leaf, topK2);
            return;
        }

        if (ab)
        {
            // Head-to-head, no training/host needed: compare --data's net against --baseline's net.
            FruitCakeAb.Run(baselineDir, dataDir, abEpisodes, seedBase: 20_000);
            return;
        }

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new FruitCakeDqnCampaign(seed, chunkSteps, targetSteps, evalEpisodes, learningRate, explore, hidden, gamma, noisy, nStep, shape, grow, growEvery),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "fruitcake-dqn.csv")));
    }
}
