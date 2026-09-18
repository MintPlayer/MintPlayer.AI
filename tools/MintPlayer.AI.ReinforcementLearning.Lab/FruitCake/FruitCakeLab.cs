using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

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
        string dataDir = a.Str("--data", "data");
        int abEpisodes = a.Int("--ab-episodes", 200);   // paired greedy games per net (averages out eval noise)

        if (a.Has("--search-eval"))
        {
            // F1: does forward-model search beat the plain net on max-tier? No training/host needed.
            FruitCakeSearchEval.Run(dataDir, abEpisodes, a.Int("--depth", 2), a.Int("--topk", 5),
                                    seedBase: 20_000, a.Str("--leaf", "net"), a.Int("--topk2", 3));
            return;
        }

        if (a.Has("--ab"))
        {
            // Head-to-head, no training/host needed: compare --data's net against --baseline's net.
            FruitCakeAb.Run(a.Str("--baseline", ""), dataDir, abEpisodes, seedBase: 20_000);
            return;
        }

        var options = Parse(a, out double hours, out _, out bool evalOnly, out bool shape);
        double gamma = options.Gamma;

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            // Shaping lives on the training env (ShapingGamma matches the learner's γ for policy-invariance);
            // the eval env stays a plain game so keep-best/A/B judge real merge points, never the shaped signal.
            services => services.AddFruitCakeDqnCampaign(
                trainEnv: new FruitCakeEnv { ShapeRewards = shape, ShapingGamma = gamma },
                evalEnv: new FruitCakeEnv(),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "fruitcake-dqn.csv")));
    }

    /// <summary>
    /// The DQN options this entry point's flags resolve to, plus the three values the host needs
    /// (<paramref name="hours"/>, <paramref name="dataDir"/>, <paramref name="evalOnly"/>) and
    /// <paramref name="shape"/>, which selects the shaped TRAINING env while eval stays a plain game.
    /// </summary>
    /// <remarks>
    /// M63.5: extracted from <see cref="Run"/>, where every default was reachable only by starting a real
    /// training run. The <c>--grow</c> interaction moves with it — a growing run REPLACES the requested
    /// <c>--hidden</c> with the tiny first stage, which is the least obvious rule here.
    /// </remarks>
    internal static FruitCakeDqnOptions Parse(CliArgs a, out double hours, out string dataDir,
                                              out bool evalOnly, out bool shape)
    {
        hours = a.Dbl("--hours", 1);
        dataDir = a.Str("--data", "data");
        evalOnly = a.Has("--eval-only");
        shape = a.Has("--shape");                       // tier-reached bonus + potential-based adjacency/height

        int[] hidden = a.Ints("--hidden", [256, 256]);  // trunk widths for the Dueling Q-net
        bool grow = a.Has("--grow");                    // grow the net wider+deeper mid-training (Net2Net demo)
        // A growing run starts from the tiny first stage and adds capacity mid-training (Net2Wider/DeeperNet).
        if (grow) hidden = DqnGrowth.Start;

        return new FruitCakeDqnOptions
        {
            Seed = a.ULong("--seed", 1),
            ChunkSteps = a.Int("--chunk-steps", 2_000), // drops per chunk (each drop = simulate-to-rest)
            TargetSteps = a.Long("--steps", 0),         // 0 = time-bounded only; a hard drop cap otherwise
            EvalEpisodes = a.Int("--episodes", 10),
            LearningRate = a.Flt("--lr", 5e-4f),
            EpsilonStart = a.Flt("--explore", 1.0f),    // ε-start; low (e.g. 0.2) refines a warm-started net
            Hidden = hidden,
            Gamma = a.Dbl("--gamma", 0.99),             // high for the long drop horizon
            Noisy = a.Has("--noisy"),                   // NoisyNets (learned σ) instead of ε-greedy
            NStep = a.Int("--nstep", 1),                // n-step return horizon (1 = single-step DQN)
            Grow = grow,
            GrowEvery = a.Int("--grow-every", 2000),    // drops between growth steps (with --grow)
        };
    }
}
