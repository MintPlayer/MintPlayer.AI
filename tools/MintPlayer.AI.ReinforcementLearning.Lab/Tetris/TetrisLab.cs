using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

/// <summary>
/// `--game tetris` entry point (PLAN M54): runs the lines-maximizing <see cref="TetrisDqnCampaign"/> on the
/// shared <see cref="CampaignRunner"/>. CPU-only (a 454→128→128 MLP is far below the GPU threshold —
/// TETRIS_PRD.md §3.7). `--baselines N` skips training and prints the scripted-policy table over BOTH eval
/// protocols (PRD §3.8): (A) uniform no-garbage 500-piece lines, and (B) garbage/10 survival — the primary,
/// discriminative gate. Tiers: random / Dellacherie / Dellacherie-search, plus the trained net when
/// `--net` exists.
/// </summary>
internal static class TetrisLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        int pieceBudget = a.Int("--piece-budget", 500);
        int chunkSteps = a.Int("--chunk-steps", 5_000);
        long targetSteps = a.Long("--steps", 400_000);
        int evalEpisodes = a.Int("--episodes", 20);
        float learningRate = a.Flt("--lr", 1e-3f);
        float explore = a.Flt("--explore", 1.0f);
        int[] hidden = a.Ints("--hidden", [128, 128]);
        double gamma = a.Dbl("--gamma", 0.995);
        bool evalOnly = a.Has("--eval-only");
        int baselines = a.Int("--baselines", 0);
        string netPath = a.Str("--net", Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "tetris.dqn.ckpt"));

        if (baselines > 0)
        {
            RunBaselines(baselines, pieceBudget, seed, netPath);
            return;
        }

        var options = new TetrisDqnOptions
        {
            Seed = seed, ChunkSteps = chunkSteps, TargetSteps = targetSteps, EvalEpisodes = evalEpisodes,
            LearningRate = learningRate, EpsilonStart = explore, Hidden = hidden, Gamma = gamma,
            Grow = a.Has("--grow"), GrowEvery = a.Int("--grow-every", 5000),
            NStep = a.Int("--nstep", 3),
            Noisy = a.Has("--noisy"),
            // The M49/M51 recipe (γ=0 only): dense all-action regression toward the Dellacherie-basis
            // value read back from the observation planes.
            DenseRegression = a.Has("--dense"),
            DenseTargetWeight = a.Flt("--dense-weight", 1.0f),
            EpsilonEnd = a.Flt("--eps-end", 0.05f),
            BufferCapacity = a.Int("--buffer", 100_000),
        };
        // Training + eval both uniform-random pieces, no garbage (the benchmark-honest protocol; garbage is
        // an eval protocol and a web mode, not a training distribution — PRD §3.6). PBRS shaping defaults ON
        // (M54.3 escalation: the bare reward is too sparse — 180K steps measured near-random) and lives on
        // the TRAIN env only, so gates stay honest; --no-pbrs reverts to the bare reward.
        bool pbrs = !a.Has("--no-pbrs");
        // Mixed garbage on/off per training episode. MEASURED WORSE on both protocols (tet5train head-to-
        // head vs tet4train, 30 seeds: A 17,022 vs 21,739 · B survival 101.3 vs 105.0): the dense target is
        // the same function on any board, so the clean-trained net already generalizes to garbage — the
        // garbage ceiling is γ=0 MYOPIA, which search fixes, not state coverage. Kept as an opt-in flag.
        bool mixGarbage = a.Has("--mix-garbage");
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddTetrisDqnCampaign(
                trainEnv: new TetrisEnv(pieceBudget)
                {
                    ShapeBoardPotential = pbrs,
                    PotentialGamma = gamma,
                    MixedGarbageTraining = mixGarbage,
                },
                evalEnv: new TetrisEnv(pieceBudget),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "tetris-dqn.csv")));
    }

    /// <summary>
    /// The falsifiable eval protocols (PRD §3.8): every policy plays the SAME seeded games (seeds 5000+e —
    /// the campaign's held-out eval line). Protocol A: uniform, no garbage, 500-piece cap, metric = lines.
    /// Protocol B (primary): garbage every 10, 5000-piece safety cap, metric = pieces survived.
    /// M54.3 gates: net survival ≥ 100 pieces, ≥ 4× random, CI-separated; gap-share vs Dellacherie ≥ 25%;
    /// protocol A net ≥ 50 lines. M54.4 gate: search > plain, CI-separated, ≥ Dellacherie on B.
    /// </summary>
    private static void RunBaselines(int episodes, int pieceBudget, ulong seed, string netPath)
    {
        Console.WriteLine($"Tetris baselines: {episodes} episodes (eval seeds 5000+e), protocol A = {pieceBudget}-piece lines, protocol B = garbage/10 survival");

        DuelingQNet? net = null;
        if (File.Exists(netPath))
        {
            using var stream = File.OpenRead(netPath);
            net = DuelingQNetCheckpoint.Load(stream);
            if (net.InputSize != TetrisEnv.ObservationSize)
            {
                Console.WriteLine($"  (net at {netPath} has input width {net.InputSize} ≠ {TetrisEnv.ObservationSize} — stale checkpoint, skipped)");
                net = null;
            }
        }
        else
        {
            Console.WriteLine($"  (no net at {netPath} — scripted baselines only)");
        }
        var agent = net is null ? null : new GreedyQAgent(net, TetrisEnv.ActionCount);

        // The search tier costs ~7K board sims per move — it gets fewer episodes and a lower protocol-B
        // safety cap (a capped row reads "≥ cap": still conclusive against Dellacherie's ~393).
        int searchEpisodes = Math.Min(20, episodes);
        var policies = new List<(string Name, Func<TetrisBoard, int, int> Act, int Episodes, int CapB)>
        {
            ("random", (b, s) => b.RandomAction(seed ^ 0xC0FFEE, s), episodes, 5_000),
            ("dellacherie", (b, _) => b.DellacherieAction(), episodes, 5_000),
            ("della-search(8,5)", (b, _) => b.DellaSearchAction(8, 5), searchEpisodes, 1_500),
        };
        if (agent is not null)
            policies.Add(("net", (b, _) => agent.Act(b.BuildObservation(), b.LegalMask(), greedy: true), episodes, 5_000));
        // Net+search runs the GENERATED f64 forward (the browser's exact tier) via the facade-loaded net.
        byte[]? ckptBytes = agent is not null ? File.ReadAllBytes(netPath) : null;
        if (ckptBytes is not null)
            policies.Add(("net-search(8)", (b, _) =>
            {
                if (b.NetInputSize < 0) b.LoadNet(new MemoryStream(ckptBytes));
                return b.NetSearchAction(8);
            }, searchEpisodes, 1_500));

        Console.WriteLine("Protocol A — uniform pieces, no garbage, capped: NES score (lines · tetrises annotated):");
        var linesA = new List<(string Name, double Mean, double Ci)>();
        foreach (var (name, act, eps, _) in policies)
            linesA.Add(RunProtocol(name, eps, act, garbageEvery: 0, pieceCap: pieceBudget, metricScore: true));

        Console.WriteLine("Protocol B — garbage every 10, survival (pieces placed):");
        var survB = new List<(string Name, double Mean, double Ci)>();
        foreach (var (name, act, eps, capB) in policies)
            survB.Add(RunProtocol(name, eps, act, garbageEvery: 10, pieceCap: capB, metricScore: false));

        var randomB = survB[0];
        var dellaB = survB[1];
        var searchB = survB[2];
        Console.WriteLine($"della vs random (B): {(dellaB.Mean - dellaB.Ci > randomB.Mean + randomB.Ci ? "CI-SEPARATED" : "OVERLAPPING")} " +
                          $"({dellaB.Mean / randomB.Mean:F1}×; spike measured 18×)");
        Console.WriteLine($"della-search vs della (B): {100 * (searchB.Mean - dellaB.Mean) / dellaB.Mean:+0.0;-0.0}% " +
                          $"({(searchB.Mean - searchB.Ci > dellaB.Mean + dellaB.Ci ? "CI-SEPARATED" : "OVERLAPPING")}; M54.4 gate: separated)");
        if (survB.Count >= 4)
        {
            var netB = survB[3];
            var netA = linesA[3];
            double gapShare = (netB.Mean - randomB.Mean) / (dellaB.Mean - randomB.Mean);
            Console.WriteLine($"net survival (B): {netB.Mean:F1} (gates: ≥ 100 · ≥ 4× random [{4 * randomB.Mean:F0}] · " +
                              $"{(netB.Mean - netB.Ci > randomB.Mean + randomB.Ci ? "CI-SEPARATED" : "OVERLAPPING")} vs random)");
            Console.WriteLine($"net gap share random→della (B): {gapShare:P0} (gate ≥ 25%)");
            Console.WriteLine($"net score (A): {netA.Mean:F0} (gate ≥ 5000 — ≈50 single-line clears with the NES level multiplier)");
            if (survB.Count >= 5)
            {
                var netSearchB = survB[4];
                Console.WriteLine($"net+search vs net (B): {100 * (netSearchB.Mean - netB.Mean) / netB.Mean:+0.0;-0.0}% " +
                                  $"({(netSearchB.Mean - netSearchB.Ci > netB.Mean + netB.Ci ? "CI-SEPARATED" : "OVERLAPPING")}; M54.4 gate: separated)");
                Console.WriteLine($"net+search vs dellacherie (B): {100 * (netSearchB.Mean - dellaB.Mean) / dellaB.Mean:+0.0;-0.0}% (M54.4 headline gate: ≥ 0%)");
            }
        }
    }

    private static (string, double, double) RunProtocol(string name, int episodes,
        Func<TetrisBoard, int, int> policy, int garbageEvery, int pieceCap, bool metricScore)
    {
        double sum = 0, sumSq = 0, lines = 0, tetrises = 0;
        int topOuts = 0;
        for (int e = 0; e < episodes; e++)
        {
            // Reset through the env seed path so every policy sees the same games as the net eval.
            var env = new TetrisEnv(pieceCap, sevenBag: false, garbageEvery: garbageEvery);
            env.Reset((ulong)(5_000 + e));
            var b = env.Board;
            for (int step = 0; step < pieceCap && !b.GameOver; step++)
            {
                int action = policy(b, e * pieceCap + step);
                if (action < 0 || b.ApplyPlacement(action) < 0) break;
            }
            if (b.GameOver) topOuts++;
            double metric = metricScore ? b.Score : b.PiecesPlaced;
            sum += metric;
            sumSq += metric * metric;
            lines += b.Lines;
            tetrises += b.Tetrises;
        }
        double mean = sum / episodes;
        double ci = 1.96 * Math.Sqrt(Math.Max(0, sumSq / episodes - mean * mean) / episodes);
        Console.WriteLine($"  {name,-20} mean {mean,9:F1} ± {ci:F1} (95% CI), " +
                          $"lines {lines / episodes:F1} · tetrises {tetrises / episodes:F2} · top-outs {topOuts}/{episodes}");
        return (name, mean, ci);
    }
}
