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

        int probe = a.Int("--probe", 0);
        if (probe > 0)
        {
            RunProbe(probe, moveBudget, netPath);
            return;
        }

        var options = new CrazyFruitsDqnOptions
        {
            Seed = seed, ChunkSteps = chunkSteps, TargetSteps = targetSteps, EvalEpisodes = evalEpisodes,
            LearningRate = learningRate, EpsilonStart = explore, Hidden = hidden, Gamma = gamma,
            Grow = grow, GrowEvery = growEvery,
            NStep = a.Int("--nstep", 1),
            // RANKING PRD M51.2: dense all-action regression toward the shaped observation plane (γ=0 only).
            DenseRegression = a.Has("--dense"),
            DenseTargetWeight = a.Flt("--dense-weight", 1.0f),
        };
        // Shaping on the TRAIN env only; the eval env scores the bare game, so gates stay honest.
        // Default (γ=0 recipe): creation bonuses. --pbrs (the §3.6 escalation, with --gamma 0.5 --nstep 3):
        // potential-based Φ over on-board specials instead — creation bonuses OFF, PotentialGamma = γ.
        bool pbrs = a.Has("--pbrs");
        bool shape = !a.Has("--no-shape") && !pbrs;
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddCrazyFruitsDqnCampaign(
                trainEnv: new CrazyFruitsEnv(moveBudget)
                {
                    ShapeCreationRewards = shape,
                    ShapeSpecialsPotential = pbrs,
                    PotentialGamma = gamma,
                    // Combo curriculum (M52, COMBO_CURRICULUM PRD): train env only — eval stays natural.
                    SeedSpecialsProb = a.Dbl("--seed-specials", 0),
                    ComboExploreBias = a.Dbl("--combo-explore", 0),
                },
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
            long cs = 0, cw = 0, cb = 0, fired = 0;
            for (int e = 0; e < episodes; e++)
            {
                var (obs, _) = env.Reset((ulong)(5_000 + e));
                while (true)
                {
                    var step = env.Step(agent.Act(obs, env.CurrentActionMask(), greedy: true));
                    cs += env.Board.MoveCreatedStriped; cw += env.Board.MoveCreatedWrapped; cb += env.Board.MoveCreatedBombs;
                    fired += env.Board.MoveSpecialsFired;
                    obs = step.Observation;
                    if (step.Done) break;
                }
                sum += env.Score;
                sumSq += (double)env.Score * env.Score;
            }
            PrintSpecialsLine("net", episodes, cs, cw, cb, fired);
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

    /// <summary>
    /// The M51.0 missed-opportunity probe (RANKING PRD §4): every policy is asked what it WOULD play on the
    /// SAME random-walk states (seeded, never the eval line 5000+e), without applying. A "creating" swap is
    /// one whose shaped immediate score exceeds its plain immediate score (it makes a special at step 0 —
    /// the owner's reported scenario). Metrics per policy:
    /// - opportune take-rate = P(picks a creating swap | a creating swap is the SHAPED-DETERMINISTIC-OPTIMAL
    ///   move) — the gate metric: an opportunity only counts as "missed" when creating was the best move
    ///   (the raw any-creating take-rate punished good play: expectimax-2 scores 8098 at a 38% raw rate,
    ///   while specials-greedy's immediate-only 91% scores 3903);
    /// - oracle match = P(action == argmax deterministicValueShaped) — plan fidelity, reported not gated;
    /// - raw take-rate + combo take-rate, kept for continuity with the M51.0 baseline table.
    /// </summary>
    private static void RunProbe(int episodes, int moveBudget, string netPath)
    {
        DuelingQNet? net = null;
        if (File.Exists(netPath))
        {
            using var stream = File.OpenRead(netPath);
            net = DuelingQNetCheckpoint.Load(stream);
            if (net.InputSize != CrazyFruitsEnv.ObservationSize)
            {
                Console.WriteLine($"  (net at {netPath} has input width {net.InputSize} ≠ {CrazyFruitsEnv.ObservationSize} — skipped)");
                net = null;
            }
        }
        var agent = net is null ? null : new GreedyQAgent(net, CrazyFruitsEnv.ActionCount);

        var policies = new List<(string Name, Func<CrazyFruitsBoard, int, int> Act)>
        {
            ("random", (b, s) => b.RandomAction(0xABBAUL, s)),
            ("greedy", (b, _) => b.GreedyAction()),
            ("specials-greedy", (b, _) => b.SpecialsGreedyAction()),
            ("expectimax-1", (b, _) => b.ExpectimaxAction()),
            ("expectimax-2", (b, _) => b.Expectimax2Action()),
        };
        if (agent is not null)
            policies.Add(("net", (b, _) => agent.Act(b.BuildObservation(), b.LegalMask(), greedy: true)));

        int opportuneStates = 0, opportunityStates = 0, comboStates = 0, statesSeen = 0;
        var opportuneTaken = new long[policies.Count];
        var opportunityTaken = new long[policies.Count];
        var comboTaken = new long[policies.Count];
        var oracleMatch = new long[policies.Count];
        // Opportune states bucketed by the best action's creation bonus (shaped − plain immediate):
        // 40 = striped (4-run), 60 = wrapped (L/T), ≥100 = bomb (5-run) or multi-creation. Owner report
        // 2026-07-25 round 2: a skipped 5-in-a-row — the bomb bucket answers whether that's systemic.
        var kindStates = new int[3];
        var kindTaken = new long[3 * policies.Count];

        var env = new CrazyFruitsEnv(moveBudget);
        var creating = new bool[CrazyFruitsEnv.ActionCount];
        for (int e = 0; e < episodes; e++)
        {
            env.Reset((ulong)(9_000 + e));
            var b = env.Board;
            for (int move = 0; move < moveBudget; move++)
            {
                int stateIndex = e * moveBudget + move;
                var mask = b.LegalMask();
                bool anyCreating = false, anyCombo = false;
                int bestAction = -1, bestShaped = 0;
                for (int a = 0; a < CrazyFruitsEnv.ActionCount; a++)
                {
                    creating[a] = mask[a] && b.ImmediateScoreShaped(a) > b.ImmediateScore(a);
                    if (creating[a]) anyCreating = true;
                    if (!mask[a]) continue;
                    int shaped = b.DeterministicValueShaped(a);
                    if (shaped > bestShaped) { bestShaped = shaped; bestAction = a; }
                }
                for (int a = 0; a < CrazyFruitsEnv.ActionCount && !anyCombo; a++)
                {
                    if (!mask[a]) continue;
                    var (cellA, cellB) = b.SwapCells(a);
                    anyCombo = b.Kind(cellA) > 0 && b.Kind(cellB) > 0;
                }

                statesSeen++;
                bool opportune = bestAction >= 0 && creating[bestAction];
                int kind = -1;
                if (opportune)
                {
                    opportuneStates++;
                    int bonus = b.ImmediateScoreShaped(bestAction) - b.ImmediateScore(bestAction);
                    kind = bonus >= 100 ? 2 : bonus >= 60 ? 1 : 0;
                    kindStates[kind]++;
                }
                if (anyCreating) opportunityStates++;
                if (anyCombo) comboStates++;

                for (int p = 0; p < policies.Count; p++)
                {
                    int action = policies[p].Act(b, stateIndex);
                    if (action < 0) continue;
                    if (action == bestAction) oracleMatch[p]++;
                    if (anyCreating && creating[action])
                    {
                        opportunityTaken[p]++;
                        if (opportune) { opportuneTaken[p]++; kindTaken[kind * policies.Count + p]++; }
                    }
                    if (anyCombo)
                    {
                        var (cellA, cellB) = b.SwapCells(action);
                        if (b.Kind(cellA) > 0 && b.Kind(cellB) > 0) comboTaken[p]++;
                    }
                }

                // The walk itself is uniform-random over legal swaps — policy-neutral state coverage.
                b.ApplySwap(b.RandomAction(0xCF51UL, stateIndex));
            }
        }

        Console.WriteLine($"Crazy Fruits opportunity probe: {episodes} episodes × {moveBudget} random-walk moves (seeds 9000+e)");
        Console.WriteLine($"  states {statesSeen}: creating swap available {opportunityStates}, creating swap is the " +
                          $"shaped-optimal move {opportuneStates} | special+special legal {comboStates}");
        Console.WriteLine($"  opportune states by best-action creation: striped {kindStates[0]} · wrapped {kindStates[1]} · bomb/multi {kindStates[2]}");
        Console.WriteLine($"  {"policy",-18} {"opportune take",16} {"striped",9} {"wrapped",9} {"bomb",9} {"raw take",10} {"oracle match",14} {"combo take",12}");
        for (int p = 0; p < policies.Count; p++)
        {
            string Rate(long taken, int total) => total == 0 ? "n/a" : $"{(double)taken / total:P1}";
            Console.WriteLine($"  {policies[p].Name,-18} {Rate(opportuneTaken[p], opportuneStates),16} " +
                              $"{Rate(kindTaken[0 * policies.Count + p], kindStates[0]),9} " +
                              $"{Rate(kindTaken[1 * policies.Count + p], kindStates[1]),9} " +
                              $"{Rate(kindTaken[2 * policies.Count + p], kindStates[2]),9} " +
                              $"{Rate(opportunityTaken[p], opportunityStates),10} {Rate(oracleMatch[p], statesSeen),14} " +
                              $"{Rate(comboTaken[p], comboStates),12}");
        }
        Console.WriteLine("  gate (RANKING PRD M51.2, final form): net opportune take-rate ≥ expectimax-1 − 5 pts.");
    }

    private static (string, double, double) RunPolicy(string name, int episodes, int moveBudget, Func<CrazyFruitsBoard, int, int> policy)
    {
        double sum = 0, sumSq = 0;
        long cs = 0, cw = 0, cb = 0, fired = 0;
        for (int e = 0; e < episodes; e++)
        {
            // Reset through the env seed path so every policy sees the same boards as the net eval.
            var env = new CrazyFruitsEnv(moveBudget);
            env.Reset((ulong)(5_000 + e));
            var b = env.Board;
            for (int move = 0; move < moveBudget; move++)
            {
                b.ApplySwap(policy(b, move));
                cs += b.MoveCreatedStriped; cw += b.MoveCreatedWrapped; cb += b.MoveCreatedBombs;
                fired += b.MoveSpecialsFired;
            }
            sum += b.Score;
            sumSq += (double)b.Score * b.Score;
        }
        PrintSpecialsLine(name, episodes, cs, cw, cb, fired);
        return Summarize(name, episodes, sum, sumSq);
    }

    // Per-kind created counts (M52 gate: kind-mix must not regress, ESPECIALLY bomb-created — seeded boards
    // hand the net free specials, exactly the pressure to under-create the rarest kind).
    private static void PrintSpecialsLine(string name, int episodes, long cs, long cw, long cb, long fired)
        => Console.WriteLine($"  {name,-28} specials/episode: created {(double)(cs + cw + cb) / episodes:F2} " +
                             $"(striped {(double)cs / episodes:F2} · wrapped {(double)cw / episodes:F2} · bomb {(double)cb / episodes:F2}), " +
                             $"fired {(double)fired / episodes:F2}");

    private static (string, double, double) Summarize(string name, int n, double sum, double sumSq)
    {
        double mean = sum / n;
        double variance = Math.Max(0, sumSq / n - mean * mean);
        double ci = 1.96 * Math.Sqrt(variance / n);
        return (name, mean, ci);
    }
}
