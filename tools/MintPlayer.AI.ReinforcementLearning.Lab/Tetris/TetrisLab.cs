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

        // M62.3b diagnostic: how many placements does each technique actually reach, per level? The mask
        // is only believable if these counts match hand-arithmetic on the frame budget.
        if (a.Has("--reach-census"))
        {
            // Summed over ALL SEVEN pieces — a single seed can land on O (one rotation, spawn-centred) and
            // report "everything reachable" no matter the gravity, which is exactly the false negative that
            // made the first run of this census useless.
            int[][] models = [[6, 6], [5, 5], [3, 3]];
            foreach (int stackHeight in new[] { 0, 10 })
            {
                Console.WriteLine($"\nReachable / legal placements, summed over all 7 pieces, stack height {stackHeight}:");
                Console.WriteLine($"{"level",6} {"g",3} {"legal",7} {"DAS 6",9} {"hyper 5",9} {"roll 3",9}");
                foreach (int lvl in new[] { 0, 9, 18, 19, 22, 29 })
                {
                    int legal = 0;
                    var counts = new int[3];
                    for (int piece = 0; piece < TetrisBoard.PieceCount; piece++)
                    {
                        for (int m = -1; m < 3; m++)
                        {
                            var b2 = new TetrisBoard();
                            b2.Reset(5);
                            if (stackHeight > 0)
                            {
                                var rows = new int[20];
                                // A flat floor with column 9 left open — the classic tetris well.
                                for (int y = 20 - stackHeight; y < 20; y++) rows[y] = 1023 - (1 << 9);
                                b2.LoadRows(rows);
                            }
                            b2.LoadPieces(piece, piece);
                            b2.SetStartLevel(lvl);
                            if (m >= 0)
                            {
                                b2.SetReachEnforced(true);
                                b2.SetTapModel(models[m][0], models[m][1]);
                            }
                            for (int act = 0; act < TetrisBoard.ActionCount; act++)
                            {
                                bool ok = m < 0 ? b2.IsLegal(act) : b2.PlacementReachable(act);
                                if (!ok) continue;
                                if (m < 0) legal++; else counts[m]++;
                            }
                        }
                    }
                    Console.WriteLine($"{lvl,6} {new TetrisBoard().GravityFramesAt(lvl),3} {legal,7} {counts[0],9} {counts[1],9} {counts[2],9}");
                }
            }
            return;
        }

        // M62.4 diagnostic (owner-reported): "the AI gets a long bar, could grab a tetris, and instead
        // puts it somewhere else — about 1 in 4 times". Counts exactly that: steps where SOME legal
        // placement would clear 4 rows, and whether the policy took one.
        if (a.Has("--decline-census"))
        {
            RunDeclineCensus(a.Int("--decline-census", 12), pieceBudget, a.Int("--tap", 0), a.Has("--reach"), a.Dbl("--ready-dig", 0));
            return;
        }

        if (baselines > 0)
        {
            // M62.3: --tap <framesPerShift> measures the technique dial's effect on the AI's own play
            // (6 = DAS, 5 = hypertapping, 3 = rolling). 0 leaves the engine default (6).
            // --start-level matters for --tap: the tap budget is measured against GRAVITY, so at level 0
            // (48 frames/row) every technique reaches every column and the dial is a no-op. It only bites
            // from ~L19 (2 frames/row) upward, which is exactly where real players change technique.
            // --reach turns on M62.3b reachability enforcement (opt-in, PRD D7) so the mask's effect is a
            // measurable delta against the same command without it.
            RunBaselines(baselines, pieceBudget, seed, netPath, a.Int("--tap", 0), a.Int("--start-level", 0), a.Has("--reach"));
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
                    // --mandatory-tetris: declining a reachable tetris on a CLEAN stack ends the episode.
                    // Train env only — the eval env below deliberately leaves it off, so the gates keep
                    // measuring the same game they always did.
                    MandatoryTetris = a.Has("--mandatory-tetris"),
                },
                evalEnv: new TetrisEnv(pieceBudget),
                options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "tetris-dqn.csv")));
    }

    /// <summary>
    /// M62.4: how often is a 4-line clear on the table and declined? Reports, per tier, the number of
    /// steps where at least one legal placement cleared 4 rows, how many of those the policy took, and —
    /// when it declined — what it did instead. Measures the owner's observation directly instead of
    /// reasoning about the evaluator's arithmetic.
    /// </summary>
    private static void RunDeclineCensus(int episodes, int pieceBudget, int tapRate, bool reach, double readyDig)
    {
        Console.WriteLine($"Tetris tetris-decline census: {episodes} episodes, seeds 5000+e" +
                          (tapRate > 0 ? $", tap {tapRate}" : "") + (reach ? ", reachability ENFORCED" : ""));

        var tiers = new (string Name, Func<TetrisBoard, int> Act)[]
        {
            ("dellacherie", b => b.DellacherieAction()),
            ("della-search(8,5)", b => b.DellaSearchAction(8, 5)),
        };

        foreach (var (name, act) in tiers)
        {
            int offered = 0, taken = 0, declinedClearedSomething = 0, declinedClearedNothing = 0;
            // The hypothesis under test: EvalReady (and observation plane 6) are both switched off while
            // dig == holes() > 0, so on a holed board nothing values the well and the tetris gets declined.
            // If that is right, declines should cluster almost entirely on holed boards.
            int declinedWithHoles = 0, declinedClean = 0, takenWithHoles = 0;
            for (int e = 0; e < episodes; e++)
            {
                var env = new TetrisEnv(pieceBudget, sevenBag: false, garbageEvery: 0);
                env.Reset((ulong)(5_000 + e));
                var b = env.Board;
                if (tapRate > 0) b.SetTapModel(tapRate, tapRate);
                if (reach) b.SetReachEnforced(true);
                if (readyDig > 0) b.SetReadyDigScale(readyDig);

                for (int step = 0; step < pieceBudget && !b.GameOver; step++)
                {
                    // Which placements would clear four rows? Probe each on a scratch copy of the board.
                    var rows = new int[TetrisBoard.Height];
                    for (int y = 0; y < TetrisBoard.Height; y++) rows[y] = b.Row(y);
                    bool anyTetris = false;
                    for (int actn = 0; actn < TetrisBoard.ActionCount && !anyTetris; actn++)
                    {
                        if (reach ? !b.PlacementReachable(actn) : !b.IsLegal(actn)) continue;
                        var probe = new TetrisBoard();
                        probe.Reset(1);
                        probe.LoadRows(rows);
                        probe.LoadPieces(b.CurrentPiece, b.NextPiece);
                        if (probe.ApplyPlacement(actn) == 4) anyTetris = true;
                    }

                    int holesBefore = b.Holes();
                    int chosen = act(b);
                    if (chosen < 0) break;
                    int clearedNow = b.ApplyPlacement(chosen);
                    if (clearedNow < 0) break;

                    if (anyTetris)
                    {
                        offered++;
                        if (clearedNow == 4)
                        {
                            taken++;
                            if (holesBefore > 0) takenWithHoles++;
                        }
                        else
                        {
                            if (clearedNow > 0) declinedClearedSomething++; else declinedClearedNothing++;
                            if (holesBefore > 0) declinedWithHoles++; else declinedClean++;
                        }
                    }
                }
            }
            double rate = offered == 0 ? 0 : 100.0 * taken / offered;
            int declined = declinedClearedSomething + declinedClearedNothing;
            double holedShare = declined == 0 ? 0 : 100.0 * declinedWithHoles / declined;
            Console.WriteLine($"  {name,-20} tetris available on {offered,5} steps · TOOK {taken,5} ({rate,5:F1}%) · " +
                              $"declined-and-burned {declinedClearedSomething,4} · declined-and-stacked {declinedClearedNothing,4}");
            Console.WriteLine($"  {"",-20} of {declined,5} declines, {declinedWithHoles,5} were on a HOLED board ({holedShare,5:F1}%) " +
                              $"and {declinedClean,4} on a clean one · takes on holed boards {takenWithHoles,4}");
        }
    }

    /// <summary>
    /// The falsifiable eval protocols (PRD §3.8): every policy plays the SAME seeded games (seeds 5000+e —
    /// the campaign's held-out eval line). Protocol A: uniform, no garbage, 500-piece cap, metric = lines.
    /// Protocol B (primary): garbage every 10, 5000-piece safety cap, metric = pieces survived.
    /// M54.3 gates: net survival ≥ 100 pieces, ≥ 4× random, CI-separated; gap-share vs Dellacherie ≥ 25%;
    /// protocol A net ≥ 50 lines. M54.4 gate: search > plain, CI-separated, ≥ Dellacherie on B.
    /// </summary>
    private static void RunBaselines(int episodes, int pieceBudget, ulong seed, string netPath, int tapRate = 0, int startLevel = 0, bool reach = false)
    {
        Console.WriteLine($"Tetris baselines: {episodes} episodes (eval seeds 5000+e), protocol A = {pieceBudget}-piece lines, protocol B = garbage/10 survival");
        if (tapRate > 0)
        {
            string name = tapRate == 6 ? "DAS" : tapRate == 5 ? "hypertapping" : tapRate == 3 ? "rolling" : "custom";
            Console.WriteLine($"  tap budget: {tapRate} frames/shift ({60.0988 / tapRate:F1} Hz — {name})");
        }
        if (startLevel > 0)
            Console.WriteLine($"  start level: {startLevel} (gravity {new TetrisBoard().GravityFramesAt(startLevel)} frames/row)");
        if (reach)
            Console.WriteLine("  reachability: ENFORCED at the root (M62.3b) — unreachable placements are masked out");

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
            // M62.3b: the net tier must consume the SAME mask the scripted tiers obey, or --reach measures
            // two different games and the comparison is meaningless.
            policies.Add(("net", (b, _) => agent.Act(b.BuildObservation(), reach ? b.ReachableMask() : b.LegalMask(), greedy: true), episodes, 5_000));
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
            linesA.Add(RunProtocol(name, eps, act, garbageEvery: 0, pieceCap: pieceBudget, metricScore: true, tapRate: tapRate, startLevel: startLevel, reach: reach));

        Console.WriteLine("Protocol B — garbage every 10, survival (pieces placed):");
        var survB = new List<(string Name, double Mean, double Ci)>();
        foreach (var (name, act, eps, capB) in policies)
            survB.Add(RunProtocol(name, eps, act, garbageEvery: 10, pieceCap: capB, metricScore: false, tapRate: tapRate, startLevel: startLevel, reach: reach));

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
        Func<TetrisBoard, int, int> policy, int garbageEvery, int pieceCap, bool metricScore, int tapRate = 0, int startLevel = 0, bool reach = false)
    {
        double sum = 0, sumSq = 0, lines = 0, tetrises = 0;
        int topOuts = 0;
        var moveTicks = new List<long>();
        for (int e = 0; e < episodes; e++)
        {
            // Reset through the env seed path so every policy sees the same games as the net eval.
            var env = new TetrisEnv(pieceCap, sevenBag: false, garbageEvery: garbageEvery);
            env.Reset((ulong)(5_000 + e));
            var b = env.Board;
            // M62.3. Reset does not clear the tap rate (only startLevel), but each episode builds a fresh
            // TetrisEnv, so the rate has to be set per episode regardless.
            // M62.3: DAS is the only technique that pays an auto-shift charge; 6/16 is the ROM model and
            // the tap techniques charge nothing (chargeFrames == framesPerShift).
            if (tapRate > 0) b.SetTapModel(tapRate, tapRate);
            if (startLevel > 0) b.SetStartLevel(startLevel);
            if (reach) b.SetReachEnforced(true);
            for (int step = 0; step < pieceCap && !b.GameOver; step++)
            {
                // G6 is the gate most at risk once reachability runs at the root, so time the DECISION
                // itself (not the placement) and report the tail, which is what a visitor actually feels.
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                int action = policy(b, e * pieceCap + step);
                moveTicks.Add(System.Diagnostics.Stopwatch.GetTimestamp() - t0);
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
        moveTicks.Sort();
        double msPerTick = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        double p50 = moveTicks.Count == 0 ? 0 : moveTicks[moveTicks.Count / 2] * msPerTick;
        double p99 = moveTicks.Count == 0 ? 0 : moveTicks[(int)(moveTicks.Count * 0.99)] * msPerTick;
        Console.WriteLine($"  {name,-20} mean {mean,9:F1} ± {ci:F1} (95% CI), " +
                          $"lines {lines / episodes:F1} · tetrises {tetrises / episodes:F2} · top-outs {topOuts}/{episodes} · " +
                          $"ms/move p50 {p50:F2} p99 {p99:F2}");
        return (name, mean, ci);
    }
}
