using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>
/// Scores a trained net against the ELEVEN SHIPPED LEVELS — the actual game content, as opposed to the generated
/// boards the curriculum gates on.
/// </summary>
/// <remarks>
/// <para>This exists because the training gate only ever measures generated boards at the current rung, so a run
/// can report a healthy gate while nobody has ever checked what the net does with the levels a player will see.
/// "Ship a net" is not a meaningful claim until this number exists.</para>
///
/// <para>Expect it to be low, and that is not a defect: most shipped levels are far beyond the exact oracle that
/// produced the training labels (level 11 is 42 blocks over 551 cells), so the net was never shown anything of
/// that size. The point is to KNOW the number rather than assume it.</para>
/// </remarks>
internal static class BlockDudeLevelBench
{
    /// <summary>Records one solved level in the web recorder's line format, keeping the SHORTEST line per level
    /// when several tiers solve it — the tiers differ, and reporting the worse one would understate the net.</summary>
    private static void Emit(List<string> lines, string levelName, IReadOnlyList<int> moves)
    {
        string line = $"{levelName} · {moves.Count} moves · {string.Concat(moves)}";

        int existing = lines.FindIndex(l => l.StartsWith($"{levelName} · ", StringComparison.Ordinal));
        if (existing < 0) lines.Add(line);
        else if (moves.Count < ExistingLength(lines[existing])) lines[existing] = line;
    }

    private static int ExistingLength(string line)
    {
        var parts = line.Split(" · ");
        return parts.Length >= 2 && int.TryParse(parts[1].AsSpan(0, parts[1].IndexOf(' ')), out int n) ? n : int.MaxValue;
    }

    /// <summary>
    /// Scores the net on generated gate boards — positions it was never trained on.
    /// </summary>
    /// <remarks>
    /// <para>Read the result with the generator's known bias in mind: it cannot express 53% of shipped
    /// topologies and tops out around 25 optimal moves (§2 of the rebuild PRD), so this is a *different*
    /// distribution rather than a harder sample of the same one. A low score here does not straightforwardly
    /// mean "did not generalise" — but a high one is real evidence that the mechanics were learned, because
    /// nothing in phase 2's training data is anywhere near these boards.</para>
    ///
    /// <para>The oracle also gives the exact optimal length for each, so unlike the shipped levels this reports
    /// how far from optimal the net's solutions actually are.</para>
    /// </remarks>
    private static void HeldOut(BlockDudePolicyNet net, int stage, int count, int budget, int beamWidth, int seconds)
    {
        var boards = BlockDudeCurriculum.GateBoardsFor(stage, count);
        Console.WriteLine($"held-out: {boards.Count} generated gate boards at stage {stage} — positions phase 2 " +
                          $"never trained on (the 15 shipped levels are its training set)");
        Console.WriteLine();

        int greedy = 0, beam = 0, optimal = 0, excess = 0, scored = 0, blindBeam = 0;
        foreach (var board in boards)
        {
            if (BlockDudeGreedy.Run(net, board, budget).Solved) greedy++;

            // The control, for the same reason it exists on the shipped levels — and it matters MORE here.
            // These boards are small (roughly 25 optimal moves), so a 256-wide beam may be close to exhaustive,
            // and "98% optimal" would then be a fact about the search rather than about the net.
            if (BlockDudeSearch.SolveByBeam(BlockDudeSearch.UniformPriors, board, beamWidth, 400,
                                            TimeSpan.FromSeconds(seconds)).Solved) blindBeam++;

            var found = BlockDudeSearch.SolveByBeam(net, board, beamWidth, 400, TimeSpan.FromSeconds(seconds));
            if (!found.Solved) continue;
            beam++;

            var oracle = new BlockDudeOracle(board);
            int exact = oracle.OptimalFromStart;
            if (oracle.Truncated || exact <= 0) continue;   // unlabelable board: no honest comparison exists
            scored++;
            excess += found.Length - exact;
            if (found.Length == exact) optimal++;
        }

        Console.WriteLine($"greedy (policy alone)   {greedy,3}/{boards.Count} ({greedy / (double)boards.Count:P0})");
        Console.WriteLine($"policy beam search      {beam,3}/{boards.Count} ({beam / (double)boards.Count:P0})");
        Console.WriteLine($"  same beam, UNIFORM prior (control)  {blindBeam,3}/{boards.Count} " +
                          $"({blindBeam / (double)boards.Count:P0}) — these boards are small, so the beam may be " +
                          $"close to exhaustive on its own");
        if (scored > 0)
            Console.WriteLine($"of the beam solutions the oracle could score: {optimal}/{scored} were OPTIMAL, " +
                              $"average {excess / (double)scored:F1} moves over optimal");
    }

    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        string dataDir = a.Str("--data", "data");
        int phase = a.Int("--phase", 1);
        int budget = a.Int("--step-budget", 2000);
        bool useResumeNet = a.Has("--resume-net");

        // --search adds a second, net-guided A* pass per level. Off by default so the headline number stays the
        // policy-alone one the gate also reports.
        bool search = a.Has("--search");
        int expansions = a.Int("--expansions", 200_000);
        float weight = a.Flt("--weight", 2f);
        int seconds = a.Int("--search-seconds", 20);

        // --zero-h runs the SAME search with h = 0, i.e. uninformed breadth-first, as a control. Without it
        // "search solves N/15" cannot be attributed: part of that N is the search and part is the net.
        bool zeroH = a.Has("--zero-h");

        // --policy-search adds a third pass that uses the policy head as a prior as well as the value head as
        // cost-to-go. Reported next to the value-only number, because the comparison is the point.
        bool policySearch = a.Has("--policy-search");
        float policyWeight = a.Flt("--policy-weight", 5f);

        // --beam adds the long-level tier: policy beam search, which costs width × depth rather than holding an
        // exponential frontier, so it is the only tier that can reach a multi-hundred-move solution at all.
        bool beam = a.Has("--beam");
        int beamWidth = a.Int("--beam-width", 256);
        int beamDepth = a.Int("--beam-depth", 1_200);

        // --emit-solutions writes what the net actually played, in the SAME one-line-per-level format the web
        // game's own recorder produces ("Level 1 · 19 moves · 0303…"). That makes an AI solution paste-able
        // straight back into the game to be watched, and directly comparable with the human line for the same
        // level — which is the only honest way to read "solved it in 172 moves".
        string? emitPath = a.Str("--emit-solutions", "") is { Length: > 0 } p ? p : null;
        var emitted = new List<string>();

        var ids = BlockDudeIds.ForPhase(phase);
        string netId = useResumeNet ? ids.Policy : ids.PolicyBest;

        var store = new FileModelStore(dataDir);
        using var stream = store.TryOpenRead(BlockDudeIds.Environment, netId);
        if (stream is null)
        {
            Console.WriteLine($"No net at {dataDir}/{BlockDudeIds.Environment}.{netId}.ckpt.");
            if (!useResumeNet)
                Console.WriteLine("The deployable net appears only once an eval has improved on the best seen; " +
                                  "pass --resume-net to benchmark the latest weights instead.");
            return;
        }

        var net = BlockDudePolicyNet.Load(stream);
        Console.WriteLine($"net: {dataDir}/{BlockDudeIds.Environment}.{netId}.ckpt  " +
                          $"(trunk [{string.Join(",", net.Trunk)}], step budget {budget:N0})");
        Console.WriteLine();

        // --held-out answers a question the shipped-level bench structurally CANNOT. Phase 2 trains on the 15
        // shipped levels, so those levels are the curriculum and the benchmark at once and every "solves N/15"
        // is a training-set score. Generated gate boards are the only Block Dude positions this net has not been
        // trained on, so they are the only evidence about whether it learned the game or memorised 15 paths.
        if (a.Has("--held-out"))
        {
            HeldOut(net, a.Int("--held-out-stage", 4), a.Int("--held-out-boards", 64), budget, beamWidth, seconds);
            return;
        }

        var levels = BlockDudeLevels.All;
        int solved = 0, noRevisitSolved = 0, searchSolved = 0, zeroSolved = 0, policySolved = 0, beamSolved = 0, zeroBeamSolved = 0;
        for (int i = 0; i < levels.Length; i++)
        {
            var board = BlockDudeBoard.FromGrid(levels[i].Grid);
            var outcome = BlockDudeGreedy.Run(net, board, budget, recordMoves: emitPath is not null);
            if (outcome.Solved) solved++;
            // Emitted FIRST, so the greedy line wins ties against the search tiers: same length, but it is the
            // one the net plays on its own, and that is the line worth watching.
            if (outcome.Solved && outcome.Moves is not null) Emit(emitted, levels[i].Name, outcome.Moves);

            var noRevisit = BlockDudeGreedy.RunAvoidingRevisits(net, board, budget);
            if (noRevisit.Solved) noRevisitSolved++;

            string line = $"  {i + 1,2}. {levels[i].Name,-24} {board.Width,3}x{board.Height,-3} " +
                          $"{(outcome.Solved ? "SOLVED" : "-     ")} " +
                          $"{outcome.Ending,-8} after {outcome.Steps,5:N0} steps  | no-revisit " +
                          $"{(noRevisit.Solved ? "SOLVED" : "-     ")} {noRevisit.Ending,-7} " +
                          $"{noRevisit.Steps,5:N0} steps";

            if (search)
            {
                var found = BlockDudeSearch.Solve(net, board, expansions, weight, TimeSpan.FromSeconds(seconds));
                if (found.Solved) searchSolved++;
                line += found.Solved ? $"  | search SOLVED in {found.Length,4:N0} moves" : "  | search -";
                if (found.Solved) Emit(emitted, levels[i].Name, found.Moves!);
            }

            if (policySearch)
            {
                var found = BlockDudeSearch.SolveWithPolicy(net, board, expansions, weight, policyWeight,
                                                           TimeSpan.FromSeconds(seconds));
                if (found.Solved) policySolved++;
                line += found.Solved ? $"  | policy+value SOLVED in {found.Length,4:N0} moves" : "  | policy+value -";
                if (found.Solved) Emit(emitted, levels[i].Name, found.Moves!);
            }

            if (beam)
            {
                var found = BlockDudeSearch.SolveByBeam(net, board, beamWidth, beamDepth,
                                                        TimeSpan.FromSeconds(seconds));
                if (found.Solved) beamSolved++;
                line += found.Solved ? $"  | beam SOLVED in {found.Length,4:N0} moves" : "  | beam -";
                if (found.Solved) Emit(emitted, levels[i].Name, found.Moves!);
            }

            if (zeroH)
            {
                var blind = BlockDudeSearch.Solve(BlockDudeSearch.ZeroHeuristic, board, expansions, weight,
                                                  TimeSpan.FromSeconds(seconds));
                if (blind.Solved) zeroSolved++;
                line += blind.Solved ? $"  | blind SOLVED in {blind.Length,4:N0} moves" : "  | blind -";

                // Beam and A* are different search SHAPES, so the h = 0 number cannot attribute a beam result —
                // it would credit the policy for the change of shape. The beam tier needs its own control.
                if (beam)
                {
                    var blindBeam = BlockDudeSearch.SolveByBeam(BlockDudeSearch.UniformPriors, board, beamWidth,
                                                                beamDepth, TimeSpan.FromSeconds(seconds));
                    if (blindBeam.Solved) zeroBeamSolved++;
                    line += blindBeam.Solved ? $"  | blind-beam SOLVED in {blindBeam.Length,4:N0}" : "  | blind-beam -";
                }
            }

            Console.WriteLine(line);
        }

        Console.WriteLine();
        Console.WriteLine($"solved {solved}/{levels.Length} shipped levels " +
                          $"({solved / (double)levels.Length:P0}), greedy, no search");
        Console.WriteLine($"solved {noRevisitSolved}/{levels.Length} shipped levels " +
                          $"({noRevisitSolved / (double)levels.Length:P0}), greedy + never re-enter a visited state " +
                          $"(a tie-break, still no lookahead)");

        if (search)
        {
            // Printed side by side on purpose. The two numbers answer different questions — "has the policy
            // learned the game" and "what is this net worth with lookahead" — and reporting only the first is
            // what let a net that loops on every shipped level look like a pure training failure.
            Console.WriteLine($"solved {searchSolved}/{levels.Length} shipped levels " +
                              $"({searchSolved / (double)levels.Length:P0}), net-guided A* " +
                              $"(weight {weight}, ≤{expansions:N0} expansions, ≤{seconds}s per level)");
        }

        if (emitPath is not null && emitted.Count > 0)
        {
            File.WriteAllLines(emitPath, emitted);
            Console.WriteLine();
            Console.WriteLine($"wrote {emitted.Count} solution line(s) to {emitPath} — paste them into the game to watch them.");
        }

        if (policySearch)
        {
            Console.WriteLine($"solved {policySolved}/{levels.Length} shipped levels " +
                              $"({policySolved / (double)levels.Length:P0}), policy-prior + value A* " +
                              $"(weight {weight}, policy weight {policyWeight}, ≤{expansions:N0} expansions, ≤{seconds}s per level)");
        }

        if (beam)
        {
            Console.WriteLine($"solved {beamSolved}/{levels.Length} shipped levels " +
                              $"({beamSolved / (double)levels.Length:P0}), policy beam search " +
                              $"(width {beamWidth:N0}, ≤{beamDepth:N0} moves, ≤{seconds}s per level) — " +
                              $"the tier that can reach the LONG levels at all");
        }

        if (zeroH)
        {
            // The control. Whatever the net-guided number is, the honest claim about the heuristic is the
            // DIFFERENCE between the two lines — not the net-guided line on its own.
            Console.WriteLine($"solved {zeroSolved}/{levels.Length} shipped levels " +
                              $"({zeroSolved / (double)levels.Length:P0}), SAME search with h = 0 " +
                              $"(uninformed breadth-first — the control for what the value head is worth)");

            if (beam)
                Console.WriteLine($"solved {zeroBeamSolved}/{levels.Length} shipped levels " +
                                  $"({zeroBeamSolved / (double)levels.Length:P0}), SAME beam with a UNIFORM prior " +
                                  $"(the control for what the policy head is worth — beam and A* are different " +
                                  $"search shapes, so the h = 0 line above cannot attribute the beam result)");
        }
    }
}
