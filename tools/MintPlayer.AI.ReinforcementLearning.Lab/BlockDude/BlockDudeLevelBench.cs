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
        float policyWeight = a.Flt("--policy-weight", 1f);

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

        var levels = BlockDudeLevels.All;
        int solved = 0, noRevisitSolved = 0, searchSolved = 0, zeroSolved = 0, policySolved = 0;
        for (int i = 0; i < levels.Length; i++)
        {
            var board = BlockDudeBoard.FromGrid(levels[i].Grid);
            var outcome = BlockDudeGreedy.Run(net, board, budget);
            if (outcome.Solved) solved++;

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
            }

            if (policySearch)
            {
                var found = BlockDudeSearch.SolveWithPolicy(net, board, expansions, weight, policyWeight,
                                                           TimeSpan.FromSeconds(seconds));
                if (found.Solved) policySolved++;
                line += found.Solved ? $"  | policy+value SOLVED in {found.Length,4:N0} moves" : "  | policy+value -";
            }

            if (zeroH)
            {
                var blind = BlockDudeSearch.Solve(BlockDudeSearch.ZeroHeuristic, board, expansions, weight,
                                                  TimeSpan.FromSeconds(seconds));
                if (blind.Solved) zeroSolved++;
                line += blind.Solved ? $"  | blind SOLVED in {blind.Length,4:N0} moves" : "  | blind -";
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

        if (policySearch)
        {
            Console.WriteLine($"solved {policySolved}/{levels.Length} shipped levels " +
                              $"({policySolved / (double)levels.Length:P0}), policy-prior + value A* " +
                              $"(weight {weight}, policy weight {policyWeight}, ≤{expansions:N0} expansions, ≤{seconds}s per level)");
        }

        if (zeroH)
        {
            // The control. Whatever the net-guided number is, the honest claim about the heuristic is the
            // DIFFERENCE between the two lines — not the net-guided line on its own.
            Console.WriteLine($"solved {zeroSolved}/{levels.Length} shipped levels " +
                              $"({zeroSolved / (double)levels.Length:P0}), SAME search with h = 0 " +
                              $"(uninformed breadth-first — the control for what the value head is worth)");
        }
    }
}
