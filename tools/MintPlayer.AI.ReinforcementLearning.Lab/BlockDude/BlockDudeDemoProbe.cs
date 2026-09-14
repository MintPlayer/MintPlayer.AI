using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>
/// Measures how far back along the human demonstrations the current net + A* can still finish a level.
/// </summary>
/// <remarks>
/// <para>This is the number that decides whether a reverse curriculum is worth building. The premise is that a
/// suffix of a human solution is a real position on real shipped terrain that is only N moves from the door, so
/// search should handle it even on levels it cannot touch from the opening. If that holds, fifteen
/// demonstrations become thousands of trainable tasks; if search fails even 10 moves from the door on a big
/// level, the premise is wrong and the plan needs rethinking.</para>
///
/// <para>Read the depth at which each level stops being solvable as the ceiling the curriculum would start
/// beneath — the frontier training should push outward, not the answer itself.</para>
/// </remarks>
internal static class BlockDudeDemoProbe
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        string dataDir = a.Str("--data", "data");
        int phase = a.Int("--phase", 1);
        int expansions = a.Int("--expansions", 60_000);
        float weight = a.Flt("--weight", 2f);
        int seconds = a.Int("--search-seconds", 10);
        bool useResumeNet = a.Has("--resume-net");

        int[] depths = [.. (a.Str("--depths", "5,10,20,40,80,160") ?? "").Split(',').Select(int.Parse)];

        var ids = BlockDudeIds.ForPhase(phase);
        var store = new FileModelStore(dataDir);
        using var stream = store.TryOpenRead(BlockDudeIds.Environment, useResumeNet ? ids.Policy : ids.PolicyBest);
        if (stream is null) { Console.WriteLine($"No net under {dataDir}."); return; }

        var net = BlockDudePolicyNet.Load(stream);
        var human = BlockDudeDemonstrations.HumanMoveCounts();

        Console.WriteLine($"net trunk [{string.Join(",", net.Trunk)}], A* weight {weight}, " +
                          $"≤{expansions:N0} expansions, ≤{seconds}s per attempt");
        Console.WriteLine();
        Console.Write($"  {"level",-10} {"human",6}  ");
        foreach (int d in depths) Console.Write($"{d,6}");
        Console.WriteLine("   ← moves from the door");

        foreach (var solution in BlockDudeSolutions.All)
        {
            Console.Write($"  {solution.Name,-10} {human[solution.Name],6}  ");
            foreach (int depth in depths)
            {
                // Past the level's own length the task IS the level, so there is nothing further to report.
                if (depth > solution.Moves.Length) { Console.Write($"{"·",6}"); continue; }

                var start = BlockDudeDemonstrations.ReverseCurriculumStarts(depth, solution.Name).Single();
                var found = BlockDudeSearch.Solve(net, start, expansions, weight, TimeSpan.FromSeconds(seconds));
                Console.Write($"{(found.Solved ? found.Length.ToString() : "-"),6}");
            }
            Console.WriteLine();
        }

        Console.WriteLine();
        Console.WriteLine("  A number is the solution length A* found from that many moves out; '-' is a miss,");
        Console.WriteLine("  '·' means the level is shorter than that. The rightmost solved column per row is");
        Console.WriteLine("  where a reverse curriculum would currently sit for that level.");
    }
}
