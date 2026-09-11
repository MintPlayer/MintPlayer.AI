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
        int solved = 0;
        for (int i = 0; i < levels.Length; i++)
        {
            var board = BlockDudeBoard.FromGrid(levels[i].Grid);
            var outcome = BlockDudeGreedy.Run(net, board, budget);
            if (outcome.Solved) solved++;

            Console.WriteLine($"  {i + 1,2}. {levels[i].Name,-24} {board.Width,3}x{board.Height,-3} " +
                              $"{(outcome.Solved ? "SOLVED" : "-     ")} " +
                              $"{outcome.Ending,-8} after {outcome.Steps,5:N0} steps, " +
                              $"{outcome.DistinctStates,5:N0} distinct states");
        }

        Console.WriteLine();
        Console.WriteLine($"solved {solved}/{levels.Length} shipped levels " +
                          $"({solved / (double)levels.Length:P0}), greedy, no search");
    }
}
