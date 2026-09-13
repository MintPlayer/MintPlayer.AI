using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>
/// Measures the value head against ground truth: predicted distance-to-goal vs the exact oracle's optimal move
/// count, over labelled states drawn from the curriculum's own hold-out boards.
/// </summary>
/// <remarks>
/// <para><b>Why this number matters more than the loss.</b> The Huber term reports error in SCALED units against
/// a moving training distribution; it cannot tell you whether the estimate is biased, or biased differently for
/// the states that matter. And the value head is not decoration — it is the heuristic
/// <see cref="BlockDudeSearch"/> steers by, so a systematic bias is a search that walks confidently in the wrong
/// direction.</para>
///
/// <para><b>The specific hypothesis this tests (owner, 2026-09-13):</b> in Block Dude the player often has to
/// walk AWAY from the door first, to fetch blocks, before any progress toward it is possible. If the value head
/// has learned "close to the door ⇒ nearly done", it will UNDER-estimate exactly those states — the ones where a
/// long detour is still required — and mis-rank them against states that are genuinely nearly finished. The
/// breakdown by true distance is what shows that: bias should be roughly flat across bands if the head has
/// learned real cost-to-go, and increasingly negative at large true distances if it has learned geometry.</para>
/// </remarks>
internal static class BlockDudeValueCalibration
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        string dataDir = a.Str("--data", "data");
        int phase = a.Int("--phase", 1);
        int perStage = a.Int("--boards", 6);
        int maxStage = a.Int("--max-stage", BlockDudeCurriculum.LastStage);
        bool useResumeNet = a.Has("--resume-net");

        var ids = BlockDudeIds.ForPhase(phase);
        string netId = useResumeNet ? ids.Policy : ids.PolicyBest;
        var store = new FileModelStore(dataDir);
        using var stream = store.TryOpenRead(BlockDudeIds.Environment, netId);
        if (stream is null) { Console.WriteLine($"No net at {dataDir}/{BlockDudeIds.Environment}.{netId}.ckpt."); return; }

        var net = BlockDudePolicyNet.Load(stream);
        Console.WriteLine($"net: {dataDir}/{BlockDudeIds.Environment}.{netId}.ckpt (trunk [{string.Join(",", net.Trunk)}])");
        Console.WriteLine();
        Console.WriteLine("  stage  states    MAE   bias   |  bias by TRUE distance band");
        Console.WriteLine("                                |   1-10   11-25   26-50    51+");

        // Bands over the TRUE distance: a head that has learned geometry rather than cost-to-go goes wrong in the
        // far bands, where the remaining work is a detour the geometry cannot see.
        (int Lo, int Hi)[] bands = [(1, 10), (11, 25), (26, 50), (51, int.MaxValue)];

        for (int stage = 0; stage <= maxStage; stage++)
        {
            var spec = BlockDudeCurriculum.Stages[stage].Spec;
            double sumAbs = 0, sumSigned = 0;
            int n = 0;
            var bandSum = new double[bands.Length];
            var bandCount = new int[bands.Length];

            foreach (var board in BlockDudeCurriculum.GateBoardsFor(stage, perStage))
            {
                var oracle = new BlockDudeOracle(board, spec.OracleMaxStates);
                if (oracle.Truncated) continue;

                foreach (var (state, trueDistance, _) in oracle.LabelledStates())
                {
                    if (trueDistance <= 0) continue;
                    float predicted = net.Evaluate(state).Distance;
                    double error = predicted - trueDistance;      // negative ⇒ the net thinks it is closer than it is

                    sumAbs += Math.Abs(error);
                    sumSigned += error;
                    n++;

                    for (int b = 0; b < bands.Length; b++)
                        if (trueDistance >= bands[b].Lo && trueDistance <= bands[b].Hi)
                        {
                            bandSum[b] += error;
                            bandCount[b]++;
                            break;
                        }
                }
            }

            if (n == 0) { Console.WriteLine($"  {stage,5}  (no labelled states)"); continue; }

            string Band(int b) => bandCount[b] == 0 ? "     -" : $"{bandSum[b] / bandCount[b],6:+0.0;-0.0}";
            Console.WriteLine($"  {stage,5} {n,7:N0} {sumAbs / n,6:F1} {sumSigned / n,6:+0.0;-0.0}   | " +
                              $"{Band(0)}  {Band(1)}  {Band(2)} {Band(3)}");
        }

        Console.WriteLine();
        Console.WriteLine("  MAE/bias are in MOVES. Negative bias = the net believes it is closer to the door than it is.");
        Console.WriteLine("  A bias that grows more negative with true distance is the geometry failure mode:");
        Console.WriteLine("  states needing a long detour are scored as nearly finished, which mis-steers the search.");
    }
}
