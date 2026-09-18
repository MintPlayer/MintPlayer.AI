using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;
using Xunit.Abstractions;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Dead ends: states reachable from the start with NO path to the door. Block Dude has them because it is
/// irreversible — drop a block into a pit you cannot climb out of and the level is lost from that state, with
/// nothing in the rules to recover it. Rush Hour has none at all, which is the structural difference between the
/// two games (PRD §8.1a).
/// </summary>
public class BlockDudeDeadEndTests
{
    private readonly ITestOutputHelper _output;

    public BlockDudeDeadEndTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ASubstantialShareOfReachableStatesIsUnwinnable_AndTrainingDiscardsAllOfThem()
    {
        // The oracle already knows which states are lost: it enumerates forward from the start, then runs a
        // BACKWARD BFS from the won states, so anything the backward pass never reaches keeps dist = -1.
        //
        // CollectSamples then filters `distance > 0 && mask != 0`, which drops exactly those. The net is
        // therefore trained only on positions from which the level is still winnable, and has never once seen a
        // state labelled lost. This test measures how much of the state space that blind spot covers.
        int totalLive = 0, totalDead = 0;

        // M64: every stage -> {0, mid, last}. This is a MEASUREMENT test (it reports the
        // dead-end share of the state space); three rungs characterise the trend as well as
        // seven, and it cost 199s instrumented.
        foreach (int stage in new[] { 0, BlockDudeCurriculum.LastStage / 2, BlockDudeCurriculum.LastStage })
        {
            var spec = BlockDudeCurriculum.Stages[stage].Spec;
            int live = 0, dead = 0;

            foreach (var board in BlockDudeCurriculum.GateBoardsFor(stage, count: 2))
            {
                var oracle = new BlockDudeOracle(board, spec.OracleMaxStates);
                if (oracle.Truncated) continue;

                foreach (var (_, distance, _) in oracle.LabelledStates())
                {
                    if (distance < 0) dead++;
                    else live++;
                }
            }

            _output.WriteLine($"stage {stage}: {live,7:N0} winnable, {dead,7:N0} dead " +
                              $"({(live + dead == 0 ? 0 : dead / (double)(live + dead)):P1} of reachable states)");
            totalLive += live;
            totalDead += dead;
        }

        _output.WriteLine($"TOTAL: {totalLive:N0} winnable, {totalDead:N0} dead " +
                          $"({totalDead / (double)(totalLive + totalDead):P1})");

        // The claim this test exists to pin: dead ends are not a rounding error, so discarding them is a real
        // blind spot rather than a harmless filter. If this ever drops to zero, either the generator stopped
        // producing irreversible boards or the oracle stopped labelling them — both worth knowing.
        Assert.True(totalDead > 0, "expected reachable-but-lost states; Block Dude is supposed to be irreversible");
        Assert.True(totalDead / (double)(totalLive + totalDead) > 0.01,
            "expected dead ends to be a meaningful share of the state space");
    }
}
