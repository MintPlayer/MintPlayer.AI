using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// These three checks share one subject but live in THREE classes, deliberately. xUnit's unit of
// parallelism is the test *collection*, and by default that is the class — tests inside one class
// run strictly serially, however many cores are free. As one class these three were 272s of serial
// work against a 279s wall clock for the whole 707-test suite, i.e. they WERE the critical path
// (measured 2026-09-18, M63). Split, they run concurrently and the suite is bounded by the longest
// single test instead of their sum.
//
// Keep them apart. Merging them back into one class silently re-serialises ~90 seconds. Splitting
// the FILE is unnecessary — the collection is keyed on the class, not the file.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// The gate hold-out must be drawn from the SAME distribution the net trains on. Training discards any board
/// whose exact oracle exceeds the rung's state cap; until 2026-09-13 the gate did not, so it scored the policy on
/// a class of board that could never appear in its training data — and the gap widened with every rung
/// (truncations ran 0 → 1 → 8 → 53 → 181 → 313 over one measured run).
/// </summary>
public class BlockDudeGateBoardLabellingTests
{
    [Fact]
    public void EveryGateBoardIsOneTheExactOracleCanFullyLabel()
    {
        // The invariant. A board the oracle truncates on is one the net was never trained for, so scoring it
        // measures the generator's reach rather than the policy's.
        //
        // This re-solves each returned board even though `GateBoardsFor` already filtered on exactly this
        // condition, so today the assertion cannot fail. That is the POINT: it is the regression guard on the
        // filter itself. Drop the re-solve and the body becomes an empty foreach that would not notice the
        // `if (oracle.Truncated) continue` being removed from `GateBoardsFor`.
        for (int stage = 0; stage <= BlockDudeCurriculum.LastStage; stage++)
        {
            var spec = BlockDudeCurriculum.Stages[stage].Spec;
            foreach (var board in BlockDudeCurriculum.GateBoardsFor(stage, count: 8))
            {
                var oracle = new BlockDudeOracle(board, spec.OracleMaxStates);
                Assert.False(oracle.Truncated,
                    $"stage {stage}: gate board {board.Width}x{board.Height} truncates the oracle, so the net " +
                    "is being graded on a board it could never have been trained on");
            }
        }
    }
}

/// <summary>
/// The hold-out is seeded from the stage alone, so it must be identical run to run. See
/// <see cref="BlockDudeGateBoardLabellingTests"/> for why this is its own class.
/// </summary>
public class BlockDudeGateBoardReproducibilityTests
{
    [Fact]
    public void TheHoldOutIsStillReproducibleFromTheStageAlone()
    {
        // GateRng is seeded from the stage, deliberately independent of --seed, so the hold-out is the same set
        // for every run at that rung. Adding the oracle filter must not have introduced run-to-run variation.
        for (int stage = 0; stage <= 2; stage++)
        {
            var first = BlockDudeCurriculum.GateBoardsFor(stage, count: 6);
            var second = BlockDudeCurriculum.GateBoardsFor(stage, count: 6);

            Assert.Equal(first.Count, second.Count);
            for (int i = 0; i < first.Count; i++)
                Assert.Equal(string.Join("\n", first[i].ToGrid()), string.Join("\n", second[i].ToGrid()));
        }
    }
}

/// <summary>
/// The oracle filter rejects boards, so it could in principle empty the hold-out. See
/// <see cref="BlockDudeGateBoardLabellingTests"/> for why this is its own class.
/// </summary>
public class BlockDudeGateBoardStarvationTests
{
    [Fact]
    public void TheHoldOutIsNotStarvedByTheFilter()
    {
        // The filter rejects boards, so it could in principle empty the hold-out at the hardest rung and turn the
        // gate into a silent 0. The attempt budget must be generous enough that it does not.
        var boards = BlockDudeCurriculum.GateBoardsFor(BlockDudeCurriculum.LastStage, count: 8);

        Assert.NotEmpty(boards);
    }
}
