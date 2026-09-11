using MintPlayer.AI.ReinforcementLearning.Campaigns;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The curriculum advance rule. It is a pure function of persisted state on purpose: that is what lets a run be
/// restarted from a blank slate and replay the same stages in the same order (PRD §7.1).
/// </summary>
public class BlockDudeCurriculumTests
{
    [Fact]
    public void AStageIsNeverLeftBeforeItsMinimumSamples()
    {
        var rung = BlockDudeCurriculum.Stages[0];

        // Even a perfect gate rate does not promote early — a net that memorised a small hold-out should still
        // see the rung's full sample budget.
        Assert.Equal(0, BlockDudeCurriculum.Advance(0, rung.MinStageSamples - 1, 1.0, out bool forced));
        Assert.False(forced);
    }

    [Fact]
    public void PassingTheGateAfterTheMinimumPromotes()
    {
        var rung = BlockDudeCurriculum.Stages[0];
        Assert.Equal(1, BlockDudeCurriculum.Advance(0, rung.MinStageSamples, rung.PromoteSolveRate, out bool forced));
        Assert.False(forced);
    }

    [Fact]
    public void FailingTheGateHoldsTheStage_UntilTheSampleCeiling()
    {
        var rung = BlockDudeCurriculum.Stages[0];
        double failing = rung.PromoteSolveRate - 0.01;

        Assert.Equal(0, BlockDudeCurriculum.Advance(0, rung.MinStageSamples, failing, out bool held));
        Assert.False(held);

        // At the ceiling the rung is left anyway, flagged — a stage that cannot be passed must stall visibly,
        // not silently consume the whole run.
        Assert.Equal(1, BlockDudeCurriculum.Advance(0, rung.MaxStageSamples, failing, out bool forced));
        Assert.True(forced);
    }

    [Fact]
    public void TheLastStageIsTerminal()
    {
        int last = BlockDudeCurriculum.LastStage;
        Assert.Equal(last, BlockDudeCurriculum.Advance(last, long.MaxValue, 1.0, out bool forced));
        Assert.False(forced);
    }

    [Fact]
    public void AdvanceIsPure_AndMonotone()
    {
        // Same inputs, same answer, and never a demotion — a stage the run has reached is never given back.
        for (int stage = 0; stage <= BlockDudeCurriculum.LastStage; stage++)
            foreach (long samples in new long[] { 0, 100_000, 1_000_000, 10_000_000 })
                foreach (double rate in new[] { 0.0, 0.5, 0.9, 1.0 })
                {
                    int first = BlockDudeCurriculum.Advance(stage, samples, rate, out _);
                    int second = BlockDudeCurriculum.Advance(stage, samples, rate, out _);
                    Assert.Equal(first, second);
                    Assert.InRange(first, stage, Math.Min(stage + 1, BlockDudeCurriculum.LastStage));
                }
    }

    [Fact]
    public void StagesGrowMonotonically()
    {
        // A curriculum whose rungs are not ordered would teach nothing in particular.
        for (int i = 1; i < BlockDudeCurriculum.Stages.Length; i++)
        {
            var previous = BlockDudeCurriculum.Stages[i - 1].Spec;
            var current = BlockDudeCurriculum.Stages[i].Spec;

            Assert.True(current.MaxWidth >= previous.MaxWidth, $"stage {i} is not wider");
            Assert.True(current.MaxHeight >= previous.MaxHeight, $"stage {i} is not taller");
            Assert.True(current.MaxBlocks >= previous.MaxBlocks, $"stage {i} does not allow more blocks");
            Assert.True(current.MaxFreeCells >= previous.MaxFreeCells, $"stage {i} is not more open");
        }
    }

    [Fact]
    public void EveryStageStaysInsideTheMeasuredOracleFrontier()
    {
        // Exact labelling dies around 6-7 blocks over ~150 free cells. A rung past that would spend the whole
        // run generating boards the oracle then refuses.
        foreach (var stage in BlockDudeCurriculum.Stages)
        {
            Assert.True(stage.Spec.MaxBlocks <= 7, "a stage exceeds the measured block frontier");
            Assert.True(stage.Spec.MaxFreeCells <= 150, "a stage exceeds the measured free-cell frontier");
        }
    }

    [Fact]
    public void TheGateHoldOutIsFixedPerStage_AndIndependentOfTheTrainingSeed()
    {
        // Regenerating a stage's hold-out must give the identical set, so gate rates are comparable across runs
        // and across seeds.
        var first = BlockDudeCurriculum.GateBoardsFor(0, count: 6);
        var second = BlockDudeCurriculum.GateBoardsFor(0, count: 6);

        Assert.NotEmpty(first);
        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
            Assert.Equal(string.Join("\n", first[i].ToGrid()), string.Join("\n", second[i].ToGrid()));
    }
}
