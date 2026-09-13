using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The generic <see cref="SaturationGrowth"/> machinery bound to a real net. The detector and the ladder have
/// their own unit tests; these check the parts that only show up against an actual growable network.
/// </summary>
public class BlockDudeGrowthTests
{
    private static BlockDudePolicyNet Net(int[] trunk) => new(new Xoshiro256StarStar(7), trunk);
    private static GrowthSettings Eager => new(Enabled: true, Patience: 2, MinImprovement: 0.04);

    private static SaturationGrowth.Result<BlockDudePolicyNet> Saturate(
        BlockDudePolicyNet net, GrowthProgress progress, Action<string>? log = null)
    {
        // Three identical observations: the first sets the bar, the next two spend the patience of 2.
        var rng = new Xoshiro256StarStar(1);
        var result = new SaturationGrowth.Result<BlockDudePolicyNet>(net, progress, false);
        for (int i = 0; i < 3; i++)
            result = SaturationGrowth.Observe(result.Net, result.Progress, 0.50, 1_000, BlockDudeGrowth.Ladder,
                                              Eager, rng, log ?? (_ => { }));
        return result;
    }

    [Fact]
    public void RungZeroIsTheNetsDefaultTrunk_SoGrowingNeverStartsSmallerThanNotGrowing()
    {
        // The bug the ladder type exists to prevent: the shared DqnGrowth schedule starts at [16] and TOPS OUT
        // at [128,128,128], below this game's default [512,512]. Growing on it shrank the net.
        Assert.Equal(new BlockDudePolicyNet(new Xoshiro256StarStar(7)).Trunk, BlockDudeGrowth.Ladder.TrunkFor(0));
        Assert.True(BlockDudeGrowth.Ladder.TrunkFor(0)[0] > DqnGrowth.Stages[^1][^1],
            "rung 0 must already be wider than the shared ladder's top rung");
    }

    [Fact]
    public void ASaturatedMetricGrowsExactlyOneRung_AndResetsTheWindow()
    {
        var result = Saturate(Net(BlockDudeGrowth.Ladder.TrunkFor(0)), GrowthProgress.Start);

        Assert.True(result.Grew);
        Assert.Equal(1, result.Progress.Rung);
        Assert.Equal(BlockDudeGrowth.Ladder.TrunkFor(1), result.Net.Trunk);

        // The window must be empty afterwards, or a single plateau cascades up the whole ladder: the grown net
        // inherits the old one's function, so it would inherit its best gate too and never look like progress.
        Assert.Equal(0, result.Progress.PlateauEvalsSinceBest());
        Assert.True(double.IsNaN(result.Progress.Plateau.Best));
    }

    [Fact]
    public void SteadyImprovementNeverGrows_HoweverManyObservations()
    {
        // Steps of 0.05 against a 0.04 threshold: genuine progress, so patience never accrues. (A 0.02 step
        // would NOT count — that is below the noise floor by design, and the net would rightly be grown.)
        var rng = new Xoshiro256StarStar(1);
        var result = new SaturationGrowth.Result<BlockDudePolicyNet>(
            Net(BlockDudeGrowth.Ladder.TrunkFor(0)), GrowthProgress.Start, false);

        for (int i = 0; i < 15; i++)
            result = SaturationGrowth.Observe(result.Net, result.Progress, 0.05 + 0.05 * i, i, BlockDudeGrowth.Ladder,
                                              Eager, rng, _ => { });

        Assert.False(result.Grew);
        Assert.Equal(0, result.Progress.Rung);
    }

    [Fact]
    public void ACreepSmallerThanTheNoiseFloorDoesGrow_WhichIsThePoint()
    {
        // The mirror of the test above, and the behaviour that surfaced it: a metric inching up by less than
        // --grow-min-improvement per observation is jitter, not progress, and a net sitting under it is exactly
        // what capacity is for. If this ever stops growing, the threshold has been set below the noise and the
        // trigger will silently never fire.
        var rng = new Xoshiro256StarStar(1);
        var result = new SaturationGrowth.Result<BlockDudePolicyNet>(
            Net(BlockDudeGrowth.Ladder.TrunkFor(0)), GrowthProgress.Start, false);

        for (int i = 0; i < 4; i++)
            result = SaturationGrowth.Observe(result.Net, result.Progress, 0.50 + 0.01 * i, i, BlockDudeGrowth.Ladder,
                                              Eager, rng, _ => { });

        Assert.Equal(1, result.Progress.Rung);
    }

    [Fact]
    public void GrowthIsDisabledEntirely_WhenTheSettingIsOff()
    {
        var rng = new Xoshiro256StarStar(1);
        var net = Net(BlockDudeGrowth.Ladder.TrunkFor(0));

        var result = SaturationGrowth.Observe(net, GrowthProgress.Start, 0.5, 0, BlockDudeGrowth.Ladder,
                                              new GrowthSettings(Enabled: false), rng, _ => { });

        Assert.False(result.Grew);
        Assert.Equal(GrowthProgress.Start, result.Progress);   // not even the window advances
    }

    [Fact]
    public void GrowthIsRefusedAndReported_WhenTheRecordedRungDisagreesWithTheLiveTrunk()
    {
        // The failure the persisted rung exists to prevent. A net built outside the ladder matches no rung, and
        // the old trunk-matching helper silently reported rung 0 for it — which at a high step count meant a jump
        // straight to the top. Here it must refuse, and say so rather than no-op'ing silently.
        string? message = null;
        var result = Saturate(Net([300, 300]), GrowthProgress.Start, m => message = m);

        Assert.False(result.Grew);
        Assert.Equal(0, result.Progress.Rung);
        Assert.NotNull(message);
        Assert.Contains("does not match recorded rung", message);
    }

    [Fact]
    public void TheTopOfTheLadderIsReported_NotSilentlyIgnored()
    {
        // A silent no-op here is indistinguishable from a trigger that never fired — which is exactly how the
        // old clock-driven growth hid the fact that it was doing nothing useful.
        string? message = null;
        int top = BlockDudeGrowth.Ladder.Top;

        var result = Saturate(Net(BlockDudeGrowth.Ladder.TrunkFor(top)),
                              GrowthProgress.Start with { Rung = top }, m => message = m);

        Assert.False(result.Grew);
        Assert.NotNull(message);
        Assert.Contains("TOP of the growth ladder", message);
    }

    [Fact]
    public void GrowthIsFunctionPreserving_SoCapacityArrivesWithoutALossSpike()
    {
        // The point of Net2Net: the grown net computes what the old one computed. If this drifts, growth becomes
        // a reset and the saturation trigger would make training worse exactly when it fires.
        var net = Net(BlockDudeGrowth.Ladder.TrunkFor(0));
        var observation = new float[BlockDudeBoard.ObservationSize];
        BlockDudeLevels.Load(0).WriteObservation(observation);

        var (beforeLogits, beforeValue) = net.Forward(new Tensor((float[])observation.Clone(), 1, observation.Length));
        float[] before = [.. beforeLogits.Data, beforeValue.Data[0]];

        var result = Saturate(net, GrowthProgress.Start);
        Assert.True(result.Grew);

        var (afterLogits, afterValue) = result.Net.Forward(
            new Tensor((float[])observation.Clone(), 1, observation.Length));
        float[] after = [.. afterLogits.Data, afterValue.Data[0]];

        for (int i = 0; i < before.Length; i++)
            Assert.Equal(before[i], after[i], 3);
    }
}

internal static class GrowthProgressTestExtensions
{
    public static int PlateauEvalsSinceBest(this GrowthProgress progress) => progress.Plateau.EvalsSinceBest;
}
