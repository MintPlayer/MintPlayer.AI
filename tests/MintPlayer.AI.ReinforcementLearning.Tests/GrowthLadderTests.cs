using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The ladder type exists to make one class of bug unrepresentable: a growth schedule whose rungs hold LESS
/// capacity than the net it is supposed to grow. That shipped — the shared DQN schedule topped out at
/// [128,128,128] while the games using it defaulted to [384,384] and [512,512] — so the invariants are tests.
/// </summary>
public class GrowthLadderTests
{
    [Fact]
    public void FromTrunk_RootsTheLadderAtTheNetsOwnDefault()
    {
        // The invariant that kills the downgrade bug by construction: rung 0 IS the non-growing shape, so
        // enabling growth can never start a run smaller than leaving it off.
        var ladder = GrowthLadder.FromTrunk([512, 512], steps: 3);

        Assert.Equal([512, 512], ladder.TrunkFor(0));
        Assert.Equal(3, ladder.Top);
    }

    [Fact]
    public void FromTrunk_AlternatesWidenAndDeepen_SoCapacityGrowsBothWays()
    {
        var ladder = GrowthLadder.FromTrunk([512, 512], steps: 3);

        Assert.Equal([768, 768], ladder.TrunkFor(1));           // widen
        Assert.Equal([768, 768, 768], ladder.TrunkFor(2));      // deepen
        Assert.Equal([1152, 1152, 1152], ladder.TrunkFor(3));   // widen
    }

    [Fact]
    public void EveryRungIsStrictlyAtLeastAsLargeAsTheOneBelow()
    {
        var ladder = GrowthLadder.FromTrunk([384, 384], steps: 5);

        for (int i = 1; i <= ladder.Top; i++)
        {
            int[] below = ladder.TrunkFor(i - 1);
            int[] here = ladder.TrunkFor(i);
            Assert.True(here.Length >= below.Length);
            Assert.True(here.Sum() > below.Sum(), $"rung {i} holds no more capacity than rung {i - 1}");
        }
    }

    [Fact]
    public void AHandWrittenLadderMustStepOneWidenOrOneDeepenAtATime()
    {
        // Net2Net does one or the other in a single function-preserving step. A rung that does both at once
        // cannot be reached without a loss spike, so it is refused at construction rather than at 3am mid-run.
        _ = new GrowthLadder([64, 64], [96, 96]);              // widen — fine
        _ = new GrowthLadder([64, 64], [64, 64, 64]);          // deepen — fine

        Assert.Throws<ArgumentException>(() => new GrowthLadder([64, 64], [96, 96, 96]));   // both at once
        Assert.Throws<ArgumentException>(() => new GrowthLadder([64, 64], [32, 32]));       // narrower
        Assert.Throws<ArgumentException>(() => new GrowthLadder([64, 64], [64, 64]));       // no change
        Assert.Throws<ArgumentException>(() => new GrowthLadder([64, 64], [64, 64, 64, 64]));// two layers at once
    }

    [Fact]
    public void ADegenerateLadderIsRefused()
    {
        Assert.Throws<ArgumentException>(() => new GrowthLadder());
        Assert.Throws<ArgumentException>(() => new GrowthLadder([]));
        Assert.Throws<ArgumentException>(() => new GrowthLadder([0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => GrowthLadder.FromTrunk([64], steps: 1, widenFactor: 1.0));
    }

    [Fact]
    public void ASingleRungLadderIsValid_AndImmediatelyAtItsTop()
    {
        // "Growth enabled but nowhere to go" must be expressible: it is what a game with no ladder yet looks
        // like, and SaturationGrowth reports it rather than pretending to grow.
        var ladder = GrowthLadder.FromTrunk([256, 256], steps: 0);

        Assert.Equal(0, ladder.Top);
        Assert.Equal([256, 256], ladder.TrunkFor(0));
    }

    [Fact]
    public void TrunkForClampsRatherThanThrowing()
    {
        var ladder = GrowthLadder.FromTrunk([128, 128], steps: 2);

        Assert.Equal(ladder.TrunkFor(0), ladder.TrunkFor(-5));
        Assert.Equal(ladder.TrunkFor(ladder.Top), ladder.TrunkFor(99));
    }
}
