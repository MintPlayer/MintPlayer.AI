using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The saturation detector behind capacity growth. The whole reason it is a running maximum with a patience
/// counter, rather than a comparison of consecutive values, is that the signal it watches is noisy: Block Dude's
/// 64-board gate was measured swinging ~10 points between consecutive evaluations while the loss fell throughout.
/// These tests pin the behaviour that noise level demands.
/// </summary>
public class GrowthPlateauTests
{
    private static GrowthPlateau.State Feed(double minImprovement, params double[] values)
    {
        var state = GrowthPlateau.State.Empty;
        foreach (double v in values) state = GrowthPlateau.Observe(state, v, minImprovement);
        return state;
    }

    [Fact]
    public void SteadyImprovementNeverSaturates()
    {
        var state = Feed(0.04, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90);

        Assert.Equal(0, state.EvalsSinceBest);
        Assert.False(GrowthPlateau.Saturated(state, patience: 6));
    }

    [Fact]
    public void AFlatMetricSaturatesExactlyAtThePatienceCount()
    {
        // First value sets the bar and does not count as a miss; six more without a new high do.
        var state = Feed(0.04, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50);

        Assert.Equal(6, state.EvalsSinceBest);
        Assert.False(GrowthPlateau.Saturated(state, patience: 7));
        Assert.True(GrowthPlateau.Saturated(state, patience: 6));
    }

    [Fact]
    public void TheMeasuredGateSwingDoesNotTriggerGrowth_ForANetThatIsStillImproving()
    {
        // The real shape from the 2026-09-11 measurement: ~10-point swings around a genuinely rising trend. A
        // detector comparing consecutive values would see three "declines" here and fire; this one must not.
        var state = Feed(0.04, 0.61, 0.50, 0.52, 0.66, 0.55, 0.58, 0.72);

        Assert.Equal(0.72, state.Best, 3);
        Assert.Equal(0, state.EvalsSinceBest);
        Assert.False(GrowthPlateau.Saturated(state, patience: 6));
    }

    [Fact]
    public void DownwardNoiseAloneCanNeverTriggerGrowth()
    {
        // A collapse is not saturation, and must not be read as one: the running maximum holds, so patience
        // advances at the same rate it would for a flat metric — never faster.
        var crashing = Feed(0.04, 0.80, 0.10, 0.05, 0.02);
        var flat = Feed(0.04, 0.80, 0.80, 0.80, 0.80);

        Assert.Equal(0.80, crashing.Best, 3);
        Assert.Equal(flat.EvalsSinceBest, crashing.EvalsSinceBest);
    }

    [Fact]
    public void AnImprovementSmallerThanTheNoiseFloorDoesNotResetPatience()
    {
        // Creeping up by a point a time is what upward jitter looks like. If that counted as progress the net
        // would never be called saturated, and the trigger would silently never fire.
        var state = Feed(0.04, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56);

        Assert.Equal(6, state.EvalsSinceBest);
        Assert.True(GrowthPlateau.Saturated(state, patience: 6));

        // The same series with a threshold below the step size reads as steady progress instead.
        Assert.Equal(0, Feed(0.005, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56).EvalsSinceBest);
    }

    [Fact]
    public void AnUngatedRungIsIgnoredRatherThanCountedAsAMiss()
    {
        // GateRates carries NaN for a rung that has not been gated yet. Folding that in as a miss would spend
        // patience on an observation that never happened.
        var state = Feed(0.04, 0.50, double.NaN, double.NaN);

        Assert.Equal(0.50, state.Best, 3);
        Assert.Equal(0, state.EvalsSinceBest);
    }

    [Fact]
    public void ResetForgetsTheWindow_SoAHarderRungStartsOnItsOwnScale()
    {
        var saturated = Feed(0.04, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50);
        Assert.True(GrowthPlateau.Saturated(saturated, patience: 6));

        var afterReset = GrowthPlateau.Reset();
        Assert.False(GrowthPlateau.Saturated(afterReset, patience: 6));

        // A promotion drops the gate to 0.20. Against the OLD best of 0.50 that reads as six more misses and
        // would grow the net immediately; on a fresh window it is simply the new rung's starting point.
        var onNewRung = GrowthPlateau.Observe(afterReset, 0.20, 0.04);
        Assert.Equal(0.20, onNewRung.Best, 3);
        Assert.Equal(0, onNewRung.EvalsSinceBest);
    }

    [Fact]
    public void AnEmptyWindowIsNeverSaturated_HoweverLongThePatience()
    {
        Assert.False(GrowthPlateau.Saturated(GrowthPlateau.State.Empty, patience: 0));
        Assert.False(GrowthPlateau.Saturated(GrowthPlateau.State.Empty, patience: 6));
    }
}
