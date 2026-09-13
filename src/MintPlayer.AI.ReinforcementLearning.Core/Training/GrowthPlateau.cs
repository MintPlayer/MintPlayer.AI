namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Saturation detection for progressive net growth: decides WHEN a net has stopped getting better, so capacity can
/// be added because the net needs it rather than because a sample counter said so.
/// </summary>
/// <remarks>
/// <para><b>Why not simply "the metric did not improve".</b> The signal this watches is a solve rate over a finite
/// hold-out, and it is noisy: Block Dude's 64-board gate was measured swinging ~10 points between consecutive
/// evaluations (61% → 50% → 52%) while the loss fell throughout. A detector comparing consecutive values, or even
/// consecutive windows, fires on that noise almost immediately and would grow a net that is still learning fine.</para>
///
/// <para><b>The rule.</b> Track the BEST value seen so far and a patience counter, exactly as early stopping does.
/// An evaluation that beats the best by more than <c>minImprovement</c> resets patience; anything else increments
/// it. Saturation is patience reaching <c>patience</c> evaluations. Because the reference is a running maximum, a
/// downward swing can never trigger growth — only a genuine absence of new highs can. Upward noise sets a new high
/// and DELAYS growth, which is the safe direction to be wrong in: a late grow costs samples, an early grow costs
/// samples AND resets the optimizer on a net that was still improving.</para>
///
/// <para><b>What the caller must do.</b> The maximum is only meaningful while the thing being measured holds still.
/// A curriculum that promotes to a harder rung makes the gate legitimately drop, which is indistinguishable here
/// from "no new highs" — so the caller <see cref="Reset"/>s on any change that redefines the metric (a stage
/// change), and after growing (the new net must earn its own high before it can be called saturated again).</para>
/// </remarks>
public static class GrowthPlateau
{
    /// <summary>Best value seen since the last reset, and how many evaluations have passed without beating it.
    /// <see cref="Best"/> is <see cref="double.NaN"/> before the first observation.</summary>
    public readonly record struct State(double Best, int EvalsSinceBest)
    {
        public static State Empty => new(double.NaN, 0);
    }

    /// <summary>Folds one evaluation into the plateau state. Pure.</summary>
    /// <param name="minImprovement">How much a value must beat the running best by to count as progress rather
    /// than noise. Sized ABOVE the metric's own jitter, or every upward swing resets patience forever.</param>
    public static State Observe(State current, double value, double minImprovement)
    {
        if (double.IsNaN(value)) return current;                       // a rung that has not been gated yet
        if (double.IsNaN(current.Best)) return new(value, 0);          // first observation sets the bar
        if (value > current.Best + minImprovement) return new(value, 0);
        return new(Math.Max(current.Best, value), current.EvalsSinceBest + 1);
    }

    /// <summary>Whether the metric has gone <paramref name="patience"/> evaluations without a new high.</summary>
    public static bool Saturated(State state, int patience)
        => !double.IsNaN(state.Best) && state.EvalsSinceBest >= patience;

    /// <summary>Forgets the history. Call on anything that redefines the metric, and after growing.</summary>
    public static State Reset() => State.Empty;
}
