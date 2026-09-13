using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>Everything a growing run must persist to resume mid-ladder. Carry it in the campaign's own state.</summary>
/// <param name="Rung">Which rung of the <see cref="GrowthLadder"/> the net is on. RECORDED, never recovered from
/// the live trunk: shape-matching cannot distinguish "rung 0" from "an architecture this ladder never described",
/// and guessing the latter as the former makes a resumed run jump straight to the top of the ladder.</param>
/// <param name="Plateau">The saturation window, reset on growth and whenever the metric is redefined.</param>
/// <param name="LastGrowthAt">Step counter at the last growth, for the log and for reading the curve afterwards.</param>
public readonly record struct GrowthProgress(int Rung, GrowthPlateau.State Plateau, long LastGrowthAt)
{
    public static GrowthProgress Start => new(0, GrowthPlateau.State.Empty, 0);
}

/// <summary>How eagerly a run adds capacity.</summary>
/// <param name="Enabled">Master switch; when false <see cref="SaturationGrowth.Observe"/> is a pure no-op.</param>
/// <param name="Patience">Metric observations without a new best before the net is called saturated.</param>
/// <param name="MinImprovement">How much an observation must beat the window's best by to count as progress
/// rather than noise. Must sit ABOVE the metric's jitter or upward noise resets patience forever.</param>
public readonly record struct GrowthSettings(bool Enabled, int Patience = 6, double MinImprovement = 0.04);

/// <summary>
/// Adds capacity to a net when its quality metric SATURATES, rather than on a fixed step cadence.
/// </summary>
/// <remarks>
/// <para><b>Feed it a held-out quality metric, not the loss.</b> Falling loss with a flat quality metric is the
/// saturation signature: the net is still learning to fit its targets and still failing at the task. A
/// loss-driven trigger cannot see that — it reads the falling loss as healthy progress and never fires.</para>
///
/// <para><b>Feed it on a DETERMINISTIC cadence.</b> Observations must be driven by a step/sample counter, never
/// by the wall clock. <see cref="CampaignRunner"/> fires <see cref="ITrainingCampaign.Evaluate"/> on a timer, so
/// driving growth from that would make the architecture — and therefore the whole trajectory — depend on how
/// fast the machine is, and a resumed or re-run job would grow at different points. A campaign whose only
/// quality signal comes from the timed eval needs a metric on its own sample cadence before it can adopt this.</para>
///
/// <para><b>Reset the window when the metric changes meaning.</b> A curriculum that promotes to a harder rung
/// makes its solve rate legitimately drop, which is indistinguishable from "no new highs" and would grow the net
/// for the one reason that is not a capacity problem. Call <see cref="ResetWindow"/> on any such change. Growth
/// resets it too: the grown net inherits the old one's function exactly, so it would otherwise inherit its best
/// and have to beat the plateau it was added to break — and one event could cascade up the whole ladder.</para>
/// </remarks>
public static class SaturationGrowth
{
    /// <summary>The outcome of folding one observation in: the (possibly grown) net and the progress to persist.</summary>
    public readonly record struct Result<TNet>(TNet Net, GrowthProgress Progress, bool Grew);

    /// <summary>
    /// Folds one metric observation into the saturation window and grows the net a single rung if it saturated.
    /// Pure with respect to training: the net is replaced, never mutated, and nothing else is touched.
    /// </summary>
    /// <param name="step">Current step/sample counter — recorded on growth, never used to decide it.</param>
    /// <param name="log">Told what happened, including the cases that deliberately do nothing: a silent no-op at
    /// the top of the ladder is indistinguishable from a trigger that never fired.</param>
    public static Result<TNet> Observe<TNet>(
        TNet net, GrowthProgress progress, double metric, long step,
        GrowthLadder ladder, GrowthSettings settings, Xoshiro256StarStar rng, Action<string> log)
        where TNet : IGrowableTrunkNet<TNet>
    {
        ArgumentNullException.ThrowIfNull(ladder);
        ArgumentNullException.ThrowIfNull(log);

        if (!settings.Enabled) return new(net, progress, false);

        var plateau = GrowthPlateau.Observe(progress.Plateau, metric, settings.MinImprovement);
        progress = progress with { Plateau = plateau };

        if (!GrowthPlateau.Saturated(plateau, settings.Patience)) return new(net, progress, false);

        if (progress.Rung >= ladder.Top)
        {
            log($"metric saturated at {plateau.Best:F3} but the net is at the TOP of the growth ladder " +
                $"([{string.Join(",", net.Trunk)}]) — no capacity left to add");
            return new(net, progress with { Plateau = GrowthPlateau.Reset() }, false);
        }

        int[] expected = ladder.TrunkFor(progress.Rung);
        if (!expected.AsSpan().SequenceEqual(net.Trunk))
        {
            log($"growth SKIPPED: net trunk [{string.Join(",", net.Trunk)}] does not match recorded rung " +
                $"{progress.Rung} ([{string.Join(",", expected)}]) — refusing to grow from an unknown architecture");
            return new(net, progress with { Plateau = GrowthPlateau.Reset() }, false);
        }

        int[] next = ladder.TrunkFor(progress.Rung + 1);
        bool deeper = next.Length > net.Trunk.Length;
        var grown = deeper ? net.Deepen(rng) : net.WidenTo(next, rng);

        log($"metric saturated: {settings.Patience} observations without beating {plateau.Best:F3} by " +
            $"{settings.MinImprovement:F3} at step {step:N0} — {(deeper ? "deepened" : "widened")} net → " +
            $"[{string.Join(",", grown.Trunk)}] (rung {progress.Rung} → {progress.Rung + 1}, function-preserving)");

        return new(grown, new GrowthProgress(progress.Rung + 1, GrowthPlateau.Reset(), step), true);
    }

    /// <summary>Starts the saturation window again, keeping the rung. Call when the metric changes meaning.</summary>
    public static GrowthProgress ResetWindow(GrowthProgress progress)
        => progress with { Plateau = GrowthPlateau.Reset() };
}
