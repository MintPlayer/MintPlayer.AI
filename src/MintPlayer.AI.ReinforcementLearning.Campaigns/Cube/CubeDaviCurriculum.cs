namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// The DAVI curriculum's decision rules, as a pure function — extracted from
/// <c>CubeDaviCampaign.AdvanceCurriculumOrGrow</c> so they can be tested without a campaign, a net, a
/// backend or a checkpoint store (M69/A1; asked for in <c>CAMPAIGN_TESTABILITY_PRD.md</c> §6).
/// </summary>
/// <remarks>
/// <para><b>Why this is worth extracting.</b> The campaign's 286 lines were entirely untested because its
/// constructor demands a concrete GPU backend, and these rules — advance, auto-widen, stall-note, grow —
/// are the part where a mistake is both easy and invisible: a wrong comparison does not throw, it just
/// trains at the wrong scramble depth or at the wrong width, and the run still looks healthy.</para>
///
/// <para><b>What deliberately stayed behind.</b> Everything with a side effect: the net swap, the fresh
/// <c>Adam</c>, <c>BuildStack</c>, <c>SaveCheckpoint</c>, the log lines — and, importantly, the
/// <c>MeanValueAtDepth</c> call that produces <paramref name="frontierRatio"/>. That runs net forwards,
/// so taking it as a parameter is what keeps this function pure.</para>
///
/// <para><b>Three details that are load-bearing, not incidental.</b>
/// (1) <paramref name="lastLoss"/> is <c>float</c> while <paramref name="bestLossSinceReset"/> is
/// <c>double</c>, and the literal is <c>0.98f</c>. The comparison promotes to <c>double</c>; writing
/// <c>0.98</c> instead would move the threshold, because <c>0.98f</c> is not <c>0.98</c>.
/// (2) The first three outcomes are an <c>if / else if / else if</c> chain — at most one fires.
/// (3) <see cref="Decision.Grow"/> is a <b>separate</b> decision evaluated independently, not a fourth
/// arm: a run can advance the curriculum and hit its progressive-growth threshold in the same call.</para>
/// </remarks>
public static class CubeDaviCurriculum
{
    /// <summary>Which of the three mutually exclusive curriculum outcomes a step selected.</summary>
    public enum Outcome
    {
        /// <summary>Nothing to do: the frontier is not mastered and nothing has stalled long enough.</summary>
        None,
        /// <summary>Mean predicted value at the frontier cleared the gate — train one shell deeper.</summary>
        Advance,
        /// <summary>Loss flatlined at the frontier while still short of the gate — capacity-bound, widen.</summary>
        Widen,
        /// <summary>Stalled long enough to be worth saying so, but auto-widen is off or already at max width.</summary>
        StallNote,
    }

    /// <param name="Outcome">The one arm of the chain that fired.</param>
    /// <param name="NewWidth">Target trunk width when <see cref="Outcome.Widen"/>; 0 otherwise.</param>
    /// <param name="BestLossSinceReset">The plateau tracker's new value — <see cref="double.MaxValue"/>
    /// whenever the shell or the width changed, because a new shell's loss is not comparable to the old
    /// one's and carrying it over would suppress the next widen.</param>
    /// <param name="SamplesAtBestLoss">The plateau timer's new anchor.</param>
    /// <param name="SamplesSinceAdvance">Reset to 0 on Advance, Widen and StallNote (the note throttles
    /// itself); carried through unchanged on None.</param>
    /// <param name="LossStagnantSamples">Samples since the last ≥2% loss improvement, for the log line.</param>
    /// <param name="Grow">Whether progressive growing fires — independent of <see cref="Outcome"/>.</param>
    /// <param name="GrowWidth">Target width when <paramref name="Grow"/>; 0 otherwise.</param>
    public readonly record struct Decision(
        Outcome Outcome,
        int NewWidth,
        double BestLossSinceReset,
        long SamplesAtBestLoss,
        long SamplesSinceAdvance,
        long LossStagnantSamples,
        bool Grow,
        int GrowWidth);

    /// <summary>
    /// Decides what the curriculum does this step. <paramref name="netWidth"/> is the trunk width when the
    /// net is a widenable residual trunk and <c>null</c> otherwise — that null is how the campaign's two
    /// <c>is ResidualMlp</c> pattern checks become data, so a non-widenable net can never select Widen or
    /// Grow.
    /// </summary>
    public static Decision Step(
        int curriculumDepth,
        long currentSamples,
        long samplesSinceAdvance,
        long samplesAtBestLoss,
        double bestLossSinceReset,
        float lastLoss,
        int? netWidth,
        double frontierRatio,
        int maxDepthCap,
        double advanceRatio,
        bool autoWiden,
        int maxWidth,
        long widenStallSamples,
        long stallWarnSamples,
        int growToWidth,
        long growAtSamples)
    {
        // ≥2% improvement resets the plateau timer. Types kept exactly as the campaign had them: the
        // float/double mix and the `f` suffix both matter (see the class remarks).
        if (lastLoss < bestLossSinceReset * 0.98f)
        {
            bestLossSinceReset = lastLoss;
            samplesAtBestLoss = currentSamples;
        }
        long lossStagnantSamples = currentSamples - samplesAtBestLoss;

        var outcome = Outcome.None;
        int newWidth = 0;

        if (curriculumDepth < maxDepthCap && frontierRatio >= advanceRatio)
        {
            outcome = Outcome.Advance;
            samplesSinceAdvance = 0;
            bestLossSinceReset = double.MaxValue;   // new shell → track its loss afresh
            samplesAtBestLoss = currentSamples;
        }
        else if (autoWiden && netWidth is int w && w < maxWidth && lossStagnantSamples >= widenStallSamples)
        {
            outcome = Outcome.Widen;
            newWidth = Math.Min(maxWidth, w * 2);
            bestLossSinceReset = double.MaxValue;
            samplesAtBestLoss = currentSamples;
            samplesSinceAdvance = 0;
        }
        else if (curriculumDepth < maxDepthCap && samplesSinceAdvance >= stallWarnSamples)
        {
            outcome = Outcome.StallNote;
            samplesSinceAdvance = 0;                // throttle the note
        }

        // Progressive growing — a SEPARATE decision, not a fourth arm of the chain above.
        bool grow = growToWidth > 0 && netWidth is int g && g < growToWidth && currentSamples >= growAtSamples;

        return new Decision(
            outcome, newWidth, bestLossSinceReset, samplesAtBestLoss,
            samplesSinceAdvance, lossStagnantSamples, grow, grow ? growToWidth : 0);
    }
}
