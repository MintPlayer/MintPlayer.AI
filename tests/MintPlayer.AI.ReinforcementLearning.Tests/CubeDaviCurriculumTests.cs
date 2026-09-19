using MintPlayer.AI.ReinforcementLearning.Campaigns;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The DAVI curriculum's advance / auto-widen / stall-note / grow rules (M69/A1), extracted from
/// <c>CubeDaviCampaign</c> so they can be tested at all.
/// </summary>
/// <remarks>
/// <para>Until this extraction <b>the entire campaign was 0/286 covered</b> — not one statement had ever
/// executed in a test, because its constructor demands a concrete GPU backend. These rules were the part
/// worth rescuing first: a mistake in them does not throw, it silently trains at the wrong scramble depth
/// or the wrong width, and the run still looks healthy for hours.</para>
/// <para>Every test below fixes one input and varies one other, so a failure names the rule that broke
/// rather than "the curriculum changed".</para>
/// </remarks>
public class CubeDaviCurriculumTests
{
    /// <summary>A step where nothing fires: mid-curriculum, gate not met, no stall, growth off.</summary>
    private static CubeDaviCurriculum.Decision Step(
        int depth = 5, long samples = 1_000_000, long sinceAdvance = 0, long atBestLoss = 1_000_000,
        double bestLoss = 1.0, float lastLoss = 1.0f, int? width = 256, double frontierRatio = 0.5,
        int maxDepthCap = 20, double advanceRatio = 0.9, bool autoWiden = false, int maxWidth = 1024,
        long widenStall = 5_000_000, long stallWarn = 4_000_000, int growTo = 0, long growAt = 0)
        => CubeDaviCurriculum.Step(depth, samples, sinceAdvance, atBestLoss, bestLoss, lastLoss, width,
            frontierRatio, maxDepthCap, advanceRatio, autoWiden, maxWidth, widenStall, stallWarn, growTo, growAt);

    // ── advancing the frontier ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Clearing_the_gate_advances_the_curriculum()
    {
        var d = Step(frontierRatio: 0.95, advanceRatio: 0.9);

        Assert.Equal(CubeDaviCurriculum.Outcome.Advance, d.Outcome);
    }

    [Fact]
    public void Missing_the_gate_does_not_advance()
    {
        // There is deliberately NO forced advance: a persistent stall is the honest "needs more training
        // or capacity" signal, and papering over it would train the next shell on untrustworthy targets.
        Assert.Equal(CubeDaviCurriculum.Outcome.None, Step(frontierRatio: 0.89, advanceRatio: 0.9).Outcome);
    }

    [Fact]
    public void The_gate_is_inclusive_at_the_threshold()
    {
        Assert.Equal(CubeDaviCurriculum.Outcome.Advance, Step(frontierRatio: 0.9, advanceRatio: 0.9).Outcome);
    }

    [Fact]
    public void The_depth_cap_stops_advancing()
    {
        // At the cap the gate is irrelevant — and note the stall-note arm is also capped, so a capped
        // curriculum goes quiet rather than logging forever.
        var d = Step(depth: 20, maxDepthCap: 20, frontierRatio: 1.0, sinceAdvance: long.MaxValue / 2);

        Assert.Equal(CubeDaviCurriculum.Outcome.None, d.Outcome);
    }

    [Fact]
    public void Advancing_resets_the_plateau_tracker_to_an_incomparable_loss()
    {
        // A new shell's loss is not comparable to the old shell's, so carrying the old best over would
        // suppress the next auto-widen indefinitely. MaxValue is how "no baseline yet" is expressed.
        var d = Step(frontierRatio: 1.0, bestLoss: 0.01, samples: 7_000_000);

        Assert.Equal(double.MaxValue, d.BestLossSinceReset);
        Assert.Equal(7_000_000, d.SamplesAtBestLoss);
        Assert.Equal(0, d.SamplesSinceAdvance);
    }

    // ── the plateau timer ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_two_percent_improvement_resets_the_plateau_timer()
    {
        var d = Step(lastLoss: 0.97f, bestLoss: 1.0, samples: 5_000_000);

        Assert.Equal(0.97, d.BestLossSinceReset, 5);
        Assert.Equal(5_000_000, d.SamplesAtBestLoss);
        Assert.Equal(0, d.LossStagnantSamples);
    }

    [Fact]
    public void A_smaller_improvement_does_not_reset_it()
    {
        // 1% better is still "plateaued" — the rule is a ≥2% drop, so noise cannot keep the timer alive
        // and prevent a widen that is genuinely needed.
        var d = Step(lastLoss: 0.99f, bestLoss: 1.0, samples: 5_000_000, atBestLoss: 1_000_000);

        Assert.Equal(1.0, d.BestLossSinceReset, 5);
        Assert.Equal(4_000_000, d.LossStagnantSamples);
    }

    [Fact]
    public void The_threshold_is_strict_and_uses_the_float_literal_it_was_written_with()
    {
        // Two things at once, both invisible in review.
        //
        // The comparison is strictly `<`, so a loss sitting EXACTLY on the threshold is not an
        // improvement and the plateau timer keeps running. And the literal is `0.98f`, which widens to
        // 0.9800000190734863 rather than 0.98 — so "exactly on the threshold" means that value, not the
        // decimal one. Rewriting the literal as `0.98` would move the boundary between these two cases.
        var atThreshold = Step(lastLoss: 0.98f, bestLoss: 1.0, samples: 3_000_000, atBestLoss: 1_000_000);
        Assert.Equal(1.0, atThreshold.BestLossSinceReset, 6);          // unchanged: not an improvement
        Assert.Equal(2_000_000, atThreshold.LossStagnantSamples);

        // A hair under it is an improvement, which is what makes the boundary meaningful rather than
        // an accident of rounding.
        var justUnder = Step(lastLoss: 0.9799f, bestLoss: 1.0, samples: 3_000_000, atBestLoss: 1_000_000);
        Assert.Equal(0.9799, justUnder.BestLossSinceReset, 4);
        Assert.Equal(0, justUnder.LossStagnantSamples);
    }

    // ── auto-widen ───────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_plateaued_frontier_widens_when_auto_widen_is_on()
    {
        var d = Step(autoWiden: true, width: 256, maxWidth: 1024,
                     widenStall: 1_000_000, samples: 5_000_000, atBestLoss: 1_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
        Assert.Equal(512, d.NewWidth);   // doubles
    }

    [Fact]
    public void Widening_is_clamped_to_the_maximum_width()
    {
        var d = Step(autoWiden: true, width: 768, maxWidth: 1024,
                     widenStall: 1_000_000, samples: 5_000_000, atBestLoss: 1_000_000);

        Assert.Equal(1024, d.NewWidth);
    }

    [Fact]
    public void A_net_already_at_maximum_width_does_not_widen()
    {
        var d = Step(autoWiden: true, width: 1024, maxWidth: 1024,
                     widenStall: 1_000_000, samples: 5_000_000, atBestLoss: 1_000_000,
                     sinceAdvance: 4_000_000);

        Assert.NotEqual(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
    }

    [Fact]
    public void A_non_widenable_net_never_widens()
    {
        // null width is how "the net is not a widenable residual trunk" is carried. A plain MLP must not
        // select Widen — the campaign would have to cast it, which is the bug class that put an
        // InvalidCastException in the time-budget path.
        var d = Step(autoWiden: true, width: null, widenStall: 1_000_000,
                     samples: 5_000_000, atBestLoss: 1_000_000, sinceAdvance: 4_000_000);

        Assert.NotEqual(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
        Assert.Equal(0, d.NewWidth);
    }

    [Fact]
    public void Widening_requires_auto_widen_to_be_enabled()
    {
        var d = Step(autoWiden: false, width: 256, widenStall: 1_000_000,
                     samples: 5_000_000, atBestLoss: 1_000_000, sinceAdvance: 4_000_000);

        Assert.NotEqual(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
    }

    [Fact]
    public void Advancing_takes_precedence_over_widening()
    {
        // Both conditions hold. Advance wins, because a frontier that just cleared its gate is not
        // capacity-bound — widening there would spend capacity on a problem the net already solved.
        var d = Step(frontierRatio: 1.0, autoWiden: true, width: 256,
                     widenStall: 1_000_000, samples: 5_000_000, atBestLoss: 1_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.Advance, d.Outcome);
    }

    // ── the stall note ───────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_long_stall_without_auto_widen_reports_a_note()
    {
        var d = Step(sinceAdvance: 4_000_000, stallWarn: 4_000_000, autoWiden: false);

        Assert.Equal(CubeDaviCurriculum.Outcome.StallNote, d.Outcome);
        Assert.Equal(0, d.SamplesSinceAdvance);   // throttled, so it does not log every chunk
    }

    [Fact]
    public void Widening_takes_precedence_over_the_note()
    {
        var d = Step(autoWiden: true, width: 256, widenStall: 1_000_000,
                     samples: 5_000_000, atBestLoss: 1_000_000, sinceAdvance: 9_000_000, stallWarn: 4_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
    }

    [Fact]
    public void A_short_stall_says_nothing()
    {
        Assert.Equal(CubeDaviCurriculum.Outcome.None,
            Step(sinceAdvance: 3_999_999, stallWarn: 4_000_000).Outcome);
    }

    [Fact]
    public void Doing_nothing_carries_the_advance_counter_through_unchanged()
    {
        var d = Step(sinceAdvance: 123_456, stallWarn: 4_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.None, d.Outcome);
        Assert.Equal(123_456, d.SamplesSinceAdvance);
    }

    // ── progressive growth: a SEPARATE decision ──────────────────────────────────────────────────

    [Fact]
    public void Growth_fires_once_the_sample_threshold_is_reached()
    {
        var d = Step(width: 128, growTo: 512, growAt: 1_000_000, samples: 1_000_000);

        Assert.True(d.Grow);
        Assert.Equal(512, d.GrowWidth);
    }

    [Fact]
    public void Growth_waits_for_its_threshold()
    {
        Assert.False(Step(width: 128, growTo: 512, growAt: 2_000_000, samples: 1_999_999).Grow);
    }

    [Fact]
    public void Growth_does_not_fire_at_or_past_the_target_width()
    {
        Assert.False(Step(width: 512, growTo: 512, growAt: 0, samples: 9_000_000).Grow);
        Assert.False(Step(width: 1024, growTo: 512, growAt: 0, samples: 9_000_000).Grow);
    }

    [Fact]
    public void Growth_is_off_when_no_target_width_is_configured()
    {
        Assert.False(Step(width: 128, growTo: 0, growAt: 0, samples: 9_000_000).Grow);
    }

    [Fact]
    public void A_non_widenable_net_never_grows()
    {
        Assert.False(Step(width: null, growTo: 512, growAt: 0, samples: 9_000_000).Grow);
    }

    [Fact]
    public void Growth_is_independent_of_the_curriculum_outcome()
    {
        // THE structural rule of this function. Growth is a separate `if`, not a fourth arm of the
        // chain, so a step can advance the curriculum AND grow the net. Folding it into the chain would
        // silently defer growth by however long the curriculum keeps advancing.
        var d = Step(frontierRatio: 1.0, width: 128, growTo: 512, growAt: 1_000_000, samples: 1_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.Advance, d.Outcome);
        Assert.True(d.Grow);
    }

    [Fact]
    public void Growth_can_accompany_a_widen_too()
    {
        var d = Step(autoWiden: true, width: 128, maxWidth: 1024, widenStall: 1_000_000,
                     samples: 5_000_000, atBestLoss: 1_000_000, growTo: 512, growAt: 1_000_000);

        Assert.Equal(CubeDaviCurriculum.Outcome.Widen, d.Outcome);
        Assert.True(d.Grow);
    }
}
