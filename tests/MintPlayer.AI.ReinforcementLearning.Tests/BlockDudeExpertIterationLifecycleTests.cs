using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The phase-2 expert-iteration campaign's lifecycle — 521 lines of which only the static
/// <c>NextFrontier</c> had any coverage.
/// </summary>
/// <remarks>
/// Like the phase-1 tests these <b>never call <c>TrainChunk</c></b>: it runs the A*/beam oracle over shipped
/// levels, which is what makes the existing BlockDude gate tests multi-minute. Everything asserted here —
/// resume, warm start, the frontier sidecar, <c>Evaluate</c>, <c>Checkpoint</c>, <c>--fresh</c> — is reachable
/// without a single search.
/// <para>
/// The failure modes these guard are all silent. The <b>frontier sidecar is this run's actual progress</b>: an
/// unreadable or mis-versioned sidecar falls back to the opening depth, which throws away hours of curriculum
/// advance while the run carries on looking healthy. Warm start reaching for the wrong phase's net, or
/// <c>--fresh</c> failing to clear the shippable best, are both invisible until a gate report is read days
/// later.
/// </para>
/// </remarks>
public class BlockDudeExpertIterationLifecycleTests
{
    private static BlockDudeExpertIterationOptions Options(
        ulong seed = 1, bool fresh = false, bool warmStart = true, int initialFrontier = 20) => new()
    {
        Seed = seed,
        Fresh = fresh,
        WarmStart = warmStart,
        InitialFrontier = initialFrontier,
    };

    private static readonly BlockDudeIds.NetIds Phase2 = BlockDudeIds.ForPhase(2);

    // ── resume ───────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void An_empty_store_is_not_a_resume()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-empty");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options());

            // False means "no phase-2 net was loaded". The campaign is still fully initialised afterwards.
            Assert.False(c.Resume(new FileModelStore(dir.FullName)));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_second_instance_resumes_the_net_the_first_checkpointed()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-resume");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeExpertIterationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var second = new BlockDudeExpertIterationCampaign(Options());
            Assert.True(second.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Checkpointing_writes_the_net_the_optimizer_and_the_frontier_sidecar()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-write");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeExpertIterationCampaign(Options());

            c.Resume(store);
            c.Checkpoint(store);   // no TrainChunk: a freshly built net is still a valid checkpoint

            Assert.True(store.Exists(BlockDudeIds.Environment, Phase2.Policy));
            Assert.True(store.Exists(BlockDudeIds.Environment, Phase2.PolicyAdam));
            Assert.True(store.Exists(BlockDudeIds.Environment, Phase2.State));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Phase_two_never_writes_over_the_phase_one_net()
    {
        // The whole point of the distinct `policy-xit*` ids: phase 1's net stays available as a fallback tier,
        // and a phase-2 run that clobbered it would destroy a multi-hour imitation result with no error.
        var dir = Directory.CreateTempSubdirectory("m65-xit-disjoint");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var phase1 = BlockDudeIds.ForPhase(1);
            using var c = new BlockDudeExpertIterationCampaign(Options());

            c.Resume(store);
            c.Checkpoint(store);

            Assert.False(store.Exists(BlockDudeIds.Environment, phase1.Policy));
            Assert.False(store.Exists(BlockDudeIds.Environment, phase1.PolicyAdam));
            Assert.False(store.Exists(BlockDudeIds.Environment, phase1.State));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Fresh_deletes_the_phase_two_checkpoints_and_starts_over()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-fresh");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeExpertIterationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }
            Assert.True(store.Exists(BlockDudeIds.Environment, Phase2.Policy));

            using var fresh = new BlockDudeExpertIterationCampaign(Options(fresh: true, warmStart: false));

            // --fresh must delete before it reads, so this is NOT a resume even though a net was on disk.
            Assert.False(fresh.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Fresh_clears_the_frontier_so_the_curriculum_restarts_at_the_opening_depth()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-fresh-frontier");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 40)))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var fresh = new BlockDudeExpertIterationCampaign(
                Options(fresh: true, warmStart: false, initialFrontier: 7));
            fresh.Resume(store);

            // The sidecar written at 40 must be gone, so every level starts at the NEW opening depth.
            var eval = fresh.Evaluate();
            Assert.Equal(7.0, Metric(eval, "frontier_min"));
            Assert.Equal(7.0, Metric(eval, "frontier_max"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Resuming_without_a_warm_start_source_still_produces_a_usable_campaign()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-nowarm");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options(warmStart: true));

            // WarmStart is on but there is no phase-1 best net to start from — the documented
            // "starting phase 2 from random weights" path, which must not throw.
            Assert.False(c.Resume(new FileModelStore(dir.FullName)));
            Assert.NotNull(c.Evaluate());
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the frontier sidecar ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Every_shipped_level_starts_on_the_frontier_at_the_configured_opening_depth()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-initial");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 13));
            c.Resume(new FileModelStore(dir.FullName));

            var eval = c.Evaluate();
            Assert.Equal(13.0, Metric(eval, "frontier_min"));
            Assert.Equal(13.0, Metric(eval, "frontier_max"));
            // A level pack that failed to load would leave an EMPTY frontier, which reports a perfectly
            // innocent-looking "0-0 | 0/0 levels whole" rather than erroring. Assert the denominator is the
            // real shipped count.
            Assert.Contains($"/{BlockDudeSolutions.All.Length} levels whole", eval.Summary);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_frontier_sidecar_round_trips_across_instances()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-sidecar");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 31)))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            // The second instance is configured with a DIFFERENT opening depth, so if it reports 31 the value
            // can only have come from the sidecar rather than from its own options.
            using var second = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 5));
            second.Resume(store);

            Assert.Equal(31.0, Metric(second.Evaluate(), "frontier_min"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void An_unreadable_sidecar_falls_back_to_the_opening_depth_instead_of_throwing()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-corrupt");
        try
        {
            var store = new FileModelStore(dir.FullName);

            // A truncated sidecar — the shape a run killed mid-write leaves behind.
            store.Save(BlockDudeIds.Environment, Phase2.State, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                writer.Write(1);          // correct format version
                writer.Write(9999L);      // total samples
                // and then nothing: the reader hits end-of-stream reading the round count
            });

            using var c = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 17));
            c.Resume(store);

            Assert.Equal(17.0, Metric(c.Evaluate(), "frontier_min"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_sidecar_from_a_future_format_version_is_ignored_rather_than_misread()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-version");
        try
        {
            var store = new FileModelStore(dir.FullName);
            store.Save(BlockDudeIds.Environment, Phase2.State, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                writer.Write(2);          // a version this build does not know
                writer.Write(123L);
                writer.Write(4);
                writer.Write(0);
            });

            using var c = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 11));
            c.Resume(store);

            // Reading a v2 sidecar as if it were v1 would silently corrupt the curriculum; the version check
            // must discard it instead.
            Assert.Equal(11.0, Metric(c.Evaluate(), "frontier_min"));
            Assert.Equal(0.0, Metric(c.Evaluate(), "samples"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_level_missing_from_an_older_sidecar_is_added_at_the_opening_depth()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-added");
        try
        {
            var store = new FileModelStore(dir.FullName);
            string known = BlockDudeSolutions.All[0].Name;

            // A sidecar naming only ONE level — exactly what an older run wrote before the pack grew.
            store.Save(BlockDudeIds.Environment, Phase2.State, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                writer.Write(1);
                writer.Write(500L);
                writer.Write(3);
                writer.Write(1);
                writer.Write(known);
                writer.Write(64);
            });

            using var c = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 9));
            c.Resume(store);
            var eval = c.Evaluate();

            // The known level keeps its advanced depth; the newly-added ones enter at the opening depth, rather
            // than the whole frontier being thrown away.
            Assert.Equal(9.0, Metric(eval, "frontier_min"));
            Assert.Equal(64.0, Metric(eval, "frontier_max"));
            Assert.Equal(500.0, Metric(eval, "samples"));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── evaluate ─────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Evaluate_reports_the_documented_metric_set()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-metrics");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options());
            c.Resume(new FileModelStore(dir.FullName));

            string[] expected =
            [
                "samples", "loss", "acc", "search_rate", "beam_hits",
                "landmark_hits", "frontier_min", "frontier_max", "levels_whole",
            ];

            Assert.Equal(expected, c.Evaluate().Metrics.Select(m => m.Name));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Evaluate_reports_an_undefined_search_rate_before_any_attempt()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-nan");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options());
            c.Resume(new FileModelStore(dir.FullName));

            // NaN rather than 0: a zero success rate and "no attempts yet" are different facts, and reporting
            // the second as the first would read as a catastrophically failing run on the very first report.
            Assert.True(double.IsNaN(Metric(c.Evaluate(), "search_rate")));
            Assert.Contains("search -", c.Evaluate().Summary);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Evaluate_resets_the_windowed_counters_so_each_report_covers_one_window()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-window");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options());
            c.Resume(new FileModelStore(dir.FullName));

            c.Evaluate();
            var second = c.Evaluate();

            // beam and landmark hits are per-window tallies, not running totals.
            Assert.Equal(0.0, Metric(second, "beam_hits"));
            Assert.Equal(0.0, Metric(second, "landmark_hits"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_report_line_names_the_round_the_samples_and_the_frontier_span()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-report");
        try
        {
            using var c = new BlockDudeExpertIterationCampaign(Options(initialFrontier: 20));
            c.Resume(new FileModelStore(dir.FullName));

            string report = c.Evaluate().Summary;

            Assert.Contains("round 0", report);
            Assert.Contains("samples", report);
            Assert.Contains("frontier 20", report);
            Assert.Contains("levels whole", report);
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── completion ───────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_campaign_with_no_target_never_reports_itself_complete()
    {
        // TargetSamples defaults to 0, which means "run until stopped". Treating 0 as an already-met target
        // would make every untargeted run exit immediately.
        using var c = new BlockDudeExpertIterationCampaign(Options());

        Assert.False(c.IsComplete);
    }

    [Fact]
    public void A_resumed_sample_count_past_the_target_reports_complete()
    {
        var dir = Directory.CreateTempSubdirectory("m65-xit-complete");
        try
        {
            var store = new FileModelStore(dir.FullName);
            store.Save(BlockDudeIds.Environment, Phase2.State, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                writer.Write(1);
                writer.Write(10_000L);   // samples already collected
                writer.Write(12);
                writer.Write(0);
            });

            using var c = new BlockDudeExpertIterationCampaign(Options() with { TargetSamples = 5_000 });
            c.Resume(store);

            Assert.True(c.IsComplete);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_environment_name_is_the_shared_block_dude_one()
    {
        // Both phases must write under the same environment folder, or --fresh and the web tier look in the
        // wrong place.
        using var c = new BlockDudeExpertIterationCampaign(Options());

        Assert.Equal(BlockDudeIds.Environment, c.Environment);
    }

    private static double Metric(CampaignEval eval, string name)
        => eval.Metrics.Single(m => m.Name == name).Value;
}
