using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The cube imitation campaign's lifecycle — resume, the width-tagged store ids, the progress sidecar and the
/// live-telemetry seam. Of its 124 lines only 7 had any coverage, because every existing cube test goes through
/// Kociemba or a trained net rather than through the campaign object.
/// </summary>
/// <remarks>
/// These <b>never call <c>TrainChunk</c> or <c>Evaluate</c></b>. <c>TrainChunk</c> streams thousands of Kociemba
/// solves; <c>Evaluate</c> is worse than it looks — it runs 8 depths × 20 episodes of greedy rollout and, on
/// every failure, an A* budgeted at 2 000 expansions, each expansion evaluating 11 children through the net. On
/// an untrained net (which is all a fast test can have) essentially every episode pays the full budget, so one
/// <c>Evaluate</c> is millions of forward passes. Everything below is reachable without either, by reading the
/// counters back out through <see cref="INetworkTelemetrySource"/> instead.
/// <para>
/// The failures guarded here are all silent. The <b>width-tagged ids</b> (<see cref="CubeIds.ForWidth"/>) are the
/// only thing standing between a ladder rung and the shipped <c>cube.policy</c> the web app serves: a rung that
/// wrote the bare id would overwrite the shipped solver with a half-trained net and report success. The
/// <b>progress sidecar</b> carries the sample counter that <c>PolicyGrowth</c> keys its stage target off, so a
/// sidecar that is silently discarded does not just lose reporting — it restarts the growth schedule, and a run
/// restarted often enough never leaves the first trunk stage while looking perfectly healthy. Every documented
/// way of discarding one (truncated, foreign kind, wrong stream count) must fall back to zeros rather than throw
/// or, worse, misread.
/// </para>
/// <para>
/// <c>Resume</c> warms the Kociemba tables, which costs seconds on the first call in the process. They are static
/// and shared with <c>CubeApiTests</c>/<c>KociembaInternalsTests</c>, so only whichever test runs first pays it.
/// </para>
/// </remarks>
public class CubeImitationLifecycleTests
{
    // A narrow trunk keeps net construction, save and load trivial, and 64 ≠ 512 puts the run on the width ladder
    // — which is what the id tests below are about.
    private const int Width = 64;
    private static readonly CubeIds.NetIds Ids = CubeIds.ForWidth(Width);
    private const string ProgressId = "policy-w64-progress";
    private const string ProgressKind = "cube-imitation-progress";

    private static CubeImitationOptions Options() => new() { Seed = 3, Width = Width, LearningRate = 1e-3f };

    private static NetworkMetrics Telemetry(CubeImitationCampaign campaign)
        => ((INetworkTelemetrySource)campaign).Sample();

    // ── resume ───────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void An_empty_store_is_not_a_resume_but_still_leaves_a_usable_net()
    {
        var dir = Directory.CreateTempSubdirectory("m65-cube-empty");
        try
        {
            using var c = new CubeImitationCampaign(Options());
            var telemetry = (INetworkTelemetrySource)c;

            // Before Resume the net field is genuinely null; the viewer polls this seam on a background thread
            // from the moment the Lab starts, so returning null rather than dereferencing is the contract.
            Assert.Null(telemetry.SnapshotParameters());

            Assert.False(c.Resume(new FileModelStore(dir.FullName)));

            // False means "nothing was loaded", NOT "the campaign is unusable" — it is fully initialised.
            Assert.NotNull(telemetry.SnapshotParameters());
            Assert.NotEmpty(telemetry.SnapshotParameters()!);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Checkpointing_writes_the_net_the_optimizer_and_the_progress_sidecar()
    {
        var dir = Directory.CreateTempSubdirectory("m65-cube-write");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new CubeImitationCampaign(Options());

            c.Resume(store);
            c.Checkpoint(store);   // no TrainChunk: a freshly built net is still a valid checkpoint

            Assert.True(store.Exists(CubeIds.Environment, Ids.Policy));
            Assert.True(store.Exists(CubeIds.Environment, Ids.PolicyAdam));
            Assert.True(store.Exists(CubeIds.Environment, ProgressId));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_second_instance_resumes_the_net_the_first_checkpointed()
    {
        var dir = Directory.CreateTempSubdirectory("m65-cube-resume");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new CubeImitationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var second = new CubeImitationCampaign(Options());
            Assert.True(second.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the width ladder's store ids ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_width_rung_never_writes_over_the_shipped_512_net()
    {
        // CubeIds.ForWidth keeps the bare `policy` id for the shipped 512 net; every other width is tagged. A rung
        // that wrote the bare id would silently replace the solver the web app serves.
        var dir = Directory.CreateTempSubdirectory("m65-cube-ids");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new CubeImitationCampaign(Options());

            c.Resume(store);
            c.Checkpoint(store);

            Assert.False(store.Exists(CubeIds.Environment, CubeIds.Policy));
            Assert.False(store.Exists(CubeIds.Environment, CubeIds.PolicyAdam));
            Assert.False(store.Exists(CubeIds.Environment, $"{CubeIds.Policy}-progress"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Two_rungs_of_the_width_ladder_do_not_resume_each_others_checkpoints()
    {
        // The rungs have incompatible shapes, so a cross-rung load is not merely wrong bookkeeping — it is a net of
        // the wrong size. The id scheme is the only thing preventing it.
        var dir = Directory.CreateTempSubdirectory("m65-cube-rungs");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var narrow = new CubeImitationCampaign(Options()))
            {
                narrow.Resume(store);
                narrow.Checkpoint(store);
            }

            using var wider = new CubeImitationCampaign(Options() with { Width = 128 });
            Assert.False(wider.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the progress sidecar ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void The_sample_counter_round_trips_through_the_progress_sidecar()
    {
        // Written by hand rather than by a TrainChunk: the point is that the ON-DISK shape (this id, this kind,
        // these two RNG streams) is what Resume reads. A mismatch on any of the three degrades to zeros, which is
        // exactly the silent loss this test exists to catch.
        var dir = Directory.CreateTempSubdirectory("m65-cube-sidecar");
        try
        {
            var store = new FileModelStore(dir.FullName);
            CampaignProgressState.Save(store, CubeIds.Environment, ProgressId, ProgressKind,
                samples: 8_192, units: 17, lastMetric: 42,
                new Xoshiro256StarStar(101), new Xoshiro256StarStar(202));

            using var c = new CubeImitationCampaign(Options());
            c.Resume(store);

            // Step is _totalSamples — the counter PolicyGrowth keys its stage target off.
            Assert.Equal(8_192, Telemetry(c).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_truncated_sidecar_restarts_the_counters_instead_of_failing_the_run()
    {
        var dir = Directory.CreateTempSubdirectory("m65-cube-truncated");
        try
        {
            var store = new FileModelStore(dir.FullName);

            // The shape a run killed mid-write leaves behind: a valid header, then end-of-stream.
            store.Save(CubeIds.Environment, ProgressId, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                CheckpointFormat.WriteHeader(writer, ProgressKind, 1);
                writer.Write(9_999L);   // samples, and then nothing
            });

            using var c = new CubeImitationCampaign(Options());

            // Progress is an optimisation, never a correctness requirement: losing it costs replayed data, so it
            // must degrade rather than throw out of Resume and kill a multi-hour run at startup.
            Assert.False(c.Resume(store));
            Assert.Equal(0, Telemetry(c).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_sidecar_written_by_a_different_campaign_is_ignored_rather_than_misread()
    {
        var dir = Directory.CreateTempSubdirectory("m65-cube-kind");
        try
        {
            var store = new FileModelStore(dir.FullName);

            // Same file layout, different kind string — the self-play sidecar. Read as if it were this campaign's,
            // its game counter would land in the sample counter and its single RNG stream would be read as two.
            CampaignProgressState.Save(store, CubeIds.Environment, ProgressId, "selfplay-progress",
                samples: 5_000, units: 3, lastMetric: 1, new Xoshiro256StarStar(7));

            using var c = new CubeImitationCampaign(Options());
            c.Resume(store);

            Assert.Equal(0, Telemetry(c).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_sidecar_with_the_wrong_number_of_rng_streams_is_ignored()
    {
        // This campaign declares TWO owner-thread streams (data generation + growth). A one-stream sidecar — what an
        // older build, or a different campaign at the same id, wrote — must be discarded whole: reading the first
        // stream and inventing the second would leave the growth RNG silently desynchronised from the checkpoint.
        var dir = Directory.CreateTempSubdirectory("m65-cube-streams");
        try
        {
            var store = new FileModelStore(dir.FullName);
            CampaignProgressState.Save(store, CubeIds.Environment, ProgressId, ProgressKind,
                samples: 6_000, units: 9, lastMetric: 11, new Xoshiro256StarStar(13));

            using var c = new CubeImitationCampaign(Options());
            c.Resume(store);

            Assert.Equal(0, Telemetry(c).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_resumed_net_without_a_sidecar_still_resumes_and_starts_its_counters_at_zero()
    {
        // The documented pre-M58 fallback: checkpoints written before the sidecar existed must still load their net.
        var dir = Directory.CreateTempSubdirectory("m65-cube-nosidecar");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new CubeImitationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }
            Assert.True(store.Delete(CubeIds.Environment, ProgressId));

            using var second = new CubeImitationCampaign(Options());

            Assert.True(second.Resume(store));
            Assert.Equal(0, Telemetry(second).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── identity + completion ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void The_campaign_trains_under_the_shared_cube_environment()
    {
        // Every cube artifact (imitation, DAVI, EfficientCube, the web app's models dir) lives under one
        // environment folder; a campaign that named its own would write where nothing looks for it.
        using var c = new CubeImitationCampaign(Options());

        Assert.Equal(CubeIds.Environment, c.Environment);
        Assert.Equal("cube-policy", ((INetworkTelemetrySource)c).NetKind);
    }

    [Fact]
    public void The_campaign_never_reports_itself_complete()
    {
        // Imitation has no sample target: it runs to the runner's wall-clock budget, so it leaves ITrainingCampaign's
        // `IsComplete => false` default in place. Reporting completion would end every run at the first cadence check.
        var dir = Directory.CreateTempSubdirectory("m65-cube-complete");
        try
        {
            var store = new FileModelStore(dir.FullName);
            CampaignProgressState.Save(store, CubeIds.Environment, ProgressId, ProgressKind,
                samples: long.MaxValue / 2, units: 1_000, lastMetric: 0,
                new Xoshiro256StarStar(1), new Xoshiro256StarStar(2));

            using var c = new CubeImitationCampaign(Options());
            c.Resume(store);

            Assert.False(((ITrainingCampaign)c).IsComplete);
        }
        finally { dir.Delete(recursive: true); }
    }
}
