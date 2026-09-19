using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M64.6 — the BlockDude imitation campaign's checkpoint lifecycle, on 526 lines that had no
/// lifecycle coverage at all.
/// <para>
/// These deliberately never call <c>TrainChunk</c>. Doing so would generate curriculum boards and run
/// the exact oracle, whose state count explodes per added block (M61) — that is what makes the
/// existing BlockDude tests multi-minute. Resume, Checkpoint and the id scheme are reachable without
/// any of it, and they are where the silent failures live: a fingerprint that fails to refuse a
/// foreign run, or a <c>--fresh</c> that leaves a stale shippable net behind.
/// </para>
/// </summary>
public class BlockDudeCampaignLifecycleTests
{
    private static BlockDudeImitationOptions Options(ulong seed = 1, int phase = 1, bool fresh = false) => new()
    {
        Seed = seed,
        Phase = phase,
        Fresh = fresh,
        PinStage = 0,
        GateEverySamples = long.MaxValue,   // never gate: gating would solve boards
        BoardsPerRound = 1,
        SamplesPerBoard = 1,
    };

    // ── the id scheme: two phases must not share a file ───────────────────────────────────────

    [Fact]
    public void The_two_phases_use_four_fully_disjoint_ids()
    {
        // Phase 2 (expert iteration) writing over phase 1's shippable net would destroy the result of
        // a multi-hour imitation run with no error anywhere.
        var one = BlockDudeIds.ForPhase(1);
        var two = BlockDudeIds.ForPhase(2);

        string[] a = [one.Policy, one.PolicyAdam, one.State, one.PolicyBest];
        string[] b = [two.Policy, two.PolicyAdam, two.State, two.PolicyBest];

        Assert.Equal(4, a.Distinct().Count());
        Assert.Equal(4, b.Distinct().Count());
        Assert.Empty(a.Intersect(b));
    }

    [Fact]
    public void Phase_zero_and_one_are_the_same_line()
    {
        // ForPhase is `phase <= 1`, so 0 and 1 are deliberately the same imitation line rather than
        // two separate ones.
        Assert.Equal(BlockDudeIds.ForPhase(1), BlockDudeIds.ForPhase(0));
    }

    // ── resume ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void An_empty_store_is_not_a_resume()
    {
        var dir = Directory.CreateTempSubdirectory("m64-bd-fresh");
        try
        {
            using var c = new BlockDudeImitationCampaign(Options());

            Assert.False(c.Resume(new FileModelStore(dir.FullName)));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Checkpointing_writes_the_net_the_optimizer_and_the_state()
    {
        var dir = Directory.CreateTempSubdirectory("m64-bd-write");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var ids = BlockDudeIds.ForPhase(1);
            using var c = new BlockDudeImitationCampaign(Options());

            c.Resume(store);
            c.Checkpoint(store);   // no TrainChunk: the freshly built net is still a valid checkpoint

            Assert.True(store.Exists(BlockDudeIds.Environment, ids.Policy));
            Assert.True(store.Exists(BlockDudeIds.Environment, ids.PolicyAdam));
            Assert.True(store.Exists(BlockDudeIds.Environment, ids.State));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_second_instance_resumes_what_the_first_wrote()
    {
        var dir = Directory.CreateTempSubdirectory("m64-bd-resume");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeImitationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var second = new BlockDudeImitationCampaign(Options());

            Assert.True(second.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_different_seed_still_loads_the_net_but_refuses_the_foreign_sidecar()
    {
        // The fingerprint guard. The net is architecture-compatible so it loads, but the curriculum
        // state belongs to a different run and adopting it would resume someone else's stage cursor.
        var dir = Directory.CreateTempSubdirectory("m64-bd-fingerprint");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new BlockDudeImitationCampaign(Options(seed: 1)))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var other = new BlockDudeImitationCampaign(Options(seed: 999));

            // Resumes (the net is there) and does not throw -- the sidecar is refused internally and
            // the curriculum restarts, which is the documented degrade-rather-than-fail behaviour.
            Assert.True(other.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── --fresh ───────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Fresh_deletes_every_id_including_the_shippable_best()
    {
        // A blank-slate run that left policy-best behind would ship a net it never trained -- the web
        // app serves that file.
        var dir = Directory.CreateTempSubdirectory("m64-bd-freshwipe");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var ids = BlockDudeIds.ForPhase(1);

            using (var first = new BlockDudeImitationCampaign(Options()))
            {
                first.Resume(store);
                first.Checkpoint(store);
            }
            // Stand in for a best-net from an earlier run.
            store.Save(BlockDudeIds.Environment, ids.PolicyBest, s => s.WriteByte(1));
            Assert.True(store.Exists(BlockDudeIds.Environment, ids.PolicyBest));

            using var wiped = new BlockDudeImitationCampaign(Options(fresh: true));
            wiped.Resume(store);

            Assert.False(store.Exists(BlockDudeIds.Environment, ids.PolicyBest));
            Assert.False(store.Exists(BlockDudeIds.Environment, ids.State));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── telemetry ─────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void The_campaign_reports_its_environment_id()
    {
        using var c = new BlockDudeImitationCampaign(Options());

        Assert.Equal(BlockDudeIds.Environment, c.Environment);
    }
}
