using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The phase-1 gate limb (M69/B4) — <c>MaybeGateAndAdvance</c> → <c>GateSolveRate</c> → best-net capture →
/// <c>MaybeGrow</c> → <c>BlockDudeCurriculum.Advance</c>.
/// </summary>
/// <remarks>
/// Every other phase-1 test sets <c>GateEverySamples = long.MaxValue</c> with the comment "never gate: gating
/// would solve boards", because the hold-out was a hard-wired 64 and the gate solves every board greedily. That
/// is what kept this whole limb untested. <see cref="BlockDudeImitationOptions.GateBoards"/> makes the hold-out
/// a knob, exactly as M64.7 did for <c>BatchSize</c>, so the limb is now reachable in milliseconds.
/// <para>
/// The failures guarded here are silent ones. The gate is what promotes a curriculum rung and what selects the
/// <b>deployable</b> net: a gate that never records, or a best-net capture that never fires, leaves a run
/// training forever on rung 0 and shipping whatever the last checkpoint happened to be, with nothing in the
/// logs looking wrong.
/// </para>
/// </remarks>
public class BlockDudeImitationGateTests
{
    /// <summary>Gate on every chunk, over a hold-out small enough to solve instantly.</summary>
    private static BlockDudeImitationOptions Gating(int gateBoards = 2) => new()
    {
        Seed = 1,
        PinStage = null,           // the limb returns immediately when a stage is pinned
        MaxStage = 0,              // stay on rung 0: promotion is asserted by BlockDudeCurriculumTests
        GateEverySamples = 1,      // gate on the first chunk rather than after 100k samples
        GateBoards = gateBoards,   // the M69/B4 seam
        BoardsPerRound = 1,
        SamplesPerBoard = 8,
        BatchSize = 8,
    };

    [Fact]
    public void The_gate_runs_on_its_sample_cadence_and_records_a_rate_for_the_current_stage()
    {
        var dir = Directory.CreateTempSubdirectory("m69-gate-runs");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeImitationCampaign(Gating());
            c.Resume(store);

            c.TrainChunk();

            // stage_samples is reset on promotion and MaxStage pins us to rung 0, so the gate having run is read
            // off the deployable net instead: BestStage starts at -1, so the FIRST gate always beats it and
            // fires SaveBestNet regardless of how badly an untrained net scores.
            Assert.True(
                store.Exists(BlockDudeIds.Environment, BlockDudeIds.ForPhase(1).PolicyBest),
                "the first gate must capture a deployable net: BestStage starts at -1, so stage 0 beats it");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_pinned_stage_skips_the_gate_entirely()
    {
        var dir = Directory.CreateTempSubdirectory("m69-gate-pinned");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeImitationCampaign(Gating() with { PinStage = 0 });
            c.Resume(store);

            c.TrainChunk();

            // Same cadence, same hold-out — the ONLY difference is the pin. No gate means no best net, which is
            // the observable half of "diagnostic only": a pinned run is not allowed to select a shippable net.
            Assert.False(
                store.Exists(BlockDudeIds.Environment, BlockDudeIds.ForPhase(1).PolicyBest),
                "a pinned stage must not gate, and therefore must not capture a deployable net");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_cadence_holds_the_gate_off_until_the_sample_budget_is_spent()
    {
        var dir = Directory.CreateTempSubdirectory("m69-gate-cadence");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeImitationCampaign(Gating() with { GateEverySamples = long.MaxValue });
            c.Resume(store);

            c.TrainChunk();

            // The early return at the top of MaybeGateAndAdvance. This is the arm every other phase-1 test sits
            // on permanently, so it is worth one explicit assertion rather than being assumed.
            Assert.False(
                store.Exists(BlockDudeIds.Environment, BlockDudeIds.ForPhase(1).PolicyBest),
                "no gate may run before GateEverySamples has elapsed");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_shrunk_hold_out_is_a_PREFIX_of_the_larger_one_not_a_different_draw()
    {
        // This is the claim that makes GateBoards safe to expose at all. GateBoardsFor reseeds per stage and
        // stops once it has `count` accepted boards, so asking for fewer walks the SAME sequence and stops
        // earlier. If it ever reseeded off `count`, a shrunk hold-out would silently measure a different
        // distribution and every gate number taken in a test would be incomparable with a real run's.
        var two = BlockDudeCurriculum.GateBoardsFor(0, count: 2);
        var three = BlockDudeCurriculum.GateBoardsFor(0, count: 3);

        Assert.Equal(2, two.Count);
        Assert.Equal(3, three.Count);
        for (int i = 0; i < two.Count; i++)
            Assert.Equal(two[i].StateHash, three[i].StateHash);
    }
}
