using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69 (COVERAGE_90_PRD §19.5 item 5 / §19.6) — <see cref="PolicyGrowth.Maybe"/>, the sample-cadence Net2Net step
/// the imitation policy campaigns (Cube, Rush Hour, Block Dude) take. <c>CAMPAIGN_TESTABILITY_PRD</c> marks M64.5
/// ✅ and names this member, but what shipped was only the null guard and the <c>grow: false</c> early-out — the
/// growth body itself was never covered.
/// </summary>
/// <remarks>
/// <para><b>The silent failures this guards.</b></para>
/// <list type="bullet">
/// <item><b>The returned <see cref="Adam"/> must be built over the GROWN net's parameters.</b> Net2Net produces
/// new tensors, so handing back the caller's old optimizer would step parameters that are no longer in the graph:
/// every later update would land on dead memory, the live net would stop moving, and the loss would simply
/// flatten with no error at all.</item>
/// <item><b>The step must be function-preserving.</b> Logits for a fixed observation are identical across the
/// grow; anything else is a loss spike that reads as ordinary training noise.</item>
/// <item><b>The rung is RECORDED, not read back out of the trunk shape (M70).</b> Shape-matching cannot tell
/// "rung 0" from "an architecture this ladder never described", so an off-ladder net read as rung 0 at a high
/// sample count was walked to the top in ONE call — the exact shape of the bug that once made Rush Hour and Cube
/// nets <i>smaller</i> when growth was enabled (their default trunks were far above the shared schedule's top
/// rung). Campaigns now pass the rung they persisted; shape-matching survives only as the fallback for stores
/// written before the field existed, and that legacy path is pinned here too.</item>
/// </list>
/// </remarks>
public class PolicyGrowthTests
{
    private const int GrowEvery = 1_000;

    /// <summary>[16,16] → [24,24] (widen) → [24,24,24] (deepen); rung 0 is the net's own default shape.</summary>
    private static GrowthLadder Ladder() => GrowthLadder.FromTrunk([16, 16], steps: 2);

    private static CubePolicyNet Net(int[] trunk, ulong seed = 1) => new(new Xoshiro256StarStar(seed), trunk);

    private static Tensor Obs(ulong seed = 42)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[RubiksCubeEnv.ObservationSize];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        return new Tensor(data, 1, data.Length);
    }

    private static (CubePolicyNet Net, Adam Adam, int Rung)? Grow(
        CubePolicyNet net, long samples, GrowthLadder ladder, int knownRung = -1)
        => PolicyGrowth.Maybe(net, samples, grow: true, GrowEvery, 1e-3f, ladder, new Xoshiro256StarStar(7),
            _ => { }, knownRung);

    // ── the early-outs: "nothing to do" must be a null, not a needless rebuild ─────────────────────

    [Fact]
    public void Growth_that_was_not_asked_for_returns_nothing()
    {
        Assert.Null(PolicyGrowth.Maybe(Net([16, 16]), 10 * GrowEvery, grow: false, GrowEvery, 1e-3f,
            Ladder(), new Xoshiro256StarStar(7), _ => { }));
    }

    [Fact]
    public void A_net_that_has_not_reached_the_next_rung_yet_returns_nothing()
    {
        // Below one cadence's worth of samples the target rung is still 0, and returning a "grown" net here would
        // reset Adam's moments on a net that was still improving.
        Assert.Null(Grow(Net([16, 16]), GrowEvery - 1, Ladder()));
    }

    [Fact]
    public void A_net_already_at_the_top_rung_returns_nothing()
    {
        Assert.Null(Grow(Net([24, 24, 24]), 99 * GrowEvery, Ladder()));
    }

    // ── the climb ────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void At_the_sample_threshold_the_trunk_is_on_the_next_rung()
    {
        var ladder = Ladder();

        var grown = Grow(Net([16, 16]), GrowEvery, ladder);

        Assert.NotNull(grown);
        Assert.Equal(ladder.TrunkFor(1), grown!.Value.Net.Trunk);   // one rung, not the top
    }

    [Fact]
    public void The_grown_net_computes_the_same_logits_as_the_net_it_replaced()
    {
        var net = Net([16, 16]);
        var x = Obs();
        var (logits0, value0) = net.Forward(x);
        var before = (float[])logits0.Data.Clone();
        float beforeValue = value0.Data[0];

        var grown = Grow(net, GrowEvery, Ladder());

        Assert.NotNull(grown);
        var (logits1, value1) = grown!.Value.Net.Forward(x);
        for (int a = 0; a < before.Length; a++) Assert.Equal(before[a], logits1.Data[a], 3);
        Assert.Equal(beforeValue, value1.Data[0], 3);
    }

    [Fact]
    public void The_returned_optimizer_steps_the_grown_nets_parameters_not_the_old_ones()
    {
        // Behavioural check, since Adam does not expose its parameter list: ClipGradNorm reports the pre-clip
        // global norm over exactly the tensors it holds. Give the GROWN net a gradient and the old net a much
        // larger one; an optimizer built over the wrong (or a shared) set reports the wrong norm.
        var net = Net([16, 16]);
        var grown = Grow(net, GrowEvery, Ladder());
        Assert.NotNull(grown);

        foreach (var p in net.Parameters()) { p.EnsureGrad(); p.Grad![0] = 100f; }
        foreach (var p in grown!.Value.Net.Parameters()) { p.EnsureGrad(); Array.Clear(p.Grad!); }
        var first = grown.Value.Net.Parameters().First();
        first.Grad![0] = 3f;

        Assert.Equal(3f, grown.Value.Adam.ClipGradNorm(1_000f), 3);
    }

    [Fact]
    public void The_growth_log_names_the_shape_it_grew_into()
    {
        var lines = new List<string>();
        var ladder = Ladder();

        PolicyGrowth.Maybe(Net([16, 16]), GrowEvery, grow: true, GrowEvery, 1e-3f, ladder,
            new Xoshiro256StarStar(7), lines.Add);

        Assert.Single(lines);
        Assert.Contains(string.Join(",", ladder.TrunkFor(1)), lines[0]);
    }

    // ── the rung: RECORDED, not recovered from the shape (M70) ───────────────────────────────────

    [Fact]
    public void A_recorded_rung_is_trusted_even_when_the_trunk_matches_no_rung_at_all()
    {
        // THE FIX. [20,20] appears on no rung, so shape-matching cannot tell it from the bottom — that is the
        // hazard. With the rung recorded, the shape is never consulted: a net known to be on the top rung does
        // not grow, however unrecognisable its trunk is.
        var ladder = Ladder();

        Assert.Null(Grow(Net([20, 20]), 99 * GrowEvery, ladder, knownRung: ladder.Top));
    }

    [Fact]
    public void A_recorded_rung_climbs_only_the_rungs_that_remain_above_it()
    {
        // Known to be on rung 1 of a 2-rung ladder at a sample count that targets the top: exactly ONE step,
        // where reading the shape would have started at 0 and taken two.
        var ladder = Ladder();

        var grown = Grow(Net([20, 20]), 99 * GrowEvery, ladder, knownRung: ladder.Top - 1);

        Assert.NotNull(grown);
        Assert.Equal(ladder.Top, grown!.Value.Rung);

        // Note what the resulting SHAPE is: [20,20,20], not the ladder's [24,24,24]. The ladder decides the
        // KIND of step — rung 2 is deeper than rung 1, so this deepens — and Deepen keeps the net's own widths.
        // An off-ladder net therefore stays off-ladder as it grows, which is worth pinning: it means the rung
        // and the trunk can disagree permanently, and is precisely why the rung has to be recorded rather than
        // read back out of the shape.
        Assert.Equal([20, 20, 20], grown.Value.Net.Trunk);
    }

    [Fact]
    public void The_returned_rung_is_what_the_caller_must_persist()
    {
        // Growth is only half the fix: a rung that is computed and then dropped leaves the next resume guessing
        // again. The value comes back so the campaign can write it to its sidecar.
        var ladder = Ladder();

        var grown = Grow(Net(ladder.TrunkFor(0)), GrowEvery, ladder, knownRung: 0);

        Assert.NotNull(grown);
        Assert.Equal(1, grown!.Value.Rung);
        Assert.Equal(ladder.TrunkFor(1), grown.Value.Net.Trunk);
    }

    [Fact]
    public void Without_a_recorded_rung_an_off_ladder_trunk_still_reads_as_the_bottom()
    {
        // The LEGACY path, pinned deliberately. A sidecar written before M70 has no rung, so -1 falls back to
        // shape-matching and the old hazard is still reachable for exactly those stores: [20,20] is on no rung,
        // reads as the bottom, and a high sample count walks it to the top in one call.
        //
        // This is kept rather than fixed because the alternative is refusing to resume pre-M70 checkpoints. It
        // is a bounded, closing hazard: the first Checkpoint after a resume writes a real rung, so a store fixes
        // itself the moment it is used. Rooting ladders at the caller's default trunk (GrowthLadder.FromTrunk,
        // which Ladder() uses) is what keeps normal cases off this path even before that.
        var ladder = Ladder();

        var grown = Grow(Net([20, 20]), 99 * GrowEvery, ladder, knownRung: -1);

        Assert.NotNull(grown);
        Assert.Equal(ladder.TrunkFor(ladder.Top), grown!.Value.Net.Trunk);
    }

    [Fact]
    public void A_null_ladder_is_rejected_before_anything_else_is_read()
    {
        Assert.Throws<ArgumentNullException>(() =>
        {
            PolicyGrowth.Maybe(Net([16, 16]), GrowEvery, grow: true, GrowEvery, 1e-3f, null!,
                new Xoshiro256StarStar(7), _ => { });
        });
    }
}
