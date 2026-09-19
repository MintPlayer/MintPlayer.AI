using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69 (COVERAGE_90_PRD §19.5 item 2) — <see cref="CubeViz"/>, which was 0/27 covered. Despite living next to the
/// live viewer it is <b>not</b> a server: it is four pure static samplers over one memoized probe board, so it is
/// testable with no socket, no thread and no running campaign.
/// </summary>
/// <remarks>
/// <para><b>The silent failure this guards.</b> The cube campaigns train on shuffled scramble batches, so there is
/// no "current observation" to show. The whole contract of the file is that <c>Probe</c> memoizes ONE fixed
/// depth-8 board into the caller's cache and every frame forwards that same board — that is what makes the
/// viewer's move-preference bars comparable from one step to the next. If the memoization were dropped the viewer
/// would show a different board every frame, every cross-step comparison would silently become meaningless, and
/// nothing anywhere would error.</para>
/// <para>The second rule is degradation: a net whose width does not match the cube observation must yield
/// <c>null</c> ("no sample") rather than throw, because these run on the telemetry path beside a live training
/// loop. Note that the two POLICY-side <c>catch</c> arms are unreachable by design — a
/// <see cref="CubePolicyNet"/> fixes its own input width at <see cref="RubiksCubeEnv.ObservationSize"/>, so no
/// legal argument can make its forward throw. Only the <see cref="IValueNet"/> side accepts an arbitrary width,
/// so only that arm is asserted here.</para>
/// </remarks>
public class CubeVizTests
{
    private static ResidualMlp CubeValueNet(int inputSize = RubiksCubeEnv.ObservationSize)
        => new ResidualMlp(inputSize, 8, 1, new Xoshiro256StarStar(4));

    // ── the null guards: a viewer attached before a net exists must degrade, not crash ────────────

    [Fact]
    public void All_four_samplers_return_no_sample_when_there_is_no_net_yet()
    {
        float[]? cache = null;

        Assert.Null(CubeViz.SampleIo(null, ref cache));
        Assert.Null(CubeViz.SampleActivations(null, ref cache));
        Assert.Null(CubeViz.SampleValueIo(null, ref cache));
        Assert.Null(CubeViz.SampleValueActivations(null, ref cache));
        Assert.Null(cache);   // and nothing was probed on the way
    }

    // ── the memoized probe board: the file's whole reason to exist ────────────────────────────────

    [Fact]
    public void The_probe_board_is_computed_once_and_reused_for_every_later_frame()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(1), [16, 16]);
        float[]? cache = null;

        var first = CubeViz.SampleIo(net, ref cache);
        Assert.NotNull(first);
        Assert.NotNull(cache);                        // the probe was memoized into the caller's cache
        Assert.Equal(cache, first!.Value.Input);      // and that IS the board the frame reported

        var second = CubeViz.SampleIo(net, ref cache);
        Assert.NotNull(second);
        Assert.Equal(first!.Value.Input, second!.Value.Input);
    }

    [Fact]
    public void The_probe_board_is_a_scramble_not_the_solved_cube()
    {
        // The viewer would be useless on a solved board (the net's job there is trivial), and a probe that
        // silently degenerated to `new FaceletCube()` would still pass every shape assertion.
        var solved = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(new FaceletCube(), solved);

        float[]? cache = null;
        var io = CubeViz.SampleIo(new CubePolicyNet(new Xoshiro256StarStar(2), [16, 16]), ref cache);

        Assert.NotNull(io);
        Assert.NotEqual(solved, io!.Value.Input);
    }

    [Fact]
    public void The_probe_board_is_the_same_board_for_the_policy_and_the_value_viewer()
    {
        // Both cube campaign families share one probe so their viewers are comparable; separate seeds would be
        // an invisible divergence.
        float[]? policyCache = null;
        float[]? valueCache = null;

        var policy = CubeViz.SampleIo(new CubePolicyNet(new Xoshiro256StarStar(3), [16, 16]), ref policyCache);
        var value = CubeViz.SampleValueIo(CubeValueNet(), ref valueCache);

        Assert.NotNull(policy);
        Assert.NotNull(value);
        Assert.Equal(policy!.Value.Input, value!.Value.Input);
    }

    // ── widths: what the viewer actually draws ────────────────────────────────────────────────────

    [Fact]
    public void The_policy_sample_is_one_cube_observation_in_and_twelve_move_logits_out()
    {
        float[]? cache = null;
        var io = CubeViz.SampleIo(new CubePolicyNet(new Xoshiro256StarStar(5), [16, 16]), ref cache);

        Assert.NotNull(io);
        Assert.Equal(RubiksCubeEnv.ObservationSize, io!.Value.Input.Length);
        Assert.Equal(RubiksCubeEnv.ActionCount, io.Value.Output.Length);
    }

    [Fact]
    public void The_policy_activations_are_one_row_per_layer()
    {
        float[]? cache = null;
        var acts = CubeViz.SampleActivations(new CubePolicyNet(new Xoshiro256StarStar(6), [16, 16]), ref cache);

        Assert.NotNull(acts);
        Assert.NotEmpty(acts!);
        Assert.All(acts!, layer => Assert.NotEmpty(layer));
    }

    [Fact]
    public void The_value_sample_is_one_cube_observation_in_and_a_single_cost_to_go_out()
    {
        float[]? cache = null;
        var io = CubeViz.SampleValueIo(CubeValueNet(), ref cache);

        Assert.NotNull(io);
        Assert.Equal(RubiksCubeEnv.ObservationSize, io!.Value.Input.Length);
        Assert.Single(io.Value.Output);   // DAVI's head is one scalar, not a per-action row
    }

    [Fact]
    public void The_value_activations_are_one_row_per_layer()
    {
        float[]? cache = null;
        var acts = CubeViz.SampleValueActivations(CubeValueNet(), ref cache);

        Assert.NotNull(acts);
        Assert.NotEmpty(acts!);
    }

    // ── degradation: "no sample" beats taking training down ───────────────────────────────────────

    [Fact]
    public void A_value_net_of_the_wrong_width_yields_no_sample_instead_of_throwing()
    {
        // The forward would throw an ArgumentException on the shape mismatch. These samplers run on the telemetry
        // path beside a live training loop, so the rule is that the VIEWER goes blank, not that the run dies.
        float[]? cache = null;

        Assert.Null(CubeViz.SampleValueIo(CubeValueNet(inputSize: 16), ref cache));
        Assert.Null(CubeViz.SampleValueActivations(CubeValueNet(inputSize: 16), ref cache));
    }

    [Fact]
    public void A_value_net_that_is_not_a_residual_tower_has_no_activations_to_show()
    {
        // LayerActivations is not on IValueNet, so the DAVI activation view is ResidualMlp-only; every other
        // implementation must read as "nothing to draw" rather than being cast blindly.
        var mlp = new Mlp([RubiksCubeEnv.ObservationSize, 8, 1], new Xoshiro256StarStar(7));
        float[]? cache = null;

        Assert.Null(CubeViz.SampleValueActivations(mlp, ref cache));
        Assert.NotNull(CubeViz.SampleValueIo(mlp, ref cache));   // but its IO sample still works
    }
}
