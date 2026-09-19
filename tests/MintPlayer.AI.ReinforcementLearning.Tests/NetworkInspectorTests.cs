using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// <see cref="NetworkInspector"/> is what turns a net's raw parameter tensors into something the live viewer can
/// draw, and it had <b>zero</b> coverage — every one of its 72 lines ran only when a human happened to open
/// <c>--viz</c> on a training run.
/// </summary>
/// <remarks>
/// That matters more than the line count suggests. The inspector is deliberately architecture-agnostic: it
/// recovers layers by <i>pairing a rank-2 weight with the rank-1 bias that follows it</i>, which is an assumption
/// about every net in this library rather than a property any one of them declares. If a net ever emits its
/// parameters in another order the viewer silently draws the wrong graph — no exception, no failing gate, just a
/// picture that lies. These tests pin the pairing rule, the topology arithmetic and the heatmap's block-mean so
/// that breakage is loud.
/// <para>
/// Everything here is pure array work: no net is constructed, no training runs, nothing touches the filesystem.
/// </para>
/// </remarks>
public class NetworkInspectorTests
{
    /// <summary>A weight matrix of the given shape, filled with <paramref name="fill"/> per element index.</summary>
    private static Tensor W(int rows, int cols, Func<int, float> fill)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = fill(i);
        return new Tensor(data, rows, cols);
    }

    private static Tensor B(params float[] values) => new(values, values.Length);

    // ── Layers: the weight/bias pairing rule ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Layers_PairsEachRank2WeightWithTheRank1BiasThatFollowsIt()
    {
        var parameters = new[] { W(3, 4, _ => 1f), B(1, 2, 3, 4), W(4, 2, _ => 1f), B(5, 6) };

        var layers = NetworkInspector.Layers(parameters);

        Assert.Equal(2, layers.Count);
        Assert.Equal(3, layers[0].Weight.Rows);
        Assert.Equal(4, layers[0].Weight.Cols);
        Assert.Equal(4, layers[0].Bias.Length);
        Assert.Equal(4, layers[1].Weight.Rows);
        Assert.Equal(2, layers[1].Weight.Cols);
        Assert.Equal(2, layers[1].Bias.Length);
    }

    [Fact]
    public void Layers_IgnoresATrailingWeightThatHasNoBias()
    {
        // A net mid-construction, or one whose final head is bias-free. The inspector must drop the dangling
        // weight rather than pair it with whatever comes next on the following call.
        var parameters = new[] { W(2, 2, _ => 1f), B(1, 2), W(2, 3, _ => 1f) };

        var layers = NetworkInspector.Layers(parameters);

        Assert.Single(layers);
        Assert.Equal(2, layers[0].Weight.Cols);
    }

    [Fact]
    public void Layers_SkipsALeadingBiasThatHasNoWeight()
    {
        // The rank-1-before-any-rank-2 case: there is no pending weight, so the bias is discarded rather than
        // silently becoming layer 0's bias.
        var parameters = new[] { B(9, 9), W(2, 2, _ => 1f), B(1, 2) };

        var layers = NetworkInspector.Layers(parameters);

        Assert.Single(layers);
        Assert.Equal(1f, layers[0].Bias.Data[0]);
    }

    [Fact]
    public void Layers_KeepsOnlyTheMostRecentWeightWhenTwoWeightsAppearBackToBack()
    {
        // Two rank-2 tensors in a row means the first has no bias of its own. The pairing is last-wins, which is
        // what the "[weight, bias] in forward order" layout implies; pinning it stops a refactor from quietly
        // making it first-wins and mislabelling every layer index downstream.
        var parameters = new[] { W(2, 5, _ => 1f), W(2, 3, _ => 1f), B(1, 2, 3) };

        var layers = NetworkInspector.Layers(parameters);

        Assert.Single(layers);
        Assert.Equal(3, layers[0].Weight.Cols);
    }

    [Fact]
    public void Layers_IsEmptyForNoParameters()
    {
        Assert.Empty(NetworkInspector.Layers([]));
    }

    // ── Describe: the fixed topology ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Describe_TakesInputWidthFromTheFirstLayerAndOutputWidthFromTheLast()
    {
        var parameters = new[] { W(6, 4, _ => 1f), B(1, 2, 3, 4), W(4, 2, _ => 1f), B(5, 6) };

        var topology = NetworkInspector.Describe(parameters, "dueling-q");

        Assert.Equal("dueling-q", topology.NetKind);
        Assert.Equal(6, topology.InputSize);
        Assert.Equal(2, topology.OutputSize);
        Assert.Equal(2, topology.Layers.Count);
    }

    [Fact]
    public void Describe_MarksOnlyTheFinalLayerAsTheOutputRole()
    {
        var parameters = new[]
        {
            W(4, 8, _ => 1f), B(new float[8]),
            W(8, 8, _ => 1f), B(new float[8]),
            W(8, 3, _ => 1f), B(new float[3]),
        };

        var topology = NetworkInspector.Describe(parameters, "mlp");

        Assert.Equal(["hidden", "hidden", "output"], topology.Layers.Select(l => l.Role));
        Assert.Equal([0, 1, 2], topology.Layers.Select(l => l.Index));
    }

    [Fact]
    public void Describe_ReportsZeroWidthsForAParameterlessNet()
    {
        // Before a net exists the campaign may still be asked to describe itself. Indexing layers[0] here would
        // throw and take the viewer's whole connection down.
        var topology = NetworkInspector.Describe([], "not-yet");

        Assert.Equal(0, topology.InputSize);
        Assert.Equal(0, topology.OutputSize);
        Assert.Empty(topology.Layers);
    }

    [Fact]
    public void Describe_PassesNeuronLabelsThroughWithoutForcingThemToMatchTheFinalLayer()
    {
        // Deliberate, and commented as such in the production file: a multi-head net's action head is NOT the last
        // column (a scalar value head follows it), so output labels shorter than OutputSize must survive intact.
        string[] inputs = ["x", "v"];
        string[] outputs = ["left", "right"];

        var topology = NetworkInspector.Describe(
            [W(2, 3, _ => 1f), B(0, 0, 0)], "policy-value", inputs, outputs);

        Assert.Equal(inputs, topology.InputLabels);
        Assert.Equal(outputs, topology.OutputLabels);
        Assert.Equal(3, topology.OutputSize);
    }

    [Fact]
    public void Describe_LeavesLabelsNullWhenTheEnvironmentSuppliesNone()
    {
        var topology = NetworkInspector.Describe([W(2, 2, _ => 1f), B(0, 0)], "anon");

        Assert.Null(topology.InputLabels);
        Assert.Null(topology.OutputLabels);
    }

    // ── CaptureFrame: the per-layer statistics ───────────────────────────────────────────────────────────────

    [Fact]
    public void CaptureFrame_ComputesMinMaxMeanAbsAndL2OverTheWholeWeightMatrix()
    {
        // Weights -2, -1, 0, 1 → min -2, max 1, mean|w| = 4/4 = 1, L2 = sqrt(4+1+0+1) = sqrt 6.
        var weight = new Tensor([-2f, -1f, 0f, 1f], 2, 2);
        var frame = NetworkInspector.CaptureFrame([weight, B(0.5f, -1.5f)], default);

        var layer = Assert.Single(frame.Layers);
        Assert.Equal(-2f, layer.WMin);
        Assert.Equal(1f, layer.WMax);
        Assert.Equal(1f, layer.WMeanAbs, 6);
        Assert.Equal(MathF.Sqrt(6f), layer.WL2, 5);
        Assert.Equal(1f, layer.BiasMeanAbs, 6); // (0.5 + 1.5) / 2 — the mean of the ABSOLUTE bias values
        Assert.Equal(2, layer.Rows);
        Assert.Equal(2, layer.Cols);
    }

    [Fact]
    public void CaptureFrame_CarriesTheSuppliedMetricsOntoTheFrame()
    {
        var metrics = new NetworkMetrics(Step: 1234, MaxSteps: 5000, Loss: 0.25, Eval: -99.5, Epsilon: 0.1);

        var frame = NetworkInspector.CaptureFrame([W(2, 2, _ => 1f), B(0, 0)], metrics);

        Assert.Equal(1234, frame.Step);
        Assert.Equal(5000, frame.MaxSteps);
        Assert.Equal(0.25, frame.Loss);
        Assert.Equal(-99.5, frame.Eval);
        Assert.Equal(0.1, frame.Epsilon);
    }

    [Fact]
    public void CaptureFrame_PreservesNaNMetrics()
    {
        // The viewer renders NaN as "—" for a metric that doesn't apply to this algorithm (ε for a supervised
        // campaign). If anything here coerced NaN to 0 the viewer would confidently display a real-looking zero.
        var metrics = new NetworkMetrics(0, 0, double.NaN, double.NaN, double.NaN);

        var frame = NetworkInspector.CaptureFrame([W(2, 2, _ => 1f), B(0, 0)], metrics);

        Assert.True(double.IsNaN(frame.Loss));
        Assert.True(double.IsNaN(frame.Eval));
        Assert.True(double.IsNaN(frame.Epsilon));
    }

    [Fact]
    public void CaptureFrame_PassesInputOutputAndActivationVectorsThrough()
    {
        float[] input = [1f, 2f];
        float[] output = [3f];
        float[][] activations = [[0.5f, 0.5f], [3f]];

        var frame = NetworkInspector.CaptureFrame(
            [W(2, 1, _ => 1f), B(0)], default, input, output, activations);

        Assert.Equal(input, frame.InputValues);
        Assert.Equal(output, frame.OutputValues);
        Assert.Equal(activations, frame.Activations);
    }

    [Fact]
    public void CaptureFrame_LeavesTheOptionalVectorsNullWhenTheSourceOmitsThem()
    {
        var frame = NetworkInspector.CaptureFrame([W(2, 2, _ => 1f), B(0, 0)], default);

        Assert.Null(frame.InputValues);
        Assert.Null(frame.OutputValues);
        Assert.Null(frame.Activations);
    }

    [Fact]
    public void CaptureFrame_ProducesOneLayerFrameInForwardOrder()
    {
        var parameters = new[]
        {
            W(4, 3, _ => 1f), B(0, 0, 0),
            W(3, 2, _ => 2f), B(0, 0),
        };

        var frame = NetworkInspector.CaptureFrame(parameters, default);

        Assert.Equal(2, frame.Layers.Count);
        Assert.Equal([0, 1], frame.Layers.Select(l => l.Index));
        Assert.Equal(1f, frame.Layers[0].WMeanAbs, 6);
        Assert.Equal(2f, frame.Layers[1].WMeanAbs, 6);
    }

    // ── Downsample (through CaptureFrame): the bounded heatmap ───────────────────────────────────────────────

    [Fact]
    public void CaptureFrame_KeepsTheHeatmapAtFullResolutionWhenTheLayerFitsUnderTheCap()
    {
        // 2×2 under a cap of 24 → no downsampling, and each cell is |weight| exactly.
        var weight = new Tensor([-1f, 2f, -3f, 4f], 2, 2);

        var layer = Assert.Single(NetworkInspector.CaptureFrame([weight, B(0, 0)], default).Layers);

        Assert.Equal(2, layer.HRows);
        Assert.Equal(2, layer.HCols);
        Assert.Equal([1f, 2f, 3f, 4f], layer.Heat);
    }

    [Fact]
    public void CaptureFrame_BlockAveragesAbsoluteWeightsWhenTheLayerExceedsTheCap()
    {
        // 4×4 → 2×2 with maxHeat 2. Each output cell is the mean of one 2×2 block of |weight|.
        // Row-major values 1..16 with alternating signs; block (0,0) covers 1,2,5,6 → mean 3.5.
        var data = new float[16];
        for (int i = 0; i < 16; i++) data[i] = (i % 2 == 0 ? 1 : -1) * (i + 1);
        var weight = new Tensor(data, 4, 4);

        var layer = Assert.Single(
            NetworkInspector.CaptureFrame([weight, B(0, 0, 0, 0)], default, maxHeat: 2).Layers);

        Assert.Equal(2, layer.HRows);
        Assert.Equal(2, layer.HCols);
        Assert.Equal(4, layer.Heat.Length);
        Assert.Equal(3.5f, layer.Heat[0], 5);   // 1, 2, 5, 6
        Assert.Equal(5.5f, layer.Heat[1], 5);   // 3, 4, 7, 8
        Assert.Equal(11.5f, layer.Heat[2], 5);  // 9, 10, 13, 14
        Assert.Equal(13.5f, layer.Heat[3], 5);  // 11, 12, 15, 16
        // Every value is a magnitude: the heatmap must never carry a sign through.
        Assert.All(layer.Heat, v => Assert.True(v >= 0f));
    }

    [Fact]
    public void CaptureFrame_BoundsTheFrameSizeRegardlessOfHowWideTheLayerIs()
    {
        // The whole point of the downsample: a 200×200 layer must stream the same handful of cells as a 24×24.
        var wide = NetworkInspector.CaptureFrame([W(200, 200, _ => 0.5f), B(new float[200])], default);
        var small = NetworkInspector.CaptureFrame([W(24, 24, _ => 0.5f), B(new float[24])], default);

        Assert.Equal(24 * 24, wide.Layers[0].Heat.Length);
        Assert.Equal(small.Layers[0].Heat.Length, wide.Layers[0].Heat.Length);
        Assert.Equal(200, wide.Layers[0].Rows); // the TRUE shape is still reported
        Assert.Equal(200, wide.Layers[0].Cols);
    }

    [Fact]
    public void CaptureFrame_PreservesAspectWithinTheCapForANonSquareLayer()
    {
        var layer = Assert.Single(
            NetworkInspector.CaptureFrame([W(100, 3, _ => 1f), B(0, 0, 0)], default, maxHeat: 8).Layers);

        Assert.Equal(8, layer.HRows);
        Assert.Equal(3, layer.HCols); // capped at the true width, not padded up to 8
        Assert.Equal(24, layer.Heat.Length);
    }

    [Fact]
    public void CaptureFrame_YieldsAnEmptyHeatmapWhenTheCapIsZero()
    {
        var layer = Assert.Single(
            NetworkInspector.CaptureFrame([W(4, 4, _ => 1f), B(0, 0, 0, 0)], default, maxHeat: 0).Layers);

        Assert.Equal(0, layer.HRows);
        Assert.Equal(0, layer.HCols);
        Assert.Empty(layer.Heat);
    }

    [Fact]
    public void CaptureFrame_ReturnsNoLayersForAParameterlessNet()
    {
        var frame = NetworkInspector.CaptureFrame([], new NetworkMetrics(7, 8, 0, 0, 0));

        Assert.Empty(frame.Layers);
        Assert.Equal(7, frame.Step); // the scalars still stream, so the viewer shows progress before the net exists
    }
}
