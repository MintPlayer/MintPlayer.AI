using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69/B3 — the <see cref="INetworkTelemetrySource"/> surface of the four supervised campaigns, sampled after a
/// bare <c>Resume</c> (no training: the net exists as soon as the campaign has resumed, which is all the M36
/// viewer needs).
/// <para>
/// This seam fails <b>silently by construction</b>, which is why it is worth pinning. <c>SampleIo</c> and
/// <c>SampleActivations</c> both end in <c>catch { return null; }</c> — deliberately, because a viewer must
/// never take a multi-hour training run down. The cost of that choice is that a probe which has drifted out of
/// step with the net (an observation encoding changed, the probe board was not updated) does not fail: the
/// viewer just shows "no sample" forever while training looks perfectly healthy. The label lists are worse
/// still — a drifted <c>OutputLabels</c> throws nothing at all and simply mislabels every axis of the graph,
/// so the picture stays plausible and is wrong.
/// </para>
/// <para>
/// So these tests assert AGREEMENT, not "it did not throw": the probe input width must equal the net's actual
/// input width, the label count must equal the net's actual action-head width, and the activation list must
/// have one entry per recovered layer. They also pin the honest-degradation rule at the other end — BEFORE
/// <c>Resume</c> there is no net, and every optional member must return <c>null</c> rather than dereference it.
/// </para>
/// </summary>
public class CampaignTelemetryTests
{
    /// <summary>
    /// The whole optional surface, checked against the environment's own widths.
    /// </summary>
    /// <param name="observationWidth">The environment's observation width — what the probe must produce.</param>
    /// <param name="actionCount">The environment's action count — what the policy head and labels must match.</param>
    private static void AssertTelemetryAgreesWithTheNet(INetworkTelemetrySource source, int observationWidth, int actionCount)
    {
        Assert.False(string.IsNullOrWhiteSpace(source.NetKind));

        var parameters = source.SnapshotParameters();
        Assert.NotNull(parameters);

        // Layers() pairs each weight matrix with the bias that follows it: trunk layers, then the policy head,
        // then the scalar value/distance head. That ordering is what lets the checks below name the heads.
        var layers = NetworkInspector.Layers(parameters!);
        Assert.True(layers.Count >= 3, $"expected a trunk plus two heads, recovered {layers.Count} layers");
        Assert.Equal(observationWidth, layers[0].Weight.Rows);
        Assert.Equal(actionCount, layers[^2].Weight.Cols);   // policy head
        Assert.Equal(1, layers[^1].Weight.Cols);             // scalar value / distance head

        // Metrics are read once per frame and must be readable before any training has happened; NaN is the
        // documented "not measured yet" here, so only the step counter is pinned.
        var metrics = source.Sample();
        Assert.True(metrics.Step >= 0);

        // None of these four supplies input labels today. The check is written as a conditional on purpose: the
        // moment one does, a list that does not match the observation width mislabels every input neuron.
        if (source.InputLabels is { } inputLabels)
            Assert.Equal(observationWidth, inputLabels.Count);

        var outputLabels = source.OutputLabels;
        Assert.NotNull(outputLabels);
        Assert.Equal(actionCount, outputLabels!.Count);

        var io = source.SampleIo();
        Assert.NotNull(io);
        Assert.Equal(observationWidth, io!.Value.Input.Length);
        Assert.Equal(actionCount, io.Value.Output.Length);
        Assert.All(io.Value.Output, v => Assert.False(float.IsNaN(v)));

        var activations = source.SampleActivations();
        Assert.NotNull(activations);
        Assert.Equal(layers.Count, activations!.Length);
        Assert.Equal(actionCount, activations[^2].Length);
        Assert.Equal(1, activations[^1].Length);
    }

    /// <summary>Before <c>Resume</c> there is no net: the viewer must be told "nothing yet", not crashed.</summary>
    private static void AssertDegradesToNoSampleBeforeResume(INetworkTelemetrySource source)
    {
        Assert.Null(source.SnapshotParameters());
        Assert.Null(source.SampleIo());
        Assert.Null(source.SampleActivations());
        source.Sample();                       // must be readable with no net behind it
        Assert.False(string.IsNullOrWhiteSpace(source.NetKind));
    }

    private static void Sweep(ITrainingCampaign campaign, int observationWidth, int actionCount)
    {
        var dir = Directory.CreateTempSubdirectory("m69-telemetry");
        try
        {
            var source = Assert.IsAssignableFrom<INetworkTelemetrySource>(campaign);
            AssertDegradesToNoSampleBeforeResume(source);

            Assert.False(campaign.Resume(new FileModelStore(dir.FullName)));   // fresh store: a fresh net

            AssertTelemetryAgreesWithTheNet(source, observationWidth, actionCount);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void BlockDude_expert_iteration_telemetry_matches_its_net()
    {
        using var c = new BlockDudeExpertIterationCampaign(new BlockDudeExpertIterationOptions
        {
            Seed = 1,
            WarmStart = false,   // nothing to warm-start from here, and skipping it keeps the resume a pure init
        });

        Sweep(c, BlockDudeBoard.ObservationSize, BlockDudeBoard.ActionCount);
    }

    [Fact]
    public void BlockDude_imitation_telemetry_matches_its_net()
    {
        using var c = new BlockDudeImitationCampaign(new BlockDudeImitationOptions
        {
            Seed = 1,
            PinStage = 0,
            GateEverySamples = long.MaxValue,   // never gate: a gate would solve boards, which is the slow part
            BoardsPerRound = 1,
            SamplesPerBoard = 1,
        });

        Sweep(c, BlockDudeBoard.ObservationSize, BlockDudeBoard.ActionCount);
    }

    [Fact]
    public void RushHour_imitation_telemetry_matches_its_net()
    {
        using var c = new RushHourImitationCampaign(new RushHourImitationOptions { Seed = 1 });

        Sweep(c, RushHourBoard.ObservationSize, RushHourBoard.ActionCount);
    }

    [Fact]
    public void Cube_imitation_telemetry_matches_its_net()
    {
        // A narrow trunk: this test is about widths agreeing, and the shipped 512 would allocate a net two
        // orders of magnitude larger for no extra assertion. Resume never touches the Kociemba tables — only
        // TrainChunk warms them — so nothing here pays that multi-second cost.
        using var c = new CubeImitationCampaign(new CubeImitationOptions { Seed = 1, Width = 32 });

        Sweep(c, RubiksCubeEnv.ObservationSize, RubiksCubeEnv.ActionCount);
    }
}
