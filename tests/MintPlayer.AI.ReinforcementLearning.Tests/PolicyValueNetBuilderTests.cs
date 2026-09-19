using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69 (COVERAGE_90_PRD §19.5 item 11) — the <see cref="IPolicyValueNetBuilder"/> factory bodies. The kind TAGS
/// were already asserted; what was not covered is that a tier actually reloads through the builder that wrote it.
/// </summary>
/// <remarks>
/// A <see cref="SelfPlayCampaign{TState}"/> only ever holds an <see cref="Core.Nn.IPolicyValueNet"/>, so the
/// builder is the single place that knows which architecture a stored checkpoint is. The silent failure is a
/// cross-tier load: an MLP builder handed conv bytes (or the reverse) must be refused by the kind tag at the
/// header, because a net that loaded with the wrong shape but no error would serve garbage moves from a
/// checkpoint that looks fine on disk.
/// </remarks>
public class PolicyValueNetBuilderTests
{
    private const int Planes = 2, BoardH = 3, BoardW = 3, Actions = 9;
    private const int ObsSize = Planes * BoardH * BoardW;

    private static ConvNetBuilder Builder()
        => new(planes: Planes, boardH: BoardH, boardW: BoardW, filters: 4, blocks: 1);

    private static Tensor Obs(ulong seed = 5)
    {
        var rng = new Xoshiro256StarStar(seed);
        var data = new float[ObsSize];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        return new Tensor(data, 1, ObsSize);
    }

    [Fact]
    public void A_conv_tier_reloads_through_the_builder_that_wrote_it()
    {
        var builder = Builder();
        var net = builder.CreateFresh(ObsSize, Actions, new Xoshiro256StarStar(1));
        var x = Obs();
        var (logits0, value0) = net.Forward(x);
        var expectedLogits = (float[])logits0.Data.Clone();
        float expectedValue = value0.Data[0];

        using var ms = new MemoryStream();
        net.Save(ms, builder.CheckpointKind);
        ms.Position = 0;
        var reloaded = builder.Load(ms, ObsSize, Actions);

        Assert.Equal(net.Describe(), reloaded.Describe());
        var (logits1, value1) = reloaded.Forward(x);
        Assert.Equal(expectedLogits, logits1.Data);
        Assert.Equal(expectedValue, value1.Data[0]);
    }

    [Fact]
    public void A_conv_checkpoint_cannot_be_loaded_as_an_mlp_one()
    {
        var builder = Builder();
        using var ms = new MemoryStream();
        builder.CreateFresh(ObsSize, Actions, new Xoshiro256StarStar(2)).Save(ms, builder.CheckpointKind);
        ms.Position = 0;

        // Distinct kind tags are the whole guard: the header mismatch throws instead of misreading the payload.
        Assert.Throws<InvalidDataException>(() =>
        {
            new MlpNetBuilder([8, 8]).Load(ms, ObsSize, Actions);
        });
    }

    [Fact]
    public void An_mlp_tier_reloads_through_its_own_builder()
    {
        var builder = new MlpNetBuilder([8, 8]);
        var net = builder.CreateFresh(ObsSize, Actions, new Xoshiro256StarStar(3));
        var x = Obs();
        var expected = (float[])net.Forward(x).Logits.Data.Clone();

        using var ms = new MemoryStream();
        net.Save(ms, builder.CheckpointKind);
        ms.Position = 0;

        Assert.Equal(expected, builder.Load(ms, ObsSize, Actions).Forward(x).Logits.Data);
    }
}
