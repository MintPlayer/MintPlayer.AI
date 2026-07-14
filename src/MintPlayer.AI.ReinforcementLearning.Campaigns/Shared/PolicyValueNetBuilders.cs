using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// How a <see cref="SelfPlayCampaign{TState}"/> constructs and reloads its two-headed net, so the campaign itself
/// stays architecture-agnostic (it only ever holds an <see cref="IPolicyValueNet"/>). Each builder owns the
/// checkpoint <see cref="CheckpointKind"/> tag its nets are written with, so a given tier reloads through the same
/// builder that produced it.
/// </summary>
public interface IPolicyValueNetBuilder
{
    string CheckpointKind { get; }
    IPolicyValueNet CreateFresh(int obsSize, int actions, Xoshiro256StarStar rng);
    IPolicyValueNet Load(Stream source, int obsSize, int actions);
}

/// <summary>The flat variable-depth MLP (<see cref="PolicyValueNet"/>) — the default/back-compat architecture and the
/// net connect-4 uses. Its kind tag ("selfplay-pv") is unchanged, so previously-shipped checkpoints still load.</summary>
public sealed class MlpNetBuilder(int[] hidden) : IPolicyValueNetBuilder
{
    public string CheckpointKind => "selfplay-pv";
    public IPolicyValueNet CreateFresh(int obsSize, int actions, Xoshiro256StarStar rng)
        => new PolicyValueNet(obsSize, hidden, actions, rng);
    public IPolicyValueNet Load(Stream source, int obsSize, int actions)
        => PolicyValueNet.Load(source, CheckpointKind, obsSize, actions);
}

/// <summary>The AlphaZero-style convolutional residual tower (<see cref="ConvResidualPolicyValueNet"/>) over a
/// <c>planes×boardH×boardW</c> board (M42). Its kind tag is distinct so a conv checkpoint can't be mistaken for an
/// MLP one. <paramref name="obsSize"/> passed to build/load must equal <c>planes·boardH·boardW</c>.</summary>
public sealed class ConvNetBuilder(int planes, int boardH, int boardW, int filters, int blocks) : IPolicyValueNetBuilder
{
    public string CheckpointKind => "selfplay-pv-conv";
    public IPolicyValueNet CreateFresh(int obsSize, int actions, Xoshiro256StarStar rng)
        => new ConvResidualPolicyValueNet(planes, boardH, boardW, actions, filters, blocks, rng);
    public IPolicyValueNet Load(Stream source, int obsSize, int actions)
        => ConvResidualPolicyValueNet.Load(source, CheckpointKind, actions);
}
