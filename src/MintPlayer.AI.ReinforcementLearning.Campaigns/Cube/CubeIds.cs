namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Model-store ids for the cube artifacts (shared with the web app's conventions).</summary>
public static class CubeIds
{
    public const string Environment = "cube";
    public const string Policy = "policy";
    public const string PolicyAdam = "policy-adam";

    /// <summary>The (net, Adam) id pair for a given trunk width.</summary>
    public readonly record struct NetIds(string Policy, string PolicyAdam);

    /// <summary>
    /// The shipped 512 net keeps the bare `policy` id; every other width (the M17 ladder) gets a width-tagged id
    /// so rungs never overwrite each other or the shipped net.
    /// </summary>
    public static NetIds ForWidth(int width)
        => width == 512
            ? new NetIds(Policy, PolicyAdam)
            : new NetIds($"policy-w{width}", $"policy-w{width}-adam");
}
