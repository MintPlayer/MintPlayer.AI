namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Checkpoint ids for Block Dude training.</summary>
/// <remarks>
/// Phase 2 (expert iteration on boards the oracle cannot label) writes under DISTINCT ids, exactly as
/// <c>CubeIds.ForWidth</c> does, so it can never overwrite the phase-1 net — which stays the shippable fallback
/// if phase 2 fails to beat it (PRD §8.1b).
/// </remarks>
public static class BlockDudeIds
{
    public const string Environment = "blockdude";

    /// <summary>Net, Adam moments, and the curriculum/progress sidecar for one training phase.</summary>
    public readonly record struct NetIds(string Policy, string PolicyAdam, string State);

    /// <summary>Phase 1 = imitation from the exact oracle; phase 2 = expert iteration.</summary>
    public static NetIds ForPhase(int phase) => phase <= 1
        ? new("policy", "policy-adam", "policy-state")
        : new("policy-xit", "policy-xit-adam", "policy-xit-state");
}
