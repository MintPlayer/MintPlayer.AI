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

    /// <summary>
    /// Net, Adam moments, the curriculum/progress sidecar, and the DEPLOYABLE net for one training phase.
    /// </summary>
    /// <remarks>
    /// <paramref name="Policy"/> and <paramref name="PolicyBest"/> are deliberately distinct files.
    /// <paramref name="Policy"/> is the RESUME net: always the latest weights, so a continuation picks up exactly
    /// where it stopped and stays consistent with the Adam moments saved beside it. <paramref name="PolicyBest"/>
    /// is the SHIPPABLE net: overwritten only when an eval improves on the best seen. Writing one file for both
    /// jobs would force a choice between resuming from stale weights and shipping whichever net the last eval
    /// happened to land on — and with a 64-board gate, the last eval is close to a random draw.
    /// </remarks>
    public readonly record struct NetIds(string Policy, string PolicyAdam, string State, string PolicyBest);

    /// <summary>Phase 1 = imitation from the exact oracle; phase 2 = expert iteration.</summary>
    public static NetIds ForPhase(int phase) => phase <= 1
        ? new("policy", "policy-adam", "policy-state", "policy-best")
        : new("policy-xit", "policy-xit-adam", "policy-xit-state", "policy-xit-best");
}
