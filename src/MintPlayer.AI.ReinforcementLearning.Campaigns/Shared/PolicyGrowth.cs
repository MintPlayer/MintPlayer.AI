using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Progressive architecture growth on a fixed SAMPLE CADENCE for the imitation policy campaigns: every
/// <c>growEvery</c> samples the net takes one function-preserving Net2Net step up its <see cref="GrowthLadder"/>.
/// </summary>
/// <remarks>
/// <para><b>Prefer <see cref="SaturationGrowth"/> for new work.</b> A cadence grows the net because a counter
/// advanced, not because the net ran out of capacity: too early and a run pays for width it cannot yet use and
/// resets Adam on a net that was still improving; too late and it burns samples on a saturated one. Saturation
/// growth is the same Net2Net mechanism driven by a held-out quality metric instead, and it persists its rung
/// rather than recovering it. This entry point remains because the campaigns using it predate that, and because
/// switching them would change trajectories that are still being compared against their own history.</para>
///
/// <para><b>The ladder is a required argument, and that is the point.</b> It used to be the shared
/// <see cref="DqnGrowth.Stages"/> schedule for every caller — <c>[16] → … → [128,128,128]</c>, sized for small
/// DQN nets — while the games passing through here default to <c>[384,384]</c> (Rush Hour) and <c>[512,512]</c>
/// (Cube). Enabling growth therefore started those runs 24–32× narrower and finished them SMALLER than not
/// enabling it. Build the ladder with <see cref="GrowthLadder.FromTrunk"/> from the caller's own default trunk so
/// rung 0 is the shape it would otherwise have trained, and the downgrade cannot recur.</para>
/// </remarks>
public static class PolicyGrowth
{
    /// <param name="knownRung">The rung the campaign RECORDED for this net, or -1 when it has none (a sidecar
    /// written before M70, or a store with no sidecar at all). -1 falls back to matching the trunk shape, which
    /// is the pre-M70 behaviour and carries the hazard described on <see cref="CurrentRung"/>; any value ≥ 0 is
    /// trusted as-is and no shape-matching happens.</param>
    /// <returns>Null when nothing grew, otherwise the grown net, a fresh Adam over it, and the rung it is now on
    /// — which the caller must persist, or the next resume is back to guessing.</returns>
    public static (TNet Net, Adam Adam, int Rung)? Maybe<TNet>(TNet net, long samples, bool grow, int growEvery,
        float learningRate, GrowthLadder ladder, Xoshiro256StarStar rng, Action<string> log, int knownRung = -1)
        where TNet : IGrowableTrunkNet<TNet>
    {
        ArgumentNullException.ThrowIfNull(ladder);
        if (!grow) return null;

        int target = Math.Min(ladder.Top, (int)(samples / Math.Max(1, growEvery)));
        int rung = knownRung >= 0 ? knownRung : CurrentRung(ladder, net.Trunk);
        if (rung >= target) return null;

        var grown = net;
        while (rung < target)
        {
            int[] next = ladder.TrunkFor(rung + 1);
            grown = next.Length > grown.Trunk.Length ? grown.Deepen(rng) : grown.WidenTo(next, rng);
            rung++;
        }
        log($"grew policy net → [{string.Join(",", grown.Trunk)}] at {samples:N0} samples (function-preserving)");
        return (grown, new Adam(grown.Parameters(), learningRate), rung);
    }

    /// <summary>The rung whose shape the live trunk matches, or 0 when it matches none. <b>Legacy fallback
    /// only</b> — reached solely when the caller has no recorded rung.</summary>
    /// <remarks>
    /// Recovering the rung from the shape is why <see cref="SaturationGrowth"/> persists it instead: an
    /// architecture the ladder never described is indistinguishable here from rung 0, so a net built outside the
    /// ladder reads as "the bottom" and, at a high sample count, is walked all the way to the top in one call.
    /// Rooting the ladder at the caller's default trunk keeps that from biting the normal cases, because the
    /// default shape IS rung 0 — but it remains a guess, and a genuinely off-ladder net still lands here.
    /// <para><b>M70 closed this for new runs.</b> The three campaigns using it now record the rung in their
    /// progress sidecar (v3) and pass it as <c>knownRung</c>, so this method is only consulted for a store
    /// written before that field existed. It is kept, rather than deleted, precisely because those stores exist
    /// and a wrong guess is still better than refusing to resume them.</para>
    /// </remarks>
    private static int CurrentRung(GrowthLadder ladder, int[] hidden)
    {
        for (int r = ladder.Top; r >= 0; r--)
            if (ladder.TrunkFor(r).AsSpan().SequenceEqual(hidden)) return r;
        return 0;
    }
}
