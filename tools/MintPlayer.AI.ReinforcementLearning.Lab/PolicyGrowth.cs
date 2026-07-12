using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

/// <summary>
/// Progressive architecture growth for the imitation policy campaigns (a two-headed <see cref="IGrowableTrunkNet{T}"/>
/// trained with a plain <see cref="Adam"/>): on a sample cadence it grows the net toward larger stages, each a single
/// function-preserving Net2WiderNet / Net2DeeperNet step. Shares the <see cref="DqnGrowth.Stages"/> schedule with the
/// DQN games. Returns the grown net + a fresh optimizer (Adam moments are keyed to the parameter set) when it grows,
/// else null — the caller reassigns its net/optimizer.
/// </summary>
internal static class PolicyGrowth
{
    public static (TNet Net, Adam Adam)? Maybe<TNet>(TNet net, long samples, bool grow, int growEvery,
        float learningRate, Xoshiro256StarStar rng, Action<string> log)
        where TNet : IGrowableTrunkNet<TNet>
    {
        if (!grow) return null;
        int target = Math.Min(DqnGrowth.Stages.Length - 1, (int)(samples / Math.Max(1, growEvery)));
        int stage = CurrentStage(net.Trunk);
        if (stage >= target) return null;

        var grown = net;
        while (stage < target)
        {
            var next = DqnGrowth.Stages[stage + 1];
            grown = next.Length > grown.Trunk.Length ? grown.Deepen(rng) : grown.WidenTo(next, rng);
            stage++;
        }
        log($"grew policy net → [{string.Join(",", grown.Trunk)}] at {samples:N0} samples (function-preserving)");
        return (grown, new Adam(grown.Parameters(), learningRate));
    }

    private static int CurrentStage(int[] hidden)
    {
        for (int s = DqnGrowth.Stages.Length - 1; s >= 0; s--)
            if (DqnGrowth.Stages[s].AsSpan().SequenceEqual(hidden)) return s;
        return 0; // an architecture not on the schedule (e.g. a non-grow resume) — treat as the start
    }
}
