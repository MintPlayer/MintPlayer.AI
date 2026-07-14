using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Progressive architecture growth for any DQN campaign (a <see cref="DuelingQNet"/> in a
/// <see cref="DqnTrainingState"/>): on a step cadence it grows the net toward larger stages, each a single
/// function-preserving Net2WiderNet (wider) or Net2DeeperNet (one more layer) step — so capacity is added mid-run
/// with no loss spike. Shared by every DQN game (Snake, FruitCake). The schedule starts tiny and alternates
/// wider → deeper so a viewer literally watches the graph grow both ways.
/// </summary>
public static class DqnGrowth
{
    /// <summary>Each stage is exactly one widen (same depth, larger widths) or deepen (one extra layer) from the
    /// previous. <see cref="Start"/> is the architecture a growing run should be constructed with.</summary>
    public static readonly int[][] Stages = [[16], [32], [32, 32], [64, 64], [64, 64, 64], [128, 128, 128]];
    public static int[] Start => Stages[0];

    /// <summary>Grows <paramref name="state"/>'s net toward the stage its step count has reached (a no-op unless
    /// <paramref name="grow"/> and the net is a non-noisy <see cref="DuelingQNet"/>). Returns the (possibly new)
    /// state; the caller reassigns its <c>_state</c>.</summary>
    public static DqnTrainingState Maybe(DqnTrainingState state, bool grow, int growEvery, float learningRate,
        Xoshiro256StarStar rng, Action<string> log)
    {
        if (!grow || state.Online is not DuelingQNet online || online.Noisy) return state;
        int target = Math.Min(Stages.Length - 1, state.StepsCompleted / Math.Max(1, growEvery));
        int stage = CurrentStage(online.Trunk);
        while (stage < target) { state = GrowTo(state, Stages[stage + 1], learningRate, rng, log); stage++; }
        return state;
    }

    private static int CurrentStage(int[] hidden)
    {
        for (int s = Stages.Length - 1; s >= 0; s--)
            if (Stages[s].AsSpan().SequenceEqual(hidden)) return s;
        return 0; // an architecture not on the schedule (e.g. a non-grow resume) — treat as the start
    }

    private static DqnTrainingState GrowTo(DqnTrainingState state, int[] next, float learningRate,
        Xoshiro256StarStar rng, Action<string> log)
    {
        var online = (DuelingQNet)state.Online;
        bool deeper = next.Length > online.Trunk.Length;
        var grownOnline = deeper ? online.Deepen(rng) : online.WidenTo(next, rng);
        var grownTarget = (DuelingQNet)grownOnline.CloneStructure();
        grownTarget.CopyFrom(grownOnline); // re-sync the target net on an architecture change
        var grown = state.WithNetwork(grownOnline, grownTarget, new Adam(grownOnline.Parameters(), learningRate));
        log($"{(deeper ? "deepened" : "widened")} net → [{string.Join(",", grownOnline.HiddenSizes)}] at {grown.StepsCompleted:N0} steps (function-preserving)");
        return grown;
    }
}
