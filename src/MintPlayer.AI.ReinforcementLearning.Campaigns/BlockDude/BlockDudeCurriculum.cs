using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>One rung of the Block Dude size curriculum: what to generate, and what it takes to move on.</summary>
/// <param name="Spec">Board shapes and puzzle quality for this rung.</param>
/// <param name="MinStageSamples">Samples that must be trained at this rung before promotion is even considered.</param>
/// <param name="MaxStageSamples">Samples after which the rung is left regardless of the gate — logged as a FORCED
/// advance, so a stage that cannot be passed stalls the run visibly instead of silently.</param>
/// <param name="PromoteSolveRate">Greedy solve rate on the rung's fixed hold-out that earns promotion.</param>
public sealed record BlockDudeStage(
    BlockDudeStageSpec Spec,
    long MinStageSamples,
    long MaxStageSamples,
    double PromoteSolveRate);

/// <summary>
/// The size curriculum: start on boards the exact oracle can label quickly, and grow them as the net improves.
/// </summary>
/// <remarks>
/// <para><b>Bounded by measurement, not taste.</b> Exact labelling dies at roughly 6–7 blocks over ~150 free
/// cells, and cost is memory-bound at 1.3–3 KB per state (PRD §4.5.1). The top rungs sit against that wall and
/// will reject most candidates — which is why the accept rate is a logged metric.</para>
///
/// <para><b>Advance is a pure function of persisted state</b> — see <see cref="Advance"/>. Nothing here reads a
/// clock, a machine property, or ambient state, so a blank-slate re-run at the same seed replays the same
/// stages in the same order. This is the whole reason the rule lives in its own static class: it is trivially
/// testable, and it cannot accidentally acquire a dependency on the campaign's mutable fields.</para>
/// </remarks>
public static class BlockDudeCurriculum
{
    /// <summary>Bumped whenever the stage table or the advance rule changes. Part of the run fingerprint, so a
    /// checkpoint from an older curriculum is refused rather than silently resumed into a different schedule.</summary>
    public const int Version = 1;

    /// <summary>Fixed hold-out size per rung — big enough for a stable rate, small enough to evaluate often.</summary>
    public const int GateBoards = 64;

    public static readonly BlockDudeStage[] Stages =
    [
        // width      height     blocks    optimal      free  oracle cap
        new(new( 9, 11,  7,  8, 1, 2,  4,  40,  55,  60_000), 150_000, 1_200_000, 0.80),
        new(new(11, 13,  7,  9, 1, 3,  6,  60,  70,  60_000), 250_000, 2_000_000, 0.75),
        new(new(13, 15,  7,  9, 2, 3,  8,  80,  80, 100_000), 400_000, 3_000_000, 0.70),
        new(new(15, 17,  8, 10, 2, 4, 10, 100, 100, 150_000), 600_000, 4_000_000, 0.65),
        new(new(17, 19,  9, 11, 3, 5, 12, 120, 140, 150_000), 800_000, 5_000_000, 0.60),
        new(new(19, 22, 10, 13, 3, 5, 15, 150, 170, 150_000), 1_000_000, 6_000_000, 0.55),
        new(new(22, 25, 12, 14, 4, 6, 18, 200, 200, 150_000), 1_200_000, 8_000_000, 0.50),
    ];

    public static int LastStage => Stages.Length - 1;

    /// <summary>
    /// The stage to train at next, given only persisted state. A pure function: same inputs, same answer,
    /// forever.
    /// </summary>
    /// <param name="stage">Current rung.</param>
    /// <param name="stageSamples">Samples trained at this rung so far.</param>
    /// <param name="lastGateRate">Most recent greedy solve rate on this rung's hold-out.</param>
    /// <param name="forced">Set when the rung was left on the sample ceiling rather than by passing its gate —
    /// the campaign logs that, because it means the net never reached the bar.</param>
    /// <remarks>
    /// Deliberately NOT a method on the campaign, and deliberately never called from <c>Evaluate()</c>:
    /// <c>CampaignRunner</c> fires evaluation on the wall clock, so a gate evaluated there would make the
    /// training trajectory depend on how fast the machine happens to be.
    /// </remarks>
    public static int Advance(int stage, long stageSamples, double lastGateRate, out bool forced)
    {
        forced = false;
        if (stage < 0) return 0;
        if (stage >= LastStage) return LastStage;

        var rung = Stages[stage];
        if (stageSamples < rung.MinStageSamples) return stage;

        if (lastGateRate >= rung.PromoteSolveRate) return stage + 1;

        if (stageSamples >= rung.MaxStageSamples)
        {
            forced = true;
            return stage + 1;
        }
        return stage;
    }

    /// <summary>
    /// RNG for a rung's fixed hold-out set.
    /// </summary>
    /// <remarks>
    /// Seeded from the stage alone, deliberately independent of <c>--seed</c> — the same trick
    /// <c>RushHourImitationCampaign</c> uses for its random hold-out. Two consequences that both matter: runs at
    /// different seeds are directly comparable, and the hold-out can never drift with training history.
    /// </remarks>
    public static Xoshiro256StarStar GateRng(int stage) => new(4242UL + (ulong)stage);

    /// <summary>Generates a rung's fixed hold-out boards. Pure in the stage index.</summary>
    public static List<BlockDudeBoard> GateBoardsFor(int stage, int count = GateBoards)
    {
        var rng = GateRng(stage);
        var spec = Stages[Math.Clamp(stage, 0, LastStage)].Spec;
        var boards = new List<BlockDudeBoard>(count);
        for (int attempt = 0; attempt < count * 40 && boards.Count < count; attempt++)
        {
            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            if (outcome == BlockDudeGenerationOutcome.Accepted) boards.Add(board!);
        }
        return boards;
    }
}
