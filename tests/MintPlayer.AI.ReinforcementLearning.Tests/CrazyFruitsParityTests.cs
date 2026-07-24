namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M49.1 C#↔TS parity gate (docs/prd/CRAZY_FRUITS_PRD.md §5): a seeded 1,000-move random-policy episode,
/// checksummed over every action, its points, and the full post-move grid. The SAME protocol runs against
/// the generated TypeScript twin (node, type stripping — the browser's exact code) and must print the SAME
/// checksum; the verified value is pinned here. All arithmetic in the engine is i32 (Schrage minstd, no
/// division shortcuts), so the twins are exactly equal, not merely close.
/// This talks to the generated internal core directly (InternalsVisibleTo) because the node harness does too.
/// </summary>
public class CrazyFruitsParityTests
{
    // Verified 2026-07-24 against the TS twin: `node tools/cf_parity.mjs` (committed harness).
    // M49 (no specials) pinned 78377593/score 70990; M50 specials re-pin: checksum=533753109, score=85650,
    // reshuffles=1 — random play gains ~21% from auto-firing specials, and the episode exercises creation,
    // chains, combos AND a mid-episode reshuffle byte-identically on both sides.
    private const long PinnedChecksum = 533_753_109;

    [Fact]
    public void RandomEpisode_ChecksumMatchesTheTsTwin()
    {
        var board = new PgCrazyFruits();
        board.reset(12345);
        var policy = new PgCfRng(999);

        const long P = 1_000_000_007;
        long h = 0;
        for (int move = 0; move < 1000; move++)
        {
            int action = board.randomAction(policy);
            int points = board.applySwap(action);
            h = (h * 31 + action) % P;
            h = (h * 31 + points) % P;
            for (int i = 0; i < 64; i++) h = (h * 31 + board.grid[i]) % P;
        }
        Assert.Equal(PinnedChecksum, h);
    }
}
