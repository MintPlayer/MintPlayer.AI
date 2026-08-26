namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M54.1 C#↔TS parity gate (docs/prd/TETRIS_PRD.md §5): a 1,000-step episode mix (uniform + 7-bag piece
/// streams, garbage every 10 and every 5, random + Dellacherie + Dellacherie-search actions), checksummed
/// over every action, its cleared lines, the full post-move row masks and the piece stream. The SAME
/// protocol runs against the generated TypeScript twin (`node tools/tetris_parity.mjs` — the browser's
/// exact code) and must print the SAME checksum. All rules arithmetic is i32 and the Dellacherie/search
/// comparisons use exactly-representable f64s, so the twins are exactly equal, not merely close.
/// </summary>
public class TetrisParityTests
{
    // Verified 2026-08-26 against the TS twin: `node tools/tetris_parity.mjs` (committed harness).
    // Pin history: M54.1 initial 472451993 (matched the TS twin on the first run).
    private const long PinnedChecksum = 472_451_993;

    [Fact]
    public void MixedEpisode_ChecksumMatchesTheTsTwin()
    {
        var board = new PgTetris();
        board.reset(1000, false, 10);
        var policy = new PgTetRng(999);

        const long P = 1_000_000_007;
        long h = 0;
        int episodes = 0;
        for (int step = 0; step < 1000; step++)
        {
            if (board.gameOver)
            {
                episodes++;
                board.reset(1000 + episodes, episodes % 2 == 1, episodes % 2 == 0 ? 10 : 5);
            }
            int action = step % 50 == 7 ? board.dellaSearchAction(8, 5)
                : step % 4 == 0 ? board.dellacherieAction()
                : board.randomAction(policy);
            int cleared = board.applyPlacement(action);
            h = (h * 31 + action) % P;
            h = (h * 31 + cleared) % P;
            for (int y = 0; y < 20; y++) h = (h * 31 + board.rows[y]) % P;
            h = (h * 31 + board.current) % P;
            h = (h * 31 + board.next) % P;
        }
        Assert.Equal(PinnedChecksum, h);
    }
}
