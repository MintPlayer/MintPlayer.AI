using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// `--game chess` entry point (PLAN M39.2): AlphaZero-style self-play on chess — the second consumer of the reusable
/// self-play stack (<see cref="Mcts"/> + <see cref="SelfPlayCampaign{TState}"/>), which is reused UNCHANGED; only the
/// perft-verified <see cref="ChessGame"/> is new. CPU-only and honestly bounded (a small MLP over a flattened board →
/// legal, steadily-improving play, not engine strength). Flags: --sims, --games, --eval-games, --hidden, plus the
/// common --hours/--data/--seed/--lr.
/// </summary>
internal static class ChessLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 1e-3f);
        int hidden = a.Int("--hidden", 256);      // the net trunk is [hidden, hidden]
        int sims = a.Int("--sims", 64);           // modest — chess movegen per node is heavy on CPU
        int gamesPerChunk = a.Int("--games", 8);
        int evalGames = a.Int("--eval-games", 10);
        double opponentRandom = a.Dbl("--opponent-random", 0); // fraction of games vs a random opponent (robustness)
        bool evalOnly = a.Has("--eval-only");

        // --demo: play one self-play game with the (trained) net and print FENs to watch — no training.
        if (a.Has("--demo")) { ChessDemo.Run(dataDir, sims, seed, a.Int("--demo-plies", 100)); return; }

        // --ladder: hands-off difficulty ladder (M40.4). Whenever the live net beats the last-promoted checkpoint by a
        // margin in a net-vs-net arena, a new tier .ckpt + updated manifest are written straight into the web app's
        // models dir — so `--game chess --ladder --hours N` grows the site's difficulty roster with no manual steps.
        LadderOptions? ladder = a.Has("--ladder")
            ? new LadderOptions(
                Dir: a.Str("--difficulty-dir", Path.Combine("src", "RLDemo.Web", "wwwroot", "models")),
                PromoteMargin: a.Dbl("--promote-margin", 0.08),  // required rise in winRate-vs-random over the champion
                ArenaMargin: a.Dbl("--arena-margin", 0.60),      // OR: head-to-head score once winRate-vs-random saturates
                ArenaGames: a.Int("--arena-games", 20),
                Sims: a.Int("--difficulty-sims", 128),
                OpeningPlies: a.Int("--opening-plies", 6))
            : null;

        // Eval/checkpoint cadence in minutes (defaults preserve CampaignOptions' 2 / 10). The ladder promotes on this
        // cadence, so a short cadence captures tiers sooner (and lets a quick run exercise the arena).
        double? firstEval = a.Has("--first-eval") ? a.Dbl("--first-eval", 2) : null;
        double? evalEvery = a.Has("--eval-every") ? a.Dbl("--eval-every", 10) : null;

        var game = new ChessGame();
        var cfg = new Mcts.Config(Simulations: sims);
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new SelfPlayCampaign<ChessState>(game, "chess", seed, learningRate, hidden, cfg, gamesPerChunk,
                tempMoves: 12, evalGames: evalGames, maxPlies: 200, opponentRandomFrac: opponentRandom, ladder: ladder),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "chess-selfplay.csv")),
            firstEvalMinutes: firstEval, evalEveryMinutes: evalEvery);
    }
}
