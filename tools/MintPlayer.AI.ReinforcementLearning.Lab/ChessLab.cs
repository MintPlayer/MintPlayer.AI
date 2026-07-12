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

        var game = new ChessGame();
        var cfg = new Mcts.Config(Simulations: sims);
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new SelfPlayCampaign<ChessState>(game, "chess", seed, learningRate, hidden, cfg, gamesPerChunk,
                tempMoves: 12, evalGames: evalGames, maxPlies: 200, opponentRandomFrac: opponentRandom),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "chess-selfplay.csv")));
    }
}
