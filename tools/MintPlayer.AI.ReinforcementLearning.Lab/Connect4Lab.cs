using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

/// <summary>
/// `--game connect4` entry point (PLAN M39.1): AlphaZero-style self-play on Connect-4 — the cheap first consumer of
/// the reusable self-play stack (<see cref="Mcts"/> + <see cref="SelfPlayCampaign{TState}"/>). Runs on the shared
/// <see cref="CampaignRunner"/>; the net bootstraps from random init and its win-rate vs a random-legal opponent
/// climbs. CPU-only (the net is tiny). Flags: --sims, --games, --eval-games, plus the common --hours/--data/--seed/--lr.
/// </summary>
internal static class Connect4Lab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 1);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 1e-3f);
        int hidden = a.Int("--hidden", 128);      // one width; the net trunk is [hidden, hidden]
        int sims = a.Int("--sims", 100);          // MCTS simulations per move
        int gamesPerChunk = a.Int("--games", 32);
        int evalGames = a.Int("--eval-games", 20);
        double opponentRandom = a.Dbl("--opponent-random", 0); // fraction of games vs a random opponent (robustness)
        bool evalOnly = a.Has("--eval-only");

        var game = new Connect4Game();
        var cfg = new Mcts.Config(Simulations: sims);
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            _ => new SelfPlayCampaign<Connect4State>(game, "connect4", new SelfPlayOptions
            {
                Seed = seed, LearningRate = learningRate, Hidden = hidden, Search = cfg,
                GamesPerChunk = gamesPerChunk, EvalGames = evalGames, OpponentRandomFrac = opponentRandom,
            }),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "connect4-selfplay.csv")));
    }
}
