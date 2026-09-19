using MintPlayer.AI.ReinforcementLearning.Campaigns;
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
        var options = Parse(new CliArgs(args), out double hours, out string dataDir, out bool evalOnly);

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddSelfPlayCampaign<Connect4State>("connect4", options),
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "connect4-selfplay.csv")));
    }

    /// <summary>The self-play options this entry point's flags resolve to (M63.5: extracted from
    /// <see cref="Run"/>, where they were only reachable by starting a real training run).</summary>
    internal static SelfPlayOptions Parse(CliArgs a, out double hours, out string dataDir, out bool evalOnly)
    {
        hours = a.Dbl("--hours", 1);
        dataDir = a.Str("--data", "data");
        evalOnly = a.Has("--eval-only");

        return new SelfPlayOptions
        {
            Seed = a.ULong("--seed", 1),
            LearningRate = a.Flt("--lr", 1e-3f),
            Hidden = a.Int("--hidden", 128),                      // one width; the net trunk is [hidden, hidden]
            Search = new Mcts.Config(Simulations: a.Int("--sims", 100)),
            GamesPerChunk = a.Int("--games", 32),
            EvalGames = a.Int("--eval-games", 20),
            OpponentRandomFrac = a.Dbl("--opponent-random", 0),    // fraction of games vs a random opponent
        };
    }
}
