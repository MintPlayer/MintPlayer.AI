// MintPlayer.AI.ReinforcementLearning.Lab — long-running training-campaign launcher.
// Each `--game` dispatches to its ITrainingCampaign on the shared CampaignRunner (PLAN M25);
// the runner owns the loop/resume/eval cadence/checkpointing, the per-game *Lab files own the flags.
//
// Usage: MintPlayer.AI.ReinforcementLearning.Lab [--game rushhour|snake|fruitcake|cube|cube-policy|cube-davi|connect4|chess|draughts]
//                                                [--hours H] [--data DIR] [--seed S] [--lr LR] [--eval-only]
//                                                [--viz [port]] ...
// Default game: rushhour (the original Kociemba-free BFS-oracle imitation campaign, PLAN M16).

// The Lab is a development tool: default to the Development environment (unless the operator set one) so the
// `--viz` live network viewer — gated to Development on purpose — works out of the box. Set
// DOTNET_ENVIRONMENT=Production to run without ever exposing the viewer socket.
if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT"))
    && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT")))
    Environment.SetEnvironmentVariable("DOTNET_ENVIRONMENT", "Development");

for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--game" && i + 1 < args.Length)
    {
        string game = args[i + 1];
        if (game.Equals("cube-davi", StringComparison.OrdinalIgnoreCase)) { CubeDaviLab.Run(args); return; }
        if (game.Equals("cube-policy", StringComparison.OrdinalIgnoreCase)) { CubePolicyLab.Run(args); return; }
        if (game.Equals("cube", StringComparison.OrdinalIgnoreCase)) { CubeLab.Run(args); return; }
        if (game.Equals("rushhour", StringComparison.OrdinalIgnoreCase)) { RushHourLab.Run(args); return; }
        if (game.Equals("snake", StringComparison.OrdinalIgnoreCase)) { SnakeLab.Run(args); return; }
        if (game.Equals("fruitcake", StringComparison.OrdinalIgnoreCase)) { FruitCakeLab.Run(args); return; }
        if (game.Equals("connect4", StringComparison.OrdinalIgnoreCase)) { Connect4Lab.Run(args); return; }
        if (game.Equals("chess", StringComparison.OrdinalIgnoreCase)) { ChessLab.Run(args); return; }
        if (game.Equals("draughts", StringComparison.OrdinalIgnoreCase)
            || game.Equals("checkers", StringComparison.OrdinalIgnoreCase)) { DraughtsLab.Run(args); return; }
    }
}

RushHourLab.Run(args);
