using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// The `--vs-minimax` console tail shared by the board-game labs: load the trained checkpoint, run the library's
/// <see cref="StrengthEval"/> (net + MCTS vs a fixed material alpha-beta — the non-saturating strength yardstick),
/// and print the verdict. Loading + printing live here; the match itself is library code.
/// </summary>
internal static class StrengthCli
{
    public static void Run<TState>(IZeroSumGame<TState> game, IMaterialScore<TState> material,
        IPolicyValueNetBuilder builder, string arch, string ckptPath, int sims, int depth,
        int games, int maxPlies, int openingPlies, ulong seed, string unit)
    {
        if (!File.Exists(ckptPath)) { Console.WriteLine($"[strength] checkpoint not found: {ckptPath}"); return; }

        IPolicyValueNet net;
        using (var fs = File.OpenRead(ckptPath))
            net = builder.Load(fs, game.ObservationSize, game.PolicySize);
        Console.WriteLine($"[strength] loaded {arch} net ({net.Describe()}) from {ckptPath}");
        Console.WriteLine($"[strength] net (+MCTS {sims} sims, argmax) vs material minimax depth {depth} | {games} games | max {maxPlies} plies");

        var r = StrengthEval.Run(game, material, net, sims, depth, games, maxPlies, openingPlies, seed);
        Console.WriteLine($"[strength] net vs minimax-d{depth}: {r.Wins}W {r.Draws}D {r.Losses}L | score {r.Score:P1} | avg end-material {r.AvgEndMaterial:+0.00;-0.00} {unit}");
    }
}
