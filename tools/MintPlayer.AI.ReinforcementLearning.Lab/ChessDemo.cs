using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// `--game chess --demo`: play one game with the self-taught net (net + MCTS, most-visited move) as White against a
/// random-legal opponent, printing the FEN after every ply — a watchable, decisive game that shows the AI exploiting
/// the opponent's blunders. Loads the trained checkpoint from the data dir if present, else plays from a fresh net.
/// Not part of training.
/// </summary>
internal static class ChessDemo
{
    public static void Run(string dataDir, int sims, ulong seed, int maxPlies)
    {
        var game = new ChessGame();
        var store = new FileModelStore(dataDir);
        PolicyValueNet net;
        using (var s = store.TryOpenRead("chess", "az"))
            net = s is not null
                ? PolicyValueNet.Load(s, "selfplay-pv", game.ObservationSize, game.PolicySize)
                : new PolicyValueNet(game.ObservationSize, [256, 256], game.PolicySize, new Xoshiro256StarStar(seed));
        Console.Error.WriteLine(store.TryOpenRead("chess", "az") is null ? "(fresh net)" : "(loaded trained net)");

        var rng = new Xoshiro256StarStar(seed);
        var cfg = new Mcts.Config(Simulations: sims, RootNoiseFrac: 0f);

        (float[] Priors, float Value) Evaluate(ChessState st)
        {
            var obs = new float[game.ObservationSize];
            game.WriteObservation(st, obs);
            using (GradMode.NoGrad())
            {
                var (logits, value) = net.Forward(new Tensor(obs, 1, obs.Length));
                var legal = game.LegalMoves(st);
                var priors = new float[game.PolicySize];
                float max = float.NegativeInfinity;
                foreach (int m in legal) if (logits.Data[m] > max) max = logits.Data[m];
                float sum = 0f;
                foreach (int m in legal) { float e = MathF.Exp(logits.Data[m] - max); priors[m] = e; sum += e; }
                if (sum > 0f) foreach (int m in legal) priors[m] /= sum;
                return (priors, MathF.Tanh(value.Data[0]));
            }
        }

        const bool aiIsWhite = true; // the AI plays White; a random-legal mover plays Black
        var state = game.Root();
        Console.WriteLine(ChessFen.ToFen(state)); // starting position first
        int ply = 0;
        while (game.Result(state) == GameResult.Ongoing && ply < maxPlies)
        {
            int move;
            if (state.WhiteToMove == aiIsWhite)
            {
                var pi = Mcts.Search(game, state, Evaluate, cfg, rng);
                move = 0;
                for (int a = 1; a < pi.Length; a++) if (pi[a] > pi[move]) move = a;
            }
            else
            {
                var legal = game.LegalMoves(state);
                move = legal[rng.NextInt(legal.Count)];
            }
            state = game.Apply(state, move);
            Console.WriteLine(ChessFen.ToFen(state));
            ply++;
        }
        // Report from White's (the AI's) perspective.
        var final = game.Result(state);
        string outcome = final switch
        {
            GameResult.Loss => state.WhiteToMove ? "AI (White) lost" : "AI (White) won",
            GameResult.Win => state.WhiteToMove ? "AI (White) won" : "AI (White) lost",
            GameResult.Draw => "draw",
            _ => "unfinished (ply cap)",
        };
        Console.Error.WriteLine($"result: {outcome} after {ply} plies ({sims} sims/move)");
    }
}
