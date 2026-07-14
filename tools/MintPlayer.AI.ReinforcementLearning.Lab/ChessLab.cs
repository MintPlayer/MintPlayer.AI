using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

/// <summary>
/// `--game chess` entry point (PLAN M39.2): AlphaZero-style self-play on chess — the second consumer of the reusable
/// self-play stack (<see cref="Mcts"/> + <see cref="SelfPlayCampaign{TState}"/>), which is reused UNCHANGED; only the
/// perft-verified <see cref="ChessGame"/> is new. CPU-only and honestly bounded (a small MLP over a flattened board →
/// legal, steadily-improving play, not engine strength). Flags: --sims, --games, --eval-games, --hidden, --max-plies,
/// plus the common --hours/--data/--seed/--lr.
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
        // Hard ply cap per self-play game. A weak net rarely mates, so games otherwise run to this cap; since a chunk's
        // wall time is bounded by its SLOWEST game (the straggler that finishes last, single-threaded), the cap — not
        // the average game — sets self-play throughput. Lower it for a heavy net (conv) to keep evals frequent.
        int maxPlies = a.Int("--max-plies", 200);
        double opponentRandom = a.Dbl("--opponent-random", 0); // fraction of games vs a random opponent (robustness)
        bool evalOnly = a.Has("--eval-only");

        // --demo: play one self-play game with the (trained) net and print FENs to watch — no training.
        if (a.Has("--demo")) { ChessDemo.Run(dataDir, sims, seed, a.Int("--demo-plies", 100)); return; }

        // --vs-minimax: NON-SATURATING strength eval — the (trained) net vs a fixed material alpha-beta of --minimax-depth.
        // Unlike winRate-vs-random (saturates) or self-play material (self-referential), this measures whether training
        // actually improved play: a genuinely stronger net beats a deeper opponent. Raise --minimax-depth as the net grows.
        if (a.Has("--vs-minimax"))
        {
            ChessStrengthEval.Run(
                ckptPath: a.Str("--ckpt", Path.Combine(dataDir, "chess.az.ckpt")),
                arch: a.Str("--arch", "conv"), filters: a.Int("--filters", 64), blocks: a.Int("--blocks", 6),
                hidden: [hidden, hidden], sims: sims, depth: a.Int("--minimax-depth", 2),
                games: a.Int("--strength-games", 40), maxPlies: maxPlies, openingPlies: a.Int("--opening-plies", 4), seed: seed);
            return;
        }

        // --ladder: hands-off difficulty ladder (M40.4). Whenever the live net beats the last-promoted checkpoint by a
        // margin in a net-vs-net arena, a new tier .ckpt + updated manifest are written straight into the web app's
        // models dir — so `--game chess --ladder --hours N` grows the site's difficulty roster with no manual steps.
        // Dense material-shaped value target (α): blend of game outcome + per-position material advantage. 0 = pure
        // outcome (old behaviour); default 0.5 gives a weak net a gradient on every capture (the anti-plateau fix).
        float materialWeight = a.Flt("--material-weight", 0.5f);
        // Value-loss weight (relative to policy loss). Default 1 = equal. Lower (e.g. 0.3) counters value-head
        // overfitting → strength regression at small scale — the observed "loss drops but play regresses" failure.
        float valueWeight = a.Flt("--value-weight", 1f);
        // Leaf-inference batch size for self-play MCTS (M42.5). 1 = sequential batch-1 (default, back-compat). >1 uses
        // virtual-loss batched MCTS so each net.Forward evaluates N leaves at once — required for any GPU utilization.
        int leafBatch = a.Int("--leaf-batch", 1);
        // De-ceiling knobs (defaults preserve current behaviour; a large run raises them). The MCTS knobs
        // (--cpuct/--dirichlet-alpha/--root-noise) go into the Mcts.Config below; --temp-moves was hardcoded at 12.
        int tempMoves = a.Int("--temp-moves", 12);
        int window = a.Int("--window", 40_000);   // replay-window capacity (AlphaZero-scale runs want ~500k+)
        int batch = a.Int("--batch", 128);
        int epochs = a.Int("--epochs", 1);         // shuffled passes over the window per chunk
        float clip = a.Flt("--clip", 5f);          // gradient-norm clip

        // Parallel self-play generation (M41.2): --parallel fans the chunk's games across cores (default cores-2),
        // --dop caps the degree of parallelism. Trained weights are identical at any DOP for a given seed.
        bool parallel = a.Has("--parallel");
        // --gpu: route Tensor ops through the ILGPU AdaptiveBackend (large GEMMs → GPU). Pays off with --leaf-batch
        // (batched inference); batch-1 self-play barely uses a GPU. The training step (batched) benefits regardless.
        bool useGpu = a.Has("--gpu");
        int? dop = a.Has("--dop") ? a.Int("--dop", System.Math.Max(1, System.Environment.ProcessorCount - 2)) : null;

        // Net architecture (M42): --arch conv builds an AlphaZero-style convolutional residual tower over the 18×8×8
        // board (the plateau fix); default "mlp" keeps the flat PolicyValueNet. --filters/--blocks size the tower.
        // The chess observation is 18 planes × 64 squares (ChessGame), laid out plane-major so it reshapes to 18×8×8.
        IPolicyValueNetBuilder? netBuilder = a.Str("--arch", "mlp").ToLowerInvariant() == "conv"
            ? new ConvNetBuilder(planes: 18, boardH: 8, boardW: 8, filters: a.Int("--filters", 64), blocks: a.Int("--blocks", 6))
            : null; // null → SelfPlayCampaign's default flat MLP with trunk [hidden, hidden]

        LadderOptions? ladder = a.Has("--ladder")
            ? new LadderOptions(
                Dir: a.Str("--difficulty-dir", Path.Combine("src", "RLDemo.Web", "wwwroot", "models")),
                PromoteMaterial: a.Dbl("--promote-material", 0.75), // primary gate: avg pawns of material over the champion
                PromoteMargin: a.Dbl("--promote-margin", 0.08),     // fallback: rise in winRate-vs-random
                ArenaMargin: a.Dbl("--arena-margin", 0.60),         // fallback: head-to-head score once winRate saturates
                ArenaGames: a.Int("--arena-games", 20),
                Sims: a.Int("--difficulty-sims", 128),
                OpeningPlies: a.Int("--opening-plies", 6))
            : null;

        // Eval/checkpoint cadence in minutes (defaults preserve CampaignOptions' 2 / 10). The ladder promotes on this
        // cadence, so a short cadence captures tiers sooner (and lets a quick run exercise the arena).
        double? firstEval = a.Has("--first-eval") ? a.Dbl("--first-eval", 2) : null;
        double? evalEvery = a.Has("--eval-every") ? a.Dbl("--eval-every", 10) : null;

        var game = new ChessGame();
        var cfg = new Mcts.Config(Simulations: sims, Cpuct: a.Flt("--cpuct", 1.25f),
            DirichletAlpha: a.Flt("--dirichlet-alpha", 0.3f), RootNoiseFrac: a.Flt("--root-noise", 0.25f));
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: useGpu,
            sp =>
            {
                var adaptive = useGpu ? sp.GetRequiredService<AdaptiveBackend>() : null;
                // GPU-resident conv forward for batched self-play (M43), when a GPU is present + the net is conv;
                // else the autograd default. All the Ilgpu knowledge stays here, out of the generic campaign.
                Func<IPolicyValueNet, IPolicyValueForward>? forwardFactory = adaptive is null ? null
                    : net => adaptive.Gpu is { } gpu && net is ConvResidualPolicyValueNet conv
                        ? gpu.CreateResidentForward(conv)
                        : new AutogradPolicyValueForward(net, game.ObservationSize);
                return new SelfPlayCampaign<ChessState>(game, "chess", new SelfPlayOptions
                {
                    Seed = seed, LearningRate = learningRate, Hidden = hidden, Search = cfg,
                    GamesPerChunk = gamesPerChunk, TempMoves = tempMoves, EvalGames = evalGames,
                    WindowCapacity = window, BatchSize = batch, EpochsPerChunk = epochs, MaxPlies = maxPlies,
                    OpponentRandomFrac = opponentRandom, Ladder = ladder, MaterialWeight = materialWeight,
                    ValueWeight = valueWeight, GradClipNorm = clip, Parallel = parallel, MaxDop = dop, LeafBatch = leafBatch,
                }, netBuilder: netBuilder, backend: adaptive, forwardFactory: forwardFactory);
            },
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", "chess-selfplay.csv")),
            firstEvalMinutes: firstEval, evalEveryMinutes: evalEvery);
    }
}
