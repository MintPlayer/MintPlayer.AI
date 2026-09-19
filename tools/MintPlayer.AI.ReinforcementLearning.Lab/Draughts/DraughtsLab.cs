using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

/// <summary>
/// `--game draughts` entry point (PLAN M47.3): AlphaZero-style self-play on draughts — the strength showcase
/// that replaces chess (M47 PRD). The reusable stack (<see cref="Mcts"/> + <see cref="SelfPlayCampaign{TState}"/>
/// + GPU wiring) is reused UNCHANGED; only the perft-verified <see cref="DraughtsGame"/> is new. Two variants
/// share this entry: international 10×10 "dammen" (default — the showcase) and `--variant checkers8` (english
/// 8×8 — the cheap pipeline-validation run M47.4 starts with; strongest field precedent). Each variant trains
/// under its own environment id (different policy sizes ⇒ different checkpoints). Defaults bake the locked M47
/// constants: lr 3e-4, material-weight 0.5, arena-games 40, conv tower, ply cap 150 (the in-engine no-progress
/// rule ends king shuffles first). Flags otherwise mirror `--game chess`.
/// </summary>
internal static class DraughtsLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        var f = Parse(a);
        var (variant, envId, board) = (f.Variant, f.EnvId, f.Board);
        var game = new DraughtsGame(variant);

        (double hours, string dataDir, ulong seed) = (f.Hours, f.DataDir, f.Seed);
        (int hidden, int sims, int maxPlies, bool evalOnly) = (f.Hidden, f.Sims, f.MaxPlies, f.EvalOnly);
        (int filters, int blocks, bool conv) = (f.Filters, f.Blocks, f.Conv);

        IPolicyValueNetBuilder? netBuilder = conv
            ? new ConvNetBuilder(planes: 5, boardH: board, boardW: board, filters: filters, blocks: blocks)
            : null; // null → SelfPlayCampaign's default flat MLP with trunk [hidden, hidden]

        // --bench-forward: the M47.3 micro-bench gate — real resident-forward numbers for THIS tower shape.
        if (a.Has("--bench-forward"))
        {
            ConvForwardBench.Run(planes: 5, board: board, actions: game.PolicySize,
                filters, blocks, a.Int("--leaf-batch", 256), a.Int("--bench-iters", 30));
            return;
        }

        // --vs-minimax: THE strength metric (locked M47 constant — winRate-vs-random and self-play material are
        // cosmetic). Runs automatically at promotions during training; this flag runs it standalone on a ckpt.
        if (a.Has("--vs-minimax"))
        {
            IPolicyValueNetBuilder strengthBuilder = conv
                ? new ConvNetBuilder(5, board, board, filters, blocks)
                : new MlpNetBuilder([hidden, hidden]);
            StrengthCli.Run(game, game, strengthBuilder, a.Str("--arch", "conv"),
                ckptPath: a.Str("--ckpt", Path.Combine(dataDir, $"{envId}.az.ckpt")),
                sims: sims, depth: a.Int("--minimax-depth", 1), games: a.Int("--strength-games", 40),
                maxPlies: maxPlies, openingPlies: a.Int("--opening-plies", 4), seed: seed, unit: "men");
            return;
        }

        var options = TrainingOptions(a, f, out bool useGpu, out string gpusSpec,
                                      out double? firstEval, out double? evalEvery);
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: useGpu,
            services =>
            {
                // The generated registration is the international game; the english pipeline-validation variant
                // overrides it here (last registration wins) so the campaign resolves the right board/action space.
                if (variant == DraughtsVariant.English8)
                {
                    services.AddSingleton<IZeroSumGame<DraughtsState>>(game);
                    services.AddSingleton<IMaterialScore<DraughtsState>>(game);
                }
                services.AddSelfPlayCampaign<DraughtsState>(envId, options, netBuilder: netBuilder, gpus: gpusSpec);
            },
            CampaignCli.ConsoleAndCsv(Path.Combine(dataDir, "logs", $"{envId}-selfplay.csv")),
            firstEvalMinutes: firstEval, evalEveryMinutes: evalEvery);
    }

    /// <summary>The flags read before any mode dispatch — the variant (and so the checkpoint id), the run's
    /// budget, and the tower shape the bench/strength modes share with training.</summary>
    /// <remarks>M63.6: extracted from <see cref="Run"/>, where the M47 locked constants (lr 3e-4,
    /// max-plies 150, conv-by-default) were reachable only by starting a real self-play run.</remarks>
    internal sealed record Flags(
        DraughtsVariant Variant, string EnvId, int Board,
        double Hours, string DataDir, ulong Seed, float LearningRate, int Hidden, int Sims,
        int GamesPerChunk, int EvalGames, int MaxPlies, double OpponentRandom, bool EvalOnly,
        int Filters, int Blocks, bool Conv);

    /// <summary>Reads the pure head of <see cref="Run"/> — no game, net or episode is touched.</summary>
    internal static Flags Parse(CliArgs a)
    {
        var (variant, envId, board) = ResolveVariant(a.Str("--variant", "international"));
        return new Flags(
            Variant: variant, EnvId: envId, Board: board,
            Hours: a.Dbl("--hours", 1),
            DataDir: a.Str("--data", "data"),
            Seed: a.ULong("--seed", 1),
            LearningRate: a.Flt("--lr", 3e-4f),   // locked (chess post-mortem: 1e-3 peaked-then-regressed)
            Hidden: a.Int("--hidden", 256),
            Sims: a.Int("--sims", 64),
            GamesPerChunk: a.Int("--games", 8),
            EvalGames: a.Int("--eval-games", 10),
            MaxPlies: a.Int("--max-plies", 150),  // backstop only — the no-progress rule draws shuffles first
            OpponentRandom: a.Dbl("--opponent-random", 0),
            EvalOnly: a.Has("--eval-only"),
            Filters: a.Int("--filters", 64),
            Blocks: a.Int("--blocks", 6),
            // Net architecture: conv is the DEFAULT here (the designed 5×N×N showcase tower); --arch mlp opts down.
            Conv: a.Str("--arch", "conv").ToLowerInvariant() == "conv");
    }

    /// <summary>The self-play options the training path's flags resolve to, plus the knobs handed to
    /// <c>LabHost.Run</c> rather than into the options record.</summary>
    /// <remarks>M63.6: extracted from <see cref="Run"/>. Read AFTER the bench/strength dispatch, exactly as
    /// before, so those read-only modes still never parse a training flag.</remarks>
    internal static SelfPlayOptions TrainingOptions(CliArgs a, Flags f, out bool useGpu, out string gpusSpec,
                                                    out double? firstEval, out double? evalEvery)
    {
        float materialWeight = a.Flt("--material-weight", 0.5f);   // locked (0.3 broke the chess gate)
        float valueWeight = a.Flt("--value-weight", 1f);
        int leafBatch = a.Int("--leaf-batch", 1);
        int tempMoves = a.Int("--temp-moves", 12);
        int window = a.Int("--window", 40_000);
        int batch = a.Int("--batch", 128);
        int epochs = a.Int("--epochs", 1);
        float clip = a.Flt("--clip", 5f);
        bool parallel = a.Has("--parallel");
        useGpu = a.Has("--gpu");
        gpusSpec = a.Str("--gpus", "all");
        int? dop = a.Has("--dop") ? a.Int("--dop", System.Math.Max(1, System.Environment.ProcessorCount - 2)) : null;

        LadderOptions? ladder = a.Has("--ladder")
            ? new LadderOptions(
                Dir: a.Str("--difficulty-dir", Path.Combine("src", "RLDemo.Web", "wwwroot", "models")),
                PromoteMaterial: a.Dbl("--promote-material", 0.75),
                PromoteMargin: a.Dbl("--promote-margin", 0.08),
                ArenaMargin: a.Dbl("--arena-margin", 0.60),
                ArenaGames: a.Int("--arena-games", 40),             // locked: 12-game arenas were ±1-unit noise
                Sims: a.Int("--difficulty-sims", 128),
                OpeningPlies: a.Int("--opening-plies", 6))
            : null;

        firstEval = a.Has("--first-eval") ? a.Dbl("--first-eval", 2) : null;
        evalEvery = a.Has("--eval-every") ? a.Dbl("--eval-every", 10) : null;

        var cfg = new Mcts.Config(Simulations: f.Sims, Cpuct: a.Flt("--cpuct", 1.25f),
            DirichletAlpha: a.Flt("--dirichlet-alpha", 0.3f), RootNoiseFrac: a.Flt("--root-noise", 0.25f));
        return new SelfPlayOptions
        {
            Seed = f.Seed, LearningRate = f.LearningRate, Hidden = f.Hidden, Search = cfg,
            GamesPerChunk = f.GamesPerChunk, TempMoves = tempMoves, EvalGames = f.EvalGames,
            WindowCapacity = window, BatchSize = batch, EpochsPerChunk = epochs, MaxPlies = f.MaxPlies,
            OpponentRandomFrac = f.OpponentRandom, Ladder = ladder, MaterialWeight = materialWeight,
            ValueWeight = valueWeight, GradClipNorm = clip, Parallel = parallel, MaxDop = dop, LeafBatch = leafBatch,
        };
    }

    /// <summary>
    /// Maps <c>--variant</c> onto the board size and the checkpoint id.
    /// </summary>
    /// <remarks>
    /// M63.5: extracted from <see cref="Run"/>. <c>envId</c> decides WHICH CHECKPOINT the run reads and
    /// writes, so a mis-mapped alias silently trains into the wrong file rather than failing. Three aliases
    /// select the 8x8 English game; anything else (including an unknown string) falls back to
    /// international 10x10, which is the documented default.
    /// </remarks>
    internal static (DraughtsVariant Variant, string EnvId, int Board) ResolveVariant(string variant)
    {
        bool english = variant.ToLowerInvariant() is "checkers8" or "english" or "english8";
        return english
            ? (DraughtsVariant.English8, "checkers8", 8)
            : (DraughtsVariant.International10, "draughts", 10);
    }
}
