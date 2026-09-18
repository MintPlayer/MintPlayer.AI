extern alias Lab;

using BlockDudeLab = Lab::BlockDudeLab;
using ChessLab = Lab::ChessLab;
using CliArgs = Lab::CliArgs;
using CrazyFruitsLab = Lab::CrazyFruitsLab;
using DraughtsLab = Lab::DraughtsLab;
using SnakeLab = Lab::SnakeLab;
using TetrisLab = Lab::TetrisLab;

using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M63.6 — the <c>Parse</c> seam for the Labs whose <c>Run</c> head was still inline after M63.5.
/// <para>Chess, Draughts, BlockDude, Snake, Tetris and CrazyFruits each opened <c>Run</c> with a block of
/// pure flag reads and then dispatched into a mode, so no default and no flag interaction could be checked
/// without starting a real training run — <c>ChessLab.Run</c> was 100% uncovered. These assert the defaults
/// (a silent change to one is a silently different training run), the flags that decide what gets WRITTEN
/// (data directory, seed, CSV, checkpoint id), and the handful of rules that are not plain flag reads:
/// Snake's <c>--grow</c> override, BlockDude's phase-2 CSV split and CrazyFruits' <c>--pbrs</c>-turns-off-
/// creation-bonuses interaction.</para>
/// <para>Nothing here runs an episode, loads a checkpoint, builds a net or binds a port — every method under
/// test is a pure function of the argument array.</para>
/// </summary>
public class LabFlagTests
{
    // ── Chess: the whole entry point was uncovered ──

    [Fact]
    public void Chess_defaults()
    {
        var f = ChessLab.Parse(new CliArgs([]));

        Assert.Equal(1, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(1e-3f, f.LearningRate);
        Assert.Equal(256, f.Hidden);
        Assert.Equal(64, f.Sims);            // modest — chess movegen per node is heavy on CPU
        Assert.Equal(8, f.GamesPerChunk);
        Assert.Equal(10, f.EvalGames);
        Assert.Equal(200, f.MaxPlies);
        Assert.Equal(0, f.OpponentRandom);
        Assert.False(f.EvalOnly);
    }

    [Fact]
    public void Chess_flags_override_the_defaults()
    {
        var f = ChessLab.Parse(new CliArgs(
            ["--hours", "3", "--data", "runs", "--seed", "9", "--sims", "400", "--max-plies", "60", "--eval-only"]));

        Assert.Equal(3, f.Hours);
        Assert.Equal("runs", f.DataDir);
        Assert.Equal(9UL, f.Seed);
        Assert.Equal(400, f.Sims);
        Assert.Equal(60, f.MaxPlies);
        Assert.True(f.EvalOnly);
    }

    [Fact]
    public void Chess_training_defaults_are_the_shipped_self_play_recipe()
    {
        var o = ChessLab.TrainingOptions(new CliArgs([]), ChessLab.Parse(new CliArgs([])),
            out bool useGpu, out string gpus, out double? firstEval, out double? evalEvery);

        Assert.False(useGpu);
        Assert.Equal("all", gpus);           // with --gpu, every detected GPU — no count needed
        Assert.Null(firstEval);              // null = keep CampaignOptions' own 2 / 10 cadence
        Assert.Null(evalEvery);
        Assert.Equal(0.5f, o.MaterialWeight); // the anti-plateau dense value target; 0 = pure outcome
        Assert.Equal(1f, o.ValueWeight);
        Assert.Equal(1, o.LeafBatch);        // 1 = sequential batch-1 MCTS (back-compat)
        Assert.Equal(12, o.TempMoves);
        Assert.Equal(40_000, o.WindowCapacity);
        Assert.Equal(128, o.BatchSize);
        Assert.Equal(1, o.EpochsPerChunk);
        Assert.Equal(5f, o.GradClipNorm);
        Assert.False(o.Parallel);
        Assert.Null(o.MaxDop);
        Assert.Null(o.Ladder);
    }

    [Fact]
    public void Chess_head_flags_flow_into_the_training_options()
    {
        var a = new CliArgs(["--seed", "5", "--sims", "128", "--games", "3", "--eval-games", "4",
                             "--hidden", "64", "--max-plies", "40", "--opponent-random", "0.25", "--lr", "2e-4"]);

        var o = ChessLab.TrainingOptions(a, ChessLab.Parse(a), out _, out _, out _, out _);

        Assert.Equal(5UL, o.Seed);
        Assert.Equal(128, o.Search.Simulations); // --sims reaches the MCTS config, not just the demo mode
        Assert.Equal(3, o.GamesPerChunk);
        Assert.Equal(4, o.EvalGames);
        Assert.Equal(64, o.Hidden);
        Assert.Equal(40, o.MaxPlies);
        Assert.Equal(0.25, o.OpponentRandomFrac);
        Assert.Equal(2e-4f, o.LearningRate);
    }

    [Fact]
    public void Chess_mcts_knob_defaults()
    {
        var o = ChessLab.TrainingOptions(new CliArgs([]), ChessLab.Parse(new CliArgs([])), out _, out _, out _, out _);

        Assert.Equal(1.25f, o.Search.Cpuct);
        Assert.Equal(0.3f, o.Search.DirichletAlpha);
        Assert.Equal(0.25f, o.Search.RootNoiseFrac);
    }

    [Fact]
    public void Chess_ladder_is_opt_in_and_carries_its_own_defaults()
    {
        // The ladder WRITES tier checkpoints straight into the web app's models dir, so both halves matter:
        // it must stay off unless asked, and its promotion thresholds must not drift.
        var a = new CliArgs(["--ladder"]);

        var o = ChessLab.TrainingOptions(a, ChessLab.Parse(a), out _, out _, out _, out _);

        Assert.NotNull(o.Ladder);
        var ladder = o.Ladder!;
        Assert.Equal(Path.Combine("src", "RLDemo.Web", "wwwroot", "models"), ladder.Dir);
        Assert.Equal(0.75, ladder.PromoteMaterial);
        Assert.Equal(0.08, ladder.PromoteMargin);
        Assert.Equal(0.60, ladder.ArenaMargin);
        Assert.Equal(20, ladder.ArenaGames);
        Assert.Equal(128, ladder.Sims);
        Assert.Equal(6, ladder.OpeningPlies);
    }

    [Fact]
    public void Chess_eval_cadence_and_dop_are_null_unless_their_flag_is_present()
    {
        // These three are "null means leave the default alone" flags: the VALUE is only read when the flag
        // is there, so a missing flag must not silently pin the cadence to the number in the call.
        var a = new CliArgs(["--first-eval", "1", "--eval-every", "4", "--dop", "3", "--parallel", "--gpu", "--gpus", "0,2"]);

        var o = ChessLab.TrainingOptions(a, ChessLab.Parse(a), out bool useGpu, out string gpus,
                                         out double? firstEval, out double? evalEvery);

        Assert.Equal(1, firstEval);
        Assert.Equal(4, evalEvery);
        Assert.Equal(3, o.MaxDop);
        Assert.True(o.Parallel);
        Assert.True(useGpu);
        Assert.Equal("0,2", gpus);
    }

    // ── Draughts: the M47 locked constants ──

    [Fact]
    public void Draughts_defaults_are_the_locked_M47_constants()
    {
        var f = DraughtsLab.Parse(new CliArgs([]));

        Assert.Equal(DraughtsVariant.International10, f.Variant);
        Assert.Equal("draughts", f.EnvId);
        Assert.Equal(10, f.Board);
        Assert.Equal(1, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(3e-4f, f.LearningRate);  // locked: 1e-3 peaked-then-regressed on chess
        Assert.Equal(256, f.Hidden);
        Assert.Equal(64, f.Sims);
        Assert.Equal(8, f.GamesPerChunk);
        Assert.Equal(10, f.EvalGames);
        Assert.Equal(150, f.MaxPlies);        // backstop only — the no-progress rule draws shuffles first
        Assert.Equal(0, f.OpponentRandom);
        Assert.False(f.EvalOnly);
        Assert.Equal(64, f.Filters);
        Assert.Equal(6, f.Blocks);
        Assert.True(f.Conv);                  // conv is the DEFAULT here, unlike chess
    }

    [Fact]
    public void Draughts_variant_flag_reaches_the_parsed_flags()
    {
        var f = DraughtsLab.Parse(new CliArgs(["--variant", "checkers8"]));

        Assert.Equal(DraughtsVariant.English8, f.Variant);
        Assert.Equal("checkers8", f.EnvId);   // and so a different checkpoint and CSV
        Assert.Equal(8, f.Board);
    }

    [Fact]
    public void Draughts_arch_mlp_opts_out_of_the_conv_tower()
    {
        Assert.False(DraughtsLab.Parse(new CliArgs(["--arch", "mlp"])).Conv);
        Assert.True(DraughtsLab.Parse(new CliArgs(["--arch", "CONV"])).Conv);
    }

    [Fact]
    public void Draughts_training_defaults_including_the_wider_arena()
    {
        var a = new CliArgs(["--ladder"]);

        var o = DraughtsLab.TrainingOptions(a, DraughtsLab.Parse(a), out bool useGpu, out string gpus,
                                            out double? firstEval, out double? evalEvery);

        Assert.False(useGpu);
        Assert.Equal("all", gpus);
        Assert.Null(firstEval);
        Assert.Null(evalEvery);
        Assert.Equal(0.5f, o.MaterialWeight);  // locked (0.3 broke the chess gate)
        Assert.Equal(1f, o.ValueWeight);
        Assert.Equal(12, o.TempMoves);
        Assert.Equal(40_000, o.WindowCapacity);
        Assert.Equal(128, o.BatchSize);
        Assert.Equal(5f, o.GradClipNorm);
        Assert.NotNull(o.Ladder);
        Assert.Equal(40, o.Ladder!.ArenaGames); // locked: 12-game arenas were ±1-unit noise (chess used 20)
    }

    [Fact]
    public void Draughts_no_ladder_by_default()
    {
        var o = DraughtsLab.TrainingOptions(new CliArgs([]), DraughtsLab.Parse(new CliArgs([])),
                                            out _, out _, out _, out _);

        Assert.Null(o.Ladder);
        Assert.Equal(3e-4f, o.LearningRate);
        Assert.Equal(150, o.MaxPlies);
    }

    // ── BlockDude: --phase decides WHICH campaign and WHICH log ──

    [Fact]
    public void BlockDude_defaults()
    {
        var f = BlockDudeLab.Parse(new CliArgs([]));

        Assert.Equal(9, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(3e-4f, f.LearningRate);
        Assert.False(f.EvalOnly);
        Assert.False(f.Grow);
        Assert.Equal(6, f.GrowPatience);
        Assert.Equal(0.04, f.GrowMinImprovement);
        Assert.False(f.Fresh);
        Assert.Equal(0, f.TargetSamples);       // 0 = no sample budget, run to the hour budget
        Assert.Equal(4, f.BoardsPerRound);
        Assert.Equal(512, f.SamplesPerBoard);
        Assert.Equal(100_000, f.GateEvery);
        Assert.Equal(BlockDudeCurriculum.LastStage, f.MaxStage);
        Assert.Equal(-1, f.Pinned);             // -1 = not pinned (becomes a null PinStage)
        Assert.Equal(1, f.Phase);
    }

    [Fact]
    public void BlockDude_phase_selects_the_csv_a_run_appends_to()
    {
        // The two phases are different campaigns with different checkpoints; sharing one log would present
        // two trajectories as one.
        Assert.Equal(Path.Combine("data", "logs", "blockdude.csv"), BlockDudeLab.Parse(new CliArgs([])).Csv);
        Assert.Equal(Path.Combine("data", "logs", "blockdude-xit.csv"),
                     BlockDudeLab.Parse(new CliArgs(["--phase", "2"])).Csv);
        Assert.Equal(Path.Combine("runs", "logs", "blockdude-xit.csv"),
                     BlockDudeLab.Parse(new CliArgs(["--phase", "3", "--data", "runs"])).Csv);
    }

    [Fact]
    public void BlockDude_stage_pin_is_null_below_zero_and_honoured_otherwise()
    {
        Assert.Null(BlockDudeLab.ImitationOptions(BlockDudeLab.Parse(new CliArgs([]))).PinStage);
        Assert.Equal(3, BlockDudeLab.ImitationOptions(BlockDudeLab.Parse(new CliArgs(["--stage", "3"]))).PinStage);
        Assert.Equal(0, BlockDudeLab.ImitationOptions(BlockDudeLab.Parse(new CliArgs(["--stage", "0"]))).PinStage);
    }

    [Fact]
    public void BlockDude_imitation_options_carry_the_growth_policy()
    {
        var a = new CliArgs(["--grow", "--grow-patience", "3", "--grow-min-improvement", "0.1", "--fresh"]);

        var o = BlockDudeLab.ImitationOptions(BlockDudeLab.Parse(a));

        Assert.True(o.Grow);
        Assert.Equal(3, o.GrowPatience);
        Assert.Equal(0.1, o.GrowMinImprovement);
        Assert.True(o.Fresh);
        Assert.Equal(1, o.Phase);
    }

    [Fact]
    public void BlockDude_expert_iteration_defaults()
    {
        var a = new CliArgs(["--phase", "2"]);

        var o = BlockDudeLab.ExpertIterationOptions(a, BlockDudeLab.Parse(a));

        Assert.True(o.WarmStart);            // --no-warm-start is the opt-out, so the default must be true
        Assert.Equal(4, o.AttemptsPerLevel);
        Assert.Equal(40_000, o.Expansions);
        Assert.Equal(6, o.SearchSeconds);
        Assert.Equal(2f, o.Weight);
        Assert.Equal(5f, o.PolicyWeight);
        Assert.Equal(256, o.BeamWidth);
        Assert.Equal(8, o.BeamSeconds);
        Assert.Equal(45, o.BeamSecondsMax);
        Assert.Equal(12, o.BeamMovesPerSecond);
        Assert.Equal(20, o.InitialFrontier);
        Assert.Equal(1.5, o.FrontierGrowth);
        Assert.Equal(0.75, o.AdvanceRate);
        Assert.Equal(0.25, o.DemoShare);
    }

    [Fact]
    public void BlockDude_no_warm_start_switches_off_the_phase_1_inheritance()
    {
        var a = new CliArgs(["--phase", "2", "--no-warm-start"]);

        Assert.False(BlockDudeLab.ExpertIterationOptions(a, BlockDudeLab.Parse(a)).WarmStart);
    }

    // ── Snake: the head above the already-seamed Options/SearchConfig ──

    [Fact]
    public void Snake_head_defaults()
    {
        var f = SnakeLab.Parse(new CliArgs([]));

        Assert.Equal(1, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(6, f.TrainGrid);        // train small, eval big — the M22 generalization protocol
        Assert.Equal(12, f.EvalGrid);
        Assert.Equal(5_000, f.ChunkSteps);
        Assert.Equal(100_000, f.TargetSteps);
        Assert.Equal(20, f.EvalEpisodes);
        Assert.Equal(5e-4f, f.LearningRate);
        Assert.Equal(1.0f, f.Explore);
        Assert.Equal([128, 128], f.Hidden);
        Assert.Equal(0.99, f.Gamma);
        Assert.Equal(-0.01f, f.StepPenalty); // ~0 removes the safe-starvation pressure
        Assert.False(f.SafeMask);
        Assert.False(f.EvalOnly);
        Assert.False(f.Grow);
        Assert.Equal(5000, f.GrowEvery);
        Assert.False(f.Search);
        Assert.False(f.Cycle);
        Assert.Equal(Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "snake-net.ckpt"), f.NetPath);
    }

    [Fact]
    public void Snake_grow_replaces_the_requested_hidden_widths()
    {
        // Same rule as FruitCake: --grow does not ADD to --hidden, it REPLACES it with the tiny first stage,
        // because a growing run must start small. This lived in Run and so was never asserted.
        Assert.Equal([512, 512], SnakeLab.Parse(new CliArgs(["--hidden", "512,512"])).Hidden);

        var growing = SnakeLab.Parse(new CliArgs(["--hidden", "512,512", "--grow"]));
        Assert.True(growing.Grow);
        Assert.NotEqual([512, 512], growing.Hidden);
    }

    [Fact]
    public void Snake_mode_switches_and_the_search_config_come_out_of_one_parse()
    {
        var f = SnakeLab.Parse(new CliArgs(["--search", "--depth", "30", "--eval-grid", "20", "--net", "x.ckpt"]));

        Assert.True(f.Search);
        Assert.False(f.Cycle);
        Assert.Equal(20, f.EvalGrid);
        Assert.Equal("x.ckpt", f.NetPath);
        Assert.Equal(30, f.SearchCfg.MaxDepth);
    }

    [Fact]
    public void Snake_env_shaping_flags_are_read_for_both_envs()
    {
        var f = SnakeLab.Parse(new CliArgs(["--step-penalty", "0", "--safe-mask", "--train-grid", "8"]));

        Assert.Equal(0f, f.StepPenalty);
        Assert.True(f.SafeMask);
        Assert.Equal(8, f.TrainGrid);
    }

    // ── Tetris: the head above the already-seamed Options ──

    [Fact]
    public void Tetris_head_defaults()
    {
        var f = TetrisLab.Parse(new CliArgs([]));

        Assert.Equal(1, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(500, f.PieceBudget);    // the uniform no-garbage protocol-A episode length
        Assert.Equal(5_000, f.ChunkSteps);
        Assert.Equal(400_000, f.TargetSteps);
        Assert.Equal(20, f.EvalEpisodes);
        Assert.Equal(1e-3f, f.LearningRate);
        Assert.Equal(1.0f, f.Explore);
        Assert.Equal([128, 128], f.Hidden);
        Assert.Equal(0.995, f.Gamma);
        Assert.False(f.EvalOnly);
        Assert.Equal(0, f.Baselines);        // 0 = train; >0 selects the scripted-policy table instead
        Assert.Equal(Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "tetris.dqn.ckpt"), f.NetPath);
    }

    [Fact]
    public void Tetris_head_flags_override_the_defaults()
    {
        var f = TetrisLab.Parse(new CliArgs(["--baselines", "30", "--piece-budget", "200", "--net", "t.ckpt",
                                             "--data", "runs", "--seed", "12"]));

        Assert.Equal(30, f.Baselines);
        Assert.Equal(200, f.PieceBudget);
        Assert.Equal("t.ckpt", f.NetPath);
        Assert.Equal("runs", f.DataDir);
        Assert.Equal(12UL, f.Seed);
    }

    [Fact]
    public void Tetris_pbrs_shaping_is_on_by_default_and_the_other_two_are_off()
    {
        // PBRS defaults ON (M54.3: the bare reward measured near-random at 180K steps) — it is the only
        // train-env switch whose default is true, so --no-pbrs is an opt-OUT while the other two are opt-in.
        var d = TetrisLab.Shaping(new CliArgs([]));

        Assert.True(d.Pbrs);
        Assert.False(d.MixGarbage);
        Assert.False(d.MandatoryTetris);
    }

    [Fact]
    public void Tetris_shaping_flags_flip_each_switch()
    {
        Assert.False(TetrisLab.Shaping(new CliArgs(["--no-pbrs"])).Pbrs);
        Assert.True(TetrisLab.Shaping(new CliArgs(["--mix-garbage"])).MixGarbage);
        Assert.True(TetrisLab.Shaping(new CliArgs(["--mandatory-tetris"])).MandatoryTetris);
    }

    // ── CrazyFruits: the head above the already-seamed GateLines/Summarize ──

    [Fact]
    public void CrazyFruits_head_defaults()
    {
        var f = CrazyFruitsLab.Parse(new CliArgs([]));

        Assert.Equal(1, f.Hours);
        Assert.Equal("data", f.DataDir);
        Assert.Equal(1UL, f.Seed);
        Assert.Equal(30, f.MoveBudget);
        Assert.Equal(5_000, f.ChunkSteps);
        Assert.Equal(150_000, f.TargetSteps);
        Assert.Equal(20, f.EvalEpisodes);
        Assert.Equal(5e-4f, f.LearningRate);
        Assert.Equal(1.0f, f.Explore);
        Assert.Equal([256, 256], f.Hidden);
        Assert.Equal(0.99, f.Gamma);
        Assert.False(f.EvalOnly);
        Assert.False(f.Grow);
        Assert.Equal(5000, f.GrowEvery);
        Assert.Equal(0, f.Baselines);
        Assert.Equal(Path.Combine("src", "RLDemo.Web", "wwwroot", "models", "crazyfruits.dqn.ckpt"), f.NetPath);
    }

    [Fact]
    public void CrazyFruits_dqn_option_defaults()
    {
        var a = new CliArgs([]);

        var o = CrazyFruitsLab.Options(a, CrazyFruitsLab.Parse(a));

        Assert.Equal(1UL, o.Seed);
        Assert.Equal(5_000, o.ChunkSteps);
        Assert.Equal(150_000, o.TargetSteps);
        Assert.Equal(20, o.EvalEpisodes);
        Assert.Equal([256, 256], o.Hidden);
        Assert.Equal(0.99, o.Gamma);
        Assert.Equal(1, o.NStep);
        Assert.False(o.DenseRegression);
        Assert.Equal(1.0f, o.DenseTargetWeight);
    }

    [Fact]
    public void CrazyFruits_dense_regression_is_opt_in_with_its_own_weight()
    {
        // M51: the dense weight must be able to drown unit conflicts, so it is a knob separate from the flag.
        var a = new CliArgs(["--dense", "--dense-weight", "25", "--nstep", "3", "--gamma", "0"]);

        var o = CrazyFruitsLab.Options(a, CrazyFruitsLab.Parse(a));

        Assert.True(o.DenseRegression);
        Assert.Equal(25f, o.DenseTargetWeight);
        Assert.Equal(3, o.NStep);
        Assert.Equal(0, o.Gamma);
    }

    [Fact]
    public void CrazyFruits_creation_shaping_is_the_default_and_pbrs_replaces_it()
    {
        // The non-obvious rule: --pbrs does not ADD a potential term, it REPLACES the creation bonuses, so it
        // turns shaping off without --no-shape. Getting this backwards double-shapes the training reward.
        var (pbrs, shape) = CrazyFruitsLab.Shaping(new CliArgs([]));
        Assert.False(pbrs);
        Assert.True(shape);

        var on = CrazyFruitsLab.Shaping(new CliArgs(["--pbrs"]));
        Assert.True(on.Pbrs);
        Assert.False(on.Shape);

        var off = CrazyFruitsLab.Shaping(new CliArgs(["--no-shape"]));
        Assert.False(off.Pbrs);
        Assert.False(off.Shape);
    }
}
