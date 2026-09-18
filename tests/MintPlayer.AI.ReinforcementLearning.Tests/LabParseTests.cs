extern alias Lab;

using CliArgs = Lab::CliArgs;
using Connect4Lab = Lab::Connect4Lab;
using CubeLab = Lab::CubeLab;
using CubePolicyLab = Lab::CubePolicyLab;
using DraughtsLab = Lab::DraughtsLab;
using FruitCakeLab = Lab::FruitCakeLab;
using RushHourLab = Lab::RushHourLab;
using SnakeLab = Lab::SnakeLab;

using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M63.5 item 2 — the <c>Parse</c> seam across the game Labs.
/// <para>Every Lab built its options record inside the lambda handed to <c>LabHost.Run</c>, so no default
/// and no flag interaction could be checked without starting a real training run. These assert the
/// defaults (a silent change to one is a silently different training run) and the handful of flags that
/// decide what gets WRITTEN — the data directory, the seed, and the checkpoint id.</para>
/// </summary>
public class LabParseTests
{
    // ── RushHour ──

    [Fact]
    public void RushHour_defaults()
    {
        var o = RushHourLab.Parse(new CliArgs([]), out double hours, out string dataDir, out bool evalOnly);

        Assert.Equal(9, hours);
        Assert.Equal("data", dataDir);
        Assert.False(evalOnly);
        Assert.Equal(1UL, o.Seed);
        Assert.Equal(3e-4f, o.LearningRate);
        Assert.False(o.Grow);
        Assert.Equal(2048, o.GrowEvery);
    }

    [Fact]
    public void RushHour_flags_override_the_defaults()
    {
        var o = RushHourLab.Parse(new CliArgs(["--hours", "2", "--data", "runs", "--seed", "7", "--grow", "--eval-only"]),
                                  out double hours, out string dataDir, out bool evalOnly);

        Assert.Equal(2, hours);
        Assert.Equal("runs", dataDir);
        Assert.True(evalOnly);
        Assert.Equal(7UL, o.Seed);
        Assert.True(o.Grow);
    }

    // ── Connect4 ──

    [Fact]
    public void Connect4_defaults_including_the_search_config()
    {
        var o = Connect4Lab.Parse(new CliArgs([]), out double hours, out _, out _);

        Assert.Equal(1, hours);
        Assert.Equal(128, o.Hidden);
        Assert.Equal(100, o.Search.Simulations);
        Assert.Equal(32, o.GamesPerChunk);
        Assert.Equal(20, o.EvalGames);
        Assert.Equal(0, o.OpponentRandomFrac);
    }

    [Fact]
    public void Connect4_sims_flows_into_the_mcts_config()
    {
        var o = Connect4Lab.Parse(new CliArgs(["--sims", "400"]), out _, out _, out _);

        Assert.Equal(400, o.Search.Simulations);
    }

    // ── Cube (imitation) and CubePolicy (EfficientCube) ──

    [Fact]
    public void Cube_defaults()
    {
        var o = CubeLab.Parse(new CliArgs([]), out double hours, out _, out _);

        Assert.Equal(9, hours);
        Assert.Equal(512, o.Width);
        Assert.Equal(4096, o.GrowEvery);
    }

    [Fact]
    public void CubePolicy_defaults_are_the_long_run_settings()
    {
        // 24h and a 2000-wide beam: this entry point is the multi-day EfficientCube run, not a smoke test.
        var o = CubePolicyLab.Parse(new CliArgs([]), out double hours, out _, out _);

        Assert.Equal(24, hours);
        Assert.Equal(512, o.Width);
        Assert.Equal(30, o.MaxScramble);
        Assert.Equal(2_000, o.BeamWidth);
        Assert.Equal(20, o.EvalEpisodes);
        Assert.Equal(50_000, o.GrowEvery);
    }

    // ── FruitCake ──

    [Fact]
    public void FruitCake_defaults()
    {
        var o = FruitCakeLab.Parse(new CliArgs([]), out double hours, out string dataDir, out bool evalOnly, out bool shape);

        Assert.Equal(1, hours);
        Assert.Equal("data", dataDir);
        Assert.False(evalOnly);
        Assert.False(shape);
        Assert.Equal([256, 256], o.Hidden);
        Assert.Equal(0.99, o.Gamma);
        Assert.Equal(1, o.NStep);
        Assert.False(o.Noisy);
        Assert.Equal(2_000, o.ChunkSteps);
    }

    [Fact]
    public void FruitCake_grow_replaces_the_requested_hidden_widths()
    {
        // The least obvious rule in the file: --grow does not ADD to --hidden, it REPLACES it with the tiny
        // first stage, because a growing run must start small and add capacity mid-training.
        var withHidden = FruitCakeLab.Parse(new CliArgs(["--hidden", "512,512"]), out _, out _, out _, out _);
        Assert.Equal([512, 512], withHidden.Hidden);

        var growing = FruitCakeLab.Parse(new CliArgs(["--hidden", "512,512", "--grow"]), out _, out _, out _, out _);
        Assert.True(growing.Grow);
        Assert.NotEqual([512, 512], growing.Hidden);
    }

    [Fact]
    public void FruitCake_shape_is_reported_separately_from_the_options()
    {
        // Shaping applies to the TRAINING env only — eval stays a plain game so A/B judges real merge points.
        FruitCakeLab.Parse(new CliArgs(["--shape"]), out _, out _, out _, out bool shape);

        Assert.True(shape);
    }

    // ── Snake ──

    [Fact]
    public void Snake_dqn_defaults()
    {
        var o = SnakeLab.Options(new CliArgs([]), [128, 128], grow: false);

        Assert.Equal(1UL, o.Seed);
        Assert.Equal(5_000, o.ChunkSteps);
        Assert.Equal(100_000, o.TargetSteps); // the proven M22 budget
        Assert.Equal(20, o.EvalEpisodes);
        Assert.Equal(0.99, o.Gamma);
    }

    [Fact]
    public void Snake_search_config_defaults_are_the_shipped_sweep()
    {
        // These weights ARE the measured snake strength (PR #11's depth/beam sweep). If a default drifts,
        // every reported food-count moves with it.
        var shipped = new SnakeSearchConfig();
        var parsed = SnakeLab.SearchConfig(new CliArgs([]));

        Assert.Equal(shipped.MaxDepth, parsed.MaxDepth);
        Assert.Equal(shipped.BeamWidth, parsed.BeamWidth);
        Assert.Equal(shipped.FoodWeight, parsed.FoodWeight);
        Assert.Equal(shipped.SpaceRatioWeight, parsed.SpaceRatioWeight);
    }

    [Fact]
    public void Snake_search_flags_override_the_shipped_weights()
    {
        var cfg = SnakeLab.SearchConfig(new CliArgs(["--depth", "30", "--beam", "64", "--w-ratio", "2.5"]));

        Assert.Equal(30, cfg.MaxDepth);
        Assert.Equal(64, cfg.BeamWidth);
        Assert.Equal(2.5, cfg.SpaceRatioWeight);
    }

    [Fact]
    public void Snake_no_rebuild_selects_the_fixed_cycle_baseline()
    {
        // --no-rebuild is the M48.1 baseline: the cycle is built once and never repaired per food.
        Assert.True(SnakeLab.CycleConfig(new CliArgs([])).Rebuild);
        Assert.False(SnakeLab.CycleConfig(new CliArgs(["--no-rebuild"])).Rebuild);
    }

    // ── Draughts: the variant decides which checkpoint the run reads and writes ──

    [Theory]
    [InlineData("checkers8")]
    [InlineData("english")]
    [InlineData("english8")]
    [InlineData("ENGLISH")]
    public void All_english_aliases_select_the_8x8_game(string alias)
    {
        var (variant, envId, board) = DraughtsLab.ResolveVariant(alias);

        Assert.Equal(DraughtsVariant.English8, variant);
        Assert.Equal("checkers8", envId);
        Assert.Equal(8, board);
    }

    [Theory]
    [InlineData("international")]
    [InlineData("")]
    [InlineData("something-unknown")]
    public void Anything_else_falls_back_to_international_10x10(string alias)
    {
        var (variant, envId, board) = DraughtsLab.ResolveVariant(alias);

        Assert.Equal(DraughtsVariant.International10, variant);
        Assert.Equal("draughts", envId);
        Assert.Equal(10, board);
    }
}
