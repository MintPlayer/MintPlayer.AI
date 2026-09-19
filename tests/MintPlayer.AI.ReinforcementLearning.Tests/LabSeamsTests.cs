extern alias Lab;

using System.Text.Json;
using CampaignCli = Lab::CampaignCli;
using CliArgs = Lab::CliArgs;
using CrazyFruitsLab = Lab::CrazyFruitsLab;
using CubeDaviConfig = Lab::CubeDaviConfig;
using CubeDaviLab = Lab::CubeDaviLab;
using EvalStats = Lab::EvalStats;

using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M63.5 — the Lab seams extracted in PRD §12.2 items 1, 3 and 6, plus the §12.5 bug fixes.
/// <para>Every one of these was previously reachable only by starting a real training run, because the
/// parsing and the verdict formatting lived inside a lambda handed to <c>LabHost.Run</c>.</para>
/// </summary>
public class LabSeamsTests
{
    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "m63-seam-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    // ── CubeDaviConfig.LoadFrom — the appsettings half of the precedence contract ──

    [Fact]
    public void Config_is_empty_when_no_appsettings_exists()
    {
        string dir = NewTempDir();
        try
        {
            var cfg = CubeDaviConfig.LoadFrom([dir], out string? source);

            Assert.Null(source);
            Assert.Null(cfg.Hours);
            Assert.Null(cfg.Width);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Config_maps_json_keys_onto_the_flag_names()
    {
        string dir = NewTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "appsettings.json"),
                """{ "cube-davi": { "hours": 3.5, "width": 2048, "layers": 4, "net": "residual", "probe-depths": [4, 8] } }""");

            var cfg = CubeDaviConfig.LoadFrom([dir], out string? source);

            Assert.NotNull(source);
            Assert.Equal(3.5, cfg.Hours);
            Assert.Equal(2048, cfg.Width);
            Assert.Equal(4, cfg.Layers);
            Assert.Equal("residual", cfg.Net);
            Assert.Equal([4, 8], cfg.ProbeDepths);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Config_tolerates_comments_and_trailing_commas_so_the_file_can_document_itself()
    {
        string dir = NewTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "appsettings.json"),
                """
                {
                  // the long-lived cube-davi run
                  "cube-davi": {
                    "hours": 9,
                  },
                }
                """);

            var cfg = CubeDaviConfig.LoadFrom([dir], out _);

            Assert.Equal(9, cfg.Hours);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_file_without_the_cube_davi_section_does_not_stop_the_search()
    {
        // The first directory has an appsettings.json that simply isn't about cube-davi; the loader must
        // keep scanning rather than treat it as "found, empty".
        string first = NewTempDir();
        string second = NewTempDir();
        try
        {
            File.WriteAllText(Path.Combine(first, "appsettings.json"), """{ "Logging": { "LogLevel": "Information" } }""");
            File.WriteAllText(Path.Combine(second, "appsettings.json"), """{ "cube-davi": { "hours": 12 } }""");

            var cfg = CubeDaviConfig.LoadFrom([first, second], out string? source);

            Assert.Equal(12, cfg.Hours);
            Assert.StartsWith(second, source);
        }
        finally { Directory.Delete(first, true); Directory.Delete(second, true); }
    }

    [Fact]
    public void A_malformed_file_falls_back_to_defaults_instead_of_crashing_the_run()
    {
        // A long-running training launch must not die on a stray comma in a config file.
        string dir = NewTempDir();
        try
        {
            File.WriteAllText(Path.Combine(dir, "appsettings.json"), "{ this is not json");

            var cfg = CubeDaviConfig.LoadFrom([dir], out _);

            Assert.Null(cfg.Hours);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ── CubeDaviLab.Resolve — code defaults → appsettings → CLI, in that order ──

    [Fact]
    public void Resolve_uses_the_in_code_defaults_when_nothing_overrides_them()
    {
        var settings = CubeDaviLab.Resolve([], new CubeDaviConfig(), out double hours, out string dataDir, out bool evalOnly);

        Assert.Equal(9, hours);
        Assert.Equal("data", dataDir);
        Assert.False(evalOnly);
        Assert.Equal(1024, settings.Width);
        Assert.Equal(2, settings.HiddenLayers);
        Assert.False(settings.Residual);
    }

    [Fact]
    public void Appsettings_overrides_the_in_code_defaults()
    {
        var cfg = JsonSerializer.Deserialize<CubeDaviConfig>("""{ "hours": 3, "width": 512, "net": "residual" }""")!;

        var settings = CubeDaviLab.Resolve([], cfg, out double hours, out _, out _);

        Assert.Equal(3, hours);
        Assert.Equal(512, settings.Width);
        Assert.True(settings.Residual);
    }

    [Fact]
    public void A_cli_flag_beats_appsettings()
    {
        // The whole point of the three-tier contract: the file holds the long-lived run, a flag still wins
        // for a one-off override.
        var cfg = JsonSerializer.Deserialize<CubeDaviConfig>("""{ "hours": 3, "width": 512 }""")!;

        var settings = CubeDaviLab.Resolve(["--hours", "1", "--width", "256"], cfg, out double hours, out _, out _);

        Assert.Equal(1, hours);
        Assert.Equal(256, settings.Width);
    }

    [Fact]
    public void Resolve_reads_the_flags_that_change_what_gets_written()
    {
        var settings = CubeDaviLab.Resolve(
            ["--data", "runs/cube", "--seed", "42", "--eval-only"],
            new CubeDaviConfig(), out _, out string dataDir, out bool evalOnly);

        Assert.Equal("runs/cube", dataDir);
        Assert.True(evalOnly);
        Assert.Equal(42UL, settings.Seed);
        Assert.Equal(Path.Combine("runs/cube", "logs"), settings.LogDirectory);
    }

    [Fact]
    public void Resolve_parses_a_comma_separated_probe_depth_list()
    {
        var settings = CubeDaviLab.Resolve(["--probe-depths", "4,8,12"], new CubeDaviConfig(), out _, out _, out _);

        Assert.Equal([4, 8, 12], settings.ProbeOverride);
    }

    // ── CrazyFruitsLab.GateLines — the M49/M50 PRD gate verdicts ──

    private static List<(string, double, double)> Rows(double random, double greedy, double e1, double e2, double? net = null)
    {
        var rows = new List<(string, double, double)>
        {
            ("random", random, 1),
            ("greedy", greedy, 1),
            ("filler", 0, 1),
            ("expectimax-1", e1, 1),
            ("expectimax-2", e2, 1),
        };
        if (net is { } n) rows.Add(("net", n, 1));
        return rows;
    }

    [Fact]
    public void GateLines_reports_nothing_for_a_short_result_set()
    {
        // The rows are positional (index 3 and 4 are the expectimax arms). Previously those were read
        // BEFORE any length check, so a truncated run threw instead of reporting nothing.
        Assert.Empty(CrazyFruitsLab.GateLines([]));
        Assert.Empty(CrazyFruitsLab.GateLines([("random", 1, 1), ("greedy", 2, 1)]));
    }

    [Fact]
    public void GateLines_separates_greedy_from_random_when_the_intervals_do_not_overlap()
    {
        var lines = CrazyFruitsLab.GateLines(Rows(random: 100, greedy: 200, e1: 300, e2: 330)).ToList();

        Assert.Contains(lines, l => l.StartsWith("greedy vs random: CI-SEPARATED"));
    }

    [Fact]
    public void GateLines_calls_out_an_environment_that_is_too_self_firing()
    {
        // The pre-training env validation: if random already scores ≥70% of expectimax-2, the specials fire
        // themselves and there is no skill landscape left to learn. Training on that is wasted.
        var tooEasy = CrazyFruitsLab.GateLines(Rows(random: 90, greedy: 95, e1: 100, e2: 100)).ToList();
        Assert.Contains(tooEasy, l => l.Contains("TOO SELF-FIRING"));

        var ok = CrazyFruitsLab.GateLines(Rows(random: 50, greedy: 90, e1: 100, e2: 100)).ToList();
        Assert.Contains(ok, l => l.Contains("OK (< 70%)"));
    }

    [Fact]
    public void GateLines_adds_the_net_rows_only_when_a_net_was_evaluated()
    {
        var withoutNet = CrazyFruitsLab.GateLines(Rows(100, 200, 300, 330)).ToList();
        Assert.DoesNotContain(withoutNet, l => l.StartsWith("net vs random"));

        var withNet = CrazyFruitsLab.GateLines(Rows(100, 200, 300, 330, net: 250)).ToList();
        Assert.Contains(withNet, l => l.StartsWith("net vs random"));
        Assert.Contains(withNet, l => l.StartsWith("net gap share"));
    }

    [Fact]
    public void Net_gap_share_measures_the_net_against_the_random_to_expectimax1_span()
    {
        // net 200 sits exactly halfway between random 100 and expectimax-1 300 -> 50%.
        var lines = CrazyFruitsLab.GateLines(Rows(random: 100, greedy: 150, e1: 300, e2: 330, net: 200)).ToList();

        Assert.Contains(lines, l => l.StartsWith("net gap share") && l.Contains("50"));
    }

    // ── EvalStats — the shared statistics behind the ship/no-ship verdict (§12.5 bugs 1 and 4) ──

    [Fact]
    public void Std_uses_the_sample_divisor_not_the_population_one()
    {
        // [2,4,4,4,5,5,7,9] has population SD 2 and sample SD ~2.1381. The old code returned 2, which
        // understated every standard error derived from it.
        double[] xs = [2, 4, 4, 4, 5, 5, 7, 9];

        Assert.Equal(2.13809, EvalStats.Std(xs, xs.Average()), 4);
    }

    [Fact]
    public void Std_of_fewer_than_two_samples_is_zero_not_NaN()
    {
        // NaN would poison every comparison downstream without ever failing loudly.
        Assert.Equal(0, EvalStats.Std([], 0));
        Assert.Equal(0, EvalStats.Std([5.0], 5));
    }

    [Fact]
    public void Median_averages_the_middle_pair_for_an_even_count()
    {
        // The old code took sorted[Length/2], the UPPER middle, biasing every even-sized run high.
        Assert.Equal(3.0, EvalStats.Median([1, 2, 4, 5]), 12);
        Assert.Equal(4.0, EvalStats.Median([1, 2, 4, 5, 9]), 12);
    }

    [Fact]
    public void Verdict_ships_only_on_a_difference_beyond_two_standard_errors()
    {
        Assert.Contains("SIGNIFICANTLY BETTER", EvalStats.Verdict(meanDiff: 10, se: 1));
        Assert.Contains("SIGNIFICANTLY WORSE", EvalStats.Verdict(meanDiff: -10, se: 1));
        Assert.Contains("NO significant difference", EvalStats.Verdict(meanDiff: 1, se: 1));
    }

    [Fact]
    public void Verdict_never_ships_on_a_degenerate_zero_standard_error()
    {
        // se == 0 means fewer than two episodes. Without the guard, any positive difference would read as
        // infinitely significant and ship a candidate measured once.
        Assert.Contains("NO significant difference", EvalStats.Verdict(meanDiff: 500, se: 0));
    }

    // ── §12.5 bug 2: CliArgs must not swallow the next flag as a value ──

    [Fact]
    public void A_flag_followed_by_another_flag_falls_back_to_its_default()
    {
        // `--data --seed 7` previously made Str("--data") return "--seed", so the run wrote its checkpoints
        // into a directory literally named "--seed".
        var a = new CliArgs(["--data", "--seed", "7"]);

        Assert.Equal("data", a.Str("--data", "data"));
        Assert.Equal(7UL, a.ULong("--seed", 1));
    }

    [Fact]
    public void A_negative_number_is_still_a_valid_value()
    {
        var a = new CliArgs(["--offset", "-5"]);

        Assert.Equal(-5, a.Int("--offset", 0));
    }

    // ── §12.5 bug 3: a zero-byte CSV is not a headered CSV ──

    [Fact]
    public void An_empty_csv_still_gets_its_header()
    {
        string dir = NewTempDir();
        try
        {
            string csv = Path.Combine(dir, "run.csv");
            File.WriteAllText(csv, string.Empty); // a crashed run leaves exactly this

            var onEval = CampaignCli.ConsoleAndCsv(csv);
            onEval(Progress(("score", 1.5)));

            string[] lines = File.ReadAllLines(csv);
            Assert.StartsWith("utc,score", lines[0]);
            Assert.Equal(2, lines.Length);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void An_existing_headered_csv_is_appended_to_rather_than_re_headered()
    {
        string dir = NewTempDir();
        try
        {
            string csv = Path.Combine(dir, "run.csv");
            File.WriteAllText(csv, "utc,score\n2026-01-01,1\n");

            var onEval = CampaignCli.ConsoleAndCsv(csv);
            onEval(Progress(("score", 2.0)));

            string[] lines = File.ReadAllLines(csv);
            Assert.Equal(3, lines.Length);
            Assert.Single(lines, l => l.StartsWith("utc,"));
        }
        finally { Directory.Delete(dir, true); }
    }

    private static CampaignProgress Progress(params (string Name, double Value)[] metrics)
        => new(
            Progress: 1,
            Eval: new CampaignEval([.. metrics.Select(m => new CampaignMetric(m.Name, m.Value))], Summary: "test"),
            IsFinal: false);
}
