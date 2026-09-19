extern alias Lab; // the Lab exe's internals (aliased so its top-level Program can't collide with RLDemo.Web's)

using VizLauncher = Lab::VizLauncher;
using CrazyFruitsLab = Lab::CrazyFruitsLab;
using BlockDudeLevelBench = Lab::BlockDudeLevelBench;
using BlockDudeLab = Lab::BlockDudeLab;
using VizServer = Lab::VizServer;

using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M63.5 — the Lab's pure helpers. Until now the Lab was 10/1,715 coverable lines (0.6%): every game's
/// flag parsing and verdict formatting sits inside a lambda handed to <c>LabHost.Run</c>, so it could only
/// be reached by booting DI and starting a training run.
/// <para>These are the helpers that needed no refactor at all — only widening from <c>private</c> to
/// <c>internal</c>, which <c>InternalsVisibleTo</c> in the Lab csproj already permits.</para>
/// <para>Deliberately absent: anything that binds a socket, loads a checkpoint, or calls
/// <c>LabHost.Run</c> (it takes an exclusive directory lock). See PRD §12.3.</para>
/// </summary>
public class LabHelpersTests
{
    // ── VizLauncher.ParsePort — decides whether the training run opens a viewer socket at all ──

    [Fact]
    public void ParsePort_returns_zero_when_viz_is_absent()
    {
        // 0 is the "--viz not passed" sentinel TryStart short-circuits on, so this is the branch that
        // keeps an ordinary training run from opening a port.
        Assert.Equal(0, VizLauncher.ParsePort([]));
        Assert.Equal(0, VizLauncher.ParsePort(["--game", "tetris", "--hours", "4"]));
    }

    [Fact]
    public void Bare_viz_defaults_to_5250()
    {
        Assert.Equal(5250, VizLauncher.ParsePort(["--viz"]));
        Assert.Equal(5250, VizLauncher.ParsePort(["--game", "snake", "--viz"]));
    }

    [Fact]
    public void Viz_takes_an_explicit_port()
    {
        Assert.Equal(1234, VizLauncher.ParsePort(["--viz", "1234"]));
        Assert.Equal(1234, VizLauncher.ParsePort(["--game", "chess", "--viz", "1234", "--hours", "2"]));
    }

    [Fact]
    public void Viz_followed_by_another_flag_falls_back_to_the_default_port()
    {
        // `--viz --seed 7` must not swallow "--seed" as a port. TryParse fails, so it degrades to 5250
        // rather than throwing in the middle of a long training launch.
        Assert.Equal(5250, VizLauncher.ParsePort(["--viz", "--seed", "7"]));
        Assert.Equal(5250, VizLauncher.ParsePort(["--viz", "not-a-number"]));
    }

    [Fact]
    public void First_viz_wins_when_it_is_repeated()
    {
        Assert.Equal(1111, VizLauncher.ParsePort(["--viz", "1111", "--viz", "2222"]));
    }

    // ── CrazyFruitsLab.Summarize — mean and 95% CI half-width behind every baseline verdict ──

    [Fact]
    public void Summarize_reports_the_mean_and_a_zero_interval_for_a_constant_sample()
    {
        // Ten episodes that all scored 5: mean 5, and no spread, so the interval must collapse to 0.
        var (name, mean, ci) = CrazyFruitsLab.Summarize("net", 10, sum: 50, sumSq: 250);

        Assert.Equal("net", name);
        Assert.Equal(5.0, mean, 12);
        Assert.Equal(0.0, ci, 12);
    }

    [Fact]
    public void Summarize_widens_the_interval_with_the_spread()
    {
        // Same mean, non-zero variance -> a strictly positive half-width. This is the quantity the
        // "CI-SEPARATED" / "OVERLAPPING" gate verdicts are computed from, so a sign or divisor slip here
        // silently changes a ship decision.
        var (_, mean, ci) = CrazyFruitsLab.Summarize("greedy", 10, sum: 50, sumSq: 300);

        Assert.Equal(5.0, mean, 12);
        Assert.True(ci > 0, $"expected a positive CI half-width, got {ci}");
    }

    [Fact]
    public void Summarize_of_a_single_episode_does_not_produce_a_negative_interval()
    {
        // n = 1 is the degenerate case: variance is undefined. Whatever it reports, it must not be a
        // negative half-width, which would invert every downstream comparison.
        var (_, mean, ci) = CrazyFruitsLab.Summarize("solo", 1, sum: 7, sumSq: 49);

        Assert.Equal(7.0, mean, 12);
        Assert.False(ci < 0, $"CI half-width must never be negative, got {ci}");
    }

    // ── BlockDudeLevelBench.Emit / ExistingLength — the shortest-solution-wins bookkeeping ──

    [Fact]
    public void Emit_adds_a_level_that_is_not_present_yet()
    {
        var lines = new List<string>();

        BlockDudeLevelBench.Emit(lines, "level-1", [0, 1, 2]);

        Assert.Single(lines);
        Assert.StartsWith("level-1 · 3 moves · ", lines[0]);
    }

    [Fact]
    public void Emit_replaces_an_existing_entry_only_when_the_new_solution_is_shorter()
    {
        var lines = new List<string>();
        BlockDudeLevelBench.Emit(lines, "level-1", [0, 1, 2, 3, 4]);

        BlockDudeLevelBench.Emit(lines, "level-1", [0, 1]);          // shorter -> wins
        Assert.Single(lines);
        Assert.StartsWith("level-1 · 2 moves · ", lines[0]);

        BlockDudeLevelBench.Emit(lines, "level-1", [0, 1, 2, 3]);    // longer -> ignored
        Assert.Single(lines);
        Assert.StartsWith("level-1 · 2 moves · ", lines[0]);
    }

    [Fact]
    public void Emit_keeps_the_incumbent_on_an_exact_tie()
    {
        // A tie must NOT overwrite: the cheap greedy pass runs first, so a tying search tier rewriting
        // the line would make the emitted solutions depend on tier order rather than on length.
        var lines = new List<string>();
        BlockDudeLevelBench.Emit(lines, "level-1", [0, 1]);
        string first = lines[0];

        BlockDudeLevelBench.Emit(lines, "level-1", [3, 3]);

        Assert.Single(lines);
        Assert.Equal(first, lines[0]);
    }

    [Fact]
    public void Emit_keeps_levels_with_distinct_names_apart()
    {
        var lines = new List<string>();

        BlockDudeLevelBench.Emit(lines, "level-1", [0]);
        BlockDudeLevelBench.Emit(lines, "level-2", [0, 1]);

        Assert.Equal(2, lines.Count);
    }

    [Fact]
    public void ExistingLength_reads_the_move_count_back_out_of_an_emitted_line()
    {
        var lines = new List<string>();
        BlockDudeLevelBench.Emit(lines, "level-7", [0, 1, 2, 3]);

        Assert.Equal(4, BlockDudeLevelBench.ExistingLength(lines[0]));
    }

    [Fact]
    public void ExistingLength_treats_an_unparseable_line_as_infinitely_long()
    {
        // MaxValue is the safe direction: a malformed incumbent always loses to a real solution rather
        // than permanently blocking it.
        Assert.Equal(int.MaxValue, BlockDudeLevelBench.ExistingLength("garbage"));
        Assert.Equal(int.MaxValue, BlockDudeLevelBench.ExistingLength(string.Empty));
    }

    // ── VizServer.Envelope — the WebSocket frame contract the viewer's JS parses ──

    [Fact]
    public void Envelope_wraps_already_serialised_json_without_re_encoding_it()
    {
        // The payload is spliced in verbatim rather than serialised a second time. If this ever started
        // double-encoding, the viewer would receive a JSON *string* where it expects an object and every
        // frame would silently stop rendering.
        byte[] bytes = VizServer.Envelope("frame", """{"step":7}""");

        Assert.Equal("""{"type":"frame","data":{"step":7}}""", Encoding.UTF8.GetString(bytes));
    }

    [Fact]
    public void Envelope_emits_utf8_bytes()
    {
        byte[] bytes = VizServer.Envelope("topology", "{}");

        Assert.Equal("""{"type":"topology","data":{}}""", Encoding.UTF8.GetString(bytes));
        Assert.Equal(Encoding.UTF8.GetByteCount("""{"type":"topology","data":{}}"""), bytes.Length);
    }

    // ── VizServer.CurrentTopology — must tolerate a net that does not exist yet ──

    [Fact]
    public void CurrentTopology_is_null_before_the_net_exists()
    {
        // SnapshotParameters() returns null until training has actually built the net. The viewer may
        // connect first, so this path runs on every real --viz launch.
        var server = new VizServer(0, new NoNetSource(), intervalMs: 1000);

        Assert.Null(server.CurrentTopology());
    }

    [Fact]
    public void CurrentTopology_survives_a_throwing_telemetry_source()
    {
        // A campaign mid-resize can throw from SnapshotParameters. The viewer must degrade to "no frame",
        // never take down the training run it is observing.
        var server = new VizServer(0, new ThrowingSource(), intervalMs: 1000);

        Assert.Null(server.CurrentTopology());
    }

    private sealed class NoNetSource : INetworkTelemetrySource
    {
        public string NetKind => "test";
        public IReadOnlyList<Tensor>? SnapshotParameters() => null;
        public NetworkMetrics Sample() => new();
    }

    private sealed class ThrowingSource : INetworkTelemetrySource
    {
        public string NetKind => "test";
        public IReadOnlyList<Tensor>? SnapshotParameters() => throw new InvalidOperationException("resizing");
        public NetworkMetrics Sample() => new();
    }

    // ── BlockDudeLab.RotateLog — stops two runs blending into one CSV ──

    [Fact]
    public void RotateLog_is_a_no_op_when_there_is_no_previous_log()
    {
        string dir = Path.Combine(Path.GetTempPath(), "m63-rotate-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string csv = Path.Combine(dir, "blockdude.csv");

            BlockDudeLab.RotateLog(csv); // must not throw on a blank slate

            Assert.Empty(Directory.GetFiles(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void RotateLog_moves_an_existing_log_aside_rather_than_deleting_it()
    {
        // CampaignCli APPENDS when the file exists, so without the rotation a --fresh run would continue
        // the previous run's log and present two different trajectories as one.
        string dir = Path.Combine(Path.GetTempPath(), "m63-rotate-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string csv = Path.Combine(dir, "blockdude.csv");
            File.WriteAllText(csv, "step,loss\n1,0.5\n");

            BlockDudeLab.RotateLog(csv);

            Assert.False(File.Exists(csv));
            var rotated = Directory.GetFiles(dir, "blockdude.*.csv");
            Assert.Single(rotated);
            Assert.Equal("step,loss\n1,0.5\n", File.ReadAllText(rotated[0]).Replace("\r\n", "\n"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
