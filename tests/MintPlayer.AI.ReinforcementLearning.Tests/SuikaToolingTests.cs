extern alias Lab;

using SuikaAb = Lab::SuikaAb;
using SuikaLab = Lab::SuikaLab;
using SuikaSearchEval = Lab::SuikaSearchEval;

using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Suika;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M71 — the Suika paths that had no test before the FruitCake → Suika rename touched every line of them:
/// the one-ply heuristic, potential-based reward shaping, the campaign's grow-input warm-start and telemetry
/// labels, and the Lab's A/B, search-eval and eval-only entry points. Every net here is a tiny <c>[8]</c>
/// trunk and every run is one or two episodes: these pin behaviour and wiring, not playing strength.
/// </summary>
public class SuikaToolingTests
{
    private static DuelingQNet TinyNet(int inputSize = SuikaEnv.ObservationSize)
        => new(inputSize, [8], SuikaEnv.ColumnCount, new Xoshiro256StarStar(7));

    /// <summary>A temp data dir holding a deployable <c>suika.dqn.ckpt</c>, as the Lab and web expect it.</summary>
    private static DirectoryInfo DirWithNet(DuelingQNet net)
    {
        var dir = Directory.CreateTempSubdirectory("m71-suika");
        new FileModelStore(dir.FullName).Save("suika", "dqn", s => DuelingQNetCheckpoint.Save(net, s));
        return dir;
    }

    // ── Environments ──

    [Fact]
    public void Heuristic_picks_a_legal_column_on_an_empty_and_a_busy_board()
    {
        var env = new SuikaEnv();
        env.Reset(3);
        var heuristic = new SuikaHeuristic();

        for (int i = 0; i < 8; i++)
        {
            int col = heuristic.ChooseColumn(env.World, env.CurrentTier);
            Assert.InRange(col, 0, SuikaEnv.ColumnCount - 1);
            if (env.Step(col).Terminated) break;
        }
    }

    [Fact]
    public void Shaped_rewards_differ_from_the_plain_game_but_stay_finite()
    {
        // Same seed, same actions: the shaped env adds the potential difference on top of the merge points.
        var plain = new SuikaEnv();
        var shaped = new SuikaEnv { ShapeRewards = true, ShapingGamma = 0.99 };
        plain.Reset(5);
        shaped.Reset(5);

        bool differed = false;
        foreach (int a in new[] { 6, 6, 7, 6, 5, 6, 7, 6 })
        {
            var p = plain.Step(a);
            var s = shaped.Step(a);
            Assert.True(double.IsFinite(s.Reward));
            if (Math.Abs(p.Reward - s.Reward) > 1e-9) differed = true;
            if (p.Terminated || s.Terminated) break;
        }
        Assert.True(differed, "shaping never changed a reward — Potential() is not being applied");
    }

    // ── Campaign ──

    private static SuikaDqnCampaign Campaign() => new(
        trainEnv: new SuikaEnv(),
        evalEnv: new SuikaEnv(),
        new SuikaDqnOptions { Seed = 1, ChunkSteps = 40, TargetSteps = 40, EvalEpisodes = 1, Hidden = [8], Gamma = 0.99 });

    [Fact]
    public void Campaign_grows_an_older_narrower_net_to_the_current_observation_on_warm_start()
    {
        var dir = DirWithNet(TinyNet(SuikaEnv.ObservationSize - 3)); // trained before 3 inputs were added
        try
        {
            using var c = Campaign();
            Assert.True(c.Resume(new FileModelStore(dir.FullName))); // warm-started from the deployable net

            var layers = NetworkInspector.Layers(((INetworkTelemetrySource)c).SnapshotParameters()!);
            Assert.Equal(SuikaEnv.ObservationSize, layers[0].Weight.Rows);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Campaign_telemetry_labels_and_probe_match_the_env_widths()
    {
        var dir = Directory.CreateTempSubdirectory("m71-suika-telemetry");
        try
        {
            using var c = Campaign();
            var source = (INetworkTelemetrySource)c;
            Assert.Equal(SuikaEnv.ObservationSize, source.InputLabels!.Count);
            Assert.Equal(SuikaEnv.ColumnCount, source.OutputLabels!.Count);

            Assert.False(c.Resume(new FileModelStore(dir.FullName)));
            c.TrainChunk(); // creates the net and a live observation for the probe

            var io = source.SampleIo();
            Assert.NotNull(io);
            Assert.Equal(SuikaEnv.ObservationSize, io!.Value.Input.Length);
            Assert.Equal(SuikaEnv.ColumnCount, io.Value.Output.Length);
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── Lab ──

    [Fact]
    public void Ab_plays_paired_episodes_between_two_checkpoints()
    {
        var baseline = DirWithNet(TinyNet());
        var candidate = DirWithNet(new DuelingQNet(SuikaEnv.ObservationSize, [8], SuikaEnv.ColumnCount, new Xoshiro256StarStar(8)));
        var saved = Backend.Current;
        try
        {
            SuikaAb.Run(baseline.FullName, candidate.FullName, episodes: 2, seedBase: 1);
        }
        finally
        {
            Backend.Current = saved;
            baseline.Delete(recursive: true);
            candidate.Delete(recursive: true);
        }
    }

    // "blend" = net leaf + tier potential, so it exercises both leaf functions; the "height" leaf is covered by
    // Lab_dispatches_ab_search_eval_and_eval_only_modes (each search game costs ~10 s, so no duplicate run here).
    [Theory]
    [InlineData("blend")]
    public void Search_eval_plays_greedy_vs_search_for_a_leaf(string leaf)
    {
        var dir = DirWithNet(TinyNet());
        var saved = Backend.Current;
        try
        {
            SuikaSearchEval.Run(dir.FullName, episodes: 1, depth: 1, topK: 2, seedBase: 1, leaf, topK2: 1);
        }
        finally
        {
            Backend.Current = saved;
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Ab_and_search_eval_report_a_missing_checkpoint_by_name()
    {
        var empty = Directory.CreateTempSubdirectory("m71-suika-empty");
        try
        {
            var ab = Assert.Throws<FileNotFoundException>(() => SuikaAb.Run(empty.FullName, empty.FullName, 1, 1));
            Assert.Contains("suika.dqn.ckpt", ab.Message);
            var se = Assert.Throws<FileNotFoundException>(() => SuikaSearchEval.Run(empty.FullName, 1, 1, 2, 1));
            Assert.Contains("suika.dqn.ckpt", se.Message);
        }
        finally { empty.Delete(recursive: true); }
    }

    [Fact]
    public void Lab_dispatches_ab_search_eval_and_eval_only_modes()
    {
        var dir = DirWithNet(TinyNet());
        var saved = Backend.Current;
        try
        {
            string d = dir.FullName;
            SuikaLab.Run(["--ab", "--baseline", d, "--data", d, "--ab-episodes", "1"]);
            SuikaLab.Run(["--search-eval", "--data", d, "--ab-episodes", "1", "--depth", "1", "--topk", "2", "--topk2", "1", "--leaf", "height"]);
            // Eval-only: builds the host + campaign from the flags, warm-starts from the checkpoint, evaluates once.
            SuikaLab.Run(["--eval-only", "--data", d, "--episodes", "1", "--hidden", "8"]);
        }
        finally
        {
            Backend.Current = saved;
            dir.Delete(recursive: true);
        }
    }
}
