using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The parts of <see cref="SelfPlayCampaign{TState}"/>'s lifecycle that <c>SelfPlayProgressTests</c> and
/// <c>SelfPlayLadderTests</c> leave out: what it does <i>before</i> the first chunk, what it reads back out of a
/// damaged progress sidecar, how <c>Evaluate</c> reports a run that has trained nothing yet, and what stops one
/// environment's checkpoints landing on another's.
/// </summary>
/// <remarks>
/// Every test here <b>avoids <c>TrainChunk</c></b> — the two existing files already cover the train-then-resume
/// path, and each chunk is a batch of full MCTS games. The states asserted below are the ones a REAL run is in at
/// its most dangerous moments: the very first checkpoint (before a single game), and the first <c>Resume</c>
/// after a crash.
/// <para>
/// The failures are silent by construction. <c>Evaluate</c> is called by the runner on a cadence and its output
/// is a CSV column set, so a renamed or dropped metric breaks every downstream comparison without an error. The
/// progress sidecar carries <c>_lastWinRate</c>, which is the number <c>MaybePromoteDifficulty</c> compares
/// against — a discarded sidecar does not just reset a counter, it makes the difficulty ladder promote (or
/// refuse to promote) on a win rate the run never measured. And the net id is the bare constant <c>"az"</c>: the
/// environment id is the ONLY separator between two self-play runs, so if it ever stopped scoping the store, a
/// draughts run would resume a chess net of an entirely different shape.
/// </para>
/// </remarks>
public class SelfPlayCampaignLifecycleTests
{
    private const string Env = "connect4";
    private const string ProgressId = "az-progress";
    private const string ProgressKind = "selfplay-progress";

    // Deliberately tiny: an 8-wide net and a single MCTS simulation. Nothing here trains, and the only games
    // played are the two Evaluate arena games, so this is the cheapest configuration that is still a real campaign.
    private static SelfPlayOptions Tiny() => new()
    {
        Seed = 5,
        LearningRate = 1e-3f,
        Hidden = 8,
        Search = new Mcts.Config(Simulations: 1),
        GamesPerChunk = 1,
        TempMoves = 0,
        EvalGames = 2,
        WindowCapacity = 64,
        BatchSize = 16,
        MaxPlies = 42,
    };

    private static SelfPlayCampaign<Connect4State> Campaign(SelfPlayOptions? options = null, string environmentId = Env)
        => new(new Connect4Game(), environmentId, options ?? Tiny());

    private static NetworkMetrics Telemetry(SelfPlayCampaign<Connect4State> campaign)
        => ((INetworkTelemetrySource)campaign).Sample();

    /// <summary>Writes a progress sidecar by hand — the on-disk shape a crashed run left behind.</summary>
    private static void SaveProgress(IModelStore store, long samples, long games, double lastWinRate,
        string kind = ProgressKind, string id = ProgressId)
        => CampaignProgressState.Save(store, Env, id, kind, samples, games, lastWinRate, new Xoshiro256StarStar(99));

    // ── before the first chunk ───────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Evaluating_before_the_net_exists_reports_no_model_instead_of_throwing()
    {
        // `--eval-only` against an empty models dir lands here. The runner calls Evaluate unconditionally, so the
        // no-net case has to be an ordinary (if empty) report rather than a NullReferenceException.
        using var c = Campaign();

        var eval = c.Evaluate();

        Assert.Equal("no model yet (train first)", eval.Summary);
        var metric = Assert.Single(eval.Metrics);
        Assert.Equal("games", metric.Name);
        Assert.Equal(0.0, metric.Value);
    }

    [Fact]
    public void A_checkpoint_taken_before_any_game_is_a_valid_resume_point()
    {
        // A run killed during its first chunk still checkpoints on the way out. A freshly built net is a perfectly
        // good checkpoint, and the resume that follows must load it rather than silently starting over — which
        // would throw away the Adam state and, with the ladder on, the champion's lineage.
        var dir = Directory.CreateTempSubdirectory("m65-sp-first");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = Campaign())
            {
                Assert.False(first.Resume(store));
                first.Checkpoint(store);          // no TrainChunk
            }

            Assert.True(store.Exists(Env, "az"));
            Assert.True(store.Exists(Env, "az-adam"));
            Assert.True(store.Exists(Env, ProgressId));

            using var second = Campaign();
            Assert.True(second.Resume(store));
            Assert.Equal(0, Telemetry(second).Step);   // zero games, and honestly reported as zero
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void The_net_exists_for_the_live_viewer_only_once_resume_has_run()
    {
        var dir = Directory.CreateTempSubdirectory("m65-sp-telemetry");
        try
        {
            using var c = Campaign();
            var telemetry = (INetworkTelemetrySource)c;

            // The viewer polls this from its own thread as soon as the Lab starts — before Resume has built a net.
            Assert.Null(telemetry.SnapshotParameters());

            c.Resume(new FileModelStore(dir.FullName));

            Assert.NotEmpty(telemetry.SnapshotParameters()!);
            Assert.Equal("policy-value", telemetry.NetKind);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Two_environments_never_share_a_self_play_checkpoint()
    {
        // The net id is the bare constant "az" for every self-play run, so the environment id is the only thing
        // keeping chess, draughts and connect4 apart in one models dir.
        var dir = Directory.CreateTempSubdirectory("m65-sp-envs");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = Campaign())
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            using var other = Campaign(environmentId: "connect4-conv");
            Assert.False(other.Resume(store));
            Assert.False(store.Exists("connect4-conv", "az"));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── the progress sidecar ─────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void The_game_counter_and_the_last_win_rate_both_come_back_from_the_sidecar()
    {
        // _lastWinRate is the one that matters: MaybePromoteDifficulty compares it against the champion's stored
        // win rate, so before it was persisted a promotion depended on whether the run had happened to evaluate
        // since the last restart.
        var dir = Directory.CreateTempSubdirectory("m65-sp-restore");
        try
        {
            var store = new FileModelStore(dir.FullName);
            SaveProgress(store, samples: 12_800, games: 250, lastWinRate: 0.625);

            using var c = Campaign();
            c.Resume(store);

            var metrics = Telemetry(c);
            Assert.Equal(250, metrics.Step);
            Assert.Equal(0.625, metrics.Eval);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_stored_win_rate_of_zero_comes_back_as_unknown_rather_than_as_zero()
    {
        // The sidecar carries "no metric yet" as 0, so Resume maps a stored 0 to NaN. That is the documented
        // behaviour and it is what the ladder wants (NaN disables the winRate promotion signal, leaving material
        // and head-to-head to decide) — but it does mean a genuine 0% win rate is indistinguishable from a run
        // that never evaluated. Pinned here so the conflation is a decision, not an accident.
        var dir = Directory.CreateTempSubdirectory("m65-sp-zero");
        try
        {
            var store = new FileModelStore(dir.FullName);
            SaveProgress(store, samples: 500, games: 4, lastWinRate: 0);

            using var c = Campaign();
            c.Resume(store);

            Assert.Equal(4, Telemetry(c).Step);
            Assert.True(double.IsNaN(Telemetry(c).Eval));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_truncated_sidecar_restarts_the_counters_instead_of_failing_the_run()
    {
        var dir = Directory.CreateTempSubdirectory("m65-sp-truncated");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = Campaign())
            {
                first.Resume(store);
                first.Checkpoint(store);
            }

            // Overwrite the sidecar with a valid header and nothing else — a run killed mid-write. The NET is
            // still intact, so the resume must keep it and only lose the counters.
            store.Save(Env, ProgressId, stream =>
            {
                using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
                CheckpointFormat.WriteHeader(writer, ProgressKind, 1);
                writer.Write(7_777L);
            });

            using var second = Campaign();

            Assert.True(second.Resume(store));
            Assert.Equal(0, Telemetry(second).Step);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_sidecar_written_by_a_different_campaign_is_ignored_rather_than_misread()
    {
        // Same file layout, different kind string. Read as if it were self-play's, the cube campaign's round
        // counter would become a game count and its second RNG stream would be read as garbage.
        var dir = Directory.CreateTempSubdirectory("m65-sp-kind");
        try
        {
            var store = new FileModelStore(dir.FullName);
            SaveProgress(store, samples: 4_096, games: 99, lastWinRate: 0.9, kind: "cube-imitation-progress");

            using var c = Campaign();
            c.Resume(store);

            Assert.Equal(0, Telemetry(c).Step);
            Assert.True(double.IsNaN(Telemetry(c).Eval));
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── evaluate ─────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Evaluate_reports_the_documented_metric_set()
    {
        // These names are CSV columns the runner writes for the whole history of a campaign; renaming one silently
        // breaks every comparison against an earlier run.
        var dir = Directory.CreateTempSubdirectory("m65-sp-metrics");
        try
        {
            using var c = Campaign();
            c.Resume(new FileModelStore(dir.FullName));

            string[] expected = ["games", "samples", "winRate", "policyLoss", "valueLoss"];

            var eval = c.Evaluate();
            Assert.Equal(expected, eval.Metrics.Select(m => m.Name));
            Assert.InRange(eval.Metrics.Single(m => m.Name == "winRate").Value, 0.0, 1.0);
            Assert.Contains("winRate-vs-random", eval.Summary);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Evaluate_publishes_its_win_rate_to_the_live_viewer_and_the_next_checkpoint()
    {
        // The single number that travels furthest in this campaign: Evaluate measures it, the viewer shows it, the
        // checkpoint persists it and the ladder promotes on it. This pins the whole hand-off in one go.
        var dir = Directory.CreateTempSubdirectory("m65-sp-winrate");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = Campaign();
            c.Resume(store);

            Assert.True(double.IsNaN(Telemetry(c).Eval));   // nothing measured yet

            double winRate = c.Evaluate().Metrics.Single(m => m.Name == "winRate").Value;
            Assert.Equal(winRate, Telemetry(c).Eval);

            c.Checkpoint(store);
            using var resumed = Campaign();
            resumed.Resume(store);

            // A 0.0 win rate is stored as "none yet" (see the zero test above), so only a non-zero one round-trips.
            if (winRate != 0) Assert.Equal(winRate, Telemetry(resumed).Eval);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Evaluate_reports_a_zero_loss_when_no_batch_has_run()
    {
        // TrainWindow's documented empty-window value is 0, not NaN — unlike the BlockDude campaigns, which report
        // an un-measured rate as NaN so that "no data" and "measured zero" stay distinguishable. Pinned as-is
        // because it is the shipped contract; worth noting that a report full of 0.0000 losses before the window
        // fills is indistinguishable here from a net whose loss genuinely collapsed.
        var dir = Directory.CreateTempSubdirectory("m65-sp-loss");
        try
        {
            using var c = Campaign();
            c.Resume(new FileModelStore(dir.FullName));

            var eval = c.Evaluate();
            Assert.Equal(0.0, eval.Metrics.Single(m => m.Name == "policyLoss").Value);
            Assert.Equal(0.0, eval.Metrics.Single(m => m.Name == "valueLoss").Value);
        }
        finally { dir.Delete(recursive: true); }
    }

    // ── completion ───────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_campaign_with_no_game_target_never_reports_itself_complete()
    {
        // TargetGames defaults to 0, meaning "run to the wall-clock budget". Treating 0 as an already-met target
        // would make every untargeted run — which is all of them by default — exit at the first cadence check.
        var dir = Directory.CreateTempSubdirectory("m65-sp-notarget");
        try
        {
            var store = new FileModelStore(dir.FullName);
            SaveProgress(store, samples: 1_000_000, games: 1_000_000, lastWinRate: 0.9);

            using var c = Campaign();
            c.Resume(store);

            Assert.Equal(1_000_000, Telemetry(c).Step);
            Assert.False(c.IsComplete);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_resumed_game_count_past_the_target_reports_complete_without_playing_a_game()
    {
        // The runner checks IsComplete before the first chunk, so a finished run that is restarted must stop
        // immediately rather than playing one more chunk past its target.
        var dir = Directory.CreateTempSubdirectory("m65-sp-target");
        try
        {
            var store = new FileModelStore(dir.FullName);
            SaveProgress(store, samples: 64_000, games: 5_000, lastWinRate: 0.8);

            using var c = Campaign(Tiny() with { TargetGames = 1_000 });
            c.Resume(store);

            Assert.True(c.IsComplete);
            Assert.Equal(1_000, Telemetry(c).MaxSteps);   // the target the viewer draws its progress bar against
        }
        finally { dir.Delete(recursive: true); }
    }
}
