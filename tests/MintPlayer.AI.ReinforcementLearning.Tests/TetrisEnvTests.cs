using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M54.2 gates (docs/prd/TETRIS_PRD.md §5): the env contract — spaces, masking, reward = lines,
/// TERMINATION on top-out (unlike the other games' never-dying boards, the Tetris mask can legitimately
/// go all-false — risk 2), truncation at the piece budget, state round-trip — and the campaign's
/// resume → train → checkpoint → resume contract on the real env (whose random exploration top-outs
/// every ~25 placements, exercising terminal transitions in the trainer).
/// </summary>
public class TetrisEnvTests
{
    [Fact]
    public void Contract_SpacesMaskRewardAndBudgetTruncation()
    {
        var env = new TetrisEnv(pieceBudget: 5);
        var (obs, _) = env.Reset(7);
        Assert.Equal(TetrisEnv.ObservationSize, obs.Length);

        for (int piece = 0; piece < 5; piece++)
        {
            var mask = env.CurrentActionMask();
            Assert.Equal(TetrisEnv.ActionCount, mask.Length);
            int action = Array.IndexOf(mask, true);
            Assert.True(action >= 0); // a fresh/low board always has placements

            int linesBefore = env.Lines;
            var step = env.Step(action);
            Assert.False(step.Terminated); // 5 pieces cannot top out a fresh board
            Assert.Equal(piece == 4, step.Truncated);
            int cleared = env.Lines - linesBefore;
            // Hybrid reward (owner amendment 2026-08-26): lines + the tetris bonus.
            Assert.Equal(cleared + (cleared == 4 ? TetrisBoard.TetrisRewardBonus : 0), step.Reward, 5);
        }
        Assert.Throws<InvalidOperationException>(() => env.Step(0));
    }

    [Fact]
    public void TopOut_IsTermination_AndTheMaskGoesAllFalse()
    {
        // Random play under garbage/10 dies in ~22 pieces — well inside the budget, so the episode must
        // end TERMINATED (not truncated), with an all-false mask on the dead board.
        var env = new TetrisEnv(pieceBudget: 5_000, garbageEvery: 10);
        env.Reset(3);
        bool terminated = false;
        for (int step = 0; step < 5_000; step++)
        {
            int action = env.Board.RandomAction(0xDEAD, step);
            var result = env.Step(action);
            if (result.Done)
            {
                terminated = result.Terminated;
                break;
            }
        }
        Assert.True(terminated, "random play under garbage must top out (terminated), never reach the budget");
        Assert.DoesNotContain(true, env.CurrentActionMask());
    }

    [Fact]
    public void IllegalAction_Throws_WithoutConsumingThePiece()
    {
        var env = new TetrisEnv();
        env.Reset(3);
        // Rotation 3 of the current piece may be legal; a guaranteed-illegal action is rot 3 col 9 for
        // pieces whose width at that rotation overflows — search the mask instead of guessing.
        int illegal = Array.IndexOf(env.CurrentActionMask(), false);
        Assert.True(illegal >= 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => env.Step(illegal));
        Assert.Equal(0, env.PiecesPlaced);
    }

    [Fact]
    public void StateRoundTrip_ContinuesIdentically_AcrossEpisodeBoundaries()
    {
        var env = new TetrisEnv(pieceBudget: 8, sevenBag: true, garbageEvery: 3);
        env.Reset(11);
        for (int piece = 0; piece < 5; piece++)
            env.Step(env.Board.DellacherieAction());

        var saved = env.SaveState();
        var restored = new TetrisEnv(pieceBudget: 8, sevenBag: true, garbageEvery: 3);
        restored.RestoreState(saved);

        for (int piece = 0; piece < 3; piece++)
        {
            int action = env.Board.DellacherieAction();
            Assert.Equal(action, restored.Board.DellacherieAction());
            Assert.Equal(env.Step(action).Reward, restored.Step(action).Reward);
        }
        var (nextA, _) = env.Reset();
        var (nextB, _) = restored.Reset();
        Assert.Equal(nextA, nextB);
    }

    // ── Campaign: DI registration + resume → train → checkpoint → resume on the real env ────────────────────

    [Fact]
    public void Campaign_Registers_TrainsCheckpointsAndResumes()
    {
        var services = new ServiceCollection();
        services.AddTetrisDqnCampaign(new TetrisEnv(), new TetrisEnv(),
            new TetrisDqnOptions { Hidden = [32, 32] });
        using (var provider = services.BuildServiceProvider())
            Assert.IsType<TetrisDqnCampaign>(provider.GetRequiredService<ITrainingCampaign>());

        var dir = Directory.CreateTempSubdirectory("tetris-campaign");
        try
        {
            var store = new FileModelStore(dir.FullName);
            // 60+60 random-heavy placements span 2–4 episodes incl. top-outs — the trainer sees terminal
            // transitions with all-false next masks (risk 2) inside this contract run.
            var options = new TetrisDqnOptions { Seed = 1, ChunkSteps = 60, TargetSteps = 120, Hidden = [32, 32], EvalEpisodes = 2 };

            var c1 = new TetrisDqnCampaign(evalEnv: new TetrisEnv(pieceBudget: 50), trainEnv: new TetrisEnv(), options: options, logger: null);
            Assert.False(c1.Resume(store));
            Assert.Equal(60, c1.TrainChunk());
            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "lines");
            c1.Checkpoint(store);
            c1.Dispose();
            using (var state = store.TryOpenRead("tetris", "dqn-state")) Assert.NotNull(state);

            var c2 = new TetrisDqnCampaign(evalEnv: new TetrisEnv(pieceBudget: 50), trainEnv: new TetrisEnv(), options: options, logger: null);
            Assert.True(c2.Resume(store));
            Assert.Equal(120, c2.TrainChunk());
            Assert.True(c2.IsComplete);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    /// <summary>
    /// M57.5 anti-drift gate. The net's dense regression target is reconstructed in C# from the
    /// observation planes, while the search tiers score placements with the engine's own
    /// <c>evalAfterstate</c>. If those two ever disagree, the net is being distilled from a DIFFERENT
    /// evaluator than the one that plays — the exact failure mode M54 carried (a hand-written inverse of
    /// the plane normalizers, correct only up to a per-state constant). This pins them together.
    /// </summary>
    [Fact]
    public void DenseTargets_ReconstructTheEnginesOwnEvaluatorExactly()
    {
        int checkedActions = 0;
        for (int e = 0; e < 12; e++)
        {
            var board = new TetrisBoard();
            board.Reset((ulong)(4100 + e), sevenBag: false, garbageEvery: e % 3 == 0 ? 10 : 0);

            // walk into a non-trivial mid-game position (garbage episodes exercise the DIG branch)
            for (int i = 0; i < 30 + e * 5 && !board.GameOver; i++) board.ApplyPlacement(board.DellacherieAction());
            if (board.GameOver) continue;

            var obs = board.BuildObservation();
            Assert.Equal(TetrisBoard.ObservationSize, obs.Length);

            var targets = TetrisDqnCampaign.DenseTargetsFromObservation(obs);
            var mask = board.LegalMask();

            for (int a = 0; a < TetrisBoard.ActionCount; a++)
            {
                if (!mask[a])
                {
                    Assert.True(float.IsNaN(targets[a]), $"action {a} is illegal but got a supervised target");
                    continue;
                }
                double engine = board.DellaScore(a / TetrisBoard.Width, a % TetrisBoard.Width);
                Assert.False(float.IsNaN(targets[a]), $"action {a} is legal but the target is NaN");
                Assert.True(Math.Abs(targets[a] * 10.0 - engine) < 2e-2,
                    $"seed {4100 + e} action {a}: dense target {targets[a] * 10.0:F4} != engine evaluator {engine:F4}");
                checkedActions++;
            }
        }
        Assert.True(checkedActions > 200, $"only {checkedActions} legal actions exercised — the fixture is too weak");
    }
}
