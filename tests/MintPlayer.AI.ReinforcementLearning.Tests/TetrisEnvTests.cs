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
            // Hybrid reward (owner amendment 2026-08-26): lines + the tetris bonus, over RewardScale.
            // NOTE: five pieces on a fresh board never clear a line, so this arm is vacuous here — the
            // scaling itself is pinned by RewardScale_DividesTheHybridReward below.
            Assert.Equal((cleared + (cleared == 4 ? TetrisBoard.TetrisRewardBonus : 0)) / TetrisEnv.RewardScale, step.Reward, 5);
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

            double sum = 0;
            int n = 0;
            var engine = new double[TetrisBoard.ActionCount];
            for (int a = 0; a < TetrisBoard.ActionCount; a++)
            {
                if (!mask[a])
                {
                    Assert.True(float.IsNaN(targets[a]), $"action {a} is illegal but got a supervised target");
                    continue;
                }
                Assert.False(float.IsNaN(targets[a]), $"action {a} is legal but the target is NaN");
                engine[a] = board.DellaScore(a / TetrisBoard.Width, a % TetrisBoard.Width);
                sum += engine[a];
                n++;
            }
            if (n == 0) continue;

            // Targets are CENTRED on the mean legal value (the dueling V head absorbs any per-state
            // constant, and centring is what keeps the target scale sane — see the campaign comment).
            // So the invariant is on the RANKING: target == (engineValue - meanLegal) / 10.
            double mean = sum / n;
            for (int a = 0; a < TetrisBoard.ActionCount; a++)
            {
                if (!mask[a]) continue;
                double expected = (engine[a] - mean) / 10.0;
                Assert.True(Math.Abs(targets[a] - expected) < 2e-2,
                    $"seed {4100 + e} action {a}: dense target {targets[a]:F4} != centred engine value {expected:F4}");
                checkedActions++;
            }
        }
        Assert.True(checkedActions > 200, $"only {checkedActions} legal actions exercised — the fixture is too weak");
    }

    /// <summary>
    /// Pins the hybrid reward on a REAL line clear. The contract test above only ever sees zero-line
    /// steps, so its reward assertion is vacuous — a RewardScale change slipped past it unnoticed during
    /// M57.5. This exercises the actual value.
    /// </summary>
    [Fact]
    public void RewardScale_DividesTheHybridReward()
    {
        // Flat left-to-right filling policy driven off the env's own mask: deterministic, needs no
        // mirror board, and reliably completes rows so a clear actually happens.
        var env = new TetrisEnv();
        env.Reset(11);

        double rewardOnClear = double.NaN;
        int clearedOnThatStep = 0;
        for (int i = 0; i < 400; i++)
        {
            var mask = env.CurrentActionMask();
            // round-robin the target column so rows fill left-to-right instead of stacking one column
            int action = -1;
            for (int k = 0; k < TetrisBoard.Width && action < 0; k++)
            {
                int col = (i + k) % TetrisBoard.Width;
                if (mask[col]) action = col;                       // rot 0, that column
            }
            for (int a = 0; a < TetrisBoard.ActionCount && action < 0; a++) if (mask[a]) action = a;
            if (action < 0) break;

            int before = env.Lines;
            var step = env.Step(action);
            int cleared = env.Lines - before;
            if (cleared > 0) { rewardOnClear = step.Reward; clearedOnThatStep = cleared; break; }
            if (step.Terminated || step.Truncated) break;
        }

        Assert.True(clearedOnThatStep > 0, "fixture never cleared a line");
        double expected = (clearedOnThatStep + (clearedOnThatStep == 4 ? TetrisBoard.TetrisRewardBonus : 0)) / TetrisEnv.RewardScale;
        Assert.Equal(expected, rewardOnClear, 5);
    }

    /// <summary>
    /// M57.5c. The warm-start path depends on ONE structural property: planes 0-5 must be the M54 basis,
    /// in the M54 order, so they occupy observation indices 214..453 — exactly the old 454-float
    /// observation. That is what lets DuelingQNet.GrowInput transplant the shipped M54 net
    /// function-preservingly (old weights keep their meaning, new planes start at zero). If a future
    /// change reorders or reinterprets planes 0-5, the transplant silently feeds the old weights
    /// different quantities — same width, so no guard catches it. This pins the layout.
    /// </summary>
    [Fact]
    public void ObservationLayout_KeepsTheM54BasisAsItsPrefix()
    {
        const int m54Planes = 6;
        const int m54ObservationSize = 454;
        int prefixBase = TetrisBoard.Width * TetrisBoard.Height + 2 * TetrisBoard.PieceCount;

        Assert.Equal(214, prefixBase);
        Assert.Equal(m54ObservationSize, prefixBase + m54Planes * TetrisBoard.ActionCount);
        Assert.True(TetrisBoard.ObservationPlanes > m54Planes,
            "the M57 planes must be ADDED after the M54 basis, never replace it");
        Assert.Equal(TetrisBoard.ObservationSize,
            prefixBase + TetrisBoard.ObservationPlanes * TetrisBoard.ActionCount);

        // plane 0 is landing height: strictly positive for a legal placement, zero for an illegal one,
        // which is the M54 meaning the transplanted weights expect.
        var board = new TetrisBoard();
        board.Reset(7);
        var obs = board.BuildObservation();
        var mask = board.LegalMask();
        for (int a = 0; a < TetrisBoard.ActionCount; a++)
        {
            float landing = obs[prefixBase + a];
            if (mask[a]) Assert.True(landing > 0f, $"legal action {a} has landing {landing} — plane 0 is not the M54 landing height");
            else Assert.Equal(0f, landing);
        }
    }
}
