using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M49.2 gate (docs/prd/CRAZY_FRUITS_PRD.md §5): the env contract (spaces, masking, reward normalization,
/// truncation-not-termination, state round-trip), the baseline ordering that proves scoring+masking
/// end-to-end BEFORE any training (greedy ≫ random, CI-separated; expectimax-1 ≥ greedy), and the campaign's
/// resume → train → checkpoint → resume contract on the real env. The full 500-episode gate table runs via
/// `--game crazyfruits --baselines 500`.
/// </summary>
public class CrazyFruitsEnvTests
{
    [Fact]
    public void Contract_SpacesMaskRewardTruncation()
    {
        var env = new CrazyFruitsEnv(moveBudget: 5);
        var (obs, _) = env.Reset(7);
        Assert.Equal(CrazyFruitsEnv.ObservationSize, obs.Length);

        for (int move = 0; move < 5; move++)
        {
            var mask = env.CurrentActionMask();
            Assert.Equal(CrazyFruitsEnv.ActionCount, mask.Length);
            int action = Array.IndexOf(mask, true);
            Assert.True(action >= 0);                       // always a legal swap (reshuffle invariant)

            int scoreBefore = env.Score;
            var step = env.Step(action);
            Assert.False(step.Terminated);                  // the board never dies
            Assert.Equal(move == 4, step.Truncated);        // budget end truncates
            Assert.Equal((env.Score - scoreBefore) / CrazyFruitsEnv.RewardScale, step.Reward, 5);
            Assert.True(step.Reward >= 1f - 1e-6);          // a legal swap clears at least a 3-line
        }
        Assert.Throws<InvalidOperationException>(() => env.Step(0));
    }

    [Fact]
    public void IllegalAction_Throws_WithoutConsumingTheMove()
    {
        var env = new CrazyFruitsEnv();
        env.Reset(3);
        int illegal = Array.IndexOf(env.CurrentActionMask(), false);
        Assert.True(illegal >= 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => env.Step(illegal));
        Assert.Equal(0, env.MovesMade);
    }

    [Fact]
    public void StateRoundTrip_ContinuesIdentically_AcrossEpisodeBoundaries()
    {
        var env = new CrazyFruitsEnv(moveBudget: 8);
        env.Reset(11);
        for (int move = 0; move < 5; move++)
            env.Step(Array.IndexOf(env.CurrentActionMask(), true));

        var saved = env.SaveState();
        var restored = new CrazyFruitsEnv(moveBudget: 8);
        restored.RestoreState(saved);

        // Finish the episode and roll into the next one: the env RNG stream must continue identically.
        for (int move = 0; move < 3; move++)
        {
            int action = Array.IndexOf(env.CurrentActionMask(), true);
            Assert.Equal(action, Array.IndexOf(restored.CurrentActionMask(), true));
            Assert.Equal(env.Step(action).Reward, restored.Step(action).Reward);
        }
        var (nextA, _) = env.Reset();
        var (nextB, _) = restored.Reset();
        Assert.Equal(nextA, nextB);
    }

    // ── Baseline ordering (the M49.2 falsifiable gate, CI-fast at 200 episodes) ─────────────────────────────

    private static double MeanScore(int episodes, Func<CrazyFruitsBoard, int, int> policy, out double ci)
    {
        double sum = 0, sumSq = 0;
        for (int e = 0; e < episodes; e++)
        {
            var env = new CrazyFruitsEnv();
            env.Reset((ulong)(5_000 + e));
            for (int move = 0; move < 30; move++)
                env.Board.ApplySwap(policy(env.Board, move));
            sum += env.Board.Score;
            sumSq += (double)env.Board.Score * env.Board.Score;
        }
        double mean = sum / episodes;
        ci = 1.96 * Math.Sqrt(Math.Max(0, sumSq / episodes - mean * mean) / episodes);
        return mean;
    }

    // 500 episodes = the gate protocol (PRD §4). Greedy's edge over random is real but small (+6%: with
    // masking every random move also clears a match, and refill-cascade luck dominates the variance), so the
    // CI separation genuinely needs the full N — measured 2026-07-24: random 2259.7±49.9, greedy 2387.0±49.3,
    // expectimax-1 4270.9±98.3 (cascade planning, not line size, is where the skill is).
    [Fact]
    public void Baselines_GreedyBeatsRandom_CiSeparated_AndExpectimaxAtLeastGreedy()
    {
        const int episodes = 500;
        double random = MeanScore(episodes, (b, move) => b.RandomAction(0xBEEF, move), out double ciR);
        double greedy = MeanScore(episodes, (b, _) => b.GreedyAction(), out double ciG);
        double expectimax = MeanScore(episodes, (b, _) => b.ExpectimaxAction(), out double ciE);

        Assert.True(greedy - ciG > random + ciR,
            $"greedy {greedy:F1}±{ciG:F1} must CI-separate above random {random:F1}±{ciR:F1}");
        Assert.True(expectimax + ciE >= greedy - ciG,
            $"expectimax {expectimax:F1}±{ciE:F1} must not fall below greedy {greedy:F1}±{ciG:F1}");
    }

    // ── Campaign: DI registration + resume → train → checkpoint → resume on the real env ────────────────────

    [Fact]
    public void Campaign_Registers_TrainsCheckpointsAndResumes()
    {
        var services = new ServiceCollection();
        services.AddCrazyFruitsDqnCampaign(new CrazyFruitsEnv(), new CrazyFruitsEnv(),
            new DqnScoreOptions { Hidden = [32, 32] });
        using (var provider = services.BuildServiceProvider())
            Assert.IsType<CrazyFruitsDqnCampaign>(provider.GetRequiredService<ITrainingCampaign>());

        var dir = Directory.CreateTempSubdirectory("crazyfruits-campaign");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var options = new DqnScoreOptions { Seed = 1, ChunkSteps = 60, TargetSteps = 120, Hidden = [32, 32], EvalEpisodes = 2 };

            var c1 = new CrazyFruitsDqnCampaign(evalEnv: new CrazyFruitsEnv(), trainEnv: new CrazyFruitsEnv(), options: options, logger: null);
            Assert.False(c1.Resume(store));
            Assert.Equal(60, c1.TrainChunk());
            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "score");
            c1.Checkpoint(store);
            c1.Dispose();
            using (var state = store.TryOpenRead("crazyfruits", "dqn-state")) Assert.NotNull(state);

            var c2 = new CrazyFruitsDqnCampaign(evalEnv: new CrazyFruitsEnv(), trainEnv: new CrazyFruitsEnv(), options: options, logger: null);
            Assert.True(c2.Resume(store));
            Assert.Equal(120, c2.TrainChunk());              // continues, not from zero
            Assert.True(c2.IsComplete);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
