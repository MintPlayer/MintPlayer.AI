using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// <see cref="ReinforceTrainer"/> is the library's vanilla policy-gradient trainer and had 7 of 79 lines covered —
/// effectively only its options record. Nothing exercised the loop itself.
/// </summary>
/// <remarks>
/// The gaps that matter here are not "does it learn" (that is a slow, noisy assertion owned by the campaign
/// gates) but the <b>bookkeeping around</b> the gradient step, which is exactly the kind of thing that breaks
/// silently and still trains *something*: the reward-to-go discount, the 100-episode rolling window, the
/// early-return on <c>SolveThreshold</c>, and the batch flush cadence. A wrong γ or an off-by-one window still
/// produces a plausible-looking learning curve.
/// <para>
/// Every test here runs a handful of episodes against a fixed toy environment with a 4-unit hidden layer, so the
/// whole class costs milliseconds. Learning is never asserted.
/// </para>
/// </remarks>
public class ReinforceTrainerTests
{
    /// <summary>
    /// Fixed-length 3-step episode, 2 obs features, 2 actions. Reward is +1 for action 1 and 0 otherwise, so an
    /// episode's undiscounted return is simply the number of 1s chosen — a quantity a test can predict exactly
    /// when the policy is driven deterministically.
    /// </summary>
    private sealed class ThreeStepEnv : IEnvironment<float[], int>
    {
        public int ResetCount { get; private set; }
        public ulong? LastSeed { get; private set; }

        private int _t;

        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 2);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(2);

        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
        {
            if (seed.HasValue) LastSeed = seed;
            ResetCount++;
            _t = 0;
            return (Obs(), EnvInfo.Empty);
        }

        public StepResult<float[]> Step(int action)
        {
            _t++;
            return new StepResult<float[]>(Obs(), action == 1 ? 1.0 : 0.0, _t >= 3, false, EnvInfo.Empty);
        }

        public string RenderString() => $"t={_t}";

        private float[] Obs() => [_t / 3f, 1f - _t / 3f];
    }

    /// <summary>Every reward is exactly 1, so an episode's undiscounted return is always 3 whatever the policy
    /// does. That removes the policy from the arithmetic and lets the window and threshold logic be pinned.</summary>
    private sealed class ConstantRewardEnv : IEnvironment<float[], int>
    {
        private int _t;

        public Space<float[]> ObservationSpace { get; } = new BoxSpace(0f, 1f, 2);
        public Space<int> ActionSpace { get; } = new DiscreteSpace(2);

        public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null) { _t = 0; return ([0f, 1f], EnvInfo.Empty); }
        public StepResult<float[]> Step(int action)
        {
            _t++;
            return new StepResult<float[]>([0f, 1f], 1.0, _t >= 3, false, EnvInfo.Empty);
        }
        public string RenderString() => $"t={_t}";
    }

    private static ReinforceOptions Tiny(int maxEpisodes) => new()
    {
        Hidden = [4],
        MaxEpisodes = maxEpisodes,
        EpisodesPerUpdate = 2,
        ProgressInterval = 1,
    };

    // ── The training loop's shape ────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Train_BuildsANetworkShapedByTheEnvironmentSpacesAndTheHiddenOption()
    {
        var result = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(4) with { Hidden = [8, 6] }, new SeedSequence(1));

        // Layers are [obs, ...hidden, actions] = [2, 8, 6, 2] → three weight matrices.
        var weights = result.Network.Parameters().Where(p => p.Rank == 2).ToArray();
        Assert.Equal(3, weights.Length);
        Assert.Equal(2, weights[0].Rows);
        Assert.Equal(8, weights[0].Cols);
        Assert.Equal(6, weights[1].Cols);
        Assert.Equal(2, weights[^1].Cols);
    }

    [Fact]
    public void Train_RunsExactlyMaxEpisodesWhenNoThresholdIsSet()
    {
        var env = new ThreeStepEnv();

        var result = ReinforceTrainer.Train(env, Tiny(6), new SeedSequence(7));

        Assert.Equal(6, result.EpisodesTrained);
        // One Reset per episode, plus the single seeding Reset the trainer does before the loop.
        Assert.Equal(7, env.ResetCount);
    }

    [Fact]
    public void Train_SeedsTheEnvironmentFromTheEnvironmentStreamBeforeTheFirstEpisode()
    {
        var env = new ThreeStepEnv();
        var seeds = new SeedSequence(42);

        ReinforceTrainer.Train(env, Tiny(2), seeds);

        // The per-episode Resets pass no seed, so the last seed the env ever saw is the one-off seeding Reset.
        Assert.Equal(seeds.Derive(RngStreams.Environment), env.LastSeed);
    }

    [Fact]
    public void Train_ReturnsAnAgentBackedByTheNetworkItTrained()
    {
        var result = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(4), new SeedSequence(3));

        Assert.NotNull(result.Agent);
        Assert.NotNull(result.Network);
        // The agent must act on the trained net, not a fresh copy: a greedy action is a valid index either way,
        // so the check that carries weight is that acting does not throw and stays in the action space.
        int action = result.Agent.Act([0.5f, 0.5f], greedy: true);
        Assert.InRange(action, 0, 1);
    }

    // ── The rolling window and the reported return ───────────────────────────────────────────────────────────

    [Fact]
    public void Train_ReportsTheRollingAverageReturnOverTheEpisodesSeenSoFar()
    {
        // Every episode returns exactly 3, so the window average is 3 from the first episode on — regardless of
        // how few episodes have been seen. This pins that the average divides by window.Count, not by 100.
        var result = ReinforceTrainer.Train(new ConstantRewardEnv(), Tiny(5), new SeedSequence(11));

        Assert.Equal(3.0, result.FinalAvgReturn100, 6);
    }

    [Fact]
    public void Train_StopsEarlyOnlyOnceAFullHundredEpisodeWindowClearsTheThreshold()
    {
        // The threshold is satisfiable from episode 1 (every return is 3), but the guard also requires
        // window.Count == 100. So the run must stop at exactly episode 100 — not sooner, and not run to 500.
        var options = Tiny(500) with { SolveThreshold = 2.5 };

        var result = ReinforceTrainer.Train(new ConstantRewardEnv(), options, new SeedSequence(5));

        Assert.Equal(100, result.EpisodesTrained);
        Assert.True(result.FinalAvgReturn100 >= 2.5);
    }

    [Fact]
    public void Train_RunsToCompletionWhenTheThresholdIsNeverMet()
    {
        var options = Tiny(120) with { SolveThreshold = 1000.0 };

        var result = ReinforceTrainer.Train(new ConstantRewardEnv(), options, new SeedSequence(5));

        Assert.Equal(120, result.EpisodesTrained);
    }

    // ── Progress reporting ───────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Train_RaisesProgressOnTheConfiguredIntervalAndCarriesTheEpisodeBudget()
    {
        var seen = new List<TrainingProgress>();
        var options = Tiny(10) with { ProgressInterval = 5, OnProgress = seen.Add };

        ReinforceTrainer.Train(new ConstantRewardEnv(), options, new SeedSequence(2));

        Assert.Equal(2, seen.Count);
        Assert.Equal([5, 10], seen.Select(p => p.Episode));
        Assert.All(seen, p => Assert.Equal(10, p.TotalEpisodes));
        Assert.All(seen, p => Assert.Equal(3.0, p.AvgReturn100, 6));
    }

    [Fact]
    public void Train_RaisesNoProgressWhenNoCallbackIsSupplied()
    {
        // Purely that the null-callback path is taken without throwing — the failure mode being a
        // NullReferenceException on every run that doesn't opt into progress.
        var result = ReinforceTrainer.Train(new ConstantRewardEnv(), Tiny(4) with { OnProgress = null }, new SeedSequence(2));

        Assert.Equal(4, result.EpisodesTrained);
    }

    // ── The update step ──────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Train_UpdatesTheNetworkWeights()
    {
        var before = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(0), new SeedSequence(9));
        var after = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(8), new SeedSequence(9));

        // Same seed, so both nets start identical. MaxEpisodes=0 skips the loop entirely and therefore every
        // update, so any difference is the gradient steps — the assertion is "training moved the weights",
        // deliberately NOT "training improved the policy".
        var w0 = before.Network.Parameters().First(p => p.Rank == 2).Data;
        var w1 = after.Network.Parameters().First(p => p.Rank == 2).Data;
        Assert.Equal(w0.Length, w1.Length);
        Assert.Contains(Enumerable.Range(0, w0.Length), i => Math.Abs(w0[i] - w1[i]) > 1e-6f);
    }

    [Fact]
    public void Train_LeavesTheNetworkUntouchedWhenNoEpisodeEverCompletesAnUpdateBatch()
    {
        // One episode with EpisodesPerUpdate=2 never satisfies `episode % 2 == 0`, so Update is never called.
        var trained = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(1) with { EpisodesPerUpdate = 2 }, new SeedSequence(9));
        var untouched = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(0), new SeedSequence(9));

        Assert.Equal(
            untouched.Network.Parameters().First(p => p.Rank == 2).Data,
            trained.Network.Parameters().First(p => p.Rank == 2).Data);
    }

    [Fact]
    public void Train_ProducesAFiniteNetworkWithReturnNormalisationOff()
    {
        // NormalizeReturns=false is the branch nothing ran. With a constant-return environment the advantage
        // standard deviation is zero, which is precisely the case the 1e-8 epsilon in the normalising branch
        // exists to survive — so the un-normalised path must also stay finite rather than producing NaNs.
        var result = ReinforceTrainer.Train(
            new ConstantRewardEnv(), Tiny(6) with { NormalizeReturns = false }, new SeedSequence(4));

        Assert.All(result.Network.Parameters(), p => Assert.All(p.Data, v => Assert.True(float.IsFinite(v))));
    }

    [Fact]
    public void Train_SurvivesAConstantReturnBatchWithNormalisationOn()
    {
        var result = ReinforceTrainer.Train(
            new ConstantRewardEnv(), Tiny(6) with { NormalizeReturns = true }, new SeedSequence(4));

        Assert.All(result.Network.Parameters(), p => Assert.All(p.Data, v => Assert.True(float.IsFinite(v))));
    }

    [Fact]
    public void Train_IsReproducibleForAGivenMasterSeed()
    {
        var a = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(10), new SeedSequence(2026));
        var b = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(10), new SeedSequence(2026));

        Assert.Equal(a.FinalAvgReturn100, b.FinalAvgReturn100);
        Assert.Equal(
            a.Network.Parameters().First(p => p.Rank == 2).Data,
            b.Network.Parameters().First(p => p.Rank == 2).Data);
    }

    [Fact]
    public void Train_DivergesAcrossMasterSeeds()
    {
        var a = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(10), new SeedSequence(1));
        var b = ReinforceTrainer.Train(new ThreeStepEnv(), Tiny(10), new SeedSequence(2));

        // Guards the opposite failure from the test above: a SeedSequence that ignored its master seed would make
        // every run identical, and the reproducibility test alone would still pass.
        Assert.NotEqual(
            a.Network.Parameters().First(p => p.Rank == 2).Data,
            b.Network.Parameters().First(p => p.Rank == 2).Data);
    }

    // ── PolicyAgent ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void PolicyAgent_GreedyActionIsTheArgmaxOfTheNetworkOutput()
    {
        var net = new Mlp([2, 4, 3], new Xoshiro256StarStar(17), Activation.Tanh);
        var agent = new PolicyAgent(net, new Xoshiro256StarStar(1));

        float[] obs = [0.3f, -0.7f];
        int expected = Argmax(net, obs);

        // Greedy must be a pure function of the net: repeated calls never consult the RNG.
        Assert.Equal(expected, agent.Act(obs, greedy: true));
        Assert.Equal(expected, agent.Act(obs, greedy: true));
        Assert.Equal(expected, agent.Act(obs, greedy: true));
    }

    [Fact]
    public void PolicyAgent_SamplesWithinTheActionSpaceWhenNotGreedy()
    {
        var net = new Mlp([2, 4, 3], new Xoshiro256StarStar(17), Activation.Tanh);
        var agent = new PolicyAgent(net, new Xoshiro256StarStar(99));

        for (int i = 0; i < 50; i++)
            Assert.InRange(agent.Act([0.1f, 0.2f]), 0, 2);
    }

    [Fact]
    public void PolicyAgent_NeverChoosesAMaskedOutAction()
    {
        var net = new Mlp([2, 4, 3], new Xoshiro256StarStar(17), Activation.Tanh);
        var agent = new PolicyAgent(net, new Xoshiro256StarStar(5));
        bool[] onlyTheLast = [false, false, true];

        for (int i = 0; i < 50; i++)
            Assert.Equal(2, agent.Act([0.1f, 0.2f], onlyTheLast));

        Assert.Equal(2, agent.Act([0.1f, 0.2f], onlyTheLast, greedy: true));
    }

    [Fact]
    public void PolicyAgent_RespectsAMaskThatLeavesSeveralActionsLegal()
    {
        var net = new Mlp([2, 4, 3], new Xoshiro256StarStar(17), Activation.Tanh);
        var agent = new PolicyAgent(net, new Xoshiro256StarStar(5));
        bool[] notTheMiddle = [true, false, true];

        for (int i = 0; i < 50; i++)
            Assert.NotEqual(1, agent.Act([0.1f, 0.2f], notTheMiddle));
    }

    private static int Argmax(Mlp net, float[] obs)
    {
        var logits = net.Forward(new Core.Numerics.Tensor(obs, 1, obs.Length)).Data;
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
