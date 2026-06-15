using MintPlayer.AI.ReinforcementLearning.Core.Agents;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

public sealed record SacOptions
{
    public int[] Hidden { get; init; } = [256, 256];
    public double Gamma { get; init; } = 0.99;
    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>Polyak coefficient for the soft target-critic update θ′ ← τθ + (1−τ)θ′, applied every step.</summary>
    public float Tau { get; init; } = 0.005f;

    public int BufferCapacity { get; init; } = 1_000_000;
    public int BatchSize { get; init; } = 256;

    /// <summary>Uniform-random exploration steps before learning starts (and before the policy is queried).</summary>
    public int WarmupSteps { get; init; } = 1_000;
    public int TrainEvery { get; init; } = 1;

    /// <summary>Critic+actor updates performed per environment step (SAC update-to-data ratio).</summary>
    public int GradientSteps { get; init; } = 1;

    public int MaxSteps { get; init; } = 100_000;
    public float MaxGradNorm { get; init; } = 10f;

    /// <summary>Auto-tune the entropy temperature against a target entropy (Haarnoja 2018b). Recommended.</summary>
    public bool AutoTuneAlpha { get; init; } = true;
    /// <summary>Initial temperature; seeds log-α when auto-tuning, or fixes α when not.</summary>
    public float InitialAlpha { get; init; } = 0.2f;
    /// <summary>Target entropy for the α objective; defaults to −actionDim (the SAC heuristic) when null.</summary>
    public float? TargetEntropy { get; init; }

    public int EvalEvery { get; init; } = 5_000;
    public int EvalEpisodes { get; init; } = 20;

    /// <summary>Training stops early once a deterministic evaluation reaches this mean return.</summary>
    public double? SolveThreshold { get; init; }

    public Action<SacProgress>? OnProgress { get; init; }
}

public readonly record struct SacProgress(
    int Step, int MaxSteps, double EvalMeanReturn, float Alpha, float CriticLoss, float ActorLoss);

public sealed record SacResult(
    ContinuousPolicyAgent Agent, Mlp Actor, Mlp Critic1, Mlp Critic2,
    int StepsTrained, double FinalEvalReturn, SacTrainingState State);

/// <summary>
/// Acts from a squashed-Gaussian actor (SAC). Greedy = tanh(mean) (deterministic); stochastic = a
/// reparameterized squashed sample. The native (−1,1) action is rescaled to the environment's
/// <see cref="BoxSpace"/> bounds, so the agent plugs straight into <see cref="Evaluator"/> and live demos.
/// </summary>
public sealed class ContinuousPolicyAgent(Mlp actor, int actionDim, float[] low, float[] high, Xoshiro256StarStar rng)
    : IAgent<float[], float[]>
{
    public float[] Act(float[] observation, bool greedy = false)
    {
        using (GradMode.NoGrad())
        {
            var netOutput = actor.Forward(new Tensor(observation, 1, observation.Length));
            float[] native = greedy
                ? netOutput.SliceCols(0, actionDim).Tanh().Data
                : Normal.FromNetOutput(netOutput, actionDim).RSample(rng).Action.Data;
            return SacTrainer.ScaleToBounds(native, low, high);
        }
    }
}

/// <summary>
/// Soft Actor-Critic (Haarnoja et al. 2018) — off-policy, maximum-entropy continuous control. A
/// squashed-Gaussian actor, twin Q-critics with clipped double-Q targets, soft (Polyak) target updates and
/// an auto-tuned entropy temperature. The off-policy collection loop mirrors <see cref="DqnTrainer"/>
/// (single env + replay, warmup, periodic eval, solve-threshold early-out); only the update differs.
/// Single-file reference implementation per the CleanRL discipline (PLAN M23).
/// </summary>
public static class SacTrainer
{
    /// <param name="resume">
    /// A previously saved training state to continue from. The same options and master seed must be passed
    /// as in the original run; on <see cref="IStatefulEnvironment"/> envs the resumed run is bitwise-identical
    /// to one that was never interrupted.
    /// </param>
    public static SacResult Train(IEnvironment<float[], float[]> env, SacOptions options, SeedSequence seeds,
        SacTrainingState? resume = null)
    {
        int obsDim = ((BoxSpace)env.ObservationSpace).Dimensions;
        var actionBox = (BoxSpace)env.ActionSpace;
        int actionDim = actionBox.Dimensions;
        float[] low = actionBox.Low, high = actionBox.High;
        float targetEntropy = options.TargetEntropy ?? -actionDim;

        SacTrainingState state;
        float[] obs;
        if (resume is null)
        {
            var initRng = seeds.CreateRng(RngStreams.Init);
            var actor = new Mlp([obsDim, .. options.Hidden, 2 * actionDim], initRng, Activation.Relu);
            Mlp MakeCritic() => new([obsDim + actionDim, .. options.Hidden, 1], initRng, Activation.Relu);
            var critic1 = MakeCritic();
            var critic2 = MakeCritic();
            var target1 = MakeCritic();
            var target2 = MakeCritic();
            target1.CopyFrom(critic1);
            target2.CopyFrom(critic2);

            var logAlpha = new Tensor([MathF.Log(options.InitialAlpha)], 1) { RequiresGrad = true };
            state = new SacTrainingState
            {
                Actor = actor,
                Critic1 = critic1,
                Critic2 = critic2,
                Target1 = target1,
                Target2 = target2,
                ActorOptimizer = new Adam(actor.Parameters(), options.LearningRate),
                CriticOptimizer = new Adam([.. critic1.Parameters(), .. critic2.Parameters()], options.LearningRate),
                LogAlpha = logAlpha,
                AlphaOptimizer = options.AutoTuneAlpha ? new Adam([logAlpha], options.LearningRate) : null,
                Buffer = new ContinuousReplayBuffer(options.BufferCapacity, obsDim, actionDim),
                PolicyRng = seeds.CreateRng(RngStreams.Policy),
                BufferRng = seeds.CreateRng(RngStreams.Buffer),
            };
            (obs, _) = env.Reset(seeds.Derive(RngStreams.Environment));
        }
        else
        {
            state = resume;
            if (state.Buffer.Capacity != options.BufferCapacity)
                throw new ArgumentException($"Resume state buffer capacity {state.Buffer.Capacity} != options {options.BufferCapacity}.");
            if (state.CurrentObs.Length != obsDim)
                throw new ArgumentException($"Resume state obs dim {state.CurrentObs.Length} != environment {obsDim}.");

            if (state.EnvState is not null && env is IStatefulEnvironment stateful)
            {
                stateful.RestoreState(state.EnvState);
                obs = state.CurrentObs;
            }
            else
            {
                (obs, _) = env.Reset(seeds.Derive(RngStreams.Environment));
            }
        }

        var agent = new ContinuousPolicyAgent(state.Actor, actionDim, low, high, state.PolicyRng);
        float lastCriticLoss = state.LastCriticLoss, lastActorLoss = state.LastActorLoss;
        double lastEval = state.LastEval;

        SacResult Finish(int stepsTrained)
        {
            state.CurrentObs = obs;
            state.StepsCompleted = stepsTrained;
            state.LastCriticLoss = lastCriticLoss;
            state.LastActorLoss = lastActorLoss;
            state.LastEval = lastEval;
            state.EnvState = (env as IStatefulEnvironment)?.SaveState();
            return new SacResult(agent, state.Actor, state.Critic1, state.Critic2, stepsTrained, lastEval, state);
        }

        for (int step = state.StepsCompleted + 1; step <= options.MaxSteps; step++)
        {
            // Native (−1,1) action: uniform random during warmup, else a stochastic policy sample.
            float[] nativeAction = state.Buffer.Count < options.WarmupSteps
                ? RandomNative(actionDim, state.PolicyRng)
                : SamplePolicyNative(state.Actor, obs, actionDim, state.PolicyRng);
            var result = env.Step(ScaleToBounds(nativeAction, low, high));

            state.Buffer.Add(obs, nativeAction, result.Reward, result.Observation, result.Terminated);
            obs = result.Done ? env.Reset().Observation : result.Observation;

            if (state.Buffer.Count >= options.WarmupSteps && state.Buffer.Count >= options.BatchSize
                && step % options.TrainEvery == 0)
                for (int g = 0; g < options.GradientSteps; g++)
                    (lastCriticLoss, lastActorLoss) = TrainStep(state, options, actionDim, targetEntropy);

            if (step % options.EvalEvery == 0)
            {
                lastEval = Evaluator.Evaluate(env, agent, options.EvalEpisodes, seeds.Derive(RngStreams.Evaluation)).MeanReturn;
                options.OnProgress?.Invoke(new SacProgress(
                    step, options.MaxSteps, lastEval, MathF.Exp(state.LogAlpha.Data[0]), lastCriticLoss, lastActorLoss));
                (obs, _) = env.Reset(); // evaluation ended mid-episode; start fresh

                if (options.SolveThreshold.HasValue && lastEval >= options.SolveThreshold.Value)
                    return Finish(step);
            }
        }

        return Finish(options.MaxSteps);
    }

    private static (float CriticLoss, float ActorLoss) TrainStep(
        SacTrainingState state, SacOptions options, int actionDim, float targetEntropy)
    {
        var batch = state.Buffer.Sample(options.BatchSize, state.BufferRng);
        int b = batch.Size;
        float alpha = MathF.Exp(state.LogAlpha.Data[0]);

        var sObs = new Tensor(batch.Obs, b, batch.ObsDim);
        var sAction = new Tensor(batch.Actions, b, actionDim);
        var sNext = new Tensor(batch.NextObs, b, batch.ObsDim);

        // Critic target y = r + γ(1−terminated)·(min(Q1′,Q2′)(s′,a′) − α·log π(a′|s′)), gradient-free.
        var targets = new float[b];
        using (GradMode.NoGrad())
        {
            var (nextAction, nextLogProb) = Normal.FromNetOutput(state.Actor.Forward(sNext), actionDim).RSample(state.PolicyRng);
            var saNext = sNext.ConcatCols(nextAction);
            var q1Next = state.Target1.Forward(saNext);
            var q2Next = state.Target2.Forward(saNext);
            for (int i = 0; i < b; i++)
            {
                double minQ = Math.Min(q1Next.Data[i], q2Next.Data[i]);
                double soft = minQ - alpha * nextLogProb.Data[i];
                targets[i] = (float)(batch.Rewards[i] + (batch.Terminated[i] ? 0 : options.Gamma * soft));
            }
        }
        var yTarget = new Tensor(targets, b);

        // Critic update: MSE(Q1,y) + MSE(Q2,y) over both critics in one step.
        state.CriticOptimizer.ZeroGrad();
        var sa = sObs.ConcatCols(sAction);
        var q1 = state.Critic1.Forward(sa).Reshape(b);
        var q2 = state.Critic2.Forward(sa).Reshape(b);
        var criticLoss = q1.MseLoss(yTarget).Add(q2.MseLoss(yTarget));
        criticLoss.Backward();
        state.CriticOptimizer.ClipGradNorm(options.MaxGradNorm);
        state.CriticOptimizer.Step();

        // Actor update: E[α·log π(a|s) − min(Q1,Q2)(s, a_reparam)].
        state.ActorOptimizer.ZeroGrad();
        var (action, logProb) = Normal.FromNetOutput(state.Actor.Forward(sObs), actionDim).RSample(state.PolicyRng);
        var saPi = sObs.ConcatCols(action);
        var minQpi = state.Critic1.Forward(saPi).Reshape(b).Min(state.Critic2.Forward(saPi).Reshape(b));
        var actorLoss = logProb.MulScalar(alpha).Sub(minQpi).Mean();
        actorLoss.Backward();
        state.ActorOptimizer.ClipGradNorm(options.MaxGradNorm);
        state.ActorOptimizer.Step();

        // Temperature update: −E[log α · (log π(a|s) + targetEntropy)], with log π treated as constant.
        if (state.AlphaOptimizer is not null)
        {
            float coef = 0f;
            for (int i = 0; i < b; i++) coef += logProb.Data[i] + targetEntropy;
            coef /= b;
            state.AlphaOptimizer.ZeroGrad();
            state.LogAlpha.MulScalar(-coef).Backward();
            state.AlphaOptimizer.Step();
        }

        SoftUpdate(state.Target1, state.Critic1, options.Tau);
        SoftUpdate(state.Target2, state.Critic2, options.Tau);

        return (criticLoss.Data[0], actorLoss.Data[0]);
    }

    /// <summary>Polyak soft update of a target net toward a source net: θ′ ← τθ + (1−τ)θ′.</summary>
    private static void SoftUpdate(IModule target, IModule source, float tau)
    {
        using var t = target.Parameters().GetEnumerator();
        using var s = source.Parameters().GetEnumerator();
        while (t.MoveNext() && s.MoveNext())
        {
            var td = t.Current.Data;
            var sd = s.Current.Data;
            for (int i = 0; i < td.Length; i++)
                td[i] = tau * sd[i] + (1f - tau) * td[i];
        }
    }

    private static float[] RandomNative(int actionDim, Xoshiro256StarStar rng)
    {
        var a = new float[actionDim];
        for (int i = 0; i < actionDim; i++) a[i] = (float)(2.0 * rng.NextDouble() - 1.0);
        return a;
    }

    private static float[] SamplePolicyNative(Mlp actor, float[] obs, int actionDim, Xoshiro256StarStar rng)
    {
        using (GradMode.NoGrad())
            return Normal.FromNetOutput(actor.Forward(new Tensor(obs, 1, obs.Length)), actionDim).RSample(rng).Action.Data;
    }

    /// <summary>Maps a native action in [−1,1]^k to the environment's per-dimension box bounds.</summary>
    public static float[] ScaleToBounds(float[] native, float[] low, float[] high)
    {
        var scaled = new float[native.Length];
        for (int i = 0; i < native.Length; i++)
            scaled[i] = low[i] + (native[i] + 1f) * 0.5f * (high[i] - low[i]);
        return scaled;
    }
}
