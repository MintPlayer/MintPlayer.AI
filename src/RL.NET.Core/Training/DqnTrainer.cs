using RLNet.Core.Agents;
using RLNet.Core.Checkpoints;
using RLNet.Core.Environments;
using RLNet.Core.Nn;
using RLNet.Core.Numerics;
using RLNet.Core.Random;
using RLNet.Core.Schedules;

namespace RLNet.Core.Training;

public sealed record DqnOptions
{
    public int[] Hidden { get; init; } = [64, 64];
    public double Gamma { get; init; } = 0.99;
    public float LearningRate { get; init; } = 1e-3f;
    public int BufferCapacity { get; init; } = 50_000;
    public int BatchSize { get; init; } = 64;
    public int WarmupSteps { get; init; } = 1_000;
    public int TrainEvery { get; init; } = 1;
    public int TargetSyncEvery { get; init; } = 500;
    public LinearSchedule Epsilon { get; init; } = new(1.0, 0.05, 10_000);
    public int MaxSteps { get; init; } = 100_000;
    public float MaxGradNorm { get; init; } = 10f;

    /// <summary>Use Double DQN: online net picks argmax, target net evaluates it.</summary>
    public bool DoubleDqn { get; init; } = true;

    public int EvalEvery { get; init; } = 5_000;
    public int EvalEpisodes { get; init; } = 20;

    /// <summary>Training stops early once a greedy evaluation reaches this mean return.</summary>
    public double? SolveThreshold { get; init; }

    public Action<DqnProgress>? OnProgress { get; init; }
}

public readonly record struct DqnProgress(int Step, int MaxSteps, double EvalMeanReturn, double Epsilon, float LastLoss);

public sealed record DqnResult(GreedyQAgent Agent, Mlp Network, int StepsTrained, double FinalEvalReturn, DqnTrainingState State);

/// <summary>Acts greedily from a Q-network (evaluation/playback); epsilon-greedy when given an RNG.</summary>
public sealed class GreedyQAgent(Mlp network, int actionCount, Xoshiro256StarStar? rng = null) : IAgent<float[], int>
{
    public double Epsilon { get; set; }

    public int Act(float[] observation, bool greedy = false) => Act(observation, null, greedy);

    /// <summary>Masked variant: exploration and argmax are restricted to legal actions.</summary>
    public int Act(float[] observation, bool[]? mask, bool greedy = false)
    {
        if (!greedy && rng is not null && rng.NextDouble() < Epsilon)
        {
            if (mask is null) return rng.NextInt(actionCount);
            int legal = mask.Count(m => m);
            int pick = rng.NextInt(legal);
            for (int a = 0; a < actionCount; a++)
                if (mask[a] && pick-- == 0) return a;
        }

        using (GradMode.NoGrad())
        {
            var q = network.Forward(new Tensor(observation, 1, observation.Length));
            int best = -1;
            for (int a = 0; a < actionCount; a++)
            {
                if (mask is not null && !mask[a]) continue;
                if (best < 0 || q.Data[a] > q.Data[best]) best = a;
            }
            return best;
        }
    }
}

/// <summary>
/// DQN (Mnih et al. 2015) with target network and optional Double-DQN action decoupling.
/// Single-file reference implementation per the CleanRL discipline (PLAN.md M3).
/// </summary>
public static class DqnTrainer
{
    /// <param name="resume">
    /// A previously saved training state to continue from. The same options and master
    /// seed must be passed as in the original run; on <see cref="IStatefulEnvironment"/>
    /// envs the resumed run is bitwise-identical to one that was never interrupted.
    /// </param>
    public static DqnResult Train(IEnvironment<float[], int> env, DqnOptions options, SeedSequence seeds,
        DqnTrainingState? resume = null)
    {
        int obsDim = ((BoxSpace)env.ObservationSpace).Dimensions;
        int actionCount = ((DiscreteSpace)env.ActionSpace).N;

        DqnTrainingState state;
        float[] obs;
        if (resume is null)
        {
            var initRng = seeds.CreateRng(RngStreams.Init);
            var online = new Mlp([obsDim, .. options.Hidden, actionCount], initRng, Activation.Relu);
            var target = new Mlp([obsDim, .. options.Hidden, actionCount], initRng, Activation.Relu);
            target.CopyFrom(online);
            state = new DqnTrainingState
            {
                Online = online,
                Target = target,
                Optimizer = new Adam(online.Parameters(), options.LearningRate),
                Buffer = new ReplayBuffer(options.BufferCapacity, obsDim, actionCount),
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
                // Env can't be restored: start a fresh episode (training continues, bitwise equality doesn't).
                (obs, _) = env.Reset(seeds.Derive(RngStreams.Environment));
            }
        }

        var agent = new GreedyQAgent(state.Online, actionCount, state.PolicyRng);
        var maskProvider = env as IActionMaskProvider;
        float lastLoss = state.LastLoss;
        double lastEval = state.LastEval;

        DqnResult Finish(int stepsTrained)
        {
            state.CurrentObs = obs;
            state.StepsCompleted = stepsTrained;
            state.LastLoss = lastLoss;
            state.LastEval = lastEval;
            state.EnvState = (env as IStatefulEnvironment)?.SaveState();
            return new DqnResult(agent, state.Online, stepsTrained, lastEval, state);
        }

        for (int step = state.StepsCompleted + 1; step <= options.MaxSteps; step++)
        {
            agent.Epsilon = options.Epsilon.Value(step);
            int action = agent.Act(obs, maskProvider?.CurrentActionMask());
            var result = env.Step(action);
            // Mask of the next state, captured BEFORE any autoreset (used by the TD-target max).
            var nextMask = maskProvider?.CurrentActionMask();

            // Store terminated only — truncated transitions must still bootstrap.
            state.Buffer.Add(obs, action, result.Reward, result.Observation, result.Terminated, nextMask);
            obs = result.Done ? env.Reset().Observation : result.Observation;

            if (state.Buffer.Count >= options.WarmupSteps && step % options.TrainEvery == 0)
                lastLoss = TrainStep(state.Online, state.Target, state.Optimizer,
                    state.Buffer.Sample(options.BatchSize, state.BufferRng), options);

            if (step % options.TargetSyncEvery == 0)
                state.Target.CopyFrom(state.Online);

            if (step % options.EvalEvery == 0)
            {
                lastEval = Evaluator.Evaluate(env, (float[] o, bool[]? m) => agent.Act(o, m, greedy: true),
                    options.EvalEpisodes, seeds.Derive(RngStreams.Evaluation)).MeanReturn;
                options.OnProgress?.Invoke(new DqnProgress(step, options.MaxSteps, lastEval, agent.Epsilon, lastLoss));
                (obs, _) = env.Reset(); // evaluation ended mid-episode; start fresh

                if (options.SolveThreshold.HasValue && lastEval >= options.SolveThreshold.Value)
                    return Finish(step);
            }
        }

        return Finish(options.MaxSteps);
    }

    private static float TrainStep(Mlp online, Mlp target, Adam adam, ReplayBuffer.Batch batch, DqnOptions options)
    {
        // TD targets, gradient-free: y = r + γ·(1−terminated)·Q_target(s', a*)
        var targets = new float[batch.Size];
        using (GradMode.NoGrad())
        {
            var nextObs = new Tensor(batch.NextObs, batch.Size, batch.ObsDim);
            var targetQ = target.Forward(nextObs);
            var onlineQ = options.DoubleDqn ? online.Forward(nextObs) : targetQ;
            int actions = targetQ.Cols;

            for (int i = 0; i < batch.Size; i++)
            {
                // Argmax restricted to the next state's legal actions.
                int best = -1;
                for (int a = 0; a < actions; a++)
                {
                    if (!batch.NextMasks[i * actions + a]) continue;
                    if (best < 0 || onlineQ.Data[i * actions + a] > onlineQ.Data[i * actions + best]) best = a;
                }

                double bootstrap = batch.Terminated[i] || best < 0
                    ? 0
                    : options.Gamma * targetQ.Data[i * actions + best];
                targets[i] = (float)(batch.Rewards[i] + bootstrap);
            }
        }

        adam.ZeroGrad();
        var q = online.Forward(new Tensor(batch.Obs, batch.Size, batch.ObsDim)).Gather(batch.Actions);
        var loss = q.HuberLoss(new Tensor(targets, batch.Size));
        loss.Backward();
        adam.ClipGradNorm(options.MaxGradNorm);
        adam.Step();
        return loss.Data[0];
    }
}
