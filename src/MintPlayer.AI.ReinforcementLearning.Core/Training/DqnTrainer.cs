using MintPlayer.AI.ReinforcementLearning.Core.Agents;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

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

    /// <summary>
    /// Use a <see cref="DuelingQNet"/> (shared trunk → value + advantage streams) instead of a plain
    /// <see cref="Mlp"/> Q-net. Sample-efficient where most actions are near-equivalent; combines with
    /// Double DQN. The hidden sizes (<see cref="Hidden"/>) become the shared trunk.
    /// </summary>
    public bool Dueling { get; init; }

    /// <summary>
    /// Use NoisyNets exploration (learned, state-dependent noise on the Dueling heads) INSTEAD of
    /// ε-greedy. Implies <see cref="Dueling"/>. The trainer resamples the online net's noise before
    /// each action and the online+target before each TD update, forces ε to 0, and disables noise
    /// during evaluation so greedy eval/serving stays deterministic.
    /// </summary>
    public bool NoisyNets { get; init; }

    /// <summary>
    /// Number of environment instances stepped in parallel to fill the replay buffer faster (synchronous
    /// vectorization). 1 (default) keeps the classic single-env loop, bitwise-resumable and unchanged.
    /// &gt; 1 requires the env-factory <see cref="DqnTrainer.Train(System.Func{Environments.IEnvironment{float[],int}},DqnOptions,Random.SeedSequence,Checkpoints.DqnTrainingState?,Nn.IValueNet?)"/>
    /// overload: the actors' expensive <c>Step</c> runs across cores while a single learner trains. Best
    /// for env-bound games (e.g. FruitCake's physics-in-the-loop), where the net is cheap but stepping is slow.
    /// </summary>
    public int Actors { get; init; } = 1;

    public int EvalEvery { get; init; } = 5_000;
    public int EvalEpisodes { get; init; } = 20;

    /// <summary>Training stops early once a greedy evaluation reaches this mean return.</summary>
    public double? SolveThreshold { get; init; }

    public Action<DqnProgress>? OnProgress { get; init; }
}

public readonly record struct DqnProgress(int Step, int MaxSteps, double EvalMeanReturn, double Epsilon, float LastLoss);

public sealed record DqnResult(GreedyQAgent Agent, IValueNet Network, int StepsTrained, double FinalEvalReturn, DqnTrainingState State);

/// <summary>Acts greedily from a Q-network (evaluation/playback); epsilon-greedy when given an RNG.</summary>
public sealed class GreedyQAgent(IValueNet network, int actionCount, Xoshiro256StarStar? rng = null) : IAgent<float[], int>
{
    public double Epsilon { get; set; }

    public int Act(float[] observation, bool greedy = false) => Act(observation, null, greedy);

    /// <summary>Q-values for every action (no-grad single forward pass) — for callers that rank actions themselves.</summary>
    public float[] QValues(float[] observation)
    {
        using (GradMode.NoGrad())
            return (float[])network.Forward(new Tensor(observation, 1, observation.Length)).Data.Clone();
    }

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
    /// <param name="warmStart">
    /// When starting fresh (<paramref name="resume"/> is null), use this network as the initial online net (its
    /// weights are copied into a fresh target) instead of a random init — i.e. continue training a previously
    /// trained net with a fresh optimizer/replay buffer/step count. Its shape must match the one
    /// <paramref name="options"/> would build (<see cref="DqnOptions.Dueling"/> + <see cref="DqnOptions.Hidden"/>).
    /// Ignored when resuming. Lets a campaign pick up a deployable (net-only) checkpoint that has no resume state.
    /// </param>
    public static DqnResult Train(IEnvironment<float[], int> env, DqnOptions options, SeedSequence seeds,
        DqnTrainingState? resume = null, IValueNet? warmStart = null)
    {
        if (options.Actors != 1)
            throw new ArgumentException(
                "DqnOptions.Actors > 1 needs the env-factory Train overload (it builds one env instance per actor).");
        return Train(() => env, options, seeds, resume, warmStart);
    }

    /// <param name="envFactory">Builds an environment instance; called once per <see cref="DqnOptions.Actors"/>.</param>
    /// <inheritdoc cref="Train(IEnvironment{float[],int},DqnOptions,SeedSequence,DqnTrainingState?,IValueNet?)"/>
    public static DqnResult Train(Func<IEnvironment<float[], int>> envFactory, DqnOptions options, SeedSequence seeds,
        DqnTrainingState? resume = null, IValueNet? warmStart = null)
    {
        if (options.Actors < 1)
            throw new ArgumentException($"DqnOptions.Actors must be >= 1 (got {options.Actors}).");
        if (options.Actors > 1)
            return TrainVectorized(envFactory, options, seeds, resume, warmStart);

        var env = envFactory();
        int obsDim = ((BoxSpace)env.ObservationSpace).Dimensions;
        int actionCount = ((DiscreteSpace)env.ActionSpace).N;

        DqnTrainingState state;
        float[] obs;
        if (resume is null)
        {
            var initRng = seeds.CreateRng(RngStreams.Init);
            // NoisyNets rides on the Dueling heads, so it implies a DuelingQNet.
            IValueNet MakeNet() => options.Dueling || options.NoisyNets
                ? new DuelingQNet(obsDim, options.Hidden, actionCount, initRng, options.NoisyNets)
                : new Mlp([obsDim, .. options.Hidden, actionCount], initRng, Activation.Relu);
            var online = warmStart ?? MakeNet();
            var target = MakeNet();
            target.CopyFrom(online); // warm-start: copies the provided net's weights (shape must match options)
            state = new DqnTrainingState
            {
                Online = online,
                Target = target,
                Optimizer = new Adam(online.Parameters(), options.LearningRate),
                Buffer = new ReplayBuffer(options.BufferCapacity, obsDim, actionCount),
                PolicyRng = seeds.CreateRng(RngStreams.Policy),
                BufferRng = seeds.CreateRng(RngStreams.Buffer),
                NoiseRng = seeds.CreateRng(RngStreams.Noise),
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

        // NoisyNets plumbing (no-ops for a plain/ε-greedy run). Noise stays ON for the training loop and
        // is switched OFF only around eval and on return, so a resumed run, eval, and the returned net are
        // all deterministic; exploration during training is the resampled, greedy-argmax-over-noise policy.
        void ResampleNoise(IValueNet net) { if (options.NoisyNets) ((DuelingQNet)net).ResampleNoise(state.NoiseRng); }
        void SetNoiseEnabled(bool on)
        {
            if (!options.NoisyNets) return;
            ((DuelingQNet)state.Online).SetNoiseEnabled(on);
            ((DuelingQNet)state.Target).SetNoiseEnabled(on);
        }

        DqnResult Finish(int stepsTrained)
        {
            SetNoiseEnabled(false); // return a deterministic net (keep-best eval / serving use the means)
            state.CurrentObs = obs;
            state.StepsCompleted = stepsTrained;
            state.LastLoss = lastLoss;
            state.LastEval = lastEval;
            state.EnvState = (env as IStatefulEnvironment)?.SaveState();
            return new DqnResult(agent, state.Online, stepsTrained, lastEval, state);
        }

        SetNoiseEnabled(true);

        for (int step = state.StepsCompleted + 1; step <= options.MaxSteps; step++)
        {
            ResampleNoise(state.Online); // fresh per-action exploration noise (NoisyNets)
            agent.Epsilon = options.NoisyNets ? 0 : options.Epsilon.Value(step); // noise replaces ε-greedy
            int action = agent.Act(obs, maskProvider?.CurrentActionMask());
            var result = env.Step(action);
            // Mask of the next state, captured BEFORE any autoreset (used by the TD-target max).
            var nextMask = maskProvider?.CurrentActionMask();

            // Store terminated only — truncated transitions must still bootstrap.
            state.Buffer.Add(obs, action, result.Reward, result.Observation, result.Terminated, nextMask);
            obs = result.Done ? env.Reset().Observation : result.Observation;

            if (state.Buffer.Count >= options.WarmupSteps && step % options.TrainEvery == 0)
            {
                // Independent fresh noise for the online and target nets used in the TD update.
                ResampleNoise(state.Online);
                ResampleNoise(state.Target);
                lastLoss = TrainStep(state.Online, state.Target, state.Optimizer,
                    state.Buffer.Sample(options.BatchSize, state.BufferRng), options);
            }

            if (step % options.TargetSyncEvery == 0)
                state.Target.CopyFrom(state.Online);

            if (step % options.EvalEvery == 0)
            {
                SetNoiseEnabled(false); // deterministic (means-only) greedy evaluation
                lastEval = Evaluator.Evaluate(env, (float[] o, bool[]? m) => agent.Act(o, m, greedy: true),
                    options.EvalEpisodes, seeds.Derive(RngStreams.Evaluation)).MeanReturn;
                SetNoiseEnabled(true);
                options.OnProgress?.Invoke(new DqnProgress(step, options.MaxSteps, lastEval, agent.Epsilon, lastLoss));
                (obs, _) = env.Reset(); // evaluation ended mid-episode; start fresh

                if (options.SolveThreshold.HasValue && lastEval >= options.SolveThreshold.Value)
                    return Finish(step);
            }
        }

        return Finish(options.MaxSteps);
    }

    /// <summary>
    /// Synchronous vectorized training (<see cref="DqnOptions.Actors"/> &gt; 1): N envs are stepped together,
    /// the expensive <c>Step</c> calls fanning out across cores while a single learner trains. One action-
    /// selection forward per actor per tick (the net is never touched concurrently — only <c>Step</c> is
    /// parallel), transitions are added in actor order (deterministic given N; the per-actor RNGs make the
    /// parallel steps reproducible), and the update:experience ratio is preserved by doing one TD update per
    /// <see cref="DqnOptions.TrainEvery"/> env-steps. Resume keeps the net/optimizer/buffer/RNGs but restarts
    /// the envs (no per-actor env snapshot), so unlike the scalar path it is not bitwise across an interruption.
    /// </summary>
    private static DqnResult TrainVectorized(Func<IEnvironment<float[], int>> envFactory, DqnOptions options,
        SeedSequence seeds, DqnTrainingState? resume, IValueNet? warmStart)
    {
        int n = options.Actors;
        var envs = new IEnvironment<float[], int>[n];
        for (int i = 0; i < n; i++) envs[i] = envFactory();
        int obsDim = ((BoxSpace)envs[0].ObservationSpace).Dimensions;
        int actionCount = ((DiscreteSpace)envs[0].ActionSpace).N;

        DqnTrainingState state;
        if (resume is null)
        {
            var initRng = seeds.CreateRng(RngStreams.Init);
            IValueNet MakeNet() => options.Dueling || options.NoisyNets
                ? new DuelingQNet(obsDim, options.Hidden, actionCount, initRng, options.NoisyNets)
                : new Mlp([obsDim, .. options.Hidden, actionCount], initRng, Activation.Relu);
            var online = warmStart ?? MakeNet();
            var target = MakeNet();
            target.CopyFrom(online);
            state = new DqnTrainingState
            {
                Online = online,
                Target = target,
                Optimizer = new Adam(online.Parameters(), options.LearningRate),
                Buffer = new ReplayBuffer(options.BufferCapacity, obsDim, actionCount),
                PolicyRng = seeds.CreateRng(RngStreams.Policy),
                BufferRng = seeds.CreateRng(RngStreams.Buffer),
                NoiseRng = seeds.CreateRng(RngStreams.Noise),
            };
        }
        else
        {
            state = resume;
            if (state.Buffer.Capacity != options.BufferCapacity)
                throw new ArgumentException($"Resume state buffer capacity {state.Buffer.Capacity} != options {options.BufferCapacity}.");
            if (state.CurrentObs.Length != 0 && state.CurrentObs.Length != obsDim)
                throw new ArgumentException($"Resume state obs dim {state.CurrentObs.Length} != environment {obsDim}.");
        }

        // Each actor gets its own seed so the parallel Step calls are independent and reproducible.
        ulong envSeedBase = seeds.Derive(RngStreams.Environment);
        var obs = new float[n][];
        var maskProviders = new IActionMaskProvider?[n];
        for (int i = 0; i < n; i++)
        {
            (obs[i], _) = envs[i].Reset(envSeedBase + (ulong)i);
            maskProviders[i] = envs[i] as IActionMaskProvider;
        }
        var evalEnv = envFactory(); // dedicated eval env — never disturbs the actors' in-flight episodes

        var agent = new GreedyQAgent(state.Online, actionCount, state.PolicyRng);
        float lastLoss = state.LastLoss;
        double lastEval = state.LastEval;

        void ResampleNoise(IValueNet net) { if (options.NoisyNets) ((DuelingQNet)net).ResampleNoise(state.NoiseRng); }
        void SetNoiseEnabled(bool on)
        {
            if (!options.NoisyNets) return;
            ((DuelingQNet)state.Online).SetNoiseEnabled(on);
            ((DuelingQNet)state.Target).SetNoiseEnabled(on);
        }

        DqnResult Finish(int stepsTrained)
        {
            SetNoiseEnabled(false);
            state.CurrentObs = obs[0];
            state.StepsCompleted = stepsTrained;
            state.LastLoss = lastLoss;
            state.LastEval = lastEval;
            state.EnvState = null; // vectorized: no per-actor env snapshot — resume restarts the envs
            return new DqnResult(agent, state.Online, stepsTrained, lastEval, state);
        }

        SetNoiseEnabled(true);
        int step = state.StepsCompleted;
        int trainAccum = 0;
        var actions = new int[n];
        var nextMasks = new bool[n][];
        var results = new StepResult<float[]>[n];

        while (step < options.MaxSteps)
        {
            // Action selection (cheap; the net is read single-threaded here). One noise draw per tick.
            ResampleNoise(state.Online);
            double eps = options.NoisyNets ? 0 : options.Epsilon.Value(step + 1);
            agent.Epsilon = eps;
            for (int i = 0; i < n; i++)
                actions[i] = agent.Act(obs[i], maskProviders[i]?.CurrentActionMask());

            // The expensive part: step every actor in parallel. Each env owns its RNG, so results are
            // independent of thread scheduling; we read them back by index for determinism.
            Parallel.For(0, n, i =>
            {
                results[i] = envs[i].Step(actions[i]);
                nextMasks[i] = maskProviders[i]?.CurrentActionMask();
            });

            for (int i = 0; i < n; i++)
            {
                state.Buffer.Add(obs[i], actions[i], results[i].Reward, results[i].Observation, results[i].Terminated, nextMasks[i]);
                obs[i] = results[i].Done ? envs[i].Reset().Observation : results[i].Observation;
            }
            step += n;

            // Keep ~one TD update per TrainEvery env-steps (so the update:experience ratio matches scalar).
            if (state.Buffer.Count >= options.WarmupSteps)
            {
                trainAccum += n;
                while (trainAccum >= options.TrainEvery)
                {
                    trainAccum -= options.TrainEvery;
                    ResampleNoise(state.Online);
                    ResampleNoise(state.Target);
                    lastLoss = TrainStep(state.Online, state.Target, state.Optimizer,
                        state.Buffer.Sample(options.BatchSize, state.BufferRng), options);
                }
            }

            // step jumps by n, so fire cadence whenever the interval boundary is crossed within the tick.
            if (Crossed(step - n, step, options.TargetSyncEvery))
                state.Target.CopyFrom(state.Online);

            if (Crossed(step - n, step, options.EvalEvery))
            {
                SetNoiseEnabled(false);
                lastEval = Evaluator.Evaluate(evalEnv, (float[] o, bool[]? m) => agent.Act(o, m, greedy: true),
                    options.EvalEpisodes, seeds.Derive(RngStreams.Evaluation)).MeanReturn;
                SetNoiseEnabled(true);
                options.OnProgress?.Invoke(new DqnProgress(step, options.MaxSteps, lastEval, eps, lastLoss));

                if (options.SolveThreshold.HasValue && lastEval >= options.SolveThreshold.Value)
                    return Finish(step);
            }
        }

        return Finish(step);
    }

    /// <summary>True if a positive multiple of <paramref name="interval"/> lies in (<paramref name="from"/>, <paramref name="to"/>].</summary>
    private static bool Crossed(int from, int to, int interval)
        => interval > 0 && to / interval > from / interval;

    private static float TrainStep(IValueNet online, IValueNet target, Adam adam, ReplayBuffer.Batch batch, DqnOptions options)
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
