using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

public sealed record PpoOptions
{
    public int[] Hidden { get; init; } = [64, 64];
    public int NumEnvs { get; init; } = 8;
    public int RolloutSteps { get; init; } = 128;
    public int UpdateEpochs { get; init; } = 4;
    public int MinibatchSize { get; init; } = 256;
    public double Gamma { get; init; } = 0.99;
    public double GaeLambda { get; init; } = 0.95;
    public float ClipCoef { get; init; } = 0.2f;
    public float EntropyCoef { get; init; } = 0.01f;
    public float ValueCoef { get; init; } = 0.5f;
    public float MaxGradNorm { get; init; } = 0.5f;
    public float LearningRate { get; init; } = 2.5e-4f;
    public bool AnnealLearningRate { get; init; } = true;
    public int TotalSteps { get; init; } = 300_000;

    /// <summary>Step envs on parallel Tasks. Results are identical to sequential (each env owns its RNG).</summary>
    public bool ParallelEnvs { get; init; }

    public int EvalEveryIterations { get; init; } = 10;
    public int EvalEpisodes { get; init; } = 20;
    public double? SolveThreshold { get; init; }

    public Action<PpoProgress>? OnProgress { get; init; }
}

public readonly record struct PpoProgress(
    int EnvSteps, int TotalSteps, double AvgReturn100, double EvalMeanReturn,
    float ApproxKl, float ClipFraction, float ExplainedVariance, float LearningRate);

public sealed record PpoResult(PolicyAgent Agent, Mlp Actor, Mlp Critic, int StepsTrained, double FinalEvalReturn);

/// <summary>
/// PPO with clipped surrogate objective (Schulman et al. 2017), following the
/// implementation details that matter (CleanRL/37-details): vectorized envs with
/// correct truncation bootstrapping, GAE(λ), per-minibatch advantage normalization,
/// orthogonal init (√2 hidden, 0.01 policy head, 1.0 value head), lr annealing,
/// grad-norm clip, and approx-KL / clip-fraction / explained-variance diagnostics.
/// </summary>
public static class PpoTrainer
{
    public static PpoResult Train(
        Func<int, IEnvironment<float[], int>> envFactory,
        IEnvironment<float[], int> evalEnv,
        PpoOptions options,
        SeedSequence seeds)
    {
        var vec = new VectorEnv(envFactory, options.NumEnvs, options.ParallelEnvs);
        int obsDim = vec.ObsDim, actionCount = vec.ActionCount;
        int t = options.RolloutSteps, n = options.NumEnvs, batch = t * n;

        var initRng = seeds.CreateRng(RngStreams.Init);
        var actor = BuildNet([obsDim, .. options.Hidden, actionCount], headGain: 0.01f, initRng);
        var critic = BuildNet([obsDim, .. options.Hidden, 1], headGain: 1.0f, initRng);
        var adam = new Adam([.. actor.Parameters(), .. critic.Parameters()], options.LearningRate);
        var policyRng = seeds.CreateRng(RngStreams.Policy);
        var bufferRng = seeds.CreateRng(RngStreams.Buffer);
        var agent = new PolicyAgent(actor, policyRng);

        // Rollout storage, [T,N] row-major.
        var obsBuf = new float[batch * obsDim];
        var actionsBuf = new int[batch];
        var logProbsBuf = new float[batch];
        var valuesBuf = new float[batch];
        var rewardsBuf = new float[batch];
        var terminatedBuf = new bool[batch];
        var doneBuf = new bool[batch];
        var finalValuesBuf = new float[batch];

        var obs = vec.Reset(seeds.Derive(RngStreams.Environment));
        var episodeReturns = new double[n];
        var returnWindow = new Queue<double>(100);
        double windowSum = 0, avg100 = 0, lastEval = double.NegativeInfinity;
        int envSteps = 0, iteration = 0;

        while (envSteps < options.TotalSteps)
        {
            iteration++;
            if (options.AnnealLearningRate)
                adam.LearningRate = options.LearningRate * (1f - (float)envSteps / options.TotalSteps);

            // ---- collect one rollout ----
            for (int step = 0; step < t; step++)
            {
                obs.CopyTo(obsBuf.AsSpan(step * n * obsDim, n * obsDim));

                int[] actions;
                using (GradMode.NoGrad())
                {
                    var obsT = new Tensor(obs, n, obsDim);
                    var dist = new Categorical(actor.Forward(obsT));
                    actions = dist.Sample(policyRng);
                    var logProbs = dist.LogProb(actions);
                    var values = critic.Forward(obsT);
                    for (int i = 0; i < n; i++)
                    {
                        logProbsBuf[step * n + i] = logProbs.Data[i];
                        valuesBuf[step * n + i] = values.Data[i];
                    }
                }
                actions.CopyTo(actionsBuf.AsSpan(step * n, n));

                var vecStep = vec.Step(actions);
                obs = vecStep.Obs;
                envSteps += n;

                for (int i = 0; i < n; i++)
                {
                    int idx = step * n + i;
                    rewardsBuf[idx] = (float)vecStep.Rewards[i];
                    terminatedBuf[idx] = vecStep.Terminated[i];
                    doneBuf[idx] = vecStep.Done[i];
                    finalValuesBuf[idx] = 0f;

                    episodeReturns[i] += vecStep.Rewards[i];
                    if (vecStep.Done[i])
                    {
                        if (returnWindow.Count == 100) windowSum -= returnWindow.Dequeue();
                        returnWindow.Enqueue(episodeReturns[i]);
                        windowSum += episodeReturns[i];
                        avg100 = windowSum / returnWindow.Count;
                        episodeReturns[i] = 0;
                    }
                }

                // Truncated episodes bootstrap from V(final_observation).
                var truncatedIdx = Enumerable.Range(0, n)
                    .Where(i => vecStep.Truncated[i] && !vecStep.Terminated[i]).ToArray();
                if (truncatedIdx.Length > 0)
                {
                    using (GradMode.NoGrad())
                    {
                        var finals = new float[truncatedIdx.Length * obsDim];
                        for (int j = 0; j < truncatedIdx.Length; j++)
                            vecStep.FinalObs.AsSpan(truncatedIdx[j] * obsDim, obsDim)
                                .CopyTo(finals.AsSpan(j * obsDim, obsDim));
                        var finalV = critic.Forward(new Tensor(finals, truncatedIdx.Length, obsDim));
                        for (int j = 0; j < truncatedIdx.Length; j++)
                            finalValuesBuf[step * n + truncatedIdx[j]] = finalV.Data[j];
                    }
                }
            }

            // ---- GAE ----
            float[] bootstrap;
            using (GradMode.NoGrad())
                bootstrap = critic.Forward(new Tensor(obs, n, obsDim)).Reshape(n).Data;

            var advantages = Gae.Compute(rewardsBuf, valuesBuf, terminatedBuf, doneBuf,
                finalValuesBuf, bootstrap, t, n, options.Gamma, options.GaeLambda);
            var returns = new float[batch];
            for (int i = 0; i < batch; i++) returns[i] = advantages[i] + valuesBuf[i];

            // ---- optimize ----
            float approxKl = 0f, clipFraction = 0f;
            int klSamples = 0;
            var indices = Enumerable.Range(0, batch).ToArray();

            for (int epoch = 0; epoch < options.UpdateEpochs; epoch++)
            {
                Shuffle(indices, bufferRng);
                for (int start = 0; start < batch; start += options.MinibatchSize)
                {
                    int size = Math.Min(options.MinibatchSize, batch - start);
                    var mb = indices.AsSpan(start, size);

                    var mbObs = new float[size * obsDim];
                    var mbActions = new int[size];
                    var mbOldLogProbs = new float[size];
                    var mbAdv = new float[size];
                    var mbReturns = new float[size];
                    for (int j = 0; j < size; j++)
                    {
                        obsBuf.AsSpan(mb[j] * obsDim, obsDim).CopyTo(mbObs.AsSpan(j * obsDim, obsDim));
                        mbActions[j] = actionsBuf[mb[j]];
                        mbOldLogProbs[j] = logProbsBuf[mb[j]];
                        mbAdv[j] = advantages[mb[j]];
                        mbReturns[j] = returns[mb[j]];
                    }
                    NormalizeInPlace(mbAdv);

                    adam.ZeroGrad();
                    var obsT = new Tensor(mbObs, size, obsDim);
                    var dist = new Categorical(actor.Forward(obsT));
                    var newLogProbs = dist.LogProb(mbActions);
                    var ratio = newLogProbs.Sub(new Tensor(mbOldLogProbs, size)).Exp();
                    var advT = new Tensor(mbAdv, size);

                    var surrogate = ratio.Mul(advT)
                        .Min(ratio.Clamp(1f - options.ClipCoef, 1f + options.ClipCoef).Mul(advT));
                    var policyLoss = surrogate.Mean().MulScalar(-1f);
                    var valueLoss = critic.Forward(obsT).Reshape(size)
                        .MseLoss(new Tensor(mbReturns, size)).MulScalar(0.5f * options.ValueCoef);
                    var entropyBonus = dist.Entropy().Mean().MulScalar(-options.EntropyCoef);

                    var loss = policyLoss.Add(valueLoss).Add(entropyBonus);
                    loss.Backward();
                    adam.ClipGradNorm(options.MaxGradNorm);
                    adam.Step();

                    // Diagnostics (no grad): kl ≈ mean((ratio−1) − log ratio), clip fraction.
                    for (int j = 0; j < size; j++)
                    {
                        float r = ratio.Data[j];
                        approxKl += (r - 1f) - MathF.Log(r);
                        if (Math.Abs(r - 1f) > options.ClipCoef) clipFraction++;
                    }
                    klSamples += size;
                }
            }

            float explainedVariance = ExplainedVariance(valuesBuf, returns);
            options.OnProgress?.Invoke(new PpoProgress(
                envSteps, options.TotalSteps, avg100, lastEval,
                approxKl / klSamples, clipFraction / klSamples, explainedVariance, adam.LearningRate));

            if (iteration % options.EvalEveryIterations == 0)
            {
                lastEval = Evaluator.Evaluate(evalEnv, agent, options.EvalEpisodes,
                    seeds.Derive(RngStreams.Evaluation)).MeanReturn;
                if (options.SolveThreshold.HasValue && lastEval >= options.SolveThreshold.Value)
                    return new PpoResult(agent, actor, critic, envSteps, lastEval);
            }
        }

        return new PpoResult(agent, actor, critic, envSteps, lastEval);
    }

    private static Mlp BuildNet(int[] sizes, float headGain, Xoshiro256StarStar rng)
    {
        var net = new Mlp(sizes, rng, Activation.Tanh);
        for (int i = 0; i < net.Layers.Count; i++)
            Init.Orthogonal(net.Layers[i].Weight, i == net.Layers.Count - 1 ? headGain : MathF.Sqrt(2f), rng);
        return net;
    }

    private static void NormalizeInPlace(float[] values)
    {
        float mean = values.Average();
        float variance = 0f;
        foreach (float v in values) variance += (v - mean) * (v - mean);
        float std = MathF.Sqrt(variance / values.Length) + 1e-8f;
        for (int i = 0; i < values.Length; i++) values[i] = (values[i] - mean) / std;
    }

    private static float ExplainedVariance(float[] values, float[] returns)
    {
        float meanReturn = returns.Average();
        float varReturn = 0f, varResidual = 0f;
        for (int i = 0; i < returns.Length; i++)
        {
            varReturn += (returns[i] - meanReturn) * (returns[i] - meanReturn);
            varResidual += (returns[i] - values[i]) * (returns[i] - values[i]);
        }
        return varReturn < 1e-8f ? 0f : 1f - varResidual / varReturn;
    }

    private static void Shuffle(int[] array, Xoshiro256StarStar rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
