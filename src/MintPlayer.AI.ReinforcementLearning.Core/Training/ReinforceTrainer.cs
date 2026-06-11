using MintPlayer.AI.ReinforcementLearning.Core.Agents;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

public sealed record ReinforceOptions
{
    public int[] Hidden { get; init; } = [64];
    public double Gamma { get; init; } = 0.99;
    public float LearningRate { get; init; } = 2.5e-3f;
    public int EpisodesPerUpdate { get; init; } = 4;
    public int MaxEpisodes { get; init; } = 5_000;
    public bool NormalizeReturns { get; init; } = true;
    public float MaxGradNorm { get; init; } = 10f;

    /// <summary>Training stops early when the rolling-100 training return reaches this.</summary>
    public double? SolveThreshold { get; init; }

    public Action<TrainingProgress>? OnProgress { get; init; }
    public int ProgressInterval { get; init; } = 100;
}

public sealed record ReinforceResult(PolicyAgent Agent, Mlp Network, int EpisodesTrained, double FinalAvgReturn100);

/// <summary>Samples (or argmaxes) actions from a policy network's categorical head.</summary>
public sealed class PolicyAgent(Mlp network, Xoshiro256StarStar rng) : IAgent<float[], int>
{
    public int Act(float[] observation, bool greedy = false)
    {
        using (GradMode.NoGrad())
        {
            var dist = new Categorical(network.Forward(new Tensor(observation, 1, observation.Length)));
            return greedy ? dist.Mode()[0] : dist.Sample(rng)[0];
        }
    }
}

/// <summary>
/// REINFORCE (vanilla policy gradient) with reward-to-go returns and per-update return
/// normalization. Loss: −E[log π(a|s) · G]. Single-file reference per the CleanRL discipline.
/// </summary>
public static class ReinforceTrainer
{
    public static ReinforceResult Train(IEnvironment<float[], int> env, ReinforceOptions options, SeedSequence seeds)
    {
        int obsDim = ((BoxSpace)env.ObservationSpace).Dimensions;
        int actionCount = ((DiscreteSpace)env.ActionSpace).N;

        var network = new Mlp([obsDim, .. options.Hidden, actionCount], seeds.CreateRng(RngStreams.Init), Activation.Tanh);
        var adam = new Adam(network.Parameters(), options.LearningRate);
        var agent = new PolicyAgent(network, seeds.CreateRng(RngStreams.Policy));

        env.Reset(seeds.Derive(RngStreams.Environment));
        var window = new Queue<double>(100);
        double windowSum = 0, avg100 = 0;

        var batchObs = new List<float[]>();
        var batchActions = new List<int>();
        var batchReturns = new List<float>();
        var episodeRewards = new List<double>();

        for (int episode = 1; episode <= options.MaxEpisodes; episode++)
        {
            var (obs, _) = env.Reset();
            episodeRewards.Clear();

            while (true)
            {
                int action = agent.Act(obs);
                var step = env.Step(action);
                batchObs.Add(obs);
                batchActions.Add(action);
                episodeRewards.Add(step.Reward);
                obs = step.Observation;
                if (step.Done) break;
            }

            // Reward-to-go: G_t = r_t + γ·G_{t+1}.
            double g = 0;
            var returns = new float[episodeRewards.Count];
            for (int t = episodeRewards.Count - 1; t >= 0; t--)
            {
                g = episodeRewards[t] + options.Gamma * g;
                returns[t] = (float)g;
            }
            batchReturns.AddRange(returns);

            double episodeReturn = episodeRewards.Sum();
            if (window.Count == 100) windowSum -= window.Dequeue();
            window.Enqueue(episodeReturn);
            windowSum += episodeReturn;
            avg100 = windowSum / window.Count;

            if (episode % options.EpisodesPerUpdate == 0)
            {
                Update(network, adam, batchObs, batchActions, batchReturns, obsDim, options);
                batchObs.Clear();
                batchActions.Clear();
                batchReturns.Clear();
            }

            if (options.OnProgress is not null && episode % options.ProgressInterval == 0)
                options.OnProgress(new TrainingProgress(episode, options.MaxEpisodes, avg100, 0));

            if (options.SolveThreshold.HasValue && window.Count == 100 && avg100 >= options.SolveThreshold.Value)
                return new ReinforceResult(agent, network, episode, avg100);
        }

        return new ReinforceResult(agent, network, options.MaxEpisodes, avg100);
    }

    private static void Update(Mlp network, Adam adam, List<float[]> obs, List<int> actions, List<float> returns,
        int obsDim, ReinforceOptions options)
    {
        int n = obs.Count;
        var obsData = new float[n * obsDim];
        for (int i = 0; i < n; i++)
            obs[i].CopyTo(obsData.AsSpan(i * obsDim, obsDim));

        var advantages = returns.ToArray();
        if (options.NormalizeReturns)
        {
            float mean = advantages.Average();
            float std = (float)Math.Sqrt(advantages.Sum(r => (r - mean) * (r - mean)) / advantages.Length) + 1e-8f;
            for (int i = 0; i < advantages.Length; i++)
                advantages[i] = (advantages[i] - mean) / std;
        }

        adam.ZeroGrad();
        var logProbs = new Categorical(network.Forward(new Tensor(obsData, n, obsDim))).LogProb([.. actions]);
        var loss = logProbs.Mul(new Tensor(advantages, n)).Mean().MulScalar(-1f);
        loss.Backward();
        adam.ClipGradNorm(options.MaxGradNorm);
        adam.Step();
    }
}
