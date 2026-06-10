using RLNet.Core.Agents;
using RLNet.Core.Environments;

namespace RLNet.Core.Training;

public sealed record EvalResult(double[] Returns, double[] Lengths)
{
    public double MeanReturn => Returns.Average();
    public double MeanLength => Lengths.Average();

    /// <summary>Fraction of episodes whose return exceeds the threshold (e.g. FrozenLake success = return &gt; 0).</summary>
    public double SuccessRate(double threshold = 0) => Returns.Count(r => r > threshold) / (double)Returns.Length;
}

/// <summary>Runs greedy (exploration-free) evaluation episodes.</summary>
public static class Evaluator
{
    public static EvalResult Evaluate<TObs, TAct>(
        IEnvironment<TObs, TAct> env,
        IAgent<TObs, TAct> agent,
        int episodes,
        ulong seed)
        => Evaluate(env, (obs, _) => agent.Act(obs, greedy: true), episodes, seed);

    /// <summary>
    /// Policy-function variant. For <see cref="IActionMaskProvider"/> environments the
    /// current action mask is passed to the policy; otherwise the mask is null.
    /// </summary>
    public static EvalResult Evaluate<TObs, TAct>(
        IEnvironment<TObs, TAct> env,
        Func<TObs, bool[]?, TAct> policy,
        int episodes,
        ulong seed)
    {
        var maskProvider = env as IActionMaskProvider;
        var returns = new double[episodes];
        var lengths = new double[episodes];

        env.Reset(seed);
        for (int i = 0; i < episodes; i++)
        {
            var (obs, _) = env.Reset();
            double episodeReturn = 0;
            int length = 0;

            while (true)
            {
                var step = env.Step(policy(obs, maskProvider?.CurrentActionMask()));
                episodeReturn += step.Reward;
                length++;
                obs = step.Observation;
                if (step.Done) break;
            }

            returns[i] = episodeReturn;
            lengths[i] = length;
        }

        return new EvalResult(returns, lengths);
    }
}
