using MintPlayer.AI.ReinforcementLearning.Core.Agents.Tabular;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

public sealed record TabularTrainingOptions
{
    public required int Episodes { get; init; }
    public LinearSchedule Epsilon { get; init; } = new(1.0, 0.05, 1);
    public LinearSchedule Alpha { get; init; } = new(0.1, 0.1, 1);

    /// <summary>Invoked every <see cref="ProgressInterval"/> episodes with rolling stats.</summary>
    public Action<TrainingProgress>? OnProgress { get; init; }
    public int ProgressInterval { get; init; } = 1000;
}

public readonly record struct TrainingProgress(int Episode, int TotalEpisodes, double AvgReturn100, double Epsilon);

public sealed record TabularTrainingResult(double[] EpisodeReturns, long TotalSteps);

/// <summary>
/// Generic episodic training loop for tabular TD agents. Note the terminated/truncated
/// distinction: <c>Observe</c> receives only <c>Terminated</c>, so targets still
/// bootstrap at time-limit truncations.
/// </summary>
public static class TabularTrainer
{
    public static TabularTrainingResult Train(
        ITabularEnvironment env,
        TabularAgent agent,
        TabularTrainingOptions options,
        ulong envSeed,
        MetricsLogger? logger = null)
    {
        var returns = new double[options.Episodes];
        var window = new Queue<double>(100);
        double windowSum = 0;
        long totalSteps = 0;

        // Seed once; subsequent resets continue the env's RNG stream deterministically.
        env.Reset(envSeed);

        for (int episode = 0; episode < options.Episodes; episode++)
        {
            agent.Epsilon = options.Epsilon.Value(episode);
            agent.Alpha = options.Alpha.Value(episode);

            var (state, _) = env.Reset();
            double episodeReturn = 0;
            int episodeLength = 0;

            while (true)
            {
                int action = agent.Act(state);
                var step = env.Step(action);
                agent.Observe(state, action, step.Reward, step.Observation, step.Terminated);

                episodeReturn += step.Reward;
                episodeLength++;
                totalSteps++;
                state = step.Observation;

                if (step.Done) break;
            }

            returns[episode] = episodeReturn;
            if (window.Count == 100) windowSum -= window.Dequeue();
            window.Enqueue(episodeReturn);
            windowSum += episodeReturn;
            double avg100 = windowSum / window.Count;

            logger?.Log(episode, episodeReturn, episodeLength, agent.Epsilon, avg100);

            if (options.OnProgress is not null && (episode + 1) % options.ProgressInterval == 0)
                options.OnProgress(new TrainingProgress(episode + 1, options.Episodes, avg100, agent.Epsilon));
        }

        return new TabularTrainingResult(returns, totalSteps);
    }
}
