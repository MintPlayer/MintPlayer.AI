using RLNet.Core.Random;

namespace RLNet.Core.Agents.Tabular;

/// <summary>
/// Shared base for tabular TD agents: a dense Q-table (double precision so exact-value
/// tests stay clean), epsilon-greedy action selection, and trainer-driven schedules.
/// </summary>
public abstract class TabularAgent : IAgent<int, int>
{
    protected TabularAgent(int stateCount, int actionCount, Xoshiro256StarStar rng)
    {
        Q = new double[stateCount, actionCount];
        ActionCount = actionCount;
        Rng = rng;
    }

    public double[,] Q { get; }
    public int ActionCount { get; }
    protected Xoshiro256StarStar Rng { get; }

    /// <summary>Exploration rate; set per step/episode by the trainer from a schedule.</summary>
    public double Epsilon { get; set; } = 1.0;

    /// <summary>Learning rate; set per step/episode by the trainer from a schedule.</summary>
    public double Alpha { get; set; } = 0.1;

    public double Gamma { get; init; } = 0.99;

    public virtual int Act(int observation, bool greedy = false)
    {
        if (!greedy && Rng.NextDouble() < Epsilon)
            return Rng.NextInt(ActionCount);
        return GreedyAction(observation);
    }

    /// <summary>Records one transition and updates the Q-table.</summary>
    public abstract void Observe(int state, int action, double reward, int nextState, bool terminated);

    /// <summary>Argmax over Q[state, ·] with deterministic first-max tie-breaking.</summary>
    public int GreedyAction(int state)
    {
        int best = 0;
        double bestValue = Q[state, 0];
        for (int a = 1; a < ActionCount; a++)
        {
            if (Q[state, a] > bestValue)
            {
                bestValue = Q[state, a];
                best = a;
            }
        }
        return best;
    }

    protected double MaxQ(int state)
    {
        double max = Q[state, 0];
        for (int a = 1; a < ActionCount; a++)
            max = Math.Max(max, Q[state, a]);
        return max;
    }
}
