namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>
/// A finite-MDP environment whose full transition model is known, enabling exact
/// solvers (value iteration) to act as test oracles for learning algorithms.
/// </summary>
public interface ITabularEnvironment : IEnvironment<int, int>
{
    int StateCount { get; }
    int ActionCount { get; }

    /// <summary>All possible outcomes of taking <paramref name="action"/> in <paramref name="state"/>.</summary>
    IEnumerable<Transition> Model(int state, int action);

    /// <summary>True if the state is terminal (its transitions are never taken).</summary>
    bool IsTerminal(int state);
}

public readonly record struct Transition(double Probability, int NextState, double Reward, bool Terminated);
