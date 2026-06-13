namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// A deterministic, goal-directed transition model — the minimum that classical search and
/// model-based learning (value iteration / DAVI, AlphaZero-style planning) share: enumerate
/// actions, apply one to get the successor, test the goal, and key a state for visited-sets.
/// <para>
/// Distinct from <see cref="Environments.IEnvironment{TObs,TAct}"/>, which is the RL
/// Reset/Step interaction loop with rewards. This is the pure forward model — no reward, no
/// episode state — that a planner queries to look ahead. An environment that knows its own
/// dynamics can expose both. <see cref="Apply"/> MUST return a fresh successor and leave
/// <paramref name="state"/> untouched (planners reuse a state across all its actions).
/// </para>
/// </summary>
public interface IDeterministicModel<TState>
{
    /// <summary>Number of discrete actions, valid in every state (0..ActionCount-1).</summary>
    int ActionCount { get; }

    /// <summary>True when <paramref name="state"/> is a goal (e.g. a solved cube).</summary>
    bool IsGoal(TState state);

    /// <summary>The successor of applying <paramref name="action"/>; does not mutate <paramref name="state"/>.</summary>
    TState Apply(TState state, int action);

    /// <summary>A key identifying <paramref name="state"/> for visited-set deduplication (equal states ⇒ equal keys).</summary>
    string StateKey(TState state);
}
