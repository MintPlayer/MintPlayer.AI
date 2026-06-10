using RLNet.Core.Environments;

namespace RLNet.Environments;

/// <summary>
/// Deterministic 4×4 grid world: start top-left, goal bottom-right.
/// Each move costs −0.04; entering the goal yields +1 and terminates.
/// Small enough that value iteration gives an exact optimal policy,
/// making it the unit-test oracle environment for tabular agents.
/// </summary>
public sealed class GridWorldEnv() : GridEnvironmentBase(DefaultMap, maxEpisodeSteps: 100)
{
    private static readonly string[] DefaultMap =
    [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ];

    public const double StepReward = -0.04;
    public const double GoalReward = 1.0;

    public override IEnumerable<Transition> Model(int state, int action)
    {
        yield return DeterministicTransition(state, action, probability: 1.0);
    }

    protected override bool IsTerminalCell(char cell) => cell == 'G';

    protected override double RewardFor(char destinationCell) => destinationCell == 'G' ? GoalReward : StepReward;
}
