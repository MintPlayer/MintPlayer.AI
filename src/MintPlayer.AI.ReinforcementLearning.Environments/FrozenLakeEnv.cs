using MintPlayer.AI.ReinforcementLearning.Core.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Environments;

/// <summary>
/// Faithful port of Gymnasium FrozenLake-v1 (4×4 map, is_slippery=true):
/// the intended action executes with probability ⅓, and each perpendicular
/// direction with probability ⅓. Reaching the goal yields +1; holes terminate
/// with 0. Time limit 100 steps (truncation, not termination).
/// Gymnasium-comparable solved threshold: success rate ≥ 0.70 over 100 episodes.
/// </summary>
public sealed class FrozenLakeEnv(bool slippery = true)
    : GridEnvironmentBase(DefaultMap, maxEpisodeSteps: 100)
{
    private static readonly string[] DefaultMap =
    [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ];

    public override IEnumerable<Transition> Model(int state, int action)
    {
        if (!slippery)
        {
            yield return DeterministicTransition(state, action, probability: 1.0);
            yield break;
        }

        // Gymnasium: intended direction and both perpendiculars, ⅓ each ((a−1)%4, a, (a+1)%4).
        for (int offset = -1; offset <= 1; offset++)
            yield return DeterministicTransition(state, (action + offset + 4) % 4, probability: 1.0 / 3.0);
    }

    protected override bool IsTerminalCell(char cell) => cell is 'G' or 'H';

    protected override double RewardFor(char destinationCell) => destinationCell == 'G' ? 1.0 : 0.0;
}
