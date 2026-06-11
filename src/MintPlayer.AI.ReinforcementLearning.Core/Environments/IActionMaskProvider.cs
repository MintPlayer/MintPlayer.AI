namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>
/// Implemented by discrete-action environments where not every action is legal in every
/// state (2048, Rush Hour, board games). Trainers detect this interface and restrict both
/// exploration and TD-target argmax/max to legal actions.
/// </summary>
public interface IActionMaskProvider
{
    /// <summary>Legality of each action in the CURRENT state (true = legal).</summary>
    bool[] CurrentActionMask();
}
