namespace MintPlayer.AI.ReinforcementLearning.Core.Agents;

public interface IAgent<TObs, TAct>
{
    /// <summary>
    /// Selects an action. <paramref name="greedy"/> disables exploration
    /// (used for evaluation and playback).
    /// </summary>
    TAct Act(TObs observation, bool greedy = false);
}
