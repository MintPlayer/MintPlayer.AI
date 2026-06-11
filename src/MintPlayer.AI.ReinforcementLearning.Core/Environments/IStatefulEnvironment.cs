namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>
/// Environments that can snapshot and restore their COMPLETE state — physics/board,
/// step counters, episode flags AND the internal RNG — so that training can resume
/// from a checkpoint bitwise-identically to a run that was never interrupted.
/// The blob is opaque and env-specific; only the same environment type reads it back.
/// </summary>
public interface IStatefulEnvironment
{
    byte[] SaveState();
    void RestoreState(byte[] state);
}
