namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// An OPTIONAL capability an <see cref="IZeroSumGame{TState}"/> may also implement to expose a dense, per-position
/// material/score signal (from the side-to-move's perspective). Self-play uses it to <b>shape the value target</b> so
/// a still-weak net gets a gradient on every capture — not just the sparse win/loss/draw outcome, which for a net
/// that can't yet force mate is ~always a draw (→ 0) and teaches nothing. It also gives the difficulty ladder a
/// non-saturating strength metric (two "drawing" nets are separable by average material). Games with no material
/// notion simply don't implement it, and self-play falls back to the pure outcome.
/// </summary>
/// <typeparam name="TState">The game state type.</typeparam>
public interface IMaterialScore<TState>
{
    /// <summary>The side-to-move's material advantage (own material − opponent's), in the game's natural units
    /// (chess: pawns). Positive = the side to move is ahead.</summary>
    float MaterialAdvantage(TState state);
}
