using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Perft: count the legal-move tree to a fixed depth and match the published node counts — the
/// non-negotiable correctness gate for the draughts move generator (PLAN M47.1; no training before
/// this passes). A node is a complete move (a full capture sequence, deduped by FMJD identity:
/// from, to, captured set), which is the convention the published tables use. Sources: the FMJD
/// perft thread (Halbersma) for international 10×10; the standard english checkers table (Fierz/Bik)
/// for 8×8. Shallow depths run in the fast bucket; the multi-million-node depths are marked Slow.
/// </summary>
public class DraughtsPerftTests
{
    [Theory]
    [InlineData(1, 9L)]
    [InlineData(2, 81L)]
    [InlineData(3, 658L)]
    [InlineData(4, 4265L)]
    [InlineData(5, 27117L)]
    [InlineData(6, 167140L)]
    public void Perft_international_start_matches_published_counts(int depth, long expected)
        => Assert.Equal(expected, DraughtsRules.Perft(DraughtsState.StartPosition(DraughtsVariant.International10), depth));

    [Theory]
    [InlineData(1, 7L)]
    [InlineData(2, 49L)]
    [InlineData(3, 302L)]
    [InlineData(4, 1469L)]
    [InlineData(5, 7361L)]
    [InlineData(6, 36768L)]
    [InlineData(7, 179740L)]
    public void Perft_english_start_matches_published_counts(int depth, long expected)
        => Assert.Equal(expected, DraughtsRules.Perft(DraughtsState.StartPosition(DraughtsVariant.English8), depth));

    [Theory]
    [Trait("Category", "Slow")]
    [InlineData(DraughtsVariant.International10, 7, 1049442L)]
    [InlineData(DraughtsVariant.International10, 8, 6483961L)]
    [InlineData(DraughtsVariant.English8, 8, 845931L)]
    [InlineData(DraughtsVariant.English8, 9, 3963680L)]
    public void Perft_deep_matches_published_counts(DraughtsVariant variant, int depth, long expected)
        => Assert.Equal(expected, DraughtsRules.Perft(DraughtsState.StartPosition(variant), depth));
}
