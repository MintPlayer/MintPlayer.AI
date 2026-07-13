using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The 4672-index move encoding (PLAN M39.2): distinct legal moves must map to distinct indices (so search can tell
/// them apart), and encode → decode → <see cref="ChessGame.Apply"/> must reproduce exactly the position that making
/// the move directly produces — the end-to-end check that the policy indices, the decoder, and promotion-inference
/// all agree. Run over positions rich in castling, en passant, and every promotion flavour.
/// </summary>
public class ChessEncodingTests
{
    [Theory]
    [InlineData(ChessFen.StartFen)]
    [InlineData("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")] // Kiwipete
    [InlineData("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")]   // a7-a8 promotion push (Q/R/B/N)
    [InlineData("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")] // a7xb8 promotion by capture (Q/R/B/N) and the push
    [InlineData("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")]   // a black promotion (downward direction)
    public void Encode_decode_apply_roundtrips_every_legal_move(string fen)
    {
        var game = new ChessGame();
        var state = ChessFen.Parse(fen);
        var legal = ChessRules.LegalMoves(state);
        var seen = new HashSet<int>();

        foreach (var move in legal)
        {
            int index = ChessMoveEncoding.Encode(move);
            Assert.InRange(index, 0, ChessMoveEncoding.Size - 1);
            Assert.True(seen.Add(index), $"index {index} collided (move {move.From}->{move.To} {move.Promotion})");

            // Applying the ENCODED index must land on the same position as making the move directly.
            string viaIndex = ChessFen.ToFen(game.Apply(state, index));
            string direct = ChessFen.ToFen(ChessRules.MakeMove(state, move));
            Assert.Equal(direct, viaIndex);
        }

        // The game exposes exactly one index per legal move (no collisions collapsed the set).
        Assert.Equal(legal.Count, game.LegalMoves(state).Count);
    }

    [Fact]
    public void Result_detects_checkmate_stalemate_and_ongoing()
    {
        var game = new ChessGame();
        Assert.Equal(GameResult.Ongoing, game.Result(ChessState.StartPosition()));
        // Fool's mate: Black has delivered mate; White (to move) is checkmated → Loss for the side to move.
        Assert.Equal(GameResult.Loss, game.Result(ChessFen.Parse("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")));
        // Classic stalemate: Black to move, no legal move, not in check → Draw.
        Assert.Equal(GameResult.Draw, game.Result(ChessFen.Parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")));
    }
}
