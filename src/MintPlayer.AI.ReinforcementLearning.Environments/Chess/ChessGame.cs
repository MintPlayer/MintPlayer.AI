using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// Chess as an <see cref="IZeroSumGame{TState}"/> — the headline consumer of the self-play stack (PLAN M39.2). It
/// adapts the perft-verified <see cref="ChessRules"/> engine to the seam: the action space is the 4672-index
/// <see cref="ChessMoveEncoding"/>, and the observation is an 18-plane board stack (12 piece planes + side-to-move +
/// 4 castling-rights + en passant), flattened. MCTS, the two-headed net, and the self-play campaign are all reused
/// unchanged from M39.1. (Board geometry is absolute — the net learns both colours via the side-to-move plane;
/// perspective canonicalization is a later efficiency lever, PLAN M39.3.)
/// </summary>
public sealed class ChessGame : IZeroSumGame<ChessState>, IMaterialScore<ChessState>
{
    private const int Planes = 18;

    // Standard relative piece values, indexed by |piece| (none, P, N, B, R, Q, K). The king isn't "captured"
    // (reaching it is checkmate — the ±1 game outcome), so it scores 0 and cancels between the two sides.
    private static readonly int[] PieceValues = [0, 1, 3, 3, 5, 9, 0];

    public int PolicySize => ChessMoveEncoding.Size;      // 4672
    public int ObservationSize => Planes * 64;            // 1152

    /// <summary>The side-to-move's material advantage in pawns (its pieces − the opponent's). Dense reward signal
    /// for self-play shaping + the difficulty ladder's strength metric (<see cref="IMaterialScore{TState}"/>).</summary>
    public float MaterialAdvantage(ChessState state)
    {
        int white = 0, black = 0;
        foreach (sbyte p in state.Squares)
        {
            int v = PieceValues[Math.Abs(p)];
            if (p > 0) white += v; else if (p < 0) black += v;
        }
        int diff = white - black;
        return state.WhiteToMove ? diff : -diff;
    }

    public ChessState Root(ulong? seed = null) => ChessState.StartPosition();

    // The seam methods delegate to the single-source core so training and the browser client share one
    // implementation of "legal indices / apply an index / terminal result".
    public IReadOnlyList<int> LegalMoves(ChessState state) => state.Core.legalMoveIndices();

    public ChessState Apply(ChessState state, int move) => new(state.Core.applyIndex(move));

    public GameResult Result(ChessState state) => state.Core.result() switch
    {
        1 => GameResult.Loss,   // side to move is checkmated
        2 => GameResult.Draw,   // stalemate / 50-move / insufficient material
        _ => GameResult.Ongoing,
    };

    public void WriteObservation(ChessState state, Span<float> destination)
    {
        destination.Clear();
        // Piece planes 0..5 = White P,N,B,R,Q,K ; 6..11 = Black. Plane p, square sq → index p*64 + sq.
        for (int sq = 0; sq < 64; sq++)
        {
            sbyte piece = state.Squares[sq];
            if (piece == 0) continue;
            int type = Math.Abs(piece) - 1;          // 0..5
            int plane = piece > 0 ? type : 6 + type; // white vs black block
            destination[plane * 64 + sq] = 1f;
        }
        if (state.WhiteToMove) Fill(destination, 12);
        if ((state.Castling & ChessState.CastleWK) != 0) Fill(destination, 13);
        if ((state.Castling & ChessState.CastleWQ) != 0) Fill(destination, 14);
        if ((state.Castling & ChessState.CastleBK) != 0) Fill(destination, 15);
        if ((state.Castling & ChessState.CastleBQ) != 0) Fill(destination, 16);
        if (state.EnPassant >= 0) destination[17 * 64 + state.EnPassant] = 1f;
    }

    private static void Fill(Span<float> dest, int plane) => dest.Slice(plane * 64, 64).Fill(1f);
}
