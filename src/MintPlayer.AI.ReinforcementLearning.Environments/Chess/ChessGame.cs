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
public sealed class ChessGame : IZeroSumGame<ChessState>
{
    private const int Planes = 18;

    public int PolicySize => ChessMoveEncoding.Size;      // 4672
    public int ObservationSize => Planes * 64;            // 1152

    public ChessState Root(ulong? seed = null) => ChessState.StartPosition();

    public IReadOnlyList<int> LegalMoves(ChessState state)
    {
        var moves = ChessRules.LegalMoves(state);
        var indices = new int[moves.Count];
        for (int i = 0; i < moves.Count; i++) indices[i] = ChessMoveEncoding.Encode(moves[i]);
        return indices;
    }

    public ChessState Apply(ChessState state, int move)
    {
        var decoded = ChessMoveEncoding.Decode(move);
        // A queen-promotion rides the queen planes and decodes as None → promote the pawn to a Queen by default.
        if (decoded.Promotion == PieceType.None
            && (PieceType)Math.Abs(state.Squares[decoded.From]) == PieceType.Pawn
            && (decoded.To >> 3) is 0 or 7)
            decoded = decoded with { Promotion = PieceType.Queen };
        return ChessRules.MakeMove(state, decoded);
    }

    public GameResult Result(ChessState state)
    {
        if (ChessRules.LegalMoves(state).Count == 0)
            return ChessRules.InCheck(state, state.WhiteToMove) ? GameResult.Loss : GameResult.Draw; // mate vs stalemate
        if (ChessRules.IsFiftyMove(state) || ChessRules.IsInsufficientMaterial(state))
            return GameResult.Draw;
        return GameResult.Ongoing;
    }

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
