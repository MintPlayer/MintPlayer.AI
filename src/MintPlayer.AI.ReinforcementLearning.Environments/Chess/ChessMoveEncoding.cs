namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// The AlphaZero move encoding: a fixed <c>64 × 73 = 4672</c> action space (from-square × move-type plane), which is
/// the policy-head width. The 73 planes are 56 "queen" moves (8 directions × 7 distances — this also carries
/// queen-promotions and every sliding/king/pawn move), 8 knight moves, and 9 underpromotions (Knight/Bishop/Rook ×
/// {capture-left, straight, capture-right}). Distinct legal moves always map to distinct indices, so the search can
/// tell them apart; queen-promotions ride the queen planes (decoded promotion is inferred as Queen when a pawn
/// reaches the last rank — see <see cref="ChessGame.Apply"/>).
/// <para>A thin facade over the single-source encoder (<c>chess_solver.pg</c> → <c>PgChessState.encode/decode</c>),
/// so the browser client and the training seam share one implementation.</para>
/// </summary>
public static class ChessMoveEncoding
{
    public const int PlanesPerSquare = 73;
    public const int Size = 64 * PlanesPerSquare; // 4672

    /// <summary>The action index (0..4671) for a legal move. Queen-promotions use the queen planes; N/B/R
    /// promotions use the underpromotion planes.</summary>
    public static int Encode(ChessMove move) => PgChessState.encode(ChessState.ToPg(move));

    /// <summary>Decodes an action index to a move. A queen-promotion decodes with <see cref="PieceType.None"/> — the
    /// caller promotes a pawn reaching the last rank to a Queen by default.</summary>
    public static ChessMove Decode(int index) => ChessState.FromPg(PgChessState.decode(index));
}
