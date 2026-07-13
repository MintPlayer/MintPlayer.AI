using System.Text;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Chess;

/// <summary>
/// Forsyth–Edwards Notation (FEN) ↔ <see cref="ChessState"/>. Used to set up standard test positions (perft) and to
/// render a position for debugging. Only the fields the engine tracks are parsed (board, side, castling, en passant,
/// halfmove clock); the fullmove number is ignored on read and written as 1.
/// </summary>
public static class ChessFen
{
    public const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    public static ChessState Parse(string fen)
    {
        string[] parts = fen.Split(' ');
        var squares = new sbyte[64];

        string[] ranks = parts[0].Split('/'); // rank 8 first
        for (int i = 0; i < 8; i++)
        {
            int rank = 7 - i; // ranks[0] is the 8th rank (rank index 7)
            int file = 0;
            foreach (char c in ranks[i])
            {
                if (char.IsDigit(c)) { file += c - '0'; continue; }
                squares[rank * 8 + file] = PieceOf(c);
                file++;
            }
        }

        bool whiteToMove = parts[1] == "w";
        byte castling = 0;
        if (parts[2] != "-")
            foreach (char c in parts[2])
                castling |= c switch
                {
                    'K' => ChessState.CastleWK,
                    'Q' => ChessState.CastleWQ,
                    'k' => ChessState.CastleBK,
                    'q' => ChessState.CastleBQ,
                    _ => (byte)0,
                };

        sbyte enPassant = parts[3] == "-" ? (sbyte)-1 : (sbyte)SquareOf(parts[3]);
        byte halfmove = parts.Length > 4 && byte.TryParse(parts[4], out byte hm) ? hm : (byte)0;
        return new ChessState(squares, whiteToMove, castling, enPassant, halfmove);
    }

    public static string ToFen(ChessState s)
    {
        var sb = new StringBuilder();
        for (int rank = 7; rank >= 0; rank--)
        {
            int empty = 0;
            for (int file = 0; file < 8; file++)
            {
                sbyte piece = s.Squares[rank * 8 + file];
                if (piece == 0) { empty++; continue; }
                if (empty > 0) { sb.Append(empty); empty = 0; }
                sb.Append(CharOf(piece));
            }
            if (empty > 0) sb.Append(empty);
            if (rank > 0) sb.Append('/');
        }
        sb.Append(s.WhiteToMove ? " w " : " b ");
        string rights = string.Concat(
            (s.Castling & ChessState.CastleWK) != 0 ? "K" : "",
            (s.Castling & ChessState.CastleWQ) != 0 ? "Q" : "",
            (s.Castling & ChessState.CastleBK) != 0 ? "k" : "",
            (s.Castling & ChessState.CastleBQ) != 0 ? "q" : "");
        sb.Append(rights.Length == 0 ? "-" : rights);
        sb.Append(' ').Append(s.EnPassant < 0 ? "-" : SquareName(s.EnPassant));
        sb.Append(' ').Append(s.HalfmoveClock).Append(" 1");
        return sb.ToString();
    }

    private static sbyte PieceOf(char c)
    {
        sbyte type = char.ToLowerInvariant(c) switch
        {
            'p' => 1, 'n' => 2, 'b' => 3, 'r' => 4, 'q' => 5, 'k' => 6,
            _ => throw new FormatException($"Bad FEN piece '{c}'."),
        };
        return char.IsUpper(c) ? type : (sbyte)-type;
    }

    private static char CharOf(sbyte piece)
    {
        char c = (PieceType)Math.Abs(piece) switch
        {
            PieceType.Pawn => 'p', PieceType.Knight => 'n', PieceType.Bishop => 'b',
            PieceType.Rook => 'r', PieceType.Queen => 'q', PieceType.King => 'k',
            _ => '?',
        };
        return piece > 0 ? char.ToUpperInvariant(c) : c;
    }

    private static int SquareOf(string s) => (s[1] - '1') * 8 + (s[0] - 'a');
    private static string SquareName(int sq) => $"{(char)('a' + (sq & 7))}{(char)('1' + (sq >> 3))}";
}
