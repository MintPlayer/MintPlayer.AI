using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Facelet-level 3×3×3 cube: 54 stickers in Kociemba face order (U R F D L B, 9 each,
/// row-major top-left → bottom-right when looking at the face). A sticker's value is the
/// face it belongs to on a solved cube (0..5 = U R F D L B), so the Kociemba string is a
/// direct character lookup. The quarter-turn sticker cycles are ported from the owner's
/// Rubiksolver front-end (<c>rubiksCube.ts</c>), whose conventions the Kociemba port was
/// validated against; the round-trip is re-asserted by the solver gate test.
/// </summary>
public sealed class FaceletCube
{
    public const int FaceletCount = 54;
    public const int FaceCount = 6;

    /// <summary>Face order, matching Kociemba facelet order and the wire DTO.</summary>
    public const string FaceNames = "URFDLB";

    /// <summary>Sticker color letter per home face: U=White, R=Red, F=Green, D=Yellow, L=Orange, B=Blue.</summary>
    public const string FaceColors = "WRGYOB";

    /// <summary>All 18 face moves in standard notation.</summary>
    public static readonly string[] AllMoves =
        ["U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2"];

    /// <summary>The 12 quarter-turns — the RL action space (PRD §11). Index = action id.</summary>
    public static readonly string[] QuarterTurnMoves =
        ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"];

    private readonly byte[] _facelets;

    public FaceletCube()
    {
        _facelets = new byte[FaceletCount];
        for (int i = 0; i < FaceletCount; i++)
            _facelets[i] = (byte)(i / 9);
    }

    private FaceletCube(byte[] facelets) => _facelets = facelets;

    /// <summary>Sticker home-face value (0..5 = U R F D L B) at a flat facelet index.</summary>
    public byte this[int index] => _facelets[index];

    public ReadOnlySpan<byte> Facelets => _facelets;

    public FaceletCube Clone() => new((byte[])_facelets.Clone());

    public bool IsSolved
    {
        get
        {
            for (int i = 0; i < FaceletCount; i++)
                if (_facelets[i] != i / 9)
                    return false;
            return true;
        }
    }

    /// <summary>The 54-char Kociemba definition string (e.g. "UUUUUUUUURRR…").</summary>
    public string ToKociembaString()
    {
        Span<char> chars = stackalloc char[FaceletCount];
        for (int i = 0; i < FaceletCount; i++)
            chars[i] = FaceNames[_facelets[i]];
        return new string(chars);
    }

    /// <summary>Builds a cube from six 9-sticker color-letter arrays (W/Y/G/B/R/O — the wire format).</summary>
    public static FaceletCube FromColorFaces(string[] u, string[] r, string[] f, string[] d, string[] l, string[] b)
    {
        var facelets = new byte[FaceletCount];
        string[][] faces = [u, r, f, d, l, b];
        for (int face = 0; face < FaceCount; face++)
        {
            if (faces[face] is not { Length: 9 })
                throw new ArgumentException($"Face {FaceNames[face]} needs exactly 9 stickers.");
            for (int i = 0; i < 9; i++)
            {
                int color = faces[face][i] is { Length: 1 } s ? FaceColors.IndexOf(s[0]) : -1;
                if (color < 0)
                    throw new ArgumentException(
                        $"Face {FaceNames[face]} sticker {i} has color '{faces[face][i]}'; expected one of W/R/G/Y/O/B.");
                facelets[face * 9 + i] = (byte)color;
            }
        }
        return new FaceletCube(facelets);
    }

    /// <summary>The six 9-sticker color-letter arrays in U R F D L B order (the wire format).</summary>
    public string[][] ToColorFaces()
    {
        var faces = new string[FaceCount][];
        for (int face = 0; face < FaceCount; face++)
        {
            faces[face] = new string[9];
            for (int i = 0; i < 9; i++)
                faces[face][i] = FaceColors[_facelets[face * 9 + i]].ToString();
        }
        return faces;
    }

    /// <summary>Applies a move in standard notation ("U", "U'", "U2", …).</summary>
    public void Apply(string move)
    {
        if (move is not { Length: 1 or 2 })
            throw new ArgumentException($"Unknown move '{move}'.");
        int face = FaceNames.IndexOf(move[0]);
        if (face < 0 || (move.Length == 2 && move[1] is not ('\'' or '2')))
            throw new ArgumentException($"Unknown move '{move}'.");

        int turns = move.Length == 1 ? 1 : move[1] == '2' ? 2 : 3;
        for (int t = 0; t < turns; t++)
            ApplyClockwise(face);
    }

    public void Apply(IEnumerable<string> moves)
    {
        foreach (string move in moves)
            Apply(move);
    }

    /// <summary>Applies a quarter-turn by RL action id (index into <see cref="QuarterTurnMoves"/>).</summary>
    public void ApplyQuarterTurn(int action) => Apply(QuarterTurnMoves[action]);

    public static string InverseMove(string move) => move.Length switch
    {
        1 => move + "'",
        2 when move[1] == '\'' => move[..1],
        _ => move, // half turns are self-inverse
    };

    /// <summary>
    /// A seeded random scramble: no two consecutive moves on the same face, and no third
    /// consecutive move on the same axis (so "R L R" style non-progress is excluded too).
    /// <paramref name="quarterTurnsOnly"/> restricts to the 12-move RL action space, making
    /// the scramble length an upper bound on the solution depth in quarter-turn metric.
    /// </summary>
    public static List<string> ScrambleMoves(Xoshiro256StarStar rng, int length, bool quarterTurnsOnly = false)
    {
        var moves = new List<string>(length);
        int prevFace = -1, prevPrevFace = -1;
        for (int i = 0; i < length; i++)
        {
            int face;
            do
            {
                face = rng.NextInt(FaceCount);
            } while (face == prevFace || (prevFace >= 0 && Axis(face) == Axis(prevFace) && prevPrevFace >= 0 && Axis(face) == Axis(prevPrevFace)));

            string suffix = quarterTurnsOnly
                ? rng.NextInt(2) == 0 ? "" : "'"
                : rng.NextInt(3) switch { 0 => "", 1 => "'", _ => "2" };
            moves.Add(FaceNames[face] + suffix);
            (prevPrevFace, prevFace) = (prevFace, face);
        }
        return moves;

        static int Axis(int face) => face % 3; // U/D = 0, R/L = 1, F/B = 2 in URFDLB order
    }

    // ── Move mechanics ────────────────────────────────────────────────────────────

    // Flat-index offsets of each face in URFDLB order.
    private const int U = 0, R = 9, F = 18, D = 27, L = 36, B = 45;

    /// <summary>
    /// The five 4-cycles of one clockwise quarter turn per face: two on the turning face
    /// (corners, edges) and three side strips. (a,b,c,d) means the sticker at a moves to b,
    /// b to c, c to d, d to a. Ported from rotateFace in Rubiksolver/Scripts/rubiksCube.ts.
    /// </summary>
    private static readonly int[][][] Cycles =
    [
        [ // U
            [U + 0, U + 2, U + 8, U + 6], [U + 1, U + 5, U + 7, U + 3],
            [F + 0, L + 0, B + 0, R + 0], [F + 1, L + 1, B + 1, R + 1], [F + 2, L + 2, B + 2, R + 2],
        ],
        [ // R
            [R + 0, R + 2, R + 8, R + 6], [R + 1, R + 5, R + 7, R + 3],
            [U + 2, B + 6, D + 2, F + 2], [U + 5, B + 3, D + 5, F + 5], [U + 8, B + 0, D + 8, F + 8],
        ],
        [ // F
            [F + 0, F + 2, F + 8, F + 6], [F + 1, F + 5, F + 7, F + 3],
            [U + 6, R + 0, D + 2, L + 8], [U + 7, R + 3, D + 1, L + 5], [U + 8, R + 6, D + 0, L + 2],
        ],
        [ // D
            [D + 0, D + 2, D + 8, D + 6], [D + 1, D + 5, D + 7, D + 3],
            [F + 6, R + 6, B + 6, L + 6], [F + 7, R + 7, B + 7, L + 7], [F + 8, R + 8, B + 8, L + 8],
        ],
        [ // L
            [L + 0, L + 2, L + 8, L + 6], [L + 1, L + 5, L + 7, L + 3],
            [U + 0, F + 0, D + 0, B + 8], [U + 3, F + 3, D + 3, B + 5], [U + 6, F + 6, D + 6, B + 2],
        ],
        [ // B
            [B + 0, B + 2, B + 8, B + 6], [B + 1, B + 5, B + 7, B + 3],
            [U + 2, L + 0, D + 6, R + 8], [U + 1, L + 3, D + 7, R + 5], [U + 0, L + 6, D + 8, R + 2],
        ],
    ];

    private void ApplyClockwise(int face)
    {
        foreach (int[] cycle in Cycles[face])
        {
            byte last = _facelets[cycle[3]];
            _facelets[cycle[3]] = _facelets[cycle[2]];
            _facelets[cycle[2]] = _facelets[cycle[1]];
            _facelets[cycle[1]] = _facelets[cycle[0]];
            _facelets[cycle[0]] = last;
        }
    }

    /// <summary>Flattened-net rendering (console-oriented, per the IEnvironment convention).</summary>
    public string RenderString()
    {
        var sb = new System.Text.StringBuilder();
        for (int row = 0; row < 3; row++)
            sb.Append("      ").Append(FaceRow(U, row)).AppendLine();
        for (int row = 0; row < 3; row++)
            sb.Append(FaceRow(L, row)).Append(' ').Append(FaceRow(F, row)).Append(' ')
              .Append(FaceRow(R, row)).Append(' ').Append(FaceRow(B, row)).AppendLine();
        for (int row = 0; row < 3; row++)
            sb.Append("      ").Append(FaceRow(D, row)).AppendLine();
        return sb.ToString();

        string FaceRow(int offset, int row)
            => $"{FaceColors[_facelets[offset + row * 3]]} {FaceColors[_facelets[offset + row * 3 + 1]]} {FaceColors[_facelets[offset + row * 3 + 2]]}";
    }
}
