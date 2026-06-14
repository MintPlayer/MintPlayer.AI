namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Structural validation of a sticker arrangement with human diagnostics — which physical
/// edge/corner pieces are missing, duplicated or impossible (e.g. two stickers of the same
/// color on one piece). Ported from the owner's Rubiksolver (<c>Models/CubeState.cs</c>).
/// A structurally valid cube can still be unsolvable (flipped edge, twisted corner,
/// parity); those cases come back as Kociemba error codes from the solver itself.
/// </summary>
public static class CubeValidation
{
    private const int U = 0, R = 9, F = 18, D = 27, L = 36, B = 45;

    // The 12 physical edges and 8 corners of a solved cube, as alphabetically sorted
    // color-letter keys (B < G < O < R < W < Y).
    private static readonly HashSet<string> ValidEdges =
    [
        "GW", "RW", "BW", "OW",  // U layer
        "GY", "RY", "BY", "OY",  // D layer
        "GR", "GO", "BR", "BO",  // middle layer
    ];

    private static readonly HashSet<string> ValidCorners =
    [
        "GRW", "GOW", "BRW", "BOW",  // U layer
        "GRY", "GOY", "BRY", "BOY",  // D layer
    ];

    // Facelet indices of each edge position (two stickers) and its conventional name.
    private static readonly (int A, int B, string Pos)[] EdgePositions =
    [
        (U + 7, F + 1, "UF"), (U + 5, R + 1, "UR"), (U + 1, B + 1, "UB"), (U + 3, L + 1, "UL"),
        (D + 1, F + 7, "DF"), (D + 5, R + 7, "DR"), (D + 7, B + 7, "DB"), (D + 3, L + 7, "DL"),
        (F + 5, R + 3, "FR"), (F + 3, L + 5, "FL"), (B + 3, R + 5, "BR"), (B + 5, L + 3, "BL"),
    ];

    // Facelet indices of each corner position (three stickers) and its conventional name.
    private static readonly (int A, int B, int C, string Pos)[] CornerPositions =
    [
        (U + 8, F + 2, R + 0, "UFR"), (U + 6, F + 0, L + 2, "UFL"),
        (U + 2, B + 0, R + 2, "UBR"), (U + 0, B + 2, L + 0, "UBL"),
        (D + 2, F + 8, R + 6, "DFR"), (D + 0, F + 6, L + 8, "DFL"),
        (D + 8, B + 6, R + 8, "DBR"), (D + 6, B + 8, L + 6, "DBL"),
    ];

    /// <summary>A detailed problem description, or null when the arrangement is structurally valid.</summary>
    public static string? FindStructuralError(FaceletCube cube)
    {
        List<string> invalidEdges = [], missingEdges = [], duplicateEdges = [];
        List<string> invalidCorners = [], missingCorners = [], duplicateCorners = [];

        char Color(int facelet) => FaceletCube.FaceColors[cube[facelet]];

        var edgeCounts = new Dictionary<string, List<string>>();
        foreach (var (a, b, pos) in EdgePositions)
        {
            char c1 = Color(a), c2 = Color(b);
            string key = c1 < c2 ? $"{c1}{c2}" : $"{c2}{c1}";
            (edgeCounts.TryGetValue(key, out var at) ? at : edgeCounts[key] = []).Add(pos);
            if (!ValidEdges.Contains(key))
                invalidEdges.Add($"{c1}-{c2} at {pos}");
        }
        foreach (string edge in ValidEdges)
        {
            if (!edgeCounts.TryGetValue(edge, out var at))
                missingEdges.Add($"{edge[0]}-{edge[1]}");
            else if (at.Count > 1)
                duplicateEdges.Add($"{edge[0]}-{edge[1]} at {string.Join(", ", at)}");
        }

        var cornerCounts = new Dictionary<string, List<string>>();
        foreach (var (a, b, c, pos) in CornerPositions)
        {
            char[] colors = [Color(a), Color(b), Color(c)];
            string raw = new(colors);
            Array.Sort(colors);
            string key = new(colors);
            (cornerCounts.TryGetValue(key, out var at) ? at : cornerCounts[key] = []).Add(pos);
            if (!ValidCorners.Contains(key))
                invalidCorners.Add($"{raw[0]}-{raw[1]}-{raw[2]} at {pos}");
        }
        foreach (string corner in ValidCorners)
        {
            if (!cornerCounts.TryGetValue(corner, out var at))
                missingCorners.Add($"{corner[0]}-{corner[1]}-{corner[2]}");
            else if (at.Count > 1)
                duplicateCorners.Add($"{corner[0]}-{corner[1]}-{corner[2]} at {string.Join(", ", at)}");
        }

        List<string> parts = [];
        if (invalidEdges.Count > 0) parts.Add($"Invalid edges: {string.Join("; ", invalidEdges)}");
        if (missingEdges.Count > 0) parts.Add($"Missing edges: {string.Join(", ", missingEdges)}");
        if (duplicateEdges.Count > 0) parts.Add($"Duplicate edges: {string.Join("; ", duplicateEdges)}");
        if (invalidCorners.Count > 0) parts.Add($"Invalid corners: {string.Join("; ", invalidCorners)}");
        if (missingCorners.Count > 0) parts.Add($"Missing corners: {string.Join(", ", missingCorners)}");
        if (duplicateCorners.Count > 0) parts.Add($"Duplicate corners: {string.Join("; ", duplicateCorners)}");

        return parts.Count > 0 ? string.Join(". ", parts) : null;
    }
}
