using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube.Kociemba;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

public sealed record CubeSolveResult(bool Solved, string[] Moves, string? Error);

/// <summary>
/// The cube's algorithmic oracle: Herbert Kociemba's two-phase solver (the C# port under
/// <c>Kociemba/</c>, taken verbatim from the owner's Rubiksolver app). Always answers on
/// any valid cube in ≤ 22 moves — the same role the BFS solver plays for Rush Hour, and
/// the imitation-data source (PRD §11 / PLAN M16). Lookup/pruning tables are built in
/// memory on first use (CLR static initialization, thread-safe); call <see cref="WarmUp"/>
/// off the request path to pay that cost at startup. Solves are fully concurrent — each
/// runs on its own <c>SearchRunTime</c> instance over the shared read-only tables, which
/// is what lets the Lab generate imitation data on all cores.
/// </summary>
public static class CubeSolver
{
    public const int MaxDepth = 22;
    public const int TimeoutMilliseconds = 10_000;

    /// <summary>Builds the lookup/pruning tables by solving a trivial cube; idempotent.</summary>
    public static void WarmUp()
    {
        var cube = new FaceletCube();
        cube.Apply("U");
        Solve(cube);
    }

    /// <summary>Solves the cube; an already-solved cube yields an empty move list.</summary>
    public static CubeSolveResult Solve(FaceletCube cube)
    {
        if (cube.IsSolved)
            return new(true, [], null);

        string? structural = CubeValidation.FindStructuralError(cube);
        if (structural is not null)
            return new(false, [], $"Invalid cube: {structural}");

        string solution = SearchRunTime.solution(cube.ToKociembaString(), out _,
            maxDepth: MaxDepth, timeOut: TimeoutMilliseconds, useSeparator: false, buildTables: false);

        if (solution.StartsWith("Error"))
            return new(false, [], ErrorMessage(solution));

        return new(true, solution.Split(' ', StringSplitOptions.RemoveEmptyEntries), null);
    }

    /// <summary>Kociemba error codes → user-facing messages (ported from Rubiksolver's CubeSolver).</summary>
    private static string ErrorMessage(string errorCode) => errorCode switch
    {
        "Error 1" => "Invalid cube: Not exactly one facelet of each color",
        "Error 2" => "Invalid cube: Not all 12 edges exist exactly once",
        "Error 3" => "Invalid cube: One edge has to be flipped",
        "Error 4" => "Invalid cube: Not all 8 corners exist exactly once",
        "Error 5" => "Invalid cube: One corner has to be twisted",
        "Error 6" => "Invalid cube: Two corners or two edges have to be exchanged (parity error)",
        "Error 7" => "No solution found within max depth",
        "Error 8" => "Timeout - no solution found within time limit",
        _ => errorCode,
    };
}
