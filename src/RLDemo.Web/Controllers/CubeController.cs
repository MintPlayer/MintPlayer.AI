using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>
/// A cube as drawn/tracked in the browser: six 9-sticker faces of color letters
/// (W/Y/G/B/R/O), row-major top-left → bottom-right looking at the face. Same wire
/// shape as the owner's original Rubiksolver app, so the ported front-end state
/// tracker carries over unchanged.
/// </summary>
public sealed record CubeStateDto(string[] U, string[] R, string[] F, string[] D, string[] L, string[] B);

public sealed record CubeSolveRequest(CubeStateDto? State);

public sealed record CubeSolveResponse(string[] Solution, int MoveCount, long SolveTimeMs, string? Error);

[ApiController]
[Route("api/cube")]
public sealed class CubeController(GalleryStore gallery) : ControllerBase
{
    /// <summary>Solves the cube with the Kociemba two-phase algorithm (the oracle, PRD §11).</summary>
    [HttpPost("solve")]
    public ActionResult<CubeSolveResponse> Solve(CubeSolveRequest request)
    {
        var sw = Stopwatch.StartNew();
        if (!TryBuildCube(request, out var cube, out string? error))
            return BadRequest(new CubeSolveResponse([], 0, sw.ElapsedMilliseconds, error));

        var result = CubeSolver.Solve(cube);
        if (!result.Solved)
            return BadRequest(new CubeSolveResponse([], 0, sw.ElapsedMilliseconds, result.Error));

        var response = new CubeSolveResponse(result.Moves, result.Moves.Length, sw.ElapsedMilliseconds, null);
        if (result.Moves.Length > 0)
            gallery.Add("cube", $"Kociemba solved it in {result.Moves.Length} moves ({sw.ElapsedMilliseconds} ms)",
                request, response);
        return response;
    }

    internal static bool TryBuildCube(CubeSolveRequest request, out FaceletCube cube, out string? error)
    {
        cube = null!;
        error = null;
        if (request.State is null)
        {
            error = "State is required.";
            return false;
        }

        try
        {
            var s = request.State;
            cube = FaceletCube.FromColorFaces(s.U, s.R, s.F, s.D, s.L, s.B);
            return true;
        }
        catch (ArgumentException ex)
        {
            error = ex.Message;
            return false;
        }
    }
}
