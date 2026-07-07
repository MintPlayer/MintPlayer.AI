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

/// <summary>
/// The AI's attempt, reported honestly (PRD §11): <paramref name="Solved"/> is false when
/// beam search exhausted its depth budget without solving. <paramref name="AiMode"/> is
/// "efficient" — the teacher-free EfficientCube policy net solved by beam search (the
/// website's only AI solver). <paramref name="AlgorithmMoveCount"/> is the Kociemba
/// reference for comparison.
/// </summary>
public sealed record CubeSolveAiResponse(bool Solved, string[] Solution, int MoveCount, int AlgorithmMoveCount, string AiMode);

[ApiController]
[Route("api/cube")]
public sealed class CubeController(CubeModelService model, GalleryStore gallery) : ControllerBase
{
    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint so status reflects it
        return new(model.Status.ToString().ToLowerInvariant(), model.Error);
    }

    /// <summary>
    /// Runs the teacher-free EfficientCube policy net (the website's only AI solver) on the drawn cube via
    /// beam search ranked by cumulative move log-probability — self-supervised on scramble reversals, no
    /// Kociemba and no value-iteration bootstrap. Honest about failure, though the trained net solves any
    /// solvable scramble in practice. The DAVI value net (batch-weighted A*) and the Kociemba-imitation
    /// policy remain in the repo (the <c>cube-davi</c> / <c>cube</c> Lab modes) but are no longer exposed here.
    /// </summary>
    [HttpPost("solve-efficient")]
    public ActionResult<CubeSolveAiResponse> SolveEfficient(CubeSolveRequest request)
    {
        if (!TryBuildCube(request, out var cube, out string? error))
            return BadRequest(new CubeSolveResponse([], 0, 0, error));

        var net = model.EfficientPolicyNet;
        if (net is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        // The Kociemba reference both vets the cube (orientation/parity errors the structural check misses)
        // and gives the QTM baseline the self-taught solver is measured against.
        var reference = CubeSolver.Solve(cube);
        if (!reference.Solved)
            return BadRequest(new CubeSolveResponse([], 0, 0, reference.Error));

        // Beam search over the policy. The resident GPU forward (policy head on the device, weights uploaded
        // once) runs each step's batch; the CPU autograd forward is the fallback on a GPU-less host.
        // Width 5000 (M34 W5): the sweep showed it near-optimalizes mid-depth solutions (d14 17.8→14.4qt, d16–d22
        // −1–2qt) over the old 2000, at ~2.4× the search — a deliberate quality-over-latency choice. Value-guidance
        // (M34 W3) was measured worse per-compute than simply widening, so the beam stays pure-policy.
        const int beamWidth = 5_000;
        var resident = model.ResidentEfficientForward;
        var search = resident is not null
            ? CubePolicySearch.BeamSearch(resident, cube, beamWidth)
            : CubePolicySearch.BeamSearch(net, cube, beamWidth);
        var response = new CubeSolveAiResponse(
            search.Solved, search.Moves, search.Moves.Length, reference.Moves.Length, "efficient");

        if (search.Solved && search.Moves.Length > 0)
            gallery.Add("cube",
                $"Self-taught AI solved it in {search.Moves.Length} quarter-turns (Kociemba QTM reference {reference.Moves.Length})",
                request, new CubeSolveResponse(search.Moves, search.Moves.Length, 0, null));
        return response;
    }

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
