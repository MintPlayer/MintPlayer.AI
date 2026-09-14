using Microsoft.AspNetCore.Mvc;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>
/// Plays a Block Dude level with the shipped net, for the game page's "Watch AI" button.
/// </summary>
/// <remarks>
/// The board comes in as a grid rather than a level name on purpose: the net is not restricted to the shipped
/// deck, and a level someone adds to <c>blockdude-levels.json</c> should be playable without a server change.
/// The engine parses the same grid strings the browser twin does — the two level files are byte-identical.
/// </remarks>
[ApiController]
[Route("api/blockdude")]
public sealed class BlockDudeController(BlockDudeModelService model) : ControllerBase
{
    /// <param name="Grid">The level, one string per row, exactly as it appears in the levels JSON.</param>
    /// <param name="MaxSeconds">Ceiling for the beam fallback. Ignored by the greedy path, which is the one
    /// that runs for the shipped levels and takes well under a second.</param>
    public sealed record SolveRequest(string[]? Grid, int? MaxSeconds);

    /// <param name="Tier">How it was solved: <c>greedy</c> means the policy played it unaided, with no search
    /// of any kind. Reported so the UI can say which, rather than implying search was needed when it was not.</param>
    public sealed record SolveResponse(bool Solved, int[] Moves, int MoveCount, string Tier);

    public sealed record StatusResponse(bool Ready);

    [HttpGet("status")]
    public ActionResult<StatusResponse> Status() => Ok(new StatusResponse(model.IsReady));

    [HttpPost("solve")]
    public ActionResult<SolveResponse> Solve([FromBody] SolveRequest request)
    {
        if (request.Grid is not { Length: > 0 }) return BadRequest("A level grid is required.");
        if (request.Grid.Any(string.IsNullOrEmpty)) return BadRequest("Grid rows cannot be empty.");

        // Same width on every row, checked here rather than in the engine: FromGrid indexes by row/column, so a
        // ragged grid would throw deep inside the solver and surface as a 500 on what is really a bad request.
        if (request.Grid.Any(row => row.Length != request.Grid[0].Length))
            return BadRequest("All grid rows must be the same width.");

        if (!model.IsReady) return StatusCode(503, new StatusResponse(false));

        // Clamped: the ceiling is a guard against one pathological board tying up a request thread, so it must
        // not be settable to something unbounded from the client.
        int seconds = Math.Clamp(request.MaxSeconds ?? 15, 1, 30);

        var solution = model.Solve(request.Grid, seconds);
        return Ok(new SolveResponse(solution.Solved, [.. solution.Moves], solution.Moves.Count,
                                    solution.Tier.ToString().ToLowerInvariant()));
    }
}
