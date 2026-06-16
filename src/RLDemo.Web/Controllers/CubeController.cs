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
/// even the lookahead ran out of budget — expected on scrambles deeper than the trained
/// band. <paramref name="AiMode"/> says which mode produced the answer ("greedy" =
/// reactive policy rollout, "search" = net-guided lookahead, "dqn" = legacy DQN
/// fallback when no policy checkpoint exists — the M11 pattern).
/// <paramref name="AlgorithmMoveCount"/> is the Kociemba reference for comparison.
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
        return new(model.Status.ToString().ToLowerInvariant(),
            model.TrainingStep, model.TrainingMaxSteps, model.LastEvalReturn, model.Error);
    }

    /// <summary>Runs the trained AI on the drawn cube; honest about failure.</summary>
    [HttpPost("solve-ai")]
    public ActionResult<CubeSolveAiResponse> SolveAi(CubeSolveRequest request)
    {
        if (!TryBuildCube(request, out var cube, out string? error))
            return BadRequest(new CubeSolveResponse([], 0, 0, error));

        var policy = model.PolicyNet;
        var agent = model.Agent;
        if (policy is null && agent is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        // The Kociemba reference also vets the cube (orientation/parity errors that the
        // structural check cannot see) — never let the AI loose on an unsolvable cube.
        var reference = CubeSolver.Solve(cube);
        if (!reference.Solved)
            return BadRequest(new CubeSolveResponse([], 0, 0, reference.Error));

        // Reactive play first; when that fails, the same net guides a best-first search —
        // still "the AI", now with lookahead, and honest about which mode produced the
        // answer (the Rush Hour M11 pattern). The imitation policy net is preferred;
        // the masked DQN is the fallback when no policy checkpoint exists yet.
        bool solved;
        List<string> moves;
        string aiMode;
        if (policy is not null)
        {
            var greedy = CubePolicySearch.GreedyRollout(policy, cube);
            if (greedy.Solved)
            {
                (solved, aiMode) = (true, "greedy");
                moves = [.. greedy.Actions.Select(a => FaceletCube.QuarterTurnMoves[a])];
            }
            else
            {
                var search = CubePolicySearch.Solve(policy, cube);
                (solved, aiMode) = (search.Solved, "search");
                moves = search.Solved
                    ? [.. search.Moves]
                    : [.. greedy.Actions.Select(a => FaceletCube.QuarterTurnMoves[a])];
            }
        }
        else
        {
            (solved, moves) = CubeModelService.Rollout(agent!, cube);
            aiMode = "dqn";
            if (!solved)
            {
                var search = CubeQSearch.Solve(agent!, cube);
                aiMode = "search";
                if (search.Solved)
                    (solved, moves) = (true, [.. search.Moves]);
            }
        }
        var response = new CubeSolveAiResponse(solved, [.. moves], moves.Count, reference.Moves.Length, aiMode);

        // Only solved attempts are replayable from the gallery (replay reconstructs the
        // submitted cube by inverting the solution), so failures are not persisted.
        if (solved && moves.Count > 0)
        {
            string how = aiMode == "search" ? "AI (with lookahead)" : "AI";
            gallery.Add("cube", $"{how} solved it in {moves.Count} moves (Kociemba reference {reference.Moves.Length})",
                request, new CubeSolveResponse([.. moves], moves.Count, 0, null));
        }
        return response;
    }

    /// <summary>
    /// Runs the teacher-free DAVI value net (the "self-taught AI", PLAN M21) on the drawn cube via
    /// batch-weighted A*: shortest-move search guided purely by the learned cost-to-go (no Kociemba).
    /// Honest about failure — a scramble past the net's accurate band returns Solved = false.
    /// </summary>
    [HttpPost("solve-davi")]
    public ActionResult<CubeSolveAiResponse> SolveDavi(CubeSolveRequest request)
    {
        if (!TryBuildCube(request, out var cube, out string? error))
            return BadRequest(new CubeSolveResponse([], 0, 0, error));

        var valueNet = model.ValueNet;
        if (valueNet is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        // The Kociemba reference both vets the cube (orientation/parity errors the structural check
        // misses) and gives the QTM baseline the self-taught solver is measured against.
        var reference = CubeSolver.Solve(cube);
        if (!reference.Solved)
            return BadRequest(new CubeSolveResponse([], 0, 0, reference.Error));

        // Interactivity is bounded by a wall-clock deadline, not a fixed expansion count: an unsolvable cube
        // would otherwise burn the whole expansion budget before failing, coupling worst-case latency to cube
        // difficulty. A time budget instead caps the wait while letting each cube use as much search as fits —
        // easy cubes return fast, hard ones search to the deadline then fail honestly. The expansion count is
        // kept only as a memory-safety ceiling (nodes/visited grow per expansion); the deadline is the real
        // limit. Resident GPU forward ≈ 0.3 ms/expansion → ~20 s reaches well past the old 50k/depth-15 band;
        // the CPU fallback (~2 ms/expansion, e.g. a GPU-less Hetzner box) gets a tighter 10 s, shallower reach.
        var resident = model.ResidentValueForward;
        var search = resident is not null
            ? CubeValueSearch.Solve(resident, cube, maxExpansions: 150_000, maxTime: TimeSpan.FromSeconds(20))
            : CubeValueSearch.Solve(valueNet, cube, maxExpansions: 50_000, maxTime: TimeSpan.FromSeconds(10));
        var response = new CubeSolveAiResponse(
            search.Solved, search.Moves, search.Moves.Length, reference.Moves.Length, "davi");

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
