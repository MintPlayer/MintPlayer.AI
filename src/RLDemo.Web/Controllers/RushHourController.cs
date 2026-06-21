using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>A vehicle as drawn in the browser; index 0 in the array is the red car.</summary>
public sealed record VehicleDto(int Row, int Col, int Length, bool Horizontal);

public sealed record RushHourBoardDto(VehicleDto[] Vehicles);

public sealed record AnalyzeResponse(bool Valid, string? Error, bool Solvable, int OptimalMoves);

/// <summary>
/// One trajectory step: the action taken and the COMPLETE resulting state
/// (each vehicle's variable coordinate), per the PRD §7 solve-API contract.
/// </summary>
public sealed record TrajectoryStepDto(int Vehicle, int Direction, int[] Positions);

public sealed record SolveResponse(
    bool Solved,
    int AiMoves,
    int OptimalMoves,
    TrajectoryStepDto[] Trajectory,
    TrajectoryStepDto[] OptimalTrajectory,
    string AiMode); // "greedy" (reactive policy) | "search" (policy-guided A*) | "dqn" (legacy fallback)

public sealed record StatusResponse(string Status, string? Error);

[ApiController]
[Route("api/rushhour")]
public sealed class RushHourController(RushHourModelService model, GalleryStore gallery) : ControllerBase
{
    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint so status reflects it
        return new(model.Status.ToString().ToLowerInvariant(), model.Error);
    }

    /// <summary>Validates a drawn board and reports the BFS-optimal move count (no model needed).</summary>
    [HttpPost("analyze")]
    public ActionResult<AnalyzeResponse> Analyze(RushHourBoardDto board)
    {
        if (TryBuildPuzzle(board, out var puzzle, out string? error))
        {
            int optimal = RushHourSolver.Solve(puzzle);
            return new AnalyzeResponse(true, null, optimal >= 0, optimal);
        }
        return BadRequest(new AnalyzeResponse(false, error, false, -1));
    }

    /// <summary>Runs the trained model on the drawn board; returns a replayable trajectory + the BFS-optimal one.</summary>
    [HttpPost("solve")]
    public ActionResult<SolveResponse> Solve(RushHourBoardDto board)
    {
        if (!TryBuildPuzzle(board, out var puzzle, out string? error))
            return BadRequest(new AnalyzeResponse(false, error, false, -1));

        var policy = model.PolicyNet;
        var agent = model.Agent;
        if (policy is null && agent is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        int optimal = RushHourSolver.Solve(puzzle, maxStates: 2_000_000, out int[] optimalActions);
        if (optimal < 0)
            return BadRequest(new AnalyzeResponse(true, "This puzzle is unsolvable — the red car can never reach the exit.", false, -1));
        if (optimal == 0)
            return BadRequest(new AnalyzeResponse(true, "The red car is already at the exit.", true, 0));

        // The move budget scales with difficulty: expert boards (e.g. official card 40 =
        // 81 single-cell moves) must not be truncated below what a perfect player needs.
        int maxMoves = Math.Max(RushHourModelService.MaxMoves, 2 * optimal);
        bool solved;
        string aiMode;
        TrajectoryStepDto[] trajectory;

        if (policy is not null)
        {
            // Imitation-learned net: reactive play first; when that fails, the same net
            // guides a budgeted A* (its value head is the heuristic) — still "the AI",
            // now with lookahead, and honest about which mode produced the answer.
            var greedy = RushHourPolicySearch.GreedyRollout(policy, puzzle, maxMoves);
            if (greedy.Solved)
            {
                (solved, aiMode) = (true, "greedy");
                trajectory = ReplayActions(puzzle, [.. greedy.Actions]);
            }
            else
            {
                var search = RushHourPolicySearch.Solve(policy, puzzle, maxExpansions: 150_000);
                (solved, aiMode) = (search.Solved, "search");
                trajectory = search.Solved ? ReplayActions(puzzle, search.Actions) : ReplayActions(puzzle, [.. greedy.Actions]);
            }
        }
        else
        {
            var (dqnSolved, steps) = RushHourRollout.Run(agent!, puzzle, maxMoves);
            (solved, aiMode) = (dqnSolved, "dqn");
            trajectory = [.. steps.Select(s => new TrajectoryStepDto(s.Vehicle, s.Direction, s.Positions))];
        }

        // Compacted: same optimal move count, but commutable moves grouped into fluid slides.
        var compactedOptimal = RushHourSolver.CompactSolution(puzzle, optimalActions);
        var response = new SolveResponse(solved, trajectory.Length, optimal,
            trajectory, ReplayActions(puzzle, compactedOptimal), aiMode);

        string how = aiMode == "search" ? "AI (with lookahead)" : "AI";
        gallery.Add("rushhour",
            solved ? $"{how} solved it in {trajectory.Length} moves (optimal {optimal})"
                   : $"{how} failed within {trajectory.Length} moves (optimal {optimal})",
            board, response);
        return response;
    }

    private static TrajectoryStepDto[] ReplayActions(RushHourPuzzle puzzle, int[] actions)
    {
        var positions = RushHourBoard.InitialPositions(puzzle);
        var steps = new TrajectoryStepDto[actions.Length];
        for (int i = 0; i < actions.Length; i++)
        {
            positions[actions[i] / 2] += actions[i] % 2 == 0 ? -1 : 1;
            steps[i] = new TrajectoryStepDto(actions[i] / 2, actions[i] % 2, [.. positions]);
        }
        return steps;
    }

    private static bool TryBuildPuzzle(RushHourBoardDto board, out RushHourPuzzle puzzle, out string? error)
    {
        puzzle = null!;
        error = null;
        if (board.Vehicles is not { Length: > 0 })
        {
            error = "Draw at least the red car.";
            return false;
        }

        try
        {
            puzzle = new RushHourPuzzle([.. board.Vehicles.Select(v => new Vehicle(v.Row, v.Col, v.Length, v.Horizontal))]);
            if (puzzle.Vehicles.Any(v => v.Length is < 2 or > 3))
            {
                error = "Vehicles must have length 2 (car) or 3 (truck).";
                return false;
            }
            return true;
        }
        catch (ArgumentException ex)
        {
            error = ex.Message;
            return false;
        }
    }
}
