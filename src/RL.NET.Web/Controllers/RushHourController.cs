using Microsoft.AspNetCore.Mvc;
using RLNet.Environments.RushHour;
using RLNet.Web.Services;

namespace RLNet.Web.Controllers;

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
    TrajectoryStepDto[] OptimalTrajectory);

public sealed record StatusResponse(string Status, int TrainingStep, int TrainingMaxSteps, double LastEvalReturn, string? Error);

[ApiController]
[Route("api/rushhour")]
public sealed class RushHourController(RushHourModelService model) : ControllerBase
{
    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint so status reflects it
        return new(model.Status.ToString().ToLowerInvariant(),
            model.TrainingStep, model.TrainingMaxSteps, model.LastEvalReturn, model.Error);
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

        var agent = model.Agent;
        if (agent is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        int optimal = RushHourSolver.Solve(puzzle, maxStates: 2_000_000, out int[] optimalActions);
        if (optimal < 0)
            return BadRequest(new AnalyzeResponse(true, "This puzzle is unsolvable — the red car can never reach the exit.", false, -1));
        if (optimal == 0)
            return BadRequest(new AnalyzeResponse(true, "The red car is already at the exit.", true, 0));

        // Greedy masked rollout of the trained model on the user's puzzle. The move budget
        // scales with difficulty: expert boards (e.g. official card 40 = 81 single-cell
        // moves) must not be truncated below what even a perfect player needs.
        int maxMoves = Math.Max(RushHourModelService.MaxMoves, 2 * optimal);
        var env = new RushHourEnv([puzzle], maxMoves) { FixedPuzzleIndex = 0 };
        env.Reset(1);
        var obs = env.CurrentObservation();
        var trajectory = new List<TrajectoryStepDto>();
        bool solved = false;
        var positions = RushHourBoard.InitialPositions(puzzle);

        while (true)
        {
            int action = agent.Act(obs, env.CurrentActionMask(), greedy: true);
            var step = env.Step(action);
            obs = step.Observation;
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
            trajectory.Add(new TrajectoryStepDto(action / 2, action % 2, [.. positions]));
            if (step.Terminated) solved = true;
            if (step.Done) break;
        }

        // Compacted: same optimal move count, but commutable moves grouped into fluid slides.
        var compactedOptimal = RushHourSolver.CompactSolution(puzzle, optimalActions);
        return new SolveResponse(solved, trajectory.Count, optimal,
            [.. trajectory], ReplayActions(puzzle, compactedOptimal));
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
