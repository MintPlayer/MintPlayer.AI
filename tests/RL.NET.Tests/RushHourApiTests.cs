using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using RLNet.Core.Checkpoints;
using RLNet.Core.Random;
using RLNet.Core.Schedules;
using RLNet.Core.Training;
using RLNet.Environments.RushHour;
using RLNet.Web.Controllers;
using RLNet.Web.Services;

namespace RLNet.Tests;

/// <summary>Host fixture: Testing environment (no SPA, no auto-training), isolated temp model store.</summary>
public class PlaygroundFactory : WebApplicationFactory<Program>
{
    public string DataDirectory { get; } =
        Path.Combine(Path.GetTempPath(), "rlnet-web-test-" + Guid.NewGuid().ToString("N"));

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.UseSetting("DataDirectory", DataDirectory);
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (Directory.Exists(DataDirectory)) Directory.Delete(DataDirectory, recursive: true);
    }
}

/// <summary>API contract tests that need no trained model (fast bucket).</summary>
public class RushHourApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    // The hand-verified optimal=7 puzzle from RushHourTests: red at (2,0), vertical truck col 2 rows 0–2.
    private static readonly VehicleDto[] KnownPuzzle =
    [
        new(2, 0, 2, true),
        new(0, 2, 3, false),
    ];

    [Fact]
    public async Task Status_ReportsModelNotReady()
    {
        var status = await _client.GetFromJsonAsync<StatusResponse>("/api/rushhour/status");
        Assert.NotNull(status);
        Assert.NotEqual("ready", status.Status);
    }

    [Fact]
    public async Task Analyze_KnownPuzzle_ReturnsBfsOptimal()
    {
        var response = await _client.PostAsJsonAsync("/api/rushhour/analyze", new RushHourBoardDto(KnownPuzzle));
        response.EnsureSuccessStatusCode();
        var analysis = await response.Content.ReadFromJsonAsync<AnalyzeResponse>();
        Assert.NotNull(analysis);
        Assert.True(analysis.Valid);
        Assert.True(analysis.Solvable);
        Assert.Equal(7, analysis.OptimalMoves);
    }

    [Fact]
    public async Task Analyze_OverlappingVehicles_Returns400WithReason()
    {
        VehicleDto[] overlapping = [new(2, 0, 2, true), new(2, 1, 2, true)];
        var response = await _client.PostAsJsonAsync("/api/rushhour/analyze", new RushHourBoardDto(overlapping));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var analysis = await response.Content.ReadFromJsonAsync<AnalyzeResponse>();
        Assert.NotNull(analysis);
        Assert.False(analysis.Valid);
        Assert.Contains("overlap", analysis.Error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Analyze_FrozenExitColumn_ReportsUnsolvable()
    {
        // Column 5 fully packed with three vertical cars: none can ever move.
        VehicleDto[] unsolvable =
        [
            new(2, 0, 2, true),
            new(0, 5, 2, false),
            new(2, 5, 2, false),
            new(4, 5, 2, false),
        ];
        var response = await _client.PostAsJsonAsync("/api/rushhour/analyze", new RushHourBoardDto(unsolvable));
        response.EnsureSuccessStatusCode();
        var analysis = await response.Content.ReadFromJsonAsync<AnalyzeResponse>();
        Assert.NotNull(analysis);
        Assert.True(analysis.Valid);
        Assert.False(analysis.Solvable);
    }

    [Fact]
    public async Task Solve_WithoutTrainedModel_Returns503WithStatus()
    {
        var response = await _client.PostAsJsonAsync("/api/rushhour/solve", new RushHourBoardDto(KnownPuzzle));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        var status = await response.Content.ReadFromJsonAsync<StatusResponse>();
        Assert.NotNull(status);
        Assert.NotEqual("ready", status.Status);
    }

    [Fact]
    public async Task Solve_InvalidRedCar_Returns400()
    {
        VehicleDto[] redOffExitRow = [new(0, 0, 2, true)];
        var response = await _client.PostAsJsonAsync("/api/rushhour/solve", new RushHourBoardDto(redOffExitRow));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}

/// <summary>Host fixture with a model pre-trained into the store (the M8 API gate).</summary>
public class TrainedPlaygroundFactory : PlaygroundFactory
{
    public IReadOnlyList<RushHourPuzzle> Puzzles { get; }

    public TrainedPlaygroundFactory()
    {
        // Same recipe/seed as RushHourModelService — passes the M6 gate (100% within 2× optimal).
        Puzzles = RushHourGenerator.Generate(RushHourModelService.PuzzleSetSeed, count: 30, minOptimal: 4, maxOptimal: 10);
        var env = new RushHourEnv(Puzzles, RushHourModelService.MaxMoves);
        var result = DqnTrainer.Train(env, new DqnOptions
        {
            Hidden = [128, 128],
            Gamma = 0.98,
            LearningRate = 5e-4f,
            MaxSteps = 200_000,
            BufferCapacity = 100_000,
            Epsilon = new LinearSchedule(1.0, 0.05, 60_000),
            EvalEvery = 10_000,
            EvalEpisodes = 20,
            SolveThreshold = 88,
        }, new SeedSequence(RushHourModelService.TrainingMasterSeed));

        var store = new FileModelStore(DataDirectory);
        store.Save(RushHourModelService.EnvironmentId, RushHourModelService.AlgorithmId,
            s => MlpCheckpoint.Save(result.Network, s));
    }
}

[Trait("Category", "Slow")]
public class RushHourSolveGateTests(TrainedPlaygroundFactory factory) : IClassFixture<TrainedPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Solve_GeneratedEasyPuzzle_ReturnsSolvingTrajectoryWithinTwiceOptimal()
    {
        var puzzle = factory.Puzzles[0];
        var board = new RushHourBoardDto(
            [.. puzzle.Vehicles.Select(v => new VehicleDto(v.Row, v.Col, v.Length, v.Horizontal))]);

        var response = await _client.PostAsJsonAsync("/api/rushhour/solve", board);
        response.EnsureSuccessStatusCode();
        var solution = await response.Content.ReadFromJsonAsync<SolveResponse>();

        Assert.NotNull(solution);
        Assert.True(solution.Solved);
        Assert.Equal(puzzle.OptimalMoves, solution.OptimalMoves);
        Assert.True(solution.AiMoves <= 2 * solution.OptimalMoves,
            $"AI used {solution.AiMoves} moves, budget is {2 * solution.OptimalMoves}.");
        Assert.Equal(solution.AiMoves, solution.Trajectory.Length);
        Assert.Equal(solution.OptimalMoves, solution.OptimalTrajectory.Length);

        // The trajectory's final state must actually have the red car at the exit,
        // and every step must be a legal single-cell slide from the previous state.
        AssertTrajectoryIsValid(puzzle, solution.Trajectory, mustSolve: true);
        AssertTrajectoryIsValid(puzzle, solution.OptimalTrajectory, mustSolve: true);
    }

    private static void AssertTrajectoryIsValid(RushHourPuzzle puzzle, TrajectoryStepDto[] trajectory, bool mustSolve)
    {
        var positions = RushHourBoard.InitialPositions(puzzle);
        Span<int> grid = stackalloc int[36];
        foreach (var step in trajectory)
        {
            RushHourBoard.FillOccupancy(puzzle, positions, grid);
            Assert.True(RushHourBoard.CanMove(puzzle, positions, grid, step.Vehicle, step.Direction),
                $"Illegal move: vehicle {step.Vehicle} direction {step.Direction}.");
            positions[step.Vehicle] += step.Direction == 0 ? -1 : 1;
            Assert.Equal(positions, step.Positions); // server-reported state matches replay
        }
        if (mustSolve)
            Assert.True(RushHourBoard.IsSolved(puzzle, positions), "Trajectory does not end with the red car at the exit.");
    }
}
