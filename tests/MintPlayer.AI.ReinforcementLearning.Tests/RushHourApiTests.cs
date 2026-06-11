using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>Host fixture: Testing environment (no SPA, no auto-training), isolated temp model store.</summary>
public class PlaygroundFactory : WebApplicationFactory<Program>
{
    public string DataDirectory { get; } =
        Path.Combine(Path.GetTempPath(), "rlnet-web-test-" + Guid.NewGuid().ToString("N"));

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.UseSetting("DataDirectory", DataDirectory);
        builder.UseSetting("SeedModelsDirectory", ""); // tests control the store themselves
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
    public async Task Analyze_LengthThreeRedVehicle_IsSupported()
    {
        // Red truck at (2,1) occupies cols 1-3: two single-cell slides put its nose at the exit.
        VehicleDto[] redTruck = [new(2, 1, 3, true)];
        var response = await _client.PostAsJsonAsync("/api/rushhour/analyze", new RushHourBoardDto(redTruck));
        response.EnsureSuccessStatusCode();
        var analysis = await response.Content.ReadFromJsonAsync<AnalyzeResponse>();
        Assert.NotNull(analysis);
        Assert.True(analysis.Valid);
        Assert.True(analysis.Solvable);
        Assert.Equal(2, analysis.OptimalMoves);
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

public class RushHourRolloutTests
{
    [Fact]
    public void CycleAvoidance_SolvesTrivialPuzzle_EvenWithAnUntrainedNetwork()
    {
        // Lone red car, 4 moves to the exit. The only escape from "left undoes right"
        // shuttling is the visited-state check — so even a RANDOM network must reach the
        // exit in exactly 4 moves, never revisiting a state.
        var puzzle = new RushHourPuzzle([new Vehicle(2, 0, 2, Horizontal: true)]);
        var untrained = new GreedyQAgent(
            new MintPlayer.AI.ReinforcementLearning.Core.Nn.Mlp([72, 32, 32], new Xoshiro256StarStar(123), MintPlayer.AI.ReinforcementLearning.Core.Nn.Activation.Relu),
            RushHourBoard.ActionCount);

        var (solved, steps) = RushHourRollout.Run(untrained, puzzle, maxMoves: 60);

        Assert.True(solved);
        Assert.Equal(4, steps.Count);
        Assert.Equal(steps.Count, steps.Select(s => RushHourSolver.Encode(s.Positions)).Distinct().Count());
    }

}

/// <summary>Host fixture with a (deliberately untrained) policy net in the store.</summary>
public class PolicyPlaygroundFactory : PlaygroundFactory
{
    public PolicyPlaygroundFactory()
    {
        var net = new RushHourPolicyNet(new Xoshiro256StarStar(3), hidden: 32);
        new FileModelStore(DataDirectory).Save(
            RushHourModelService.EnvironmentId, RushHourModelService.PolicyAlgorithmId, s => net.Save(s));
    }
}

public class RushHourPolicySolveTests(PolicyPlaygroundFactory factory) : IClassFixture<PolicyPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Solve_PrefersPolicyNet_AndSearchSolvesEvenUntrained()
    {
        // The hand-verified optimal-7 board. An untrained policy almost certainly fails
        // the greedy rollout, but policy-guided A* must still solve it (it degrades
        // toward uniform-cost search) — and the response says which mode answered.
        VehicleDto[] board = [new(2, 0, 2, true), new(0, 2, 3, false)];
        var response = await _client.PostAsJsonAsync("/api/rushhour/solve", new RushHourBoardDto(board));
        response.EnsureSuccessStatusCode();
        var solution = await response.Content.ReadFromJsonAsync<SolveResponse>();

        Assert.NotNull(solution);
        Assert.True(solution.Solved);
        Assert.Contains(solution.AiMode, new[] { "greedy", "search" });
        Assert.Equal(7, solution.OptimalMoves);
        Assert.Equal(solution.AiMoves, solution.Trajectory.Length);

        // The returned trajectory must replay legally and end solved.
        var puzzle = new RushHourPuzzle([new Vehicle(2, 0, 2, true), new Vehicle(0, 2, 3, false)]);
        var positions = RushHourBoard.InitialPositions(puzzle);
        foreach (var step in solution.Trajectory)
        {
            Assert.True(RushHourBoard.ActionMask(puzzle, positions)[step.Vehicle * 2 + step.Direction]);
            positions[step.Vehicle] += step.Direction == 0 ? -1 : 1;
            Assert.Equal(positions, step.Positions);
        }
        Assert.True(RushHourBoard.IsSolved(puzzle, positions));
    }
}

/// <summary>Host fixture with a model pre-trained into the store (the M8 API gate).</summary>
public class TrainedPlaygroundFactory : PlaygroundFactory
{
    public IReadOnlyList<RushHourPuzzle> Puzzles { get; }

    public TrainedPlaygroundFactory()
    {
        // Exactly the service's recipe — shared statics so the gate can't drift from production.
        Puzzles = RushHourModelService.TrainingPuzzles();
        var env = new RushHourEnv(Puzzles, RushHourModelService.MaxMoves);
        var result = DqnTrainer.Train(env, RushHourModelService.TrainingOptions(),
            new SeedSequence(RushHourModelService.TrainingMasterSeed));

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
