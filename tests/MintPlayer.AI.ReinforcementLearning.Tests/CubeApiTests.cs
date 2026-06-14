using System.Net;
using System.Net.Http.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>Cube API contract (PRD §11). Kociemba needs no trained model — fast bucket.</summary>
public class CubeApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    private static CubeSolveRequest RequestFor(FaceletCube cube)
    {
        var faces = cube.ToColorFaces();
        return new(new CubeStateDto(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]));
    }

    [Fact]
    public async Task Solve_MissingState_Returns400()
    {
        var response = await _client.PostAsJsonAsync("/api/cube/solve", new CubeSolveRequest(null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_RepaintedSticker_Returns400WithDiagnostics()
    {
        var request = RequestFor(new FaceletCube());
        request.State!.U[0] = "G";

        var response = await _client.PostAsJsonAsync("/api/cube/solve", request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var body = await response.Content.ReadFromJsonAsync<CubeSolveResponse>();
        Assert.NotNull(body);
        Assert.StartsWith("Invalid cube:", body.Error);
        Assert.Contains("corner", body.Error);
    }

    [Fact]
    public async Task Solve_SolvedCube_ReturnsEmptySolution()
    {
        var response = await _client.PostAsJsonAsync("/api/cube/solve", RequestFor(new FaceletCube()));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveResponse>();
        Assert.NotNull(body);
        Assert.Null(body.Error);
        Assert.Empty(body.Solution);
        Assert.Equal(0, body.MoveCount);
    }

    [Fact]
    public async Task Solve_ScrambledCube_SolutionAppliesBackToSolved()
    {
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(11), 15));

        var response = await _client.PostAsJsonAsync("/api/cube/solve", RequestFor(cube));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveResponse>();
        Assert.NotNull(body);
        Assert.Null(body.Error);
        Assert.InRange(body.MoveCount, 1, CubeSolver.MaxDepth);

        cube.Apply(body.Solution);
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public async Task SolveAi_WithoutModel_Returns503WithStatus()
    {
        var cube = new FaceletCube();
        cube.Apply("R");
        var response = await _client.PostAsJsonAsync("/api/cube/solve-ai", RequestFor(cube));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        var status = await response.Content.ReadFromJsonAsync<StatusResponse>();
        Assert.NotNull(status);
        Assert.Equal("loading", status.Status);
    }

    [Fact]
    public async Task SolveDavi_WithoutValueNet_Returns503()
    {
        var cube = new FaceletCube();
        cube.Apply("R");
        var response = await _client.PostAsJsonAsync("/api/cube/solve-davi", RequestFor(cube));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
    }
}

/// <summary>Host fixture with a (deliberately untrained) DAVI residual value net in the store.</summary>
public class CubeDaviPlaygroundFactory : PlaygroundFactory
{
    public CubeDaviPlaygroundFactory()
    {
        // Small + untrained: enough to exercise the load + BWAS wiring; the search itself
        // (broad exploration under a generous budget) still solves a shallow scramble.
        var net = new ResidualMlp(RubiksCubeEnv.ObservationSize, width: 32, blocks: 1, new Xoshiro256StarStar(7));
        new FileModelStore(DataDirectory).Save(
            CubeModelService.EnvironmentId, CubeModelService.ValueDaviAlgorithmId, s => ResidualMlpCheckpoint.Save(net, s));
    }
}

public class CubeDaviSolveTests(CubeDaviPlaygroundFactory factory) : IClassFixture<CubeDaviPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    private static CubeSolveRequest RequestFor(FaceletCube cube)
    {
        var faces = cube.ToColorFaces();
        return new(new CubeStateDto(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]));
    }

    /// <summary>The self-taught solver reports mode "davi"; BWAS solves a shallow scramble even untrained, and the solution actually solves the cube.</summary>
    [Fact]
    public async Task SolveDavi_ReportsDaviMode_AndShallowScrambleSolves()
    {
        var cube = new FaceletCube();
        cube.Apply("R");

        var response = await _client.PostAsJsonAsync("/api/cube/solve-davi", RequestFor(cube));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveAiResponse>();
        Assert.NotNull(body);
        Assert.Equal("davi", body.AiMode);
        Assert.True(body.AlgorithmMoveCount >= 1);
        Assert.Equal(body.Solution.Length, body.MoveCount);
        Assert.True(body.Solved);

        cube.Apply(body.Solution);
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public async Task SolveDavi_InvalidCube_Returns400()
    {
        var request = RequestFor(new FaceletCube());
        request.State!.U[0] = "G";
        var response = await _client.PostAsJsonAsync("/api/cube/solve-davi", request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}

/// <summary>Host fixture with a (deliberately untrained) cube DQN in the store.</summary>
public class CubeModelPlaygroundFactory : PlaygroundFactory
{
    public CubeModelPlaygroundFactory()
    {
        var network = new Mlp([RubiksCubeEnv.ObservationSize, 32, RubiksCubeEnv.ActionCount], new Xoshiro256StarStar(5));
        new FileModelStore(DataDirectory).Save(
            CubeModelService.EnvironmentId, CubeModelService.AlgorithmId, s => MlpCheckpoint.Save(network, s));
    }
}

public class CubeSolveAiTests(CubeModelPlaygroundFactory factory) : IClassFixture<CubeModelPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    private static CubeSolveRequest RequestFor(FaceletCube cube)
    {
        var faces = cube.ToColorFaces();
        return new(new CubeStateDto(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]));
    }

    /// <summary>The honest-reporting contract — an untrained net must still answer within budget.</summary>
    [Fact]
    public async Task SolveAi_HonorsContract_AndOnlySolvedTrajectoriesSolve()
    {
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(21), 4, quarterTurnsOnly: true));

        var response = await _client.PostAsJsonAsync("/api/cube/solve-ai", RequestFor(cube));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveAiResponse>();
        Assert.NotNull(body);
        Assert.InRange(body.MoveCount, 0, CubeModelService.MaxMoves);
        Assert.Equal(body.Solution.Length, body.MoveCount);
        Assert.True(body.AlgorithmMoveCount >= 1);

        var replay = new FaceletCube();
        var faces = RequestFor(cube).State!;
        replay = FaceletCube.FromColorFaces(faces.U, faces.R, faces.F, faces.D, faces.L, faces.B);
        replay.Apply(body.Solution);
        Assert.Equal(body.Solved, replay.IsSolved);
    }

    [Fact]
    public async Task SolveAi_InvalidCube_Returns400()
    {
        var request = RequestFor(new FaceletCube());
        request.State!.U[0] = "G";
        var response = await _client.PostAsJsonAsync("/api/cube/solve-ai", request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task SolveAi_WithoutPolicyNet_ReportsDqnOrSearchMode()
    {
        var cube = new FaceletCube();
        cube.Apply("R");
        var response = await _client.PostAsJsonAsync("/api/cube/solve-ai", RequestFor(cube));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveAiResponse>();
        Assert.NotNull(body);
        Assert.Contains(body.AiMode, new[] { "dqn", "search" });
    }
}

/// <summary>Host fixture with a (deliberately untrained) imitation policy net in the store.</summary>
public class CubePolicyPlaygroundFactory : PlaygroundFactory
{
    public CubePolicyPlaygroundFactory()
    {
        var net = new CubePolicyNet(new Xoshiro256StarStar(9), hidden: 32);
        new FileModelStore(DataDirectory).Save(
            CubeModelService.EnvironmentId, CubeModelService.PolicyAlgorithmId, s => net.Save(s));
    }
}

public class CubePolicySolveTests(CubePolicyPlaygroundFactory factory) : IClassFixture<CubePolicyPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    /// <summary>The policy net is preferred over the (absent) DQN, and search rescues an untrained policy on depth 1.</summary>
    [Fact]
    public async Task SolveAi_PrefersPolicyNet_AndSearchSolvesShallowEvenUntrained()
    {
        var cube = new FaceletCube();
        cube.Apply("R");
        var faces = cube.ToColorFaces();
        var request = new CubeSolveRequest(new CubeStateDto(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]));

        var response = await _client.PostAsJsonAsync("/api/cube/solve-ai", request);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveAiResponse>();
        Assert.NotNull(body);
        Assert.Contains(body.AiMode, new[] { "greedy", "search" }); // policy path, not "dqn"
        Assert.True(body.Solved);

        cube.Apply(body.Solution);
        Assert.True(cube.IsSolved);
    }
}
