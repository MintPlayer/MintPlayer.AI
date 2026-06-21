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
    public async Task SolveEfficient_WithoutModel_Returns503WithStatus()
    {
        var cube = new FaceletCube();
        cube.Apply("R");
        var response = await _client.PostAsJsonAsync("/api/cube/solve-efficient", RequestFor(cube));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        var status = await response.Content.ReadFromJsonAsync<StatusResponse>();
        Assert.NotNull(status);
        Assert.Equal("loading", status.Status);
    }
}

/// <summary>Host fixture with a (deliberately untrained) EfficientCube policy net in the store.</summary>
public class CubeEfficientPlaygroundFactory : PlaygroundFactory
{
    public CubeEfficientPlaygroundFactory()
    {
        // Small + untrained: enough to exercise the load + beam-search wiring. Beam search checks every
        // generated child for the goal before pruning, so it solves a depth-1 scramble even untrained.
        var net = new CubePolicyNet(new Xoshiro256StarStar(9), hidden: 32);
        new FileModelStore(DataDirectory).Save(
            CubeModelService.EnvironmentId, CubeModelService.EfficientPolicyAlgorithmId, s => net.Save(s));
    }
}

public class CubeEfficientSolveTests(CubeEfficientPlaygroundFactory factory) : IClassFixture<CubeEfficientPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    private static CubeSolveRequest RequestFor(FaceletCube cube)
    {
        var faces = cube.ToColorFaces();
        return new(new CubeStateDto(faces[0], faces[1], faces[2], faces[3], faces[4], faces[5]));
    }

    /// <summary>The self-taught solver reports mode "efficient"; beam search solves a shallow scramble even untrained, and the solution actually solves the cube.</summary>
    [Fact]
    public async Task SolveEfficient_ReportsEfficientMode_AndShallowScrambleSolves()
    {
        var cube = new FaceletCube();
        cube.Apply("R");

        var response = await _client.PostAsJsonAsync("/api/cube/solve-efficient", RequestFor(cube));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CubeSolveAiResponse>();
        Assert.NotNull(body);
        Assert.Equal("efficient", body.AiMode);
        Assert.True(body.AlgorithmMoveCount >= 1);
        Assert.Equal(body.Solution.Length, body.MoveCount);
        Assert.True(body.Solved);

        cube.Apply(body.Solution);
        Assert.True(cube.IsSolved);
    }

    [Fact]
    public async Task SolveEfficient_InvalidCube_Returns400()
    {
        var request = RequestFor(new FaceletCube());
        request.State!.U[0] = "G";
        var response = await _client.PostAsJsonAsync("/api/cube/solve-efficient", request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
