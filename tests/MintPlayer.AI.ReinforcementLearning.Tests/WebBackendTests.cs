using System.Net;
using System.Net.Http.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;
using Xunit;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class GalleryStoreTests
{
    private static string TempRoot() => Path.Combine(Path.GetTempPath(), "rlnet-gallery-test-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void Add_List_Get_RoundTrip_AndSurvivesNewInstance()
    {
        string root = TempRoot();
        try
        {
            var store = new GalleryStore(root);
            Assert.Empty(store.List());

            var first = store.Add("rushhour", "AI solved it in 9 moves (optimal 7)",
                new { vehicles = new[] { new { row = 2, col = 0 } } }, new { solved = true });
            var second = store.Add("2048", "scored 12,345", new { cells = new int[16] }, new { score = 12345 });

            var list = store.List();
            Assert.Equal(2, list.Count);
            Assert.Equal(second.Id, list[0].Id); // newest first
            Assert.Equal(first.Id, list[1].Id);

            // A fresh instance over the same directory sees everything (restart survival).
            var reopened = new GalleryStore(root);
            Assert.Equal(2, reopened.List().Count);
            var entry = reopened.Get(first.Id);
            Assert.NotNull(entry);
            Assert.Equal("rushhour", entry.Game);
            Assert.True(entry.Response.GetProperty("solved").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void CorruptEntry_IsSkipped_NotFatal()
    {
        string root = TempRoot();
        try
        {
            var store = new GalleryStore(root);
            store.Add("2048", "ok", new { }, new { });
            File.WriteAllText(Path.Combine(root, "20990101000000000-corrupt.json"), "{ not json");

            Assert.Single(store.List());
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Get_RejectsPathEscapes()
    {
        var store = new GalleryStore(TempRoot());
        Assert.Throws<ArgumentException>(() => store.Get("../escape"));
    }
}

/// <summary>2048 API contract without a trained model (fast).</summary>
public class Game2048ApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Solve_WithoutModel_Returns503()
    {
        var cells = new int[16];
        cells[0] = 2;
        var response = await _client.PostAsJsonAsync("/api/2048/solve", new Board2048Dto(cells));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
    }

    [Theory]
    [InlineData(new int[] { 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 })] // not a power of two
    [InlineData(new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 })] // empty
    [InlineData(new int[] { 2, 4 })]                                            // wrong length
    public async Task Solve_InvalidBoard_Returns400(int[] cells)
    {
        var response = await _client.PostAsJsonAsync("/api/2048/solve", new Board2048Dto(cells));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}

/// <summary>Host fixture with a small but real n-tuple model in the store (trains ~2 s).</summary>
public class Trained2048Factory : PlaygroundFactory
{
    public Trained2048Factory()
    {
        var agent = new NTuple2048Agent();
        var rng = new Xoshiro256StarStar(1);
        for (int game = 0; game < 1000; game++)
            agent.PlayGame(rng, learn: true);

        new FileModelStore(DataDirectory).Save(
            Game2048ModelService.EnvironmentId, Game2048ModelService.AlgorithmId, s => agent.Save(s));
    }
}

public class Game2048SolveTests(Trained2048Factory factory) : IClassFixture<Trained2048Factory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Solve_ReturnsReplayableTrajectory_AndRecordsGalleryEntry()
    {
        int[] cells = new int[16];
        cells[0] = 2;
        cells[5] = 2;
        var response = await _client.PostAsJsonAsync("/api/2048/solve", new Board2048Dto(cells));
        response.EnsureSuccessStatusCode();
        var solution = await response.Content.ReadFromJsonAsync<SolveResponse2048>();
        Assert.NotNull(solution);
        Assert.NotEmpty(solution.Steps);

        // Replay client-side: every action must change the board, every spawn must land on
        // an empty cell, and the reconstruction must end exactly at FinalCells/Score.
        var board = new byte[16];
        for (int i = 0; i < 16; i++)
            board[i] = (byte)(solution.InitialCells[i] == 0 ? 0 : int.TrailingZeroCount(solution.InitialCells[i]));
        int score = 0;
        foreach (var step in solution.Steps)
        {
            Assert.True(Board2048.ApplyMove(board, step.Action, out _, out int gained), "illegal move in playout");
            Assert.Equal(step.ScoreGained, gained);
            Assert.Equal(0, board[step.SpawnIndex]);
            board[step.SpawnIndex] = (byte)int.TrailingZeroCount(step.SpawnValue);
            score += gained;
        }
        Assert.Equal(solution.Score, score);
        for (int i = 0; i < 16; i++)
            Assert.Equal(solution.FinalCells[i], board[i] == 0 ? 0 : 1 << board[i]);
        Assert.False(Board2048.AnyMoveAvailable(board)); // played to game over

        // Determinism: the same drawing produces the identical playout.
        var again = await (await _client.PostAsJsonAsync("/api/2048/solve", new Board2048Dto(cells)))
            .Content.ReadFromJsonAsync<SolveResponse2048>();
        Assert.Equal(solution.Score, again!.Score);
        Assert.Equal(solution.Steps.Length, again.Steps.Length);

        // And the solve was recorded in the public gallery.
        var gallery = await _client.GetFromJsonAsync<List<GalleryListItem>>("/api/gallery");
        Assert.NotNull(gallery);
        Assert.Contains(gallery, e => e.Game == "2048");
    }
}
