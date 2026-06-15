using System.Net;
using System.Net.Http.Json;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class RushHourDeckStoreTests
{
    private static RushHourDeckStore NewStore() =>
        new(Path.Combine(Path.GetTempPath(), "rhdeck-" + Guid.NewGuid().ToString("N"), "deck.json"));

    // A red car (row 2 = exit row) at the given column; col 0 slides 4 to the exit, col 4 is already solved.
    private static VehicleDto[] Red(int col) => [new VehicleDto(2, col, 2, true)];

    [Fact]
    public void Load_MissingFile_ReturnsEmptyDeck()
    {
        var deck = NewStore().Load();
        Assert.Equal(1, deck.Version);
        Assert.Empty(deck.Levels);
    }

    [Fact]
    public void Upsert_ValidBoard_PersistsWithComputedOptimal()
    {
        var store = NewStore();
        var (level, error) = store.Upsert(null, "Slide", Red(0));

        Assert.Null(error);
        Assert.NotNull(level);
        Assert.True(level!.OptimalMoves > 0, "a red car at col 0 must take some moves to reach the exit");

        var deck = store.Load(); // persisted to disk
        Assert.Single(deck.Levels);
        Assert.Equal("Slide", deck.Levels[0].Name);
    }

    [Fact]
    public void Upsert_ExistingId_UpdatesInPlace()
    {
        var store = NewStore();
        var (first, _) = store.Upsert(null, "L", Red(0));
        var (second, error) = store.Upsert(first!.Id, "L renamed", Red(1));

        Assert.Null(error);
        Assert.Equal(first.Id, second!.Id);
        var deck = store.Load();
        Assert.Single(deck.Levels); // updated, not appended
        Assert.Equal("L renamed", deck.Levels[0].Name);
    }

    [Fact]
    public void Upsert_EmptyName_Rejected()
    {
        var (level, error) = NewStore().Upsert(null, "  ", Red(0));
        Assert.Null(level);
        Assert.NotNull(error);
    }

    [Fact]
    public void Upsert_AlreadySolvedBoard_Rejected()
    {
        var (level, error) = NewStore().Upsert(null, "Done", Red(4)); // red already at the exit
        Assert.Null(level);
        Assert.NotNull(error);
    }

    [Fact]
    public void Delete_RemovesLevel()
    {
        var store = NewStore();
        var (level, _) = store.Upsert(null, "L", Red(0));
        Assert.True(store.Delete(level!.Id));
        Assert.Empty(store.Load().Levels);
        Assert.False(store.Delete(level.Id)); // already gone
    }
}

public class RushHourDeckApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    [Fact]
    public async Task Get_Deck_ReturnsOk()
    {
        var response = await factory.CreateClient().GetAsync("/api/rushhour/deck");
        response.EnsureSuccessStatusCode();
        var deck = await response.Content.ReadFromJsonAsync<RushHourDeck>();
        Assert.NotNull(deck);
    }

    [Fact]
    public async Task Post_OutsideDevelopment_IsRejected()
    {
        // Authoring is Development-only; the Testing host must reject writes so production can't be mutated.
        var response = await factory.CreateClient().PostAsJsonAsync("/api/rushhour/deck",
            new { name = "x", vehicles = new[] { new { row = 2, col = 0, length = 2, horizontal = true } } });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}
