using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;
using RLDemo.Web.Controllers;

namespace RLDemo.Web.Services;

/// <summary>One curated Rush Hour level: a named board plus its BFS-optimal move count.</summary>
public sealed record DeckLevel(string Id, string Name, VehicleDto[] Vehicles, int OptimalMoves);

public sealed record RushHourDeck(int Version, IReadOnlyList<DeckLevel> Levels);

/// <summary>
/// The curated Rush Hour level deck — canonical, version-controlled content shipped with the app (served
/// statically from <c>wwwroot/rushhour-deck.json</c>), distinct from the runtime, per-deployment
/// <see cref="GalleryStore"/>. Authoring (upsert/delete) is server-side so the board is validated and its
/// optimal move count computed by the BFS solver in one place; the dev-only controller writes the committed
/// file, which you then commit. Production serves the file read-only.
/// </summary>
public sealed class RushHourDeckStore(string filePath)
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    private readonly object _lock = new();

    public string FilePath { get; } = Path.GetFullPath(filePath);

    public RushHourDeck Load()
    {
        lock (_lock)
        {
            if (!File.Exists(FilePath)) return new RushHourDeck(1, []);
            try
            {
                return JsonSerializer.Deserialize<RushHourDeck>(File.ReadAllText(FilePath), Json) ?? new RushHourDeck(1, []);
            }
            catch (JsonException)
            {
                return new RushHourDeck(1, []);
            }
        }
    }

    /// <summary>
    /// Validates a drawn board, computes its BFS-optimal move count, and inserts it (new <c>id</c>) or
    /// replaces an existing level (matching <paramref name="id"/>). Returns the saved level, or an error
    /// message if the board is illegal / unsolvable / already solved.
    /// </summary>
    public (DeckLevel? Level, string? Error) Upsert(string? id, string? name, VehicleDto[] vehicles)
    {
        if (string.IsNullOrWhiteSpace(name)) return (null, "Give the level a name.");
        if (!RushHourBoardDto.TryBuildPuzzle(vehicles, out var puzzle, out string? error)) return (null, error);

        int optimal = RushHourSolver.Solve(puzzle);
        if (optimal < 0) return (null, "Unsolvable — the red car can never reach the exit.");
        if (optimal == 0) return (null, "The red car is already at the exit — add some obstacles.");

        lock (_lock)
        {
            var levels = Load().Levels.ToList();
            var level = new DeckLevel(id ?? Guid.NewGuid().ToString("N")[..8], name.Trim(), vehicles, optimal);
            int index = id is null ? -1 : levels.FindIndex(l => l.Id == id);
            if (index >= 0) levels[index] = level; else levels.Add(level);
            Write(levels);
            return (level, null);
        }
    }

    public bool Delete(string id)
    {
        lock (_lock)
        {
            var levels = Load().Levels.ToList();
            if (levels.RemoveAll(l => l.Id == id) == 0) return false;
            Write(levels);
            return true;
        }
    }

    private void Write(List<DeckLevel> levels)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        string temp = FilePath + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(new RushHourDeck(1, levels), Json));
        File.Move(temp, FilePath, overwrite: true);
    }
}
