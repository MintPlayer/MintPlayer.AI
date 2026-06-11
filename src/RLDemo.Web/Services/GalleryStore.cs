using System.Text.Json;

namespace RLDemo.Web.Services;

public sealed record GalleryEntry(
    string Id,
    string Game,
    DateTimeOffset CreatedUtc,
    string Summary,
    JsonElement Request,
    JsonElement Response);

public sealed record GalleryListItem(string Id, string Game, DateTimeOffset CreatedUtc, string Summary);

/// <summary>
/// The public submitted-games gallery (PRD §7.6): every solved board + its solution is
/// persisted as one JSON file under <c>&lt;data&gt;/gallery</c>, so the collection
/// survives restarts and rides the same Docker volume as the model store.
/// </summary>
public sealed class GalleryStore(string rootDirectory)
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web) { WriteIndented = true };

    public string RootDirectory { get; } = Path.GetFullPath(rootDirectory);

    public GalleryEntry Add(string game, string summary, object request, object response)
    {
        var entry = new GalleryEntry(
            Id: $"{DateTime.UtcNow:yyyyMMddHHmmssfff}-{Guid.NewGuid().ToString("N")[..6]}",
            Game: game,
            CreatedUtc: DateTimeOffset.UtcNow,
            Summary: summary,
            Request: JsonSerializer.SerializeToElement(request, Options),
            Response: JsonSerializer.SerializeToElement(response, Options));

        Directory.CreateDirectory(RootDirectory);
        string path = PathOf(entry.Id);
        string temp = path + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(entry, Options));
        File.Move(temp, path, overwrite: true);
        return entry;
    }

    /// <summary>Newest first.</summary>
    public IReadOnlyList<GalleryListItem> List()
    {
        if (!Directory.Exists(RootDirectory)) return [];
        var items = new List<GalleryListItem>();
        foreach (string file in Directory.EnumerateFiles(RootDirectory, "*.json"))
        {
            try
            {
                var entry = JsonSerializer.Deserialize<GalleryEntry>(File.ReadAllText(file), Options);
                if (entry is not null)
                    items.Add(new GalleryListItem(entry.Id, entry.Game, entry.CreatedUtc, entry.Summary));
            }
            catch (JsonException)
            {
                // A corrupt entry must not take down the whole gallery.
            }
        }
        return [.. items.OrderByDescending(i => i.CreatedUtc)];
    }

    public GalleryEntry? Get(string id)
    {
        string path = PathOf(id);
        if (!File.Exists(path)) return null;
        return JsonSerializer.Deserialize<GalleryEntry>(File.ReadAllText(path), Options);
    }

    private string PathOf(string id)
    {
        if (id.Any(c => !char.IsAsciiLetterOrDigit(c) && c != '-'))
            throw new ArgumentException($"Invalid gallery id '{id}'.");
        return Path.Combine(RootDirectory, id + ".json");
    }
}
