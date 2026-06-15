using Microsoft.AspNetCore.Mvc;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>A board submitted to be saved as a deck level; <c>Id</c> set ⇒ update that level, else insert.</summary>
public sealed record SaveLevelRequest(string? Id, string Name, VehicleDto[] Vehicles);

public sealed record DeckError(string? Error);

/// <summary>
/// The curated Rush Hour level deck. <c>GET</c> serves it everywhere (read-only canonical content, shipped
/// in <c>wwwroot/rushhour-deck.json</c>); <c>POST</c>/<c>DELETE</c> author it and are **Development-only** —
/// you draw levels locally, save (validated + optimal computed server-side), then commit the file. Production
/// rejects writes (404), so the deployed deck is exactly what was committed.
/// </summary>
[ApiController]
[Route("api/rushhour/deck")]
public sealed class RushHourDeckController(RushHourDeckStore deck, IWebHostEnvironment env) : ControllerBase
{
    [HttpGet]
    public RushHourDeck Get() => deck.Load();

    [HttpPost]
    public ActionResult<DeckLevel> Save(SaveLevelRequest request)
    {
        if (!env.IsDevelopment()) return NotFound(); // authoring is dev-only; prod is read-only
        var (level, error) = deck.Upsert(request.Id, request.Name, request.Vehicles);
        return level is null ? BadRequest(new DeckError(error)) : level;
    }

    [HttpDelete("{id}")]
    public IActionResult Delete(string id)
    {
        if (!env.IsDevelopment()) return NotFound();
        return deck.Delete(id) ? NoContent() : NotFound();
    }
}
