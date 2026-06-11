using Microsoft.AspNetCore.Mvc;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

[ApiController]
[Route("api/gallery")]
public sealed class GalleryController(GalleryStore gallery) : ControllerBase
{
    /// <summary>All submitted games, newest first.</summary>
    [HttpGet]
    public IReadOnlyList<GalleryListItem> List() => gallery.List();

    [HttpGet("{id}")]
    public ActionResult<GalleryEntry> Get(string id)
    {
        try
        {
            var entry = gallery.Get(id);
            return entry is null ? NotFound() : entry;
        }
        catch (ArgumentException)
        {
            return NotFound();
        }
    }
}
