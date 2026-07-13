namespace RLDemo.Web.Controllers;

/// <summary>
/// The shared model-readiness reply for every game's <c>GET /api/&lt;game&gt;/status</c>:
/// the lowercased <see cref="Services.ModelStatus"/> plus an optional load error. One shape for all
/// games so the frontend poller and the API tests treat status uniformly.
/// </summary>
public sealed record StatusResponse(string Status, string? Error);
