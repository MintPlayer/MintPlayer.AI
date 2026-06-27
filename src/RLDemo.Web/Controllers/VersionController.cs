using Microsoft.AspNetCore.Mvc;

namespace RLDemo.Web.Controllers;

/// <summary>
/// Build identity of the running container — the git commit that built the image, the image's
/// sha256 digest, and when it was rolled out — surfaced in the SPA footer so the maintainer can
/// confirm exactly which build is live on the VPS. Values are injected as container env vars by
/// the deploy workflow (BUILD_SHA / IMAGE_DIGEST / DEPLOY_TIME, read via <see cref="IConfiguration"/>
/// like DataDirectory); when unset (local dev, or a plain `docker run`) they read back as "dev".
/// </summary>
public sealed record VersionResponse(string CommitSha, string ImageDigest, string DeployTime);

[ApiController]
[Route("api/version")]
public sealed class VersionController(IConfiguration configuration) : ControllerBase
{
    [HttpGet]
    public VersionResponse Get() => new(Read("BUILD_SHA"), Read("IMAGE_DIGEST"), Read("DEPLOY_TIME"));

    // Unset OR empty (compose's `${VAR:-}` substitutes an empty string when .env lacks it) → "dev".
    private string Read(string key)
    {
        var value = configuration[key];
        return string.IsNullOrWhiteSpace(value) ? "dev" : value;
    }
}
