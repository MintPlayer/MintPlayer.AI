using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using RLDemo.Web.Controllers;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The Block Dude HTTP contract (<c>api/blockdude</c>), which had no test of any kind.
/// </summary>
/// <remarks>
/// <para>These run against a store with no Block Dude checkpoint, which is the interesting half: the shipped net
/// is a 16 MB LFS blob, so on a fresh clone, a CI runner without LFS, or a deploy where the checkpoint was left
/// behind, this is exactly the state the container boots in. What must NOT happen then is a 200 with an empty
/// move list — the page would render a silent no-op "solution" and look like the AI simply gave up. The
/// no-model answer has to be a 503 carrying <c>Ready: false</c>.</para>
/// <para>The other half is request validation. <c>BlockDudeBoard.FromGrid</c> indexes by row/column, so a ragged
/// or empty grid would throw deep inside the solver and surface as a 500 on what is really a client mistake; the
/// controller pre-checks it, and these tests are what keep that pre-check from being refactored away.</para>
/// <para>The solved happy path is deliberately not covered here: it needs the real 16 MB checkpoint, which is a
/// training-gate concern, not an HTTP-contract one.</para>
/// </remarks>
public class BlockDudeApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    // Same width on every row; the shape is all these tests need, since validation runs before the engine.
    private static readonly string[] ValidGrid =
    [
        "     ",
        "  d  ",
        "#####",
    ];

    [Fact]
    public async Task Status_ReportsNotReady_WithoutACheckpoint()
    {
        var response = await _client.GetAsync("/api/blockdude/status");
        response.EnsureSuccessStatusCode();

        var status = await response.Content.ReadFromJsonAsync<BlockDudeController.StatusResponse>();
        Assert.NotNull(status);
        Assert.False(status.Ready);
    }

    [Fact]
    public async Task Solve_MissingGrid_Returns400()
    {
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(null, null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_EmptyGrid_Returns400()
    {
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest([], null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_EmptyRow_Returns400()
    {
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(["#####", ""], null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_RaggedGrid_Returns400_RatherThanA500FromTheEngine()
    {
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(["#####", "###"], null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_WithoutACheckpoint_Returns503WithNotReady()
    {
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(ValidGrid, null));

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);

        // Shape matters: the page branches on this rather than on a bare status code.
        var status = await response.Content.ReadFromJsonAsync<BlockDudeController.StatusResponse>();
        Assert.NotNull(status);
        Assert.False(status.Ready);
    }

    [Fact]
    public async Task Solve_ValidationRunsBeforeTheReadinessCheck()
    {
        // A malformed grid must read back as a client error even while the model is unavailable — otherwise a
        // missing checkpoint would mask every bad request behind a 503 and hide the real mistake.
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(["##", "#"], null));
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Solve_OutOfRangeMaxSeconds_IsNotARequestError()
    {
        // MaxSeconds is clamped server-side (1..30), not rejected: a client asking for an unbounded ceiling gets
        // the clamp, not a 400. With no checkpoint the answer is still the 503, which proves it got past
        // validation rather than being turned away.
        var response = await _client.PostAsJsonAsync("/api/blockdude/solve",
            new BlockDudeController.SolveRequest(ValidGrid, 100_000));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
    }
}

/// <summary>Host fixture with the deploy workflow's build-identity env vars set, as the VPS container has them.</summary>
public class VersionedPlaygroundFactory : PlaygroundFactory
{
    public const string Sha = "abc1234";
    public const string Digest = "sha256:deadbeef";
    public const string Deployed = "2026-09-18T10:00:00Z";

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        base.ConfigureWebHost(builder);
        builder.UseSetting("BUILD_SHA", Sha);
        builder.UseSetting("IMAGE_DIGEST", Digest);
        builder.UseSetting("DEPLOY_TIME", Deployed);
    }
}

/// <summary>
/// The build-identity endpoint (<c>api/version</c>), which had no test at all.
/// </summary>
/// <remarks>
/// It is the maintainer's only way to confirm which image is actually live on the VPS, so its failure mode is
/// quiet and expensive: if an unset or empty env var deserialized as <c>null</c> instead of <c>"dev"</c>, the SPA
/// footer would render blank and a stale deploy would look exactly like a fresh one. Both branches of the
/// unset/empty fallback are pinned, plus the three-field JSON shape the footer binds to.
/// </remarks>
public class VersionApiTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Get_WithoutDeployVariables_ReportsDev()
    {
        var response = await _client.GetAsync("/api/version");
        response.EnsureSuccessStatusCode();

        var version = await response.Content.ReadFromJsonAsync<VersionResponse>();
        Assert.NotNull(version);
        Assert.Equal("dev", version.CommitSha);
        Assert.Equal("dev", version.ImageDigest);
        Assert.Equal("dev", version.DeployTime);
    }
}

/// <summary>The deployed-container branch of <see cref="VersionApiTests"/>: the injected values pass through verbatim.</summary>
public class VersionApiDeployedTests(VersionedPlaygroundFactory factory) : IClassFixture<VersionedPlaygroundFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task Get_WithDeployVariables_ReportsThemVerbatim()
    {
        var version = await _client.GetFromJsonAsync<VersionResponse>("/api/version");

        Assert.NotNull(version);
        Assert.Equal(VersionedPlaygroundFactory.Sha, version.CommitSha);
        Assert.Equal(VersionedPlaygroundFactory.Digest, version.ImageDigest);
        Assert.Equal(VersionedPlaygroundFactory.Deployed, version.DeployTime);
    }
}
