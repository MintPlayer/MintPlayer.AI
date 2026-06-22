using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>One streamed frame of an AI Snake game (PRD §7.1 principle B). <c>Action = -1</c> marks a freshly reset start state.</summary>
public sealed record SnakeFrameDto(int[] Body, int Food, int Action, double Reward, bool Done, int FoodEaten, int Length);

[ApiController]
[Route("api/snake")]
public sealed class SnakeController(SnakeModelService model) : ControllerBase
{
    private static int _seedCounter;

    // The live demo plays via net-guided multi-ply look-ahead (PLAN M27): the trained net evaluates leaf positions
    // while a 20-ply beam search guarantees the snake never walks into a box it can't escape — the failure that caps
    // a reactive policy. Depth 20 is the empirical sweet spot on the 12×12 grid (≈84 food vs ≈50 for the raw greedy
    // net; deeper gives no reliable gain and costs more per tick). Each move searches in a few ms — well inside the tick.
    private static readonly SnakeSearchOptions LiveSearch = new() { MaxDepth = 20, BeamWidth = 32, SpaceWeight = 50f };

    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint so status reflects it
        return new(model.Status.ToString().ToLowerInvariant(), model.Error);
    }

    /// <summary>
    /// Server-authoritative live stream of the trained AI playing Snake (PRD §7.1 principle B). The backend
    /// owns the episode + clock and pushes one frame per tick; the browser is a pure renderer. A 503 (rejected
    /// upgrade) means the model is still training — the client polls <c>status</c> and connects when ready.
    /// <para>Accepts GET (HTTP/1.1 Upgrade) AND CONNECT (HTTP/2 Extended CONNECT, RFC 8441) so the WebSocket
    /// works over both — over HTTPS the browser negotiates HTTP/2, where a GET-only route returns 405.</para>
    /// </summary>
    [AcceptVerbs("GET", "CONNECT", Route = "live")]
    public async Task Live()
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            Response.StatusCode = StatusCodes.Status400BadRequest;
            return;
        }
        var net = model.Net;
        if (net is null)
        {
            Response.StatusCode = StatusCodes.Status503ServiceUnavailable; // still training — reject the upgrade
            return;
        }

        using var socket = await HttpContext.WebSockets.AcceptWebSocketAsync();
        var env = new SnakeEnv(safeMask: true);
        var planner = new SnakeSearchAgent(env, net, LiveSearch); // net-guided look-ahead reads the env state directly
        ulong seed = (ulong)System.Threading.Interlocked.Increment(ref _seedCounter); // distinct game per connection
        await EpisodeStreamer.RunAsync(
            socket,
            env,
            act: (_, _) => planner.Act(), // obs/mask ignored — the planner searches from the live env's full state
            frame: (e, action, step) => new SnakeFrameDto(
                [.. e.Body], e.Food, action, step.Reward, step.Done, e.FoodEaten, e.Length),
            tickMs: 120,
            startSeed: seed,
            ct: HttpContext.RequestAborted);
    }
}
