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

    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint so status reflects it
        return new(model.Status.ToString().ToLowerInvariant(),
            model.TrainingStep, model.TrainingMaxSteps, model.LastEvalReturn, model.Error);
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
        var agent = model.Agent;
        if (agent is null)
        {
            Response.StatusCode = StatusCodes.Status503ServiceUnavailable; // still training — reject the upgrade
            return;
        }

        using var socket = await HttpContext.WebSockets.AcceptWebSocketAsync();
        var env = new SnakeEnv();
        ulong seed = (ulong)System.Threading.Interlocked.Increment(ref _seedCounter); // distinct game per connection
        await EpisodeStreamer.RunAsync(
            socket,
            env,
            act: (obs, mask) => agent.Act(obs, mask, greedy: true),
            frame: (e, action, step) => new SnakeFrameDto(
                [.. e.Body], e.Food, action, step.Reward, step.Done, e.FoodEaten, e.Length),
            resetAction: -1, // -1 marks the freshly reset start state
            tickMs: 120,
            startSeed: seed,
            ct: HttpContext.RequestAborted);
    }
}
