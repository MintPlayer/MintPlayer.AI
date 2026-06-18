using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Environments;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>One streamed frame of an AI Pendulum episode (PRD §7.1 principle B). Observation is the rod angle
/// as (cos θ, sin θ) plus angular velocity; <c>Torque</c> is the continuous action just applied.</summary>
public sealed record PendulumFrameDto(float CosTheta, float SinTheta, float AngularVelocity, float Torque, double Reward, bool Done);

[ApiController]
[Route("api/pendulum")]
public sealed class PendulumController(PendulumModelService model) : ControllerBase
{
    private static int _seedCounter;

    [HttpGet("status")]
    public StatusResponse Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint
        return new(model.Status.ToString().ToLowerInvariant(),
            model.TrainingStep, model.TrainingMaxSteps, model.LastEvalReturn, model.Error);
    }

    /// <summary>
    /// Server-authoritative live stream of the trained AI balancing the pendulum (PRD §7.1 principle B): the
    /// backend owns the episode + clock and pushes one frame per tick; the browser renders. 503 (rejected
    /// upgrade) = still training.
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
            Response.StatusCode = StatusCodes.Status503ServiceUnavailable;
            return;
        }

        using var socket = await HttpContext.WebSockets.AcceptWebSocketAsync();
        var env = new PendulumEnv();
        ulong seed = (ulong)System.Threading.Interlocked.Increment(ref _seedCounter);
        await EpisodeStreamer.RunAsync(
            socket,
            env,
            act: (obs, _) => agent.Act(obs, greedy: true),
            frame: (e, action, step) => new PendulumFrameDto(
                (float)Math.Cos(e.Theta), (float)Math.Sin(e.Theta), (float)e.AngularVelocity,
                action.Length > 0 ? action[0] : 0f, step.Reward, step.Done),
            resetAction: [0f], // zero torque marks the freshly reset start state
            tickMs: 50, // smooth rod motion (matches the env's 0.05s dt)
            startSeed: seed,
            ct: HttpContext.RequestAborted);
    }
}
