using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Environments;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>One streamed frame of an AI MountainCar episode (PRD §7.1 principle B). <c>Action = -1</c> marks a reset.</summary>
public sealed record MountainCarFrameDto(float Position, float Velocity, int Action, double Reward, bool Done);

[ApiController]
[Route("api/mountaincar")]
public sealed class MountainCarController(MountainCarModelService model) : ControllerBase
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
    /// Server-authoritative live stream of the trained AI driving MountainCar (PRD §7.1 principle B): the
    /// backend owns the episode + clock and pushes one frame per tick; the browser renders. 503 (rejected
    /// upgrade) = still training.
    /// </summary>
    [HttpGet("live")]
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
        var env = new MountainCarEnv(); // standard 200-step env for the live demo
        ulong seed = (ulong)System.Threading.Interlocked.Increment(ref _seedCounter);
        await EpisodeStreamer.RunAsync(
            socket,
            env,
            act: (obs, _) => agent.Act(obs, greedy: true),
            frame: (e, action, step) => new MountainCarFrameDto((float)e.Position, (float)e.Velocity, action, step.Reward, step.Done),
            tickMs: 45, // smooth car motion
            startSeed: seed,
            ct: HttpContext.RequestAborted);
    }
}
