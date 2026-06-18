using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;

namespace RLDemo.Web.Services;

/// <summary>
/// The reusable server side of the "watch the AI play" interaction (PRD §7.1, principle B):
/// <b>the backend owns the episode loop and the clock</b>. It runs a fresh env per connection
/// (Reset → loop{ policy → Step → send frame } until done, then restart for continuous play),
/// streaming one JSON frame per tick. The browser is a pure renderer with no timer of its own, so
/// pacing is server-controlled and there is no client-timer race.
/// <para>
/// Each call owns its own <typeparamref name="TEnv"/> instance, so concurrent sockets are isolated
/// automatically — the only shared thing is the (read-only) policy behind <paramref name="act"/>.
/// The loop ends when the socket closes (client navigates away → <paramref name="ct"/> trips).
/// </para>
/// </summary>
public static class EpisodeStreamer
{
    public static async Task RunAsync<TEnv, TAct>(
        WebSocket socket,
        TEnv env,
        Func<float[], bool[]?, TAct> act,
        Func<TEnv, TAct, StepResult<float[]>, object> frame,
        TAct resetAction,
        int tickMs,
        ulong startSeed,
        CancellationToken ct)
        where TEnv : IEnvironment<float[], TAct>
    {
        var masker = env as IActionMaskProvider;
        ulong seed = startSeed;
        try
        {
            while (socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var (obs, _) = env.Reset(seed++);
                // The reset-marker action tags the freshly reset start state so the client draws it.
                await Send(socket, frame(env, resetAction, new StepResult<float[]>(obs, 0, false, false, EnvInfo.Empty)), ct);
                await Task.Delay(tickMs, ct);

                while (socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
                {
                    TAct action = act(obs, masker?.CurrentActionMask());
                    var step = env.Step(action);
                    obs = step.Observation;
                    await Send(socket, frame(env, action, step), ct);
                    if (step.Done)
                    {
                        await Task.Delay(tickMs * 5, ct); // brief pause on the final frame, then restart
                        break;
                    }
                    await Task.Delay(tickMs, ct);
                }
            }
        }
        catch (OperationCanceledException) { /* client disconnected — expected */ }
        catch (WebSocketException) { /* socket dropped mid-send — expected */ }
    }

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private static Task Send(WebSocket socket, object frame, CancellationToken ct)
    {
        var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(frame, Json));
        return socket.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, ct);
    }
}
