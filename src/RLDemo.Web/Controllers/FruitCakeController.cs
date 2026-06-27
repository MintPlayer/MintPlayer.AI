using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>One streamed fruit in an AI-played FruitCake frame: center (px), orientation (rad), tier.</summary>
public readonly record struct FruitDto(float X, float Y, float Angle, int Tier);

/// <summary>
/// One streamed frame of an AI FruitCake game (PRD §7.1 principle B / §4.6). The server owns the physics
/// and the clock; the browser renders this verbatim.
/// </summary>
public sealed record FruitCakeFrameDto(FruitDto[] Fruit, int HeldTier, int NextTier, int Score, bool Danger, bool Done);

/// <summary>
/// Server-authoritative "Watch AI" for FruitCake. Unlike Snake/Mountain Car (one frame per env step), the
/// agent here decides once per <b>drop</b> but viewers watch ~30 fps of falling/rolling/merging <b>between</b>
/// decisions — so this uses a bespoke intra-drop streamer rather than <c>EpisodeStreamer</c>. The C# physics
/// is authoritative (rotation on for visual parity); the client is a pure renderer. No backend model is
/// required — the agent is the greedy heuristic baseline (A1); a trained net can replace it later (A3).
/// </summary>
[ApiController]
[Route("api/fruitcake")]
public sealed class FruitCakeController(FruitCakeModelService model) : ControllerBase
{
    private static int _seedCounter;
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private const int FrameEvery = 2;      // send a frame every 2nd sub-step (~30 fps payloads)
    private const int FrameDelayMs = 33;   // ≈ 2 sub-steps of sim time → real-time playback (this is the only place we pace to wall-clock)
    private const int BetweenDropsMs = 250;
    private const int GameOverPauseMs = 1800;

    /// <summary>The heuristic is always available, so the agent is ready immediately (no checkpoint to load).</summary>
    [HttpGet("status")]
    public StatusResponse Status() => new("ready", null);

    /// <summary>
    /// Live stream of the AI playing FruitCake. Accepts GET (HTTP/1.1 Upgrade) and CONNECT (HTTP/2, RFC 8441)
    /// so the socket works over HTTP/1.1 and HTTP/2.
    /// </summary>
    [AcceptVerbs("GET", "CONNECT", Route = "live")]
    public async Task Live()
    {
        if (!HttpContext.WebSockets.IsWebSocketRequest)
        {
            Response.StatusCode = StatusCodes.Status400BadRequest;
            return;
        }

        using var socket = await HttpContext.WebSockets.AcceptWebSocketAsync();
        var ct = HttpContext.RequestAborted;
        var rng = new Xoshiro256StarStar((ulong)System.Threading.Interlocked.Increment(ref _seedCounter)); // distinct game per connection
        var agent = model.Agent; // the trained net if a checkpoint shipped; otherwise null → heuristic leaf value

        // F1: depth-3 forward-model search amplifies the net far past its reactive ceiling. Headless 100-game eval
        // (net leaf, same seeds): greedy 994 / 0% watermelon → depth-2 search 2270 / 28% → depth-3 search 2505 /
        // 50% watermelon (current+next are known, the 3rd ply is an expectimax chance node over the unknown fruit;
        // ~45 ms/drop, inside the 250 ms budget). The leaf board value is the net's max-Q, marginalized over the
        // unknown upcoming fruit; with no net it falls back to a pile-height heuristic so the demo always plays.
        // Search clones rotation-off (proven to transfer); the live world stays rotation-on. (Hand-crafted
        // "tier-seeking" leaves were tried and lost to the net leaf.)
        Func<FruitCakeWorld, double> boardValue = agent is not null
            ? w =>
            {
                double sum = 0;
                foreach (var d in FruitCatalog.Droppable)
                    sum += agent.QValues(FruitCakeEnv.BuildObservation(w, d.Tier, d.Tier)).Max();
                return sum / FruitCatalog.Droppable.Count;
            }
            : FruitCakeSearch.HeuristicBoardValue;
        var search = new FruitCakeSearch(boardValue) { MaxDepth = 3, TopK = 5, TopK2 = 2 };
        // Rotation on so fruit visibly roll. The policy trained on rotation-off physics but transfers cleanly
        // (eval: 706 rotation-off → 982 rotation-on), so serving rotation-on is both faithful and prettier.
        var world = new FruitCakeWorld(enableRotation: true);

        try
        {
            while (socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                // Fresh game.
                world.Clear();
                int score = 0;
                int current = NextTier(rng);
                int next = NextTier(rng);
                await Send(socket, Frame(world, current, next, score, false), ct);
                await Task.Delay(400, ct);

                while (socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
                {
                    // Plan the drop with depth-2 forward search (over the known current + next), valuing leaves
                    // by the net (or the heuristic fallback). This is the single call site the policy lives at.
                    int col = search.ChooseColumn(world, current, next);
                    world.SpawnFruit(current, FruitCakeEnv.ColumnX(col, current), FruitCakeEnv.HeldY(current));

                    // Tick the physics in real time, streaming frames, until the drop settles.
                    for (int sub = 0; sub < FruitCakeEnv.MaxSubsteps; sub++)
                    {
                        int gained = world.Step(1f / 60f);
                        score += gained;
                        if (sub % FrameEvery == 0)
                        {
                            await Send(socket, Frame(world, current, next, score, false), ct);
                            await Task.Delay(FrameDelayMs, ct);
                        }
                        if (sub >= FruitCakeEnv.MinSettleSubsteps && gained == 0 && world.MaxSpeed() < FruitCakeEnv.SettleSpeedPx)
                            break;
                    }

                    current = next;
                    next = NextTier(rng);

                    bool over = world.AnyEjected() || world.AnyRestingAboveDangerLine(FruitCakeEnv.RestSpeedPx);
                    await Send(socket, Frame(world, current, next, score, over), ct);
                    if (over)
                    {
                        await Task.Delay(GameOverPauseMs, ct); // hold the game-over frame, then restart
                        break;
                    }
                    await Task.Delay(BetweenDropsMs, ct);
                }
            }
        }
        catch (OperationCanceledException) { /* client disconnected — expected */ }
        catch (WebSocketException) { /* socket dropped mid-send — expected */ }
    }

    private static int NextTier(Xoshiro256StarStar rng) =>
        FruitCatalog.Droppable[rng.NextInt(FruitCatalog.Droppable.Count)].Tier;

    private static FruitCakeFrameDto Frame(FruitCakeWorld world, int held, int next, int score, bool done)
    {
        var fruit = new FruitDto[world.Count];
        int i = 0;
        foreach (var b in world.Bodies) fruit[i++] = new FruitDto(b.X, b.Y, b.Angle, b.Tier);
        return new FruitCakeFrameDto(fruit, held, next, score, world.AnyAboveDangerLine(), done);
    }

    private static Task Send(WebSocket socket, object frame, CancellationToken ct)
    {
        var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(frame, Json));
        return socket.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, ct);
    }
}
