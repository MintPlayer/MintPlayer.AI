using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>Snake WS-upgrade contract with no model: the live stream rejects the upgrade (503).</summary>
public class SnakeApiNoModelTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    [Fact]
    public async Task Live_WithoutModel_RejectsUpgrade()
    {
        var ws = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await Assert.ThrowsAnyAsync<Exception>(() =>
            ws.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/snake/live"), cts.Token));
    }
}

/// <summary>Host fixture with a (deliberately untrained) Snake Dueling-DQN in the store.</summary>
public class SnakeModelPlaygroundFactory : PlaygroundFactory
{
    public SnakeModelPlaygroundFactory()
    {
        var net = new DuelingQNet(SnakeEnv.ObservationSize, [32], SnakeEnv.ActionCount, new Xoshiro256StarStar(7));
        new FileModelStore(DataDirectory).Save(
            SnakeModelService.EnvironmentId, SnakeModelService.AlgorithmId, s => DuelingQNetCheckpoint.Save(net, s));
    }
}

public class SnakeApiTests(SnakeModelPlaygroundFactory factory) : IClassFixture<SnakeModelPlaygroundFactory>
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    [Fact]
    public async Task Status_ReportsReady_WhenSeeded()
    {
        var status = await factory.CreateClient().GetFromJsonAsync<StatusResponse>("/api/snake/status");
        Assert.NotNull(status);
        Assert.Equal("ready", status.Status);
    }

    [Fact]
    public async Task Live_StreamsValidFrames()
    {
        var wsClient = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var socket = await wsClient.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/snake/live"), cts.Token);

        var buffer = new byte[16 * 1024];
        for (int i = 0; i < 3; i++) // first frame is the reset (action -1), then live moves
        {
            var result = await socket.ReceiveAsync(buffer, cts.Token);
            var frame = JsonSerializer.Deserialize<SnakeFrameDto>(Encoding.UTF8.GetString(buffer, 0, result.Count), Json);
            Assert.NotNull(frame);
            const int demoCells = 12 * 12; // SnakeController serves a 12×12 grid
            Assert.NotEmpty(frame.Body);                              // a real snake
            Assert.InRange(frame.Food, 0, demoCells - 1);            // food on the demo grid
            Assert.All(frame.Body, c => Assert.InRange(c, 0, demoCells - 1));
        }

        socket.Abort(); // the real teardown path: client drops, the server's next send fails and the handler exits
    }
}
