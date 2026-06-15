using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using RLDemo.Web.Controllers;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>MountainCar WS-upgrade contract with no model: the live stream rejects the upgrade (503).</summary>
public class MountainCarApiNoModelTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    [Fact]
    public async Task Live_WithoutModel_RejectsUpgrade()
    {
        var ws = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await Assert.ThrowsAnyAsync<Exception>(() =>
            ws.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/mountaincar/live"), cts.Token));
    }
}

/// <summary>Host fixture with a (deliberately untrained) MountainCar PPO actor in the store.</summary>
public class MountainCarPlaygroundFactory : PlaygroundFactory
{
    public MountainCarPlaygroundFactory()
    {
        var actor = new Mlp([2, 32, 3], new Xoshiro256StarStar(7), Activation.Tanh);
        new FileModelStore(DataDirectory).Save(
            MountainCarModelService.EnvironmentId, MountainCarModelService.AlgorithmId, s => MlpCheckpoint.Save(actor, s));
    }
}

public class MountainCarApiTests(MountainCarPlaygroundFactory factory) : IClassFixture<MountainCarPlaygroundFactory>
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    [Fact]
    public async Task Status_ReportsReady_WhenSeeded()
    {
        var status = await factory.CreateClient().GetFromJsonAsync<StatusResponse>("/api/mountaincar/status");
        Assert.NotNull(status);
        Assert.Equal("ready", status.Status);
    }

    [Fact]
    public async Task Live_StreamsValidFrames()
    {
        var wsClient = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var socket = await wsClient.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/mountaincar/live"), cts.Token);

        var buffer = new byte[8 * 1024];
        for (int i = 0; i < 3; i++)
        {
            var result = await socket.ReceiveAsync(buffer, cts.Token);
            var frame = JsonSerializer.Deserialize<MountainCarFrameDto>(Encoding.UTF8.GetString(buffer, 0, result.Count), Json);
            Assert.NotNull(frame);
            Assert.InRange(frame.Position, -1.2f, 0.6f);   // within the track
            Assert.InRange(frame.Velocity, -0.07f, 0.07f);
        }

        socket.Abort();
    }
}
