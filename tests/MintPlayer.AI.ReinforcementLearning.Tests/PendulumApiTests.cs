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

/// <summary>Pendulum WS-upgrade contract with no model: the live stream rejects the upgrade (503).</summary>
public class PendulumApiNoModelTests(PlaygroundFactory factory) : IClassFixture<PlaygroundFactory>
{
    [Fact]
    public async Task Live_WithoutModel_RejectsUpgrade()
    {
        var ws = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await Assert.ThrowsAnyAsync<Exception>(() =>
            ws.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/pendulum/live"), cts.Token));
    }
}

/// <summary>Host fixture with a (deliberately untrained) Pendulum SAC actor in the store.</summary>
public class PendulumPlaygroundFactory : PlaygroundFactory
{
    public PendulumPlaygroundFactory()
    {
        // obs dim 3 → output 2 (mean + log-σ for the single torque dimension).
        var actor = new Mlp([3, 32, 2], new Xoshiro256StarStar(7), Activation.Relu);
        new FileModelStore(DataDirectory).Save(
            PendulumModelService.EnvironmentId, PendulumModelService.AlgorithmId, s => MlpCheckpoint.Save(actor, s));
    }
}

public class PendulumApiTests(PendulumPlaygroundFactory factory) : IClassFixture<PendulumPlaygroundFactory>
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    [Fact]
    public async Task Status_ReportsReady_WhenSeeded()
    {
        var status = await factory.CreateClient().GetFromJsonAsync<StatusResponse>("/api/pendulum/status");
        Assert.NotNull(status);
        Assert.Equal("ready", status.Status);
    }

    [Fact]
    public async Task Live_StreamsValidFrames()
    {
        var wsClient = factory.Server.CreateWebSocketClient();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var socket = await wsClient.ConnectAsync(new Uri(factory.Server.BaseAddress, "api/pendulum/live"), cts.Token);

        var buffer = new byte[8 * 1024];
        for (int i = 0; i < 3; i++)
        {
            var result = await socket.ReceiveAsync(buffer, cts.Token);
            var frame = JsonSerializer.Deserialize<PendulumFrameDto>(Encoding.UTF8.GetString(buffer, 0, result.Count), Json);
            Assert.NotNull(frame);
            Assert.Equal(1.0, frame.CosTheta * frame.CosTheta + frame.SinTheta * frame.SinTheta, 3); // on the unit circle
            Assert.InRange(frame.AngularVelocity, -8f, 8f);
            Assert.InRange(frame.Torque, -2f, 2f);
        }

        socket.Abort();
    }
}
