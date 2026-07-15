using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments; // generated AddReinforcementLearningGames()
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// DI smoke tests for the M46.3 registration surface: the [Register]-generated game registrations and every
/// hand-written <c>Add&lt;Game&gt;Campaign()</c> extension produce a container whose <see cref="ITrainingCampaign"/>
/// actually resolves — on the CPU path, and (for the GPU-requiring cube campaigns / the self-play compute wiring)
/// with <c>AddGpuBackend()</c>, whose <c>AdaptiveBackend</c> degrades to CPU on GPU-less machines. Resolution only —
/// campaign *behavior* is covered by the contract tests.
/// </summary>
public class CampaignRegistrationTests : IDisposable
{
    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("campaign-di-smoke");

    public void Dispose() => _dir.Delete(recursive: true);

    private ServiceProvider Build(Action<IServiceCollection> configure, bool gpu = false)
    {
        var services = new ServiceCollection();
        services.AddReinforcementLearning(_dir.FullName);
        if (gpu) services.AddGpuBackend();
        services.AddReinforcementLearningGames(); // [Register]-generated (Environments assembly)
        configure(services);
        return services.BuildServiceProvider();
    }

    [Fact]
    public void Games_ResolveFromTheGeneratedRegistrations()
    {
        using var sp = Build(_ => { });
        Assert.IsType<ChessGame>(sp.GetRequiredService<IZeroSumGame<ChessState>>());
        Assert.IsType<ChessGame>(sp.GetRequiredService<IMaterialScore<ChessState>>());
        Assert.IsType<Connect4Game>(sp.GetRequiredService<IZeroSumGame<Connect4State>>());
        Assert.IsType<DraughtsGame>(sp.GetRequiredService<IZeroSumGame<DraughtsState>>());
        Assert.IsType<DraughtsGame>(sp.GetRequiredService<IMaterialScore<DraughtsState>>());
        // The generated registration is the international 10×10 showcase variant (PolicySize 50×50).
        Assert.Equal(2500, sp.GetRequiredService<IZeroSumGame<DraughtsState>>().PolicySize);
    }

    [Fact]
    public void SelfPlayCampaign_Resolves_CpuAndGpuPaths()
    {
        using (var cpu = Build(s => s.AddSelfPlayCampaign<Connect4State>("connect4", new SelfPlayOptions())))
            Assert.IsType<SelfPlayCampaign<Connect4State>>(cpu.GetRequiredService<ITrainingCampaign>());
        using (var cpu = Build(s => s.AddSelfPlayCampaign<DraughtsState>("draughts", new SelfPlayOptions(),
                   netBuilder: new ConvNetBuilder(planes: 5, boardH: 10, boardW: 10, filters: 8, blocks: 1))))
            Assert.IsType<SelfPlayCampaign<DraughtsState>>(cpu.GetRequiredService<ITrainingCampaign>());
        // The GPU path exercises the compute wiring moved out of ChessLab (resident forwards/trainer selection);
        // AdaptiveBackend falls back to the CPU accelerator on GPU-less machines.
        using (var gpu = Build(s => s.AddSelfPlayCampaign<ChessState>("chess", new SelfPlayOptions(),
                   netBuilder: new ConvNetBuilder(planes: 18, boardH: 8, boardW: 8, filters: 8, blocks: 1)), gpu: true))
            Assert.IsType<SelfPlayCampaign<ChessState>>(gpu.GetRequiredService<ITrainingCampaign>());
    }

    [Fact]
    public void CpuCampaigns_Resolve()
    {
        using (var sp = Build(s => s.AddSnakeDqnCampaign(new SnakeEnv(6), new SnakeEnv(12), new DqnScoreOptions())))
            Assert.IsType<SnakeDqnCampaign>(sp.GetRequiredService<ITrainingCampaign>());
        using (var sp = Build(s => s.AddFruitCakeDqnCampaign(new FruitCakeEnv(), new FruitCakeEnv(), new FruitCakeDqnOptions())))
            Assert.IsType<FruitCakeDqnCampaign>(sp.GetRequiredService<ITrainingCampaign>());
        using (var sp = Build(s => s.AddCubeImitationCampaign(new CubeImitationOptions())))
            Assert.IsType<CubeImitationCampaign>(sp.GetRequiredService<ITrainingCampaign>());
        using (var sp = Build(s => s.AddRushHourImitationCampaign(new RushHourImitationOptions())))
            Assert.IsType<RushHourImitationCampaign>(sp.GetRequiredService<ITrainingCampaign>());
    }

    [Fact]
    public void CubeEfficientCampaign_Resolves_WithTheGpuBackend()
    {
        using var sp = Build(s => s.AddCubeEfficientCampaign(new CubeEfficientOptions()), gpu: true);
        Assert.IsType<CubeEfficientCampaign>(sp.GetRequiredService<ITrainingCampaign>());
    }
}
