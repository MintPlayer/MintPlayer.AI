using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// The campaign registration surface (PLAN M46.3): one <c>Add&lt;Game&gt;Campaign(options)</c> per campaign,
/// each registering an <see cref="ITrainingCampaign"/> singleton a host resolves and hands to the
/// <see cref="CampaignRunner"/>. Runtime knobs travel as the options records; container services (the game via
/// its <c>[Register]</c>-generated <c>AddReinforcementLearningGames()</c>, the optional
/// <see cref="AdaptiveBackend"/> from <c>AddGpuBackend()</c>) are resolved at build time. These are hand-written
/// rather than <c>[Register]</c>-generated because each closes over per-run options — the generator registers
/// types, not configured factories.
/// </summary>
public static class CampaignServiceCollectionExtensions
{
    /// <summary>
    /// AlphaZero-style self-play over the container's <see cref="IZeroSumGame{TState}"/>. Owns the compute
    /// wiring the campaign itself must stay ignorant of (its Ilgpu-free factory params): with a GPU backend
    /// registered and a conv net, generation gets one GPU-resident forward per selected device (M43/M45) and
    /// training a resident step on the primary (M44); otherwise everything stays CPU autograd.
    /// </summary>
    /// <param name="gpus">Which detected CUDA GPUs to shard generation across: "all" (default), a count
    /// ("2"), or explicit ordinals ("0,2"). Ignored when no GPU backend is registered.</param>
    public static IServiceCollection AddSelfPlayCampaign<TState>(this IServiceCollection services,
        string environmentId, SelfPlayOptions options, IPolicyValueNetBuilder? netBuilder = null, string gpus = "all")
        => services.AddSingleton<ITrainingCampaign>(sp =>
        {
            var game = sp.GetRequiredService<IZeroSumGame<TState>>();
            var adaptive = sp.GetService<AdaptiveBackend>(); // null unless the host called AddGpuBackend()
            var selected = SelectGpus(adaptive?.Gpus, gpus);
            // GPU-resident conv forwards for batched self-play — ONE per selected GPU (M43/M45), so generation
            // shards across devices; else a single autograd forward. All Ilgpu knowledge stays here.
            Func<IPolicyValueNet, IReadOnlyList<IPolicyValueForward>>? forwardFactory = adaptive is null ? null
                : net => selected.Count > 0 && net is ConvResidualPolicyValueNet conv
                    ? [.. selected.Select(g => (IPolicyValueForward)g.CreateResidentForward(conv))]
                    : [new AutogradPolicyValueForward(net, game.ObservationSize)];
            // GPU-resident conv TRAINING step (M44) on the primary selected GPU; else null → the campaign's
            // autograd default. Training is not sharded (generation is the bottleneck); it runs on selected[0]
            // and the trained weights fan out to all forwards.
            Func<IPolicyValueNet, Adam, IPolicyValueTrainStep>? trainStepFactory =
                (selected.Count == 0 || netBuilder is not ConvNetBuilder) ? null
                : (net, adam) => selected[0].CreateResidentTrainer(
                    (ConvResidualPolicyValueNet)net, options.BatchSize, options.LearningRate,
                    options.GradClipNorm, game.PolicySize, options.ValueWeight);
            return new SelfPlayCampaign<TState>(game, environmentId, options,
                netBuilder: netBuilder, backend: adaptive,
                forwardFactory: forwardFactory, trainStepFactory: trainStepFactory);
        });

    /// <summary>Score-maximizing Snake DQN: small training grid, deployed-size eval grid (PLAN M22).</summary>
    public static IServiceCollection AddSnakeDqnCampaign(this IServiceCollection services,
        SnakeEnv trainEnv, SnakeEnv evalEnv, DqnScoreOptions options)
        // Named args: the campaign's ctor is [Inject]-generated (own deps first, then the base's), so the two
        // same-typed envs must never be passed positionally.
        => services.AddSingleton<ITrainingCampaign>(_ =>
            new SnakeDqnCampaign(evalEnv: evalEnv, trainEnv: trainEnv, options: options, logger: null));

    /// <summary>Score-maximizing FruitCake DQN; the training env carries any reward shaping, the eval env
    /// stays a plain game (see <see cref="FruitCakeDqnCampaign"/>).</summary>
    public static IServiceCollection AddFruitCakeDqnCampaign(this IServiceCollection services,
        FruitCakeEnv trainEnv, FruitCakeEnv evalEnv, FruitCakeDqnOptions options)
        => services.AddSingleton<ITrainingCampaign>(_ => new FruitCakeDqnCampaign(trainEnv, evalEnv, options));

    /// <summary>Kociemba-imitation cube campaign (PLAN M16).</summary>
    public static IServiceCollection AddCubeImitationCampaign(this IServiceCollection services, CubeImitationOptions options)
        => services.AddSingleton<ITrainingCampaign>(_ => new CubeImitationCampaign(options));

    /// <summary>Teacher-free EfficientCube campaign; requires the GPU backend registration (CPU-degrading).</summary>
    public static IServiceCollection AddCubeEfficientCampaign(this IServiceCollection services, CubeEfficientOptions options)
        => services.AddSingleton<ITrainingCampaign>(sp =>
            new CubeEfficientCampaign(sp.GetRequiredService<AdaptiveBackend>(), options));

    /// <summary>Teacher-free DAVI value-iteration cube campaign; requires the GPU backend registration.</summary>
    public static IServiceCollection AddCubeDaviCampaign(this IServiceCollection services, CubeDaviSettings settings)
        => services.AddSingleton<ITrainingCampaign>(sp =>
            new CubeDaviCampaign(sp.GetRequiredService<AdaptiveBackend>(), settings));

    /// <summary>BFS-oracle Rush Hour imitation campaign (PLAN M16).</summary>
    public static IServiceCollection AddRushHourImitationCampaign(this IServiceCollection services, RushHourImitationOptions options)
        => services.AddSingleton<ITrainingCampaign>(_ => new RushHourImitationCampaign(options));

    // Resolve a --gpus-style spec (M45): pick which detected GPUs to shard generation across. "all" (default) →
    // every detected GPU; an integer → the first N; explicit ordinals ("0,2") → those devices. Empty when no GPU
    // is available. A spec that matches nothing falls back to all, so a typo never silently drops to CPU.
    private static IReadOnlyList<IlgpuBackend> SelectGpus(IReadOnlyList<IlgpuBackend>? all, string spec)
    {
        if (all is null || all.Count == 0) return [];
        if (spec.Equals("all", StringComparison.OrdinalIgnoreCase)) return all;
        if (int.TryParse(spec, out int n)) return [.. all.Take(Math.Clamp(n, 1, all.Count))];
        var picked = new List<IlgpuBackend>();
        foreach (var part in spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            if (int.TryParse(part, out int idx) && idx >= 0 && idx < all.Count) picked.Add(all[idx]);
        return picked.Count > 0 ? picked : all;
    }
}
