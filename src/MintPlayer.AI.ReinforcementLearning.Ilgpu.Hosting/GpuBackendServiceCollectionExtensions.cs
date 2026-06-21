using Microsoft.Extensions.DependencyInjection;

namespace MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

/// <summary>
/// Dependency-injection registration for the ILGPU GPU compute backend — the optional GPU companion to the
/// Core runtime's <c>AddReinforcementLearning</c> (AIHost). Lives in its own package so the lean
/// <see cref="MintPlayer.AI.ReinforcementLearning.Ilgpu"/> compute backend carries no DI dependency.
/// </summary>
public static class GpuBackendServiceCollectionExtensions
{
    /// <summary>
    /// Register the <see cref="AdaptiveBackend"/> (CPU + CUDA-by-GEMM-size routing; pure CPU when no GPU is
    /// present) as a singleton. A campaign that wants GPU training resolves it and sets it as the current
    /// compute backend; the container owns its lifetime, so it is disposed when the host is. Adding this is the
    /// whole opt-in: campaigns whose nets are too small to beat the CPU simply never call it.
    /// </summary>
    public static IServiceCollection AddGpuBackend(this IServiceCollection services)
    {
        services.AddSingleton<AdaptiveBackend>();
        return services;
    }
}
