using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace RLDemo.Web.Services;

/// <summary>
/// Not a trained model, but the same startup need: the Kociemba solver builds its
/// lookup/pruning tables in memory on first use (2–5 s), so build them once at startup
/// via the shared <see cref="ModelStartupHostedService"/> instead of on the first
/// user's solve request.
/// </summary>
public sealed class CubeSolverWarmupService(ILogger<CubeSolverWarmupService> logger) : IModelStartupService
{
    public void Initialize(CancellationToken cancellationToken)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        CubeSolver.WarmUp();
        logger.LogInformation("Kociemba tables built in {Elapsed:F1} s.", sw.Elapsed.TotalSeconds);
    }
}
