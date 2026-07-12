using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

/// <summary>
/// The bootstrap tail every game Lab runs: build the <see cref="AIHost"/> DI container (optionally with the GPU
/// backend), resolve the model store + <see cref="CampaignRunner"/>, let the caller construct its campaign from the
/// resolved services, attach the live <c>--viz</c> viewer, run the campaign under the given options, and keep the
/// process alive afterward so the final net stays inspectable. A Lab is then just "parse flags → build campaign";
/// the DI wiring, GPU opt-in, and viewer lifetime live here once instead of copy-pasted per game.
/// </summary>
internal static class LabHost
{
    /// <param name="build">Constructs the campaign from the host's services — GPU labs pull the
    /// <c>AdaptiveBackend</c> from it; CPU labs ignore the argument.</param>
    /// <param name="onEval">The runner's per-eval IO hook (console+CSV for most games, console-only for the ones
    /// that own their own CSVs) — see <see cref="CampaignCli"/>.</param>
    public static void Run(string[] args, string dataDir, double hours, bool evalOnly, bool useGpu,
        Func<IServiceProvider, ITrainingCampaign> build, Action<CampaignProgress> onEval)
    {
        // DI all the way: the model store, clock, (optional) GPU backend and CampaignRunner come from the container.
        var builder = AIHost.CreateBuilder(dataDir);
        if (useGpu) builder.Services.AddGpuBackend();
        using var host = builder.Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();

        var campaign = build(host.Services);
        using var viz = VizLauncher.TryStart(args, campaign, host.Services.GetRequiredService<IHostEnvironment>());
        runner.Run(campaign, store, new CampaignOptions
        {
            Duration = TimeSpan.FromHours(hours),
            EvalOnly = evalOnly,
            OnEval = onEval,
        });
        WaitForViewer(viz);
    }

    /// <summary>Keep the process (and its live viewer) alive after training so the final net can be inspected.</summary>
    private static void WaitForViewer(VizServer? viz)
    {
        if (viz is null || Console.IsInputRedirected) return;
        Console.WriteLine($"training finished — viewer still live at {viz.Url}; press Enter to exit.");
        Console.ReadLine();
    }
}
