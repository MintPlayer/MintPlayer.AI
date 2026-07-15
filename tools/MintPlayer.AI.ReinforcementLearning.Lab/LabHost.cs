using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments; // generated AddReinforcementLearningGames()
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;

/// <summary>
/// The bootstrap tail every game Lab runs: build the <see cref="AIHost"/> DI container (optionally with the GPU
/// backend, always with the [Register]-generated game registrations), let the caller register its campaign
/// (PLAN M46.3 — an <c>Add&lt;Game&gt;Campaign(options)</c> extension from the Campaigns library), resolve the
/// campaign + model store + <see cref="CampaignRunner"/> from the container, attach the live <c>--viz</c> viewer,
/// run the campaign under the given options, and keep the process alive afterward so the final net stays
/// inspectable. A Lab is then just "parse flags → register campaign"; the DI wiring, GPU opt-in, and viewer
/// lifetime live here once instead of copy-pasted per game.
/// </summary>
internal static class LabHost
{
    /// <param name="configureCampaign">Registers the run's <see cref="ITrainingCampaign"/> (plus anything else it
    /// needs) into the container — typically one <c>Add&lt;Game&gt;Campaign(options)</c> call.</param>
    /// <param name="onEval">The runner's per-eval IO hook (console+CSV for most games, console-only for the ones
    /// that own their own CSVs) — see <see cref="CampaignCli"/>.</param>
    public static void Run(string[] args, string dataDir, double hours, bool evalOnly, bool useGpu,
        Action<IServiceCollection> configureCampaign, Action<CampaignProgress> onEval,
        double? firstEvalMinutes = null, double? evalEveryMinutes = null)
    {
        // DI all the way: the model store, clock, (optional) GPU backend, CampaignRunner, the games and the
        // campaign itself all come from the container.
        var builder = AIHost.CreateBuilder(dataDir);
        if (useGpu) builder.Services.AddGpuBackend();
        builder.Services.AddReinforcementLearningGames(); // generated: IZeroSumGame implementations as singletons
        configureCampaign(builder.Services);
        using var host = builder.Build();
        var store = host.Services.GetRequiredService<IModelStore>();
        var runner = host.Services.GetRequiredService<CampaignRunner>();

        var campaign = host.Services.GetRequiredService<ITrainingCampaign>();
        using var viz = VizLauncher.TryStart(args, campaign, host.Services.GetRequiredService<IHostEnvironment>());
        var opts = new CampaignOptions
        {
            Duration = TimeSpan.FromHours(hours),
            EvalOnly = evalOnly,
            OnEval = onEval,
        };
        if (firstEvalMinutes is double fe) opts = opts with { FirstEvalAfter = TimeSpan.FromMinutes(fe) };
        if (evalEveryMinutes is double ee) opts = opts with { EvalEvery = TimeSpan.FromMinutes(ee) };
        runner.Run(campaign, store, opts);
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
