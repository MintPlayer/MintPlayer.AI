using System.Text.RegularExpressions;
using MintPlayer.AspNetCore.SpaServices.Extensions;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;
using RLDemo.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
string dataDirectory = builder.Configuration["DataDirectory"] ?? "data";

// Seed an empty model store from the shipped pre-trained checkpoints (models/ in the
// repo, /app/models in the Docker image): fresh clones and fresh volumes start with
// the trained AI instead of training from scratch. Existing files are never touched.
string? seedDirectory = builder.Configuration["SeedModelsDirectory"];
if (!string.IsNullOrEmpty(seedDirectory) && Directory.Exists(seedDirectory))
{
    Directory.CreateDirectory(dataDirectory);
    foreach (string seed in Directory.EnumerateFiles(seedDirectory, "*.ckpt"))
    {
        string target = Path.Combine(dataDirectory, Path.GetFileName(seed));
        if (!File.Exists(target))
            File.Copy(seed, target);
    }
}
// The shared RL runtime DI — the same surface the AIHost training tools use: registers the FileModelStore
// over dataDirectory, the system TimeProvider, and the CampaignRunner. The web consumes the model store
// (its model services inject IModelStore); the runner is available but unused here.
builder.Services.AddReinforcementLearning(dataDirectory);
builder.Services.AddSingleton(new GalleryStore(Path.Combine(dataDirectory, "gallery")));
// The curated Rush Hour level deck: committed canonical content under wwwroot (ships with the app);
// served read-only everywhere, authored via the Development-only deck endpoints.
builder.Services.AddSingleton(sp => new RushHourDeckStore(
    Path.Combine(sp.GetRequiredService<IWebHostEnvironment>().ContentRootPath, "wwwroot", "rushhour-deck.json")));
// One process-wide compute backend (AddGpuBackend, Ilgpu.Hosting). It selects a discrete CUDA GPU when
// present (local dev) and falls back to the multithreaded CPU otherwise (e.g. a GPU-less Hetzner
// container) — so the self-taught cube solver gets a resident GPU forward where available, CPU elsewhere.
builder.Services.AddGpuBackend();
builder.Services.AddSingleton<RushHourModelService>();
builder.Services.AddSingleton<Game2048ModelService>();
builder.Services.AddSingleton<CubeModelService>();
builder.Services.AddSingleton<SnakeModelService>();
builder.Services.AddSingleton<MountainCarModelService>();
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<RushHourModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<Game2048ModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<CubeModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<SnakeModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<MountainCarModelService>());
builder.Services.AddSingleton<IModelStartupService, CubeSolverWarmupService>();

// Integration tests control the model store themselves and host no SPA.
bool isTesting = builder.Environment.IsEnvironment("Testing");
if (!isTesting)
    builder.Services.AddHostedService<ModelStartupHostedService>();
bool hostSpa = !isTesting;
if (hostSpa)
{
    builder.Services.AddSpaStaticFilesImproved(configuration =>
    {
        configuration.RootPath = "ClientApp/dist/ClientApp/browser";
    });
}

var app = builder.Build();

app.UseStaticFiles();
app.UseRouting();
app.UseWebSockets(); // principle B (PRD §7.1): the live "watch the AI play" streams (e.g. Snake, MountainCar)
app.MapControllers();

if (hostSpa)
{
    app.UseSpaStaticFilesImproved();
    app.MapWhen(
        context => !context.Request.Path.StartsWithSegments("/api"),
        appBuilder =>
        {
            appBuilder.UseSpaImproved(spa =>
            {
                spa.Options.SourcePath = "ClientApp";

                if (app.Environment.IsDevelopment())
                {
                    // The host spawns and proxies `npm start` itself — never run ng serve separately.
                    spa.UseAngularCliServer(npmScript: "start",
                        cliRegexes: [new Regex(@"Local\:\s+(?<openbrowser>https?\:\/\/(.+))")]);
                }
            });
        });
}

app.Run();

/// <summary>Exposed for WebApplicationFactory-based integration tests.</summary>
public partial class Program;
