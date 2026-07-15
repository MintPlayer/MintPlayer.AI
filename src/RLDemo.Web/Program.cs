using System.Text.RegularExpressions;
using Microsoft.AspNetCore.StaticFiles;
using MintPlayer.AspNetCore.SpaServices.Extensions;
using MintPlayer.AI.ReinforcementLearning.Hosting;
using MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting;
using RLDemo.Web; // generated AddRLDemoWebModelServices()
using RLDemo.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
string dataDirectory = builder.Configuration["DataDirectory"] ?? "data";

// Seed the model store from the shipped pre-trained checkpoints (models/ in the repo, /app/models in the Docker
// image). The web is load-only (PRD §14 — it never trains), so these shipped checkpoints are the single source of
// truth: REFRESH them on every startup rather than skipping existing files. Skipping was the old behaviour (to
// preserve on-the-fly-trained nets), but it meant a model update never reached a persistent /data volume — and a
// shape change (e.g. the Snake observation rework) then loaded a stale, incompatible net and crashed the game.
// Only same-named shipped .ckpt files are overwritten; the gallery (data/gallery) and any other data are untouched.
string? seedDirectory = builder.Configuration["SeedModelsDirectory"];
if (!string.IsNullOrEmpty(seedDirectory) && Directory.Exists(seedDirectory))
{
    Directory.CreateDirectory(dataDirectory);
    foreach (string seed in Directory.EnumerateFiles(seedDirectory, "*.ckpt"))
        File.Copy(seed, Path.Combine(dataDirectory, Path.GetFileName(seed)), overwrite: true);
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
// The model services register themselves via [Register] (MintPlayer.SourceGenerators, PLAN M46.3) — this
// generated call adds RushHourModelService / Game2048ModelService / CubeModelService as singletons plus the
// CubeSolverWarmupService as an IModelStartupService.
builder.Services.AddRLDemoWebModelServices();
// FruitCake has no server model service: its AI (physics + net + depth-3 search) runs entirely in the browser
// from the single-source Polyglot core + the shipped ClientApp/public/fruitcake-net.ckpt (M32). Zero server inference.
// The startup-loader forwardings must reuse the SAME singleton instances the controllers inject (the hosted
// service loads checkpoints INTO them), so they stay hand-written — [Register(typeof(IModelStartupService))]
// would register separate instances.
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<RushHourModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<Game2048ModelService>());
builder.Services.AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<CubeModelService>());

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

// Serve wwwroot, teaching the middleware the `.ckpt` type: the browser-side AI weights (wwwroot/*.ckpt) are an
// unknown extension, which ASP.NET otherwise refuses to serve → the request falls through to the SPA index, the
// client's checkpoint parser rejects the HTML, and the in-browser AI silently gets no net (the "Snake doesn't
// move" bug). Same middleware in dev and prod, so this is verifiable locally.
var contentTypes = new FileExtensionContentTypeProvider();
contentTypes.Mappings[".ckpt"] = "application/octet-stream";
app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = contentTypes });
app.UseRouting();
// No server WebSockets: every game's "watch AI" now runs client-side (M32/M33). The remaining controllers are
// request/response REST (2048/RushHour/Cube solvers + /api/version).
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
