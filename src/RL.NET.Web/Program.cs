using System.Text.RegularExpressions;
using MintPlayer.AspNetCore.SpaServices.Extensions;
using RLNet.Core.Checkpoints;
using RLNet.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
string dataDirectory = builder.Configuration["DataDirectory"] ?? "data";
builder.Services.AddSingleton<IModelStore>(_ => new FileModelStore(dataDirectory));
builder.Services.AddSingleton(new GalleryStore(Path.Combine(dataDirectory, "gallery")));
builder.Services.AddSingleton<RushHourModelService>();
builder.Services.AddSingleton<Game2048ModelService>();
builder.Services.AddSingleton<ITrainableModelService>(sp => sp.GetRequiredService<RushHourModelService>());
builder.Services.AddSingleton<ITrainableModelService>(sp => sp.GetRequiredService<Game2048ModelService>());

// Integration tests control the model store themselves and host no SPA.
bool isTesting = builder.Environment.IsEnvironment("Testing");
if (!isTesting)
    builder.Services.AddHostedService<ModelTrainingHostedService>();
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
