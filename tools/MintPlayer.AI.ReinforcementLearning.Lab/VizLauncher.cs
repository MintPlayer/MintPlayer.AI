using Microsoft.Extensions.Hosting;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Shared <c>--viz [port]</c> handling for every game Lab: parses the flag, enforces the safety gate, and — if the
/// campaign can report its network — starts the <see cref="VizServer"/> live viewer. Bare <c>--viz</c> uses port
/// 5250. Kept in one place so all games get the identical behaviour and message.
/// <para>
/// The live socket is <b>restricted to a Development host environment</b>: training is a dev-only activity, so the
/// viewer never comes up in a Production-configured process even if <c>--viz</c> is passed. (The Lab defaults to
/// Development — see <c>Program.cs</c> — so it works out of the box; set <c>DOTNET_ENVIRONMENT=Production</c> to
/// disable it.) A campaign that exposes no <see cref="INetworkTelemetrySource"/> is skipped with a note.
/// </para>
/// </summary>
internal static class VizLauncher
{
    public static VizServer? TryStart(string[] args, ITrainingCampaign campaign, IHostEnvironment environment)
    {
        int port = ParsePort(args);
        if (port == 0) return null; // --viz not passed

        if (!environment.IsDevelopment())
        {
            Console.WriteLine($"[viz] --viz ignored: the live network viewer only runs in a Development environment " +
                              $"(current: {environment.EnvironmentName}). Unset DOTNET_ENVIRONMENT to enable it.");
            return null;
        }
        if (campaign is not INetworkTelemetrySource source)
        {
            Console.WriteLine($"[viz] --viz ignored: {campaign.GetType().Name} does not expose network telemetry yet.");
            return null;
        }

        var server = VizServer.Start(port, source);
        Console.WriteLine($"live network viewer: {server.Url}  (open it to watch the net evolve)");
        return server;
    }

    /// <summary>Returns the requested viewer port, 0 when <c>--viz</c> is absent. Bare <c>--viz</c> → 5250.</summary>
    private static int ParsePort(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
            if (args[i] == "--viz")
                return i + 1 < args.Length && int.TryParse(args[i + 1], out int p) ? p : 5250;
        return 0;
    }
}
