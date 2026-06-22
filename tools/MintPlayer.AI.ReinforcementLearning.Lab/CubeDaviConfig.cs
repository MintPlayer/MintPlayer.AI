using System.Text.Json;
using System.Text.Json.Serialization;

/// <summary>
/// Persistent configuration for the cube-davi campaign, loaded from <c>appsettings.json</c> (the
/// <c>"cube-davi"</c> section). Every field is nullable: a missing key leaves the in-code default in
/// place. Precedence is <b>code defaults → appsettings.json → CLI flags</b>, so the file holds the
/// long-lived campaign config (resume a multi-day run with just <c>--game cube-davi</c>) while a CLI
/// flag still wins for a one-off override. JSON keys mirror the CLI flag names (minus the <c>--</c>).
/// </summary>
internal sealed class CubeDaviConfig
{
    /// <summary>Wall-clock budget per chunk, in hours. Re-run the command to resume the next chunk (CLI <c>--hours</c>).</summary>
    [JsonPropertyName("hours")] public double? Hours { get; init; }

    /// <summary>Hard stop after this many total states processed; resumable across sessions (0 = time-bounded only). CLI <c>--samples</c>.</summary>
    [JsonPropertyName("samples")] public long? Samples { get; init; }

    /// <summary>Scramble depths for the in-loop BWAS capability probe (logged to <c>logs/cube-davi-res-cap.csv</c>). CLI <c>--probe-depths</c>.</summary>
    [JsonPropertyName("probe-depths")] public int[]? ProbeDepths { get; init; }

    /// <summary>Model-store directory holding the value-net checkpoint and training-state files. CLI <c>--data</c>.</summary>
    [JsonPropertyName("data")] public string? Data { get; init; }

    /// <summary>RNG seed for fresh net init and the scramble sampler (restored from the checkpoint on resume). CLI <c>--seed</c>.</summary>
    [JsonPropertyName("seed")] public ulong? Seed { get; init; }

    /// <summary>Net architecture: <c>"residual"</c> (deep residual value net) or <c>"mlp"</c> (plain). CLI <c>--net</c>.</summary>
    [JsonPropertyName("net")] public string? Net { get; init; }

    /// <summary>Hidden / residual-trunk width (the start width; capacity is not the d≤14 bottleneck, so wider is slower for the same reach). CLI <c>--width</c>.</summary>
    [JsonPropertyName("width")] public int? Width { get; init; }

    /// <summary>Hidden-layer count for the plain <c>mlp</c> net; ignored by the residual net (which uses <see cref="Blocks"/>). CLI <c>--layers</c>.</summary>
    [JsonPropertyName("layers")] public int? Layers { get; init; }

    /// <summary>Residual block count (network depth) for the <c>residual</c> net. CLI <c>--blocks</c>.</summary>
    [JsonPropertyName("blocks")] public int? Blocks { get; init; }

    /// <summary>DAVI training batch size. Keep this fixed across resumes so the <see cref="Samples"/> state count stays exact. CLI <c>--batch</c>.</summary>
    [JsonPropertyName("batch")] public int? Batch { get; init; }

    /// <summary>Optimizer learning rate. CLI <c>--lr</c>.</summary>
    [JsonPropertyName("lr")] public float? Lr { get; init; }

    /// <summary>ε-loss target-sync threshold: advance the bootstrap target only once batch loss falls below this (0 = sync every interval). CLI <c>--eps-sync</c>.</summary>
    [JsonPropertyName("eps-sync")] public float? EpsSync { get; init; }

    /// <summary>Steps between bootstrap-target syncs (gated by <see cref="EpsSync"/>). CLI <c>--target-sync-interval</c>.</summary>
    [JsonPropertyName("target-sync-interval")] public int? TargetSyncInterval { get; init; }

    /// <summary>Adam β₂ (second-moment decay). CLI <c>--beta2</c>.</summary>
    [JsonPropertyName("beta2")] public float? Beta2 { get; init; }

    /// <summary>Sample scramble depth toward the curriculum frontier (triangular) instead of uniform <c>[1, depth]</c>. CLI <c>--frontier-bias</c>.</summary>
    [JsonPropertyName("frontier-bias")] public bool? FrontierBias { get; init; }

    /// <summary>Scheduled (timer-based) Net2WiderNet target width; the trunk widens to this once <see cref="GrowAt"/> samples are reached (0 = never). CLI <c>--grow-to</c>.</summary>
    [JsonPropertyName("grow-to")] public int? GrowTo { get; init; }

    /// <summary>Total-sample count at which the scheduled <see cref="GrowTo"/> widen fires. CLI <c>--grow-at</c>.</summary>
    [JsonPropertyName("grow-at")] public long? GrowAt { get; init; }

    /// <summary>On resume, override the restored curriculum depth — pin to the cap for uniform sampling, or re-pin to the accuracy frontier. CLI <c>--set-curriculum-depth</c>.</summary>
    [JsonPropertyName("set-curriculum-depth")] public int? SetCurriculumDepth { get; init; }

    /// <summary>Curriculum value-accuracy gate: advance d→d+1 only when mean <c>V(d)/d ≥</c> this. CLI <c>--advance-ratio</c>.</summary>
    [JsonPropertyName("advance-ratio")] public double? AdvanceRatio { get; init; }

    /// <summary>Auto-widen the trunk on a frontier loss plateau (capacity-bound, not under-trained). CLI <c>--auto-widen</c>.</summary>
    [JsonPropertyName("auto-widen")] public bool? AutoWiden { get; init; }

    /// <summary>Upper bound the trunk width for <see cref="AutoWiden"/>. CLI <c>--max-width</c>.</summary>
    [JsonPropertyName("max-width")] public int? MaxWidth { get; init; }

    /// <summary>Loss-plateau window (samples with no improvement) before an <see cref="AutoWiden"/> fires. CLI <c>--widen-stall-samples</c>.</summary>
    [JsonPropertyName("widen-stall-samples")] public long? WidenStallSamples { get; init; }

    /// <summary>Curriculum / scramble-depth cap (god's number is 26 in the quarter-turn metric). CLI <c>--max-depth</c>.</summary>
    [JsonPropertyName("max-depth")] public int? MaxDepth { get; init; }

    // ── eval-only knobs: the file can also drive a benchmark/eval invocation instead of training ──

    /// <summary>Run an evaluation pass instead of training. CLI <c>--eval-only</c>.</summary>
    [JsonPropertyName("eval-only")] public bool? EvalOnly { get; init; }

    /// <summary>Evaluate via value-guided A* (else greedy descent). CLI <c>--search</c>.</summary>
    [JsonPropertyName("search")] public bool? Search { get; init; }

    /// <summary>Use the batched A* (BWAS) variant for the search eval. CLI <c>--batched</c>.</summary>
    [JsonPropertyName("batched")] public bool? Batched { get; init; }

    /// <summary>Also report Kociemba's QTM solution length per depth as the Tier-2 baseline. CLI <c>--vs-kociemba</c>.</summary>
    [JsonPropertyName("vs-kociemba")] public bool? VsKociemba { get; init; }

    /// <summary>A* search weight in <c>f = g + weight·h</c>; &gt; 1 reaches deeper for possibly non-optimal solutions. CLI <c>--weight</c>.</summary>
    [JsonPropertyName("weight")] public float? Weight { get; init; }

    /// <summary>A* node-expansion budget per solve. CLI <c>--max-exp</c>.</summary>
    [JsonPropertyName("max-exp")] public int? MaxExpansions { get; init; }

    /// <summary>Eval-only wall-clock budget (seconds) for the deployed time-bounded solver benchmark (0 = off). CLI <c>--time-budget</c>.</summary>
    [JsonPropertyName("time-budget")] public double? TimeBudget { get; init; }

    /// <summary>Eval-only heuristic calibration: report mean predicted <c>V(start)</c> vs scramble depth. CLI <c>--value-curve</c>.</summary>
    [JsonPropertyName("value-curve")] public bool? ValueCurve { get; init; }

    /// <summary>Cubes per depth in eval-only passes (fewer = faster deep probes). CLI <c>--episodes</c>.</summary>
    [JsonPropertyName("episodes")] public int? Episodes { get; init; }

    /// <summary>
    /// Read the <c>"cube-davi"</c> section from the first <c>appsettings.json</c> found in the current
    /// working directory then the app's base directory (the build output, where the project copy lands).
    /// Returns an empty config (all nulls) when no file/section is present. <paramref name="source"/> is
    /// the path it loaded from, or null. Tolerates comments and trailing commas so the file can document
    /// itself.
    /// </summary>
    public static CubeDaviConfig Load(out string? source)
    {
        source = null;
        var docOpts = new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };
        var serOpts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true, ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };
        foreach (var dir in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var path = Path.Combine(dir, "appsettings.json");
            if (!File.Exists(path)) continue;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path), docOpts);
                if (!doc.RootElement.TryGetProperty("cube-davi", out var section)) continue;
                source = path;
                return section.Deserialize<CubeDaviConfig>(serOpts) ?? new CubeDaviConfig();
            }
            catch (JsonException ex)
            {
                // A malformed file shouldn't crash the run — warn and fall back to defaults/CLI.
                Console.WriteLine($"{DateTime.UtcNow:HH:mm:ss} appsettings.json at {path} is invalid ({ex.Message}); ignoring.");
                return new CubeDaviConfig();
            }
        }
        return new CubeDaviConfig();
    }
}
