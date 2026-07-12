using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Core.Telemetry;

/// <summary>
/// The read-only telemetry seam a training campaign exposes so an out-of-process viewer can watch its network
/// evolve as it learns. It is a <b>pull</b> model: a campaign publishes its live parameter tensors plus the run's
/// scalar metrics, and a viewer samples them on its own cadence. Working from the parameter tensors — not a
/// concrete net type — means every architecture is covered uniformly (the DQN <c>DuelingQNet</c>/<c>Mlp</c>, the
/// two-headed policy nets, the DAVI <c>ResidualMlp</c>), and no trainer needs to know telemetry exists.
/// <para>
/// Sampling is a pure observation of parameter <see cref="Tensor.Data"/>: it never mutates the net, the RNG, or
/// the optimizer, so a run being watched trains identically to one that isn't. On the CPU path the arrays are
/// host-resident, so a read is free; it races the training thread's writes, but a torn value only affects one
/// pixel of a magnitude heatmap — cosmetically irrelevant, and never a fault.
/// </para>
/// </summary>
public interface INetworkTelemetrySource
{
    /// <summary>A free-form label for the network kind (e.g. "dueling-q", "cube-policy", "value-davi-res").</summary>
    string NetKind { get; }

    /// <summary>The net's current parameter tensors (weights + biases, in forward order), or null until the net
    /// exists (training hasn't started / resumed yet). Returns a snapshot list so the caller can iterate safely.</summary>
    IReadOnlyList<Tensor>? SnapshotParameters();

    /// <summary>The run's scalar metrics at this instant (step, loss, eval, …). Cheap; called once per frame.</summary>
    NetworkMetrics Sample();

    // ── Optional semantics (default: none → the viewer shows generic tooltips). An environment that knows what its
    // observation features and actions MEAN can override these so a viewer can say, per neuron, what it represents
    // and — via SampleIo — its current value. Static labels live on the topology; live values on each frame. ──

    /// <summary>Human-readable name for each INPUT neuron (length = input width), or null if unknown.</summary>
    IReadOnlyList<string>? InputLabels => null;

    /// <summary>Human-readable name for each OUTPUT neuron / action (length = output width), or null if unknown.</summary>
    IReadOnlyList<string>? OutputLabels => null;

    /// <summary>The net's current input vector and the output it produces for it (e.g. per-action Q-values), so a
    /// viewer can show "what this input is right now" and "what each output computes". Null when unavailable.</summary>
    (float[] Input, float[] Output)? SampleIo() => null;

    /// <summary>Each layer's current activation vector (post-activation output), in the same order as the layers
    /// <see cref="NetworkInspector"/> recovers — so a viewer can show every hidden neuron's live value, not just the
    /// input/output. Null when the net can't (cheaply) expose its intermediate activations.</summary>
    float[][]? SampleActivations() => null;
}

/// <summary>The run's scalar read-outs at a moment in training. Any field may be <see cref="double.NaN"/> when a
/// metric doesn't apply to this algorithm (e.g. ε for a supervised campaign) — the viewer renders NaN as "—".</summary>
public readonly record struct NetworkMetrics(long Step, long MaxSteps, double Loss, double Eval, double Epsilon);

/// <summary>One fully-connected layer's shape. <paramref name="OutputSize"/> neurons fed by an
/// [<paramref name="InputSize"/>×<paramref name="OutputSize"/>] weight matrix; <paramref name="Role"/> is a
/// display hint (<c>"hidden"</c> for trunk layers, <c>"output"</c> for the final head).</summary>
public sealed record LayerInfo(int Index, int InputSize, int OutputSize, string Role);

/// <summary>A network's structure: the input width and the ordered fully-connected layers recovered from the
/// parameter tensors. Stable for the whole run — the viewer lays the graph out from this once.
/// <paramref name="InputLabels"/>/<paramref name="OutputLabels"/> name each input/output neuron when the
/// environment supplies them (null otherwise).</summary>
public sealed record NetworkTopology(
    string NetKind, int InputSize, int OutputSize, IReadOnlyList<LayerInfo> Layers,
    IReadOnlyList<string>? InputLabels = null, IReadOnlyList<string>? OutputLabels = null);

/// <summary>
/// One layer's weights at a moment in training: summary stats plus a <b>downsampled magnitude heatmap</b>
/// (<see cref="Heat"/> is <see cref="HRows"/>×<see cref="HCols"/> block-averaged |weight|, row-major) so a frame
/// stays small regardless of the true matrix size (a 1024×1024 layer streams the same handful of KB as a 16×16).
/// </summary>
public sealed record LayerFrame(
    int Index, int Rows, int Cols,
    float WMin, float WMax, float WMeanAbs, float WL2,
    float BiasMeanAbs,
    int HRows, int HCols, float[] Heat);

/// <summary>A whole-network snapshot: the training scalars at this step plus every layer's
/// <see cref="LayerFrame"/>, and (when the source supplies them) the net's current input vector and the output it
/// produces — so a viewer can show each input/output neuron's live value.</summary>
public sealed record NetworkFrame(
    long Step, long MaxSteps,
    double Loss, double Eval, double Epsilon,
    IReadOnlyList<LayerFrame> Layers,
    float[]? InputValues = null, float[]? OutputValues = null,
    float[][]? Activations = null);

/// <summary>
/// Turns a net's parameter tensors into telemetry. It pairs each rank-2 weight tensor with the rank-1 bias that
/// follows it into a fully-connected layer — the layout every net in this library emits — so the viewer needs no
/// per-architecture knowledge. Weight matrices are row-major [in, out] (<c>Data[r*Cols + c]</c>), matching
/// <see cref="Nn.Linear"/>. (Parallel heads — a dueling net's value/advantage, a policy net's policy/value — are
/// rendered as successive columns; a benign cosmetic simplification, since they share the trunk output.)
/// </summary>
public static class NetworkInspector
{
    /// <summary>The (weight, bias) layers among <paramref name="parameters"/>, in forward order.</summary>
    public static IReadOnlyList<(Tensor Weight, Tensor Bias)> Layers(IReadOnlyList<Tensor> parameters)
    {
        var layers = new List<(Tensor, Tensor)>();
        Tensor? pendingWeight = null;
        foreach (var p in parameters)
        {
            if (p.Rank == 2) pendingWeight = p;
            else if (p.Rank == 1 && pendingWeight is not null) { layers.Add((pendingWeight, p)); pendingWeight = null; }
        }
        return layers;
    }

    /// <summary>The net's fixed structure. <paramref name="netKind"/> is a free-form label (e.g. "dueling-q");
    /// <paramref name="inputLabels"/>/<paramref name="outputLabels"/> name the input/output neurons when known.</summary>
    public static NetworkTopology Describe(IReadOnlyList<Tensor> parameters, string netKind,
        IReadOnlyList<string>? inputLabels = null, IReadOnlyList<string>? outputLabels = null)
    {
        var layers = Layers(parameters);
        var infos = new LayerInfo[layers.Count];
        for (int i = 0; i < layers.Count; i++)
        {
            var w = layers[i].Weight;
            infos[i] = new LayerInfo(i, w.Rows, w.Cols, i == layers.Count - 1 ? "output" : "hidden");
        }
        int inputSize = layers.Count > 0 ? layers[0].Weight.Rows : 0;
        int outputSize = layers.Count > 0 ? layers[^1].Weight.Cols : 0;
        // Pass labels through as-is: the viewer attaches input labels to the input column and output labels to
        // whichever column matches their count. (A multi-head net's action head isn't the final column — a scalar
        // value head follows it — so we deliberately do NOT force output labels to match the last layer here.)
        return new NetworkTopology(netKind, inputSize, outputSize, infos, inputLabels, outputLabels);
    }

    /// <summary>
    /// Captures the current weights as a <see cref="NetworkFrame"/> carrying the supplied <paramref name="metrics"/>.
    /// Each weight matrix is block-averaged (of |value|) down to at most <paramref name="maxHeat"/>² cells so the
    /// frame size is bounded no matter how wide the layer is.
    /// </summary>
    public static NetworkFrame CaptureFrame(IReadOnlyList<Tensor> parameters, NetworkMetrics metrics,
        float[]? inputValues = null, float[]? outputValues = null, float[][]? activations = null, int maxHeat = 24)
    {
        var layers = Layers(parameters);
        var frames = new LayerFrame[layers.Count];
        for (int i = 0; i < layers.Count; i++)
        {
            var (weight, bias) = layers[i];
            int rows = weight.Rows, cols = weight.Cols;
            float[] w = weight.Data;

            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            double sumAbs = 0, sumSq = 0;
            for (int k = 0; k < w.Length; k++)
            {
                float v = w[k];
                if (v < min) min = v;
                if (v > max) max = v;
                sumAbs += Math.Abs(v);
                sumSq += (double)v * v;
            }
            float meanAbs = w.Length == 0 ? 0 : (float)(sumAbs / w.Length);
            float l2 = (float)Math.Sqrt(sumSq);

            double biasSumAbs = 0;
            foreach (float b in bias.Data) biasSumAbs += Math.Abs(b);
            float biasMeanAbs = bias.Length == 0 ? 0 : (float)(biasSumAbs / bias.Length);

            var (heat, hRows, hCols) = Downsample(w, rows, cols, maxHeat);
            frames[i] = new LayerFrame(i, rows, cols, min, max, meanAbs, l2, biasMeanAbs, hRows, hCols, heat);
        }
        return new NetworkFrame(metrics.Step, metrics.MaxSteps, metrics.Loss, metrics.Eval, metrics.Epsilon, frames,
            inputValues, outputValues, activations);
    }

    /// <summary>Block-mean of |weight| onto an at-most maxHeat² grid (row-major), preserving aspect within the cap.</summary>
    private static (float[] Heat, int HRows, int HCols) Downsample(float[] w, int rows, int cols, int maxHeat)
    {
        int hRows = Math.Min(rows, maxHeat);
        int hCols = Math.Min(cols, maxHeat);
        if (hRows <= 0 || hCols <= 0) return ([], 0, 0);

        var heat = new float[hRows * hCols];
        var counts = new int[hRows * hCols];
        for (int r = 0; r < rows; r++)
        {
            int hr = r * hRows / rows;
            for (int c = 0; c < cols; c++)
            {
                int hc = c * hCols / cols;
                int idx = hr * hCols + hc;
                heat[idx] += Math.Abs(w[r * cols + c]);
                counts[idx]++;
            }
        }
        for (int k = 0; k < heat.Length; k++)
            if (counts[k] > 0) heat[k] /= counts[k];
        return (heat, hRows, hCols);
    }
}
