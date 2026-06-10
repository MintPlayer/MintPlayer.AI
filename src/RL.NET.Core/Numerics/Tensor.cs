using RLNet.Core.Random;

namespace RLNet.Core.Numerics;

/// <summary>
/// A dense float32 tensor (flat row-major buffer + shape) with tape-based reverse-mode
/// autodiff at tensor granularity: each op records its parents and a backward closure;
/// <see cref="Backward"/> topologically sorts the graph and accumulates gradients.
/// </summary>
public sealed partial class Tensor
{
    private Tensor[]? _parents;
    private Action? _backwardFn;

    public Tensor(float[] data, params int[] shape)
    {
        int expected = 1;
        foreach (int d in shape) expected *= d;
        if (data.Length != expected)
            throw new ArgumentException($"Data length {data.Length} does not match shape ({string.Join('x', shape)}).");
        Data = data;
        Shape = shape;
    }

    public float[] Data { get; }
    public int[] Shape { get; }
    public int Length => Data.Length;
    public int Rank => Shape.Length;

    /// <summary>Rows of a rank-2 tensor (batch dimension by convention).</summary>
    public int Rows => Shape[0];

    /// <summary>Columns of a rank-2 tensor.</summary>
    public int Cols => Shape[^1];

    /// <summary>Set on leaf tensors (parameters) that should receive gradients.</summary>
    public bool RequiresGrad { get; init; }

    public float[]? Grad { get; private set; }

    /// <summary>True if backward needs to propagate into this tensor (parameter or interior node).</summary>
    internal bool NeedsGrad => RequiresGrad || _parents is not null;

    public static Tensor Zeros(params int[] shape)
    {
        int length = 1;
        foreach (int d in shape) length *= d;
        return new Tensor(new float[length], shape);
    }

    public static Tensor Full(float value, params int[] shape)
    {
        var t = Zeros(shape);
        Array.Fill(t.Data, value);
        return t;
    }

    public static Tensor Scalar(float value) => new([value], 1);

    /// <summary>Gaussian-filled tensor (Box–Muller), for weight initialization.</summary>
    public static Tensor RandomNormal(Xoshiro256StarStar rng, float mean, float std, params int[] shape)
    {
        var t = Zeros(shape);
        for (int i = 0; i < t.Data.Length; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            t.Data[i] = mean + std * (float)(radius * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < t.Data.Length)
                t.Data[i + 1] = mean + std * (float)(radius * Math.Sin(2.0 * Math.PI * u2));
        }
        return t;
    }

    public void EnsureGrad() => Grad ??= new float[Length];

    public void ZeroGrad()
    {
        if (Grad is not null) Array.Clear(Grad);
    }

    /// <summary>A new leaf tensor sharing this tensor's buffer but cut off from the graph.</summary>
    public Tensor Detach() => new(Data, Shape);

    /// <summary>
    /// Backpropagates from this scalar through the recorded graph,
    /// accumulating into the <see cref="Grad"/> of every tensor that needs one.
    /// </summary>
    public void Backward()
    {
        if (Length != 1)
            throw new InvalidOperationException("Backward() must be called on a scalar (loss) tensor.");

        var order = new List<Tensor>();
        var visited = new HashSet<Tensor>();
        var stack = new Stack<(Tensor Node, bool Expanded)>();
        stack.Push((this, false));
        while (stack.Count > 0)
        {
            var (node, expanded) = stack.Pop();
            if (expanded)
            {
                order.Add(node);
                continue;
            }
            if (!visited.Add(node)) continue;
            stack.Push((node, true));
            if (node._parents is not null)
                foreach (var parent in node._parents)
                    stack.Push((parent, false));
        }

        EnsureGrad();
        Grad![0] = 1f;
        for (int i = order.Count - 1; i >= 0; i--)
            order[i]._backwardFn?.Invoke();
    }

    /// <summary>
    /// Creates an op result, recording parents and the backward closure only when
    /// gradients are enabled and some parent needs them (so no-grad regions and
    /// constant subgraphs cost nothing).
    /// </summary>
    internal static Tensor MakeResult(float[] data, int[] shape, Tensor[] parents, Func<Tensor, Action> makeBackward)
    {
        var result = new Tensor(data, shape);
        if (GradMode.Enabled && Array.Exists(parents, p => p.NeedsGrad))
        {
            result._parents = parents;
            result._backwardFn = makeBackward(result);
        }
        return result;
    }
}

/// <summary>
/// Disables gradient recording for a scope (target-network evaluation, action selection):
/// <c>using (GradMode.NoGrad()) { ... }</c>.
/// </summary>
public static class GradMode
{
    [ThreadStatic] private static int _noGradDepth;

    public static bool Enabled => _noGradDepth == 0;

    public static IDisposable NoGrad()
    {
        _noGradDepth++;
        return new Scope();
    }

    private sealed class Scope : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _noGradDepth--;
            }
        }
    }
}
