using System.Numerics.Tensors;

namespace MintPlayer.AI.ReinforcementLearning.Core.Numerics;

// Autograd ops. Each forward computes into a fresh buffer and registers a closure that
// ACCUMULATES (+=) into parents' Grad — a tensor used twice receives both contributions.
public sealed partial class Tensor
{
    /// <summary>[m,k] · [k,n] → [m,n]</summary>
    public Tensor MatMul(Tensor other)
    {
        CheckRank2(this); CheckRank2(other);
        int m = Rows, k = Cols, n = other.Cols;
        if (other.Rows != k)
            throw new ArgumentException($"MatMul shape mismatch: [{m},{k}] · [{other.Rows},{n}].");

        var data = new float[m * n];
        Backend.Current.Gemm(Data, other.Data, data, m, k, n);

        return MakeResult(data, [m, n], [this, other], result => () =>
        {
            if (NeedsGrad)
            {
                EnsureGrad();
                Backend.Current.GemmTransposeB(result.Grad!, other.Data, Grad!, m, k, n); // dA[m,k] += dC[m,n]·B[k,n]ᵀ
            }
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                Backend.Current.GemmTransposeA(Data, result.Grad!, other.Grad!, m, k, n); // dB += Aᵀ·dC
            }
        });
    }

    /// <summary>Elementwise add, same shape.</summary>
    public Tensor Add(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        Backend.Current.Zip(BinaryOp.Add, Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            AccumulateGrad(this, result.Grad!);
            AccumulateGrad(other, result.Grad!);
        });
    }

    /// <summary>[B,N] + [N] (bias broadcast over rows).</summary>
    public Tensor AddBias(Tensor bias)
    {
        CheckRank2(this);
        if (bias.Length != Cols)
            throw new ArgumentException($"Bias length {bias.Length} does not match columns {Cols}.");
        int rows = Rows, cols = Cols;

        var data = new float[Length];
        Backend.Current.AddBias(Data, bias.Data, rows, cols, data);

        return MakeResult(data, Shape, [this, bias], result => () =>
        {
            AccumulateGrad(this, result.Grad!);
            if (bias.NeedsGrad)
            {
                bias.EnsureGrad();
                Backend.Current.BiasGradInto(result.Grad!, bias.Grad!, rows, cols);
            }
        });
    }

    /// <summary>Elementwise subtract, same shape.</summary>
    public Tensor Sub(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        Backend.Current.Zip(BinaryOp.Sub, Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            AccumulateGrad(this, result.Grad!);
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                Backend.Current.SubInto(other.Grad!, result.Grad!);
            }
        });
    }

    /// <summary>Elementwise multiply, same shape.</summary>
    public Tensor Mul(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        Backend.Current.Zip(BinaryOp.Mul, Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            if (NeedsGrad)
            {
                EnsureGrad();
                Backend.Current.MulAddInto(Grad!, other.Data, result.Grad!);
            }
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                Backend.Current.MulAddInto(other.Grad!, Data, result.Grad!);
            }
        });
    }

    public Tensor MulScalar(float scalar)
    {
        var data = new float[Length];
        Backend.Current.Scale(Data, scalar, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.AxpyInto(Grad!, scalar, result.Grad!);
        });
    }

    // Elementwise unary ops route their forward + backward math through the compute seam
    // (Backend.Current.Map / MapBackward), so a device backend can run them; ManagedBackend keeps the
    // exact (bitwise-identical) reference math.
    public Tensor Square() => MapOp(UnaryOp.Square);
    public Tensor Relu() => MapOp(UnaryOp.Relu);
    public Tensor Tanh() => MapOp(UnaryOp.Tanh);
    public Tensor Exp() => MapOp(UnaryOp.Exp);
    public Tensor Log() => MapOp(UnaryOp.Log);

    private Tensor MapOp(UnaryOp op)
    {
        var data = new float[Length];
        Backend.Current.Map(op, Data, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.MapBackward(op, Data, result.Data, result.Grad!, Grad!);
        });
    }

    /// <summary>Elementwise clamp; gradient flows only where the input was strictly inside the range.</summary>
    public Tensor Clamp(float min, float max)
    {
        var data = new float[Length];
        Backend.Current.Clamp(Data, min, max, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.ClampBackwardInto(Data, min, max, result.Grad!, Grad!);
        });
    }

    /// <summary>Elementwise minimum; gradient routes to the smaller input (ties → this).</summary>
    public Tensor Min(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        Backend.Current.Zip(BinaryOp.Min, Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            if (NeedsGrad)
            {
                EnsureGrad();
                Backend.Current.MinBackwardInto(Data, other.Data, result.Grad!, Grad!, forA: true);
            }
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                Backend.Current.MinBackwardInto(Data, other.Data, result.Grad!, other.Grad!, forA: false);
            }
        });
    }

    /// <summary>Per-row element selection: [B,N] with indices[B] → [B]. (Q(s,a), log π(a|s).)</summary>
    public Tensor Gather(int[] indices)
    {
        CheckRank2(this);
        if (indices.Length != Rows)
            throw new ArgumentException($"Expected {Rows} indices, got {indices.Length}.");
        int cols = Cols;

        var data = new float[Rows];
        for (int r = 0; r < data.Length; r++) data[r] = Data[r * cols + indices[r]];

        return MakeResult(data, [Rows], [this], result => () =>
        {
            EnsureGrad();
            for (int r = 0; r < result.Length; r++) Grad![r * cols + indices[r]] += result.Grad![r];
        });
    }

    /// <summary>Sum of all elements → scalar.</summary>
    public Tensor Sum()
    {
        var data = new[] { TensorPrimitives.Sum<float>(Data) };

        return MakeResult(data, [1], [this], result => () =>
        {
            EnsureGrad();
            float g = result.Grad![0];
            for (int i = 0; i < Length; i++) Grad![i] += g;
        });
    }

    /// <summary>Mean of all elements → scalar.</summary>
    public Tensor Mean()
    {
        var data = new[] { TensorPrimitives.Sum<float>(Data) / Length };

        return MakeResult(data, [1], [this], result => () =>
        {
            EnsureGrad();
            float g = result.Grad![0] / Length;
            for (int i = 0; i < Length; i++) Grad![i] += g;
        });
    }

    /// <summary>Row-wise sum: [B,N] → [B].</summary>
    public Tensor SumRows()
    {
        CheckRank2(this);
        int cols = Cols;
        var data = new float[Rows];
        for (int r = 0; r < data.Length; r++)
            data[r] = TensorPrimitives.Sum<float>(Data.AsSpan(r * cols, cols));

        return MakeResult(data, [Rows], [this], result => () =>
        {
            EnsureGrad();
            for (int r = 0; r < result.Length; r++)
            {
                float g = result.Grad![r];
                var row = Grad.AsSpan(r * cols, cols);
                for (int c = 0; c < cols; c++) row[c] += g;
            }
        });
    }

    /// <summary>Numerically stable row-wise log-softmax: x − max − log Σ exp(x − max).</summary>
    public Tensor LogSoftmax()
    {
        CheckRank2(this);
        int cols = Cols;
        var data = new float[Length];
        for (int r = 0; r < Rows; r++)
        {
            var row = Data.AsSpan(r * cols, cols);
            var outRow = data.AsSpan(r * cols, cols);
            float max = TensorPrimitives.Max<float>(row);
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                outRow[c] = row[c] - max;
                sum += MathF.Exp(outRow[c]);
            }
            float logSum = MathF.Log(sum);
            for (int c = 0; c < cols; c++) outRow[c] -= logSum;
        }

        return MakeResult(data, Shape, [this], result => () =>
        {
            // d/dx = dy − softmax(x) · Σ_row dy
            EnsureGrad();
            for (int r = 0; r < result.Rows; r++)
            {
                var dy = result.Grad.AsSpan(r * cols, cols);
                var y = result.Data.AsSpan(r * cols, cols);
                var dx = Grad.AsSpan(r * cols, cols);
                float rowSum = TensorPrimitives.Sum<float>(dy);
                for (int c = 0; c < cols; c++) dx[c] += dy[c] - MathF.Exp(y[c]) * rowSum;
            }
        });
    }

    /// <summary>Mean Huber (smooth-L1) loss between two [B] tensors → scalar.</summary>
    public Tensor HuberLoss(Tensor target, float delta = 1f)
    {
        CheckSameShape(this, target);
        float total = 0f;
        for (int i = 0; i < Length; i++)
        {
            float diff = Data[i] - target.Data[i];
            float abs = Math.Abs(diff);
            total += abs <= delta ? 0.5f * diff * diff : delta * (abs - 0.5f * delta);
        }
        var data = new[] { total / Length };

        return MakeResult(data, [1], [this, target], result => () =>
        {
            float g = result.Grad![0] / Length;
            if (NeedsGrad) EnsureGrad();
            if (target.NeedsGrad) target.EnsureGrad();
            for (int i = 0; i < Length; i++)
            {
                float diff = Data[i] - target.Data[i];
                float d = Math.Clamp(diff, -delta, delta) * g;
                if (NeedsGrad) Grad![i] += d;
                if (target.NeedsGrad) target.Grad![i] -= d;
            }
        });
    }

    /// <summary>
    /// Row-wise layer normalization with a learned per-feature scale (<paramref name="gamma"/>) and
    /// shift (<paramref name="beta"/>), both [Cols]: for each row, ŷ = (x − μ)/√(σ² + ε), out = γ·ŷ + β,
    /// where μ, σ² are the row's mean and variance over its Cols features. Unlike BatchNorm it uses no
    /// running statistics and no cross-row coupling, so it is stable under the DAVI target-net bootstrap
    /// (BatchNorm's batch statistics fight a frozen target). This is the normalizer for deep residual
    /// value nets (PLAN M21).
    /// </summary>
    public Tensor LayerNorm(Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        CheckRank2(this);
        int rows = Rows, cols = Cols;
        if (gamma.Length != cols || beta.Length != cols)
            throw new ArgumentException($"LayerNorm gamma/beta length must be {cols}, got {gamma.Length}/{beta.Length}.");

        var data = new float[Length];
        var xhat = new float[Length];   // normalized inputs, cached for backward
        var invStd = new float[rows];   // 1/√(σ²+ε) per row, cached for backward
        for (int r = 0; r < rows; r++)
        {
            var row = Data.AsSpan(r * cols, cols);
            float mean = TensorPrimitives.Sum<float>(row) / cols;
            float var = 0f;
            for (int c = 0; c < cols; c++) { float d = row[c] - mean; var += d * d; }
            var /= cols;
            float inv = 1f / MathF.Sqrt(var + eps);
            invStd[r] = inv;
            for (int c = 0; c < cols; c++)
            {
                float xh = (row[c] - mean) * inv;
                xhat[r * cols + c] = xh;
                data[r * cols + c] = gamma.Data[c] * xh + beta.Data[c];
            }
        }

        return MakeResult(data, Shape, [this, gamma, beta], result => () =>
        {
            var dy = result.Grad!;
            if (gamma.NeedsGrad)
            {
                gamma.EnsureGrad();
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) gamma.Grad![c] += dy[r * cols + c] * xhat[r * cols + c];
            }
            if (beta.NeedsGrad)
            {
                beta.EnsureGrad();
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) beta.Grad![c] += dy[r * cols + c];
            }
            if (NeedsGrad)
            {
                EnsureGrad();
                float invN = 1f / cols;
                for (int r = 0; r < rows; r++)
                {
                    // dx_i = (1/σ)·(dx̂_i − mean(dx̂) − x̂_i·mean(dx̂·x̂)),  dx̂ = dy·γ
                    float sum1 = 0f, sum2 = 0f;
                    for (int c = 0; c < cols; c++)
                    {
                        float dxh = dy[r * cols + c] * gamma.Data[c];
                        sum1 += dxh;
                        sum2 += dxh * xhat[r * cols + c];
                    }
                    float inv = invStd[r];
                    for (int c = 0; c < cols; c++)
                    {
                        float dxh = dy[r * cols + c] * gamma.Data[c];
                        Grad![r * cols + c] += inv * (dxh - sum1 * invN - xhat[r * cols + c] * sum2 * invN);
                    }
                }
            }
        });
    }

    /// <summary>Mean squared error between two same-shape tensors → scalar.</summary>
    public Tensor MseLoss(Tensor target) => Sub(target).Square().Mean();

    /// <summary>Same elements, new shape (shares the forward buffer; gradient passes through).</summary>
    public Tensor Reshape(params int[] shape)
    {
        int expected = 1;
        foreach (int d in shape) expected *= d;
        if (expected != Length)
            throw new ArgumentException($"Cannot reshape {Length} elements to ({string.Join('x', shape)}).");

        return MakeResult(Data, shape, [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.AddInto(Grad!, result.Grad!);
        });
    }

    private static void AccumulateGrad(Tensor tensor, float[] grad)
    {
        if (!tensor.NeedsGrad) return;
        tensor.EnsureGrad();
        Backend.Current.AddInto(tensor.Grad!, grad);
    }

    private static void CheckRank2(Tensor t)
    {
        if (t.Rank != 2)
            throw new ArgumentException($"Expected a rank-2 tensor, got rank {t.Rank}.");
    }

    private static void CheckSameShape(Tensor a, Tensor b)
    {
        if (!a.Shape.AsSpan().SequenceEqual(b.Shape))
            throw new ArgumentException($"Shape mismatch: ({string.Join('x', a.Shape)}) vs ({string.Join('x', b.Shape)}).");
    }
}
