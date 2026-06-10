using System.Numerics.Tensors;

namespace RLNet.Core.Numerics;

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
        TensorPrimitives.Add(Data, other.Data, data);

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
        for (int r = 0; r < rows; r++)
            TensorPrimitives.Add(Data.AsSpan(r * cols, cols), bias.Data, data.AsSpan(r * cols, cols));

        return MakeResult(data, Shape, [this, bias], result => () =>
        {
            AccumulateGrad(this, result.Grad!);
            if (bias.NeedsGrad)
            {
                bias.EnsureGrad();
                var biasGrad = bias.Grad.AsSpan();
                for (int r = 0; r < rows; r++)
                    TensorPrimitives.Add(biasGrad, result.Grad.AsSpan(r * cols, cols), biasGrad);
            }
        });
    }

    /// <summary>Elementwise subtract, same shape.</summary>
    public Tensor Sub(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        TensorPrimitives.Subtract(Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            AccumulateGrad(this, result.Grad!);
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                TensorPrimitives.Subtract(other.Grad, result.Grad, other.Grad);
            }
        });
    }

    /// <summary>Elementwise multiply, same shape.</summary>
    public Tensor Mul(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        TensorPrimitives.Multiply(Data, other.Data, data);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            if (NeedsGrad)
            {
                EnsureGrad();
                for (int i = 0; i < Length; i++) Grad![i] += other.Data[i] * result.Grad![i];
            }
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                for (int i = 0; i < Length; i++) other.Grad![i] += Data[i] * result.Grad![i];
            }
        });
    }

    public Tensor MulScalar(float scalar)
    {
        var data = new float[Length];
        TensorPrimitives.Multiply(Data, scalar, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++) Grad![i] += scalar * result.Grad![i];
        });
    }

    public Tensor Square()
    {
        var data = new float[Length];
        TensorPrimitives.Multiply(Data, Data, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++) Grad![i] += 2f * Data[i] * result.Grad![i];
        });
    }

    public Tensor Relu()
    {
        var data = new float[Length];
        for (int i = 0; i < Length; i++) data[i] = Math.Max(0f, Data[i]);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++)
                if (Data[i] > 0f) Grad![i] += result.Grad![i];
        });
    }

    public Tensor Tanh()
    {
        var data = new float[Length];
        TensorPrimitives.Tanh(Data, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++)
                Grad![i] += (1f - result.Data[i] * result.Data[i]) * result.Grad![i];
        });
    }

    public Tensor Exp()
    {
        var data = new float[Length];
        TensorPrimitives.Exp(Data, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++) Grad![i] += result.Data[i] * result.Grad![i];
        });
    }

    public Tensor Log()
    {
        var data = new float[Length];
        TensorPrimitives.Log(Data, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++) Grad![i] += result.Grad![i] / Data[i];
        });
    }

    /// <summary>Elementwise clamp; gradient flows only where the input was strictly inside the range.</summary>
    public Tensor Clamp(float min, float max)
    {
        var data = new float[Length];
        for (int i = 0; i < Length; i++) data[i] = Math.Clamp(Data[i], min, max);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            for (int i = 0; i < Length; i++)
                if (Data[i] > min && Data[i] < max) Grad![i] += result.Grad![i];
        });
    }

    /// <summary>Elementwise minimum; gradient routes to the smaller input (ties → this).</summary>
    public Tensor Min(Tensor other)
    {
        CheckSameShape(this, other);
        var data = new float[Length];
        for (int i = 0; i < Length; i++) data[i] = Math.Min(Data[i], other.Data[i]);

        return MakeResult(data, Shape, [this, other], result => () =>
        {
            if (NeedsGrad) EnsureGrad();
            if (other.NeedsGrad) other.EnsureGrad();
            for (int i = 0; i < Length; i++)
            {
                if (Data[i] <= other.Data[i]) { if (NeedsGrad) Grad![i] += result.Grad![i]; }
                else if (other.NeedsGrad) other.Grad![i] += result.Grad![i];
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

    /// <summary>Mean squared error between two same-shape tensors → scalar.</summary>
    public Tensor MseLoss(Tensor target) => Sub(target).Square().Mean();

    private static void AccumulateGrad(Tensor tensor, float[] grad)
    {
        if (!tensor.NeedsGrad) return;
        tensor.EnsureGrad();
        TensorPrimitives.Add(tensor.Grad, grad, tensor.Grad);
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
