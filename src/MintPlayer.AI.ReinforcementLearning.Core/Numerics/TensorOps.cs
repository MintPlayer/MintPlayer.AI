namespace MintPlayer.AI.ReinforcementLearning.Core.Numerics;

// Autograd ops. Each op's raw math runs through Backend.Current (the compute seam); this file only
// builds the tape — a forward into a fresh buffer plus a closure that ACCUMULATES (+=) into parents'
// Grad (a tensor used twice receives both contributions).
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
        Backend.Current.Gather(Data, indices, Rows, cols, data);

        return MakeResult(data, [Rows], [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.GatherBackwardInto(indices, result.Grad!, Rows, cols, Grad!);
        });
    }

    /// <summary>Sum of all elements → scalar.</summary>
    public Tensor Sum()
    {
        var data = new[] { Backend.Current.Sum(Data) };

        return MakeResult(data, [1], [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.AddScalarInto(Grad!, result.Grad![0]);
        });
    }

    /// <summary>Mean of all elements → scalar.</summary>
    public Tensor Mean()
    {
        var data = new[] { Backend.Current.Sum(Data) / Length };

        return MakeResult(data, [1], [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.AddScalarInto(Grad!, result.Grad![0] / Length);
        });
    }

    /// <summary>Row-wise sum: [B,N] → [B].</summary>
    public Tensor SumRows()
    {
        CheckRank2(this);
        int cols = Cols;
        var data = new float[Rows];
        Backend.Current.SumRows(Data, Rows, cols, data);

        return MakeResult(data, [Rows], [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.SumRowsBackwardInto(result.Grad!, Rows, cols, Grad!);
        });
    }

    /// <summary>Numerically stable row-wise log-softmax: x − max − log Σ exp(x − max).</summary>
    public Tensor LogSoftmax()
    {
        CheckRank2(this);
        int cols = Cols;
        var data = new float[Length];
        Backend.Current.LogSoftmax(Data, Rows, cols, data);

        return MakeResult(data, Shape, [this], result => () =>
        {
            EnsureGrad();
            Backend.Current.LogSoftmaxBackwardInto(result.Data, result.Grad!, Rows, cols, Grad!);
        });
    }

    /// <summary>Mean Huber (smooth-L1) loss between two [B] tensors → scalar.</summary>
    public Tensor HuberLoss(Tensor target, float delta = 1f)
    {
        CheckSameShape(this, target);
        var data = new[] { Backend.Current.HuberLoss(Data, target.Data, delta) };

        return MakeResult(data, [1], [this, target], result => () =>
        {
            float g = result.Grad![0] / Length;
            if (NeedsGrad) { EnsureGrad(); Backend.Current.HuberGradInto(Data, target.Data, delta, g, Grad!, negate: false); }
            if (target.NeedsGrad) { target.EnsureGrad(); Backend.Current.HuberGradInto(Data, target.Data, delta, g, target.Grad!, negate: true); }
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
        Backend.Current.LayerNorm(Data, gamma.Data, beta.Data, rows, cols, eps, data, xhat, invStd);

        return MakeResult(data, Shape, [this, gamma, beta], result => () =>
        {
            var dy = result.Grad!;
            if (gamma.NeedsGrad || beta.NeedsGrad)
            {
                gamma.EnsureGrad();
                beta.EnsureGrad();
                Backend.Current.LayerNormParamGradInto(dy, xhat, rows, cols, gamma.Grad!, beta.Grad!);
            }
            if (NeedsGrad)
            {
                EnsureGrad();
                Backend.Current.LayerNormInputGradInto(dy, xhat, invStd, gamma.Data, rows, cols, Grad!);
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

    /// <summary>
    /// Column-wise concatenation of two row-aligned tensors: [B,n₁] ⊕ [B,n₂] → [B,n₁+n₂]. Gradient
    /// routes the left columns back to <c>this</c> and the right columns to <paramref name="other"/>.
    /// (SAC's critic takes <c>concat(state, action)</c>, and the action half must carry the actor's gradient.)
    /// Pure memory layout, so it copies directly rather than dispatching to the compute backend.
    /// </summary>
    public Tensor ConcatCols(Tensor other)
    {
        CheckRank2(this); CheckRank2(other);
        if (Rows != other.Rows)
            throw new ArgumentException($"ConcatCols row mismatch: {Rows} vs {other.Rows}.");
        int rows = Rows, left = Cols, right = other.Cols, total = left + right;

        var data = new float[rows * total];
        for (int r = 0; r < rows; r++)
        {
            Array.Copy(Data, r * left, data, r * total, left);
            Array.Copy(other.Data, r * right, data, r * total + left, right);
        }

        return MakeResult(data, [rows, total], [this, other], result => () =>
        {
            var dy = result.Grad!;
            if (NeedsGrad)
            {
                EnsureGrad();
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < left; c++) Grad![r * left + c] += dy[r * total + c];
            }
            if (other.NeedsGrad)
            {
                other.EnsureGrad();
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < right; c++) other.Grad![r * right + c] += dy[r * total + left + c];
            }
        });
    }

    /// <summary>
    /// Extracts a contiguous column range: [B,N] → [B,<paramref name="count"/>] starting at
    /// <paramref name="start"/>. Gradient scatters back into the source columns. (Splits a policy net's
    /// [B,2·A] output into the mean and log-σ halves of a Gaussian.) Pure layout op.
    /// </summary>
    public Tensor SliceCols(int start, int count)
    {
        CheckRank2(this);
        if (start < 0 || count < 0 || start + count > Cols)
            throw new ArgumentOutOfRangeException(nameof(start), $"Slice [{start},{start + count}) out of [0,{Cols}).");
        int rows = Rows, cols = Cols;

        var data = new float[rows * count];
        for (int r = 0; r < rows; r++)
            Array.Copy(Data, r * cols + start, data, r * count, count);

        return MakeResult(data, [rows, count], [this], result => () =>
        {
            EnsureGrad();
            var dy = result.Grad!;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < count; c++) Grad![r * cols + start + c] += dy[r * count + c];
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
