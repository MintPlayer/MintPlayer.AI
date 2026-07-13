namespace MintPlayer.AI.ReinforcementLearning.Core.Numerics;

// 2-D convolution as an autograd op (PLAN M42.1). Implemented the standard im2col way: gather each output
// position's receptive field into a row of a [M, inC·kH·kW] matrix, then a single GEMM against the weight
// matrix produces all output channels at once. The heavy multiply therefore runs through Backend.Current
// (GPU-routable), and the reverse pass is just the two transposed GEMMs the backend already provides plus a
// col2im scatter — so a spatial conv net needs NO new backend/kernel, only this data-movement wrapper.
//
// Layout convention: an NCHW image is carried as a rank-2 tensor [N, C·H·W] (row = image, col = c*H*W + h*W + w),
// so the op never produces a rank-4 tensor and never trips the rank-2 assertions the other ops enforce. The
// weight is [inC·kH·kW, outC] and the bias is [outC].
public sealed partial class Tensor
{
    /// <summary>
    /// 2-D convolution over an NCHW input carried as [N, inC·inH·inW], producing [N, outC·outH·outW] where
    /// outH = (inH + 2·pad − kH)/stride + 1 (outW likewise). <paramref name="weight"/> is [inC·kH·kW, outC];
    /// <paramref name="bias"/> is [outC]. Autograd-recorded: the backward pass yields dInput (col2im of dCols),
    /// dWeight (colsᵀ·dOut) and dBias (Σ dOut).
    /// </summary>
    public Tensor Conv2D(Tensor weight, Tensor bias,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride, int pad)
    {
        CheckRank2(this);
        int n = Rows;
        int kk = inC * kH * kW;
        if (Cols != inC * inH * inW)
            throw new ArgumentException($"Conv2D input cols {Cols} != inC·inH·inW = {inC * inH * inW}.");
        if (weight.Rows != kk || weight.Cols != outC)
            throw new ArgumentException($"Conv2D weight must be [{kk},{outC}], got [{weight.Rows},{weight.Cols}].");
        if (bias.Length != outC)
            throw new ArgumentException($"Conv2D bias length {bias.Length} != outC {outC}.");
        if (stride < 1 || pad < 0) throw new ArgumentException("Conv2D needs stride ≥ 1 and pad ≥ 0.");

        int outH = (inH + 2 * pad - kH) / stride + 1;
        int outW = (inW + 2 * pad - kW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException("Conv2D output size is non-positive; check kernel/stride/pad.");
        int m = n * outH * outW;

        // im2col → [M, kk], then GEMM against weight [kk, outC] → [M, outC].
        var cols = new float[m * kk];
        Im2Col(Data, cols, n, inC, inH, inW, kH, kW, stride, pad, outH, outW);
        var outMat = new float[m * outC];
        Backend.Current.Gemm(cols, weight.Data, outMat, m, kk, outC);

        // Permute [M=n·oH·oW, outC] → [N, outC·oH·oW] and add the per-channel bias.
        var data = new float[n * outC * outH * outW];
        ScatterMOutCToNCHW(outMat, bias.Data, data, n, outC, outH, outW);

        return MakeResult(data, [n, outC * outH * outW], [this, weight, bias], result => () =>
        {
            // dOut [N, outC·oH·oW] → dOutMat [M, outC].
            var dOutMat = new float[m * outC];
            GatherNCHWToMOutC(result.Grad!, dOutMat, n, outC, outH, outW);

            if (weight.NeedsGrad)
            {
                weight.EnsureGrad();
                Backend.Current.GemmTransposeA(cols, dOutMat, weight.Grad!, m, kk, outC); // dW[kk,outC] += colsᵀ·dOut
            }
            if (bias.NeedsGrad)
            {
                bias.EnsureGrad();
                var db = bias.Grad!;
                for (int i = 0; i < m; i++)
                    for (int oc = 0; oc < outC; oc++) db[oc] += dOutMat[i * outC + oc];
            }
            if (NeedsGrad)
            {
                EnsureGrad();
                var dCols = new float[m * kk];
                Backend.Current.GemmTransposeB(dOutMat, weight.Data, dCols, m, kk, outC); // dCols[M,kk] += dOut·Wᵀ
                Col2Im(dCols, Grad!, n, inC, inH, inW, kH, kW, stride, pad, outH, outW);   // scatter-add into dInput
            }
        });
    }

    // Gather each output position's receptive field into a matrix row: cols[m, (c·kH+kh)·kW+kw] = x[n,c,ih,iw]
    // (0 where the padded window falls outside the image). m = (n·oH+oh)·oW+ow.
    private static void Im2Col(ReadOnlySpan<float> x, Span<float> cols,
        int n, int inC, int inH, int inW, int kH, int kW, int stride, int pad, int outH, int outW)
    {
        int kk = inC * kH * kW;
        for (int img = 0; img < n; img++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int row = ((img * outH) + oh) * outW + ow;
                    int colBase = row * kk;
                    for (int c = 0; c < inC; c++)
                        for (int kh = 0; kh < kH; kh++)
                        {
                            int ih = oh * stride - pad + kh;
                            for (int kw = 0; kw < kW; kw++)
                            {
                                int iw = ow * stride - pad + kw;
                                int kcol = (c * kH + kh) * kW + kw;
                                cols[colBase + kcol] = (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    ? x[((img * inC + c) * inH + ih) * inW + iw]
                                    : 0f;
                            }
                        }
                }
    }

    // The transpose of Im2Col: scatter-ADD each matrix row back to the input positions it was gathered from
    // (overlapping windows accumulate). Adds into dInput (does not overwrite).
    private static void Col2Im(ReadOnlySpan<float> cols, Span<float> dInput,
        int n, int inC, int inH, int inW, int kH, int kW, int stride, int pad, int outH, int outW)
    {
        int kk = inC * kH * kW;
        for (int img = 0; img < n; img++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int row = ((img * outH) + oh) * outW + ow;
                    int colBase = row * kk;
                    for (int c = 0; c < inC; c++)
                        for (int kh = 0; kh < kH; kh++)
                        {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= inH) continue;
                            for (int kw = 0; kw < kW; kw++)
                            {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= inW) continue;
                                int kcol = (c * kH + kh) * kW + kw;
                                dInput[((img * inC + c) * inH + ih) * inW + iw] += cols[colBase + kcol];
                            }
                        }
                }
    }

    // [M=n·oH·oW, outC] → [N, outC·oH·oW] (+ per-channel bias). out[n,oc,oh,ow] = mat[(n·oH+oh)·oW+ow, oc] + bias[oc].
    private static void ScatterMOutCToNCHW(ReadOnlySpan<float> mat, ReadOnlySpan<float> bias, Span<float> outp,
        int n, int outC, int outH, int outW)
    {
        int hw = outH * outW;
        for (int img = 0; img < n; img++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int row = ((img * outH) + oh) * outW + ow;
                    int sp = oh * outW + ow;
                    for (int oc = 0; oc < outC; oc++)
                        outp[(img * outC + oc) * hw + sp] = mat[row * outC + oc] + bias[oc];
                }
    }

    // The inverse index map: [N, outC·oH·oW] → [M, outC]. mat[(n·oH+oh)·oW+ow, oc] = grad[n,oc,oh,ow].
    private static void GatherNCHWToMOutC(ReadOnlySpan<float> nchw, Span<float> mat,
        int n, int outC, int outH, int outW)
    {
        int hw = outH * outW;
        for (int img = 0; img < n; img++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int row = ((img * outH) + oh) * outW + ow;
                    int sp = oh * outW + ow;
                    for (int oc = 0; oc < outC; oc++)
                        mat[row * outC + oc] = nchw[(img * outC + oc) * hw + sp];
                }
    }
}
