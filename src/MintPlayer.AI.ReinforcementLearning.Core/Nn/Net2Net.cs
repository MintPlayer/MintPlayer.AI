using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>A net with a plain trunk that can grow function-preservingly (Net2Wider/DeeperNet). Lets one shared
/// growth driver widen/deepen any implementer (the two-headed policy nets) on a schedule. Self-typed so the growth
/// operators return the concrete net type.</summary>
public interface IGrowableTrunkNet<TSelf> where TSelf : IGrowableTrunkNet<TSelf>
{
    /// <summary>The shared-trunk hidden widths.</summary>
    int[] Trunk { get; }
    /// <summary>A wider net (same depth) computing the same function.</summary>
    TSelf WidenTo(int[] newHidden, Xoshiro256StarStar rng);
    /// <summary>A deeper net (one extra trunk layer) computing the same function.</summary>
    TSelf Deepen(Xoshiro256StarStar rng);
    IEnumerable<Tensor> Parameters();
}

/// <summary>
/// Function-preserving architecture-growth primitives (Chen et al. 2016), shared by the nets that grow mid-training
/// (<see cref="DuelingQNet"/>, <see cref="PolicyValueNet"/>): both are a plain ReLU trunk feeding one or more output
/// heads, so the growth math is identical and lives here once.
/// </summary>
public static class Net2Net
{
    /// <summary>
    /// Net2WiderNet: fill <paramref name="newTrunk"/> (already built with the wider <paramref name="newHidden"/>
    /// widths) and each output head so the whole net computes the <b>same function</b>. Each widened layer's extra
    /// units duplicate a randomly-chosen existing unit; every consumer of a duplicated unit (the next trunk layer,
    /// or a head reading the last trunk layer) has its incoming weights from that unit split evenly across the
    /// copies, so all downstream sums are unchanged. Heads read the LAST trunk layer's output.
    /// </summary>
    public static void WidenTrunk(int inputSize, Linear[] oldTrunk, int[] oldHidden, Linear[] newTrunk, int[] newHidden,
        ReadOnlySpan<(Linear Old, Linear New, int OutDim)> heads, Xoshiro256StarStar rng)
    {
        int layers = oldTrunk.Length;
        var map = new int[layers][];   // new-unit → old-unit
        var count = new int[layers][]; // how many new units map to each old unit (the split factor)
        for (int i = 0; i < layers; i++)
        {
            int oldW = oldHidden[i], newW = newHidden[i];
            var g = new int[newW];
            var cnt = new int[oldW];
            for (int j = 0; j < oldW; j++) { g[j] = j; cnt[j] = 1; }
            for (int j = oldW; j < newW; j++) { int s = rng.NextInt(oldW); g[j] = s; cnt[s]++; }
            map[i] = g; count[i] = cnt;
        }

        for (int i = 0; i < layers; i++)
        {
            Linear oldL = oldTrunk[i], newL = newTrunk[i];
            int oldOut = oldHidden[i], newOut = newHidden[i];
            int newIn = i == 0 ? inputSize : newHidden[i - 1];
            for (int r = 0; r < newIn; r++)
            {
                int sr = i == 0 ? r : map[i - 1][r];
                float scale = i == 0 ? 1f : 1f / count[i - 1][sr];
                for (int c = 0; c < newOut; c++)
                    newL.Weight.Data[r * newOut + c] = oldL.Weight.Data[sr * oldOut + map[i][c]] * scale;
            }
            for (int c = 0; c < newOut; c++) newL.Bias.Data[c] = oldL.Bias.Data[map[i][c]];
        }

        foreach (var (oldHead, newHead, outDim) in heads)
        {
            int mapLastLen = map[layers - 1].Length;
            for (int r = 0; r < mapLastLen; r++)
            {
                int sr = map[layers - 1][r];
                float scale = 1f / count[layers - 1][sr];
                for (int o = 0; o < outDim; o++)
                    newHead.Weight.Data[r * outDim + o] = oldHead.Weight.Data[sr * outDim + o] * scale;
            }
            oldHead.Bias.Data.CopyTo(newHead.Bias.Data.AsSpan());
        }
    }

    /// <summary>Net2DeeperNet building block: set a square layer to identity (W = I, b = 0). Inserted after a ReLU
    /// (whose output is ≥ 0), <c>ReLU(I·x) = x</c>, so it's function-preserving.</summary>
    public static void SetIdentity(Linear layer)
    {
        int w = layer.Weight.Cols;
        Array.Clear(layer.Weight.Data);
        for (int k = 0; k < w; k++) layer.Weight.Data[k * w + k] = 1f;
        Array.Clear(layer.Bias.Data);
    }

    /// <summary>Copy a Linear's weights + bias into an identically-shaped one.</summary>
    public static void CopyLinear(Linear src, Linear dst)
    {
        src.Weight.Data.CopyTo(dst.Weight.Data.AsSpan());
        src.Bias.Data.CopyTo(dst.Bias.Data.AsSpan());
    }
}
