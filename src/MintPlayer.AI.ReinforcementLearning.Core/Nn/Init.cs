using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

public static class Init
{
    /// <summary>
    /// Fills a tensor with i.i.d. samples from the uniform distribution U[−limit, +limit] —
    /// the NoisyNets mean-weight initializer (Fortunato et al. 2017, with limit = 1/√in).
    /// </summary>
    public static void Uniform(Tensor tensor, float limit, Xoshiro256StarStar rng)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
            tensor.Data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * limit);
    }

    /// <summary>
    /// Orthogonal initialization (Saxe et al. 2014) via modified Gram–Schmidt on a random
    /// Gaussian matrix — the PPO-standard scheme (hidden gain √2, policy head 0.01, value head 1).
    /// </summary>
    public static void Orthogonal(Tensor weight, float gain, Xoshiro256StarStar rng)
    {
        int rows = weight.Shape[0], cols = weight.Shape[1];
        var gaussian = Tensor.RandomNormal(rng, 0f, 1f, rows, cols).Data;

        // Orthonormalize along the shorter dimension so the vectors can be independent.
        if (rows >= cols)
            OrthonormalizeColumns(gaussian, rows, cols);
        else
            OrthonormalizeRows(gaussian, rows, cols);

        for (int i = 0; i < gaussian.Length; i++)
            weight.Data[i] = gain * gaussian[i];
    }

    private static void OrthonormalizeColumns(float[] m, int rows, int cols)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int prev = 0; prev < j; prev++)
            {
                float dot = 0f;
                for (int r = 0; r < rows; r++) dot += m[r * cols + j] * m[r * cols + prev];
                for (int r = 0; r < rows; r++) m[r * cols + j] -= dot * m[r * cols + prev];
            }
            float norm = 0f;
            for (int r = 0; r < rows; r++) norm += m[r * cols + j] * m[r * cols + j];
            norm = MathF.Sqrt(norm);
            for (int r = 0; r < rows; r++) m[r * cols + j] /= norm;
        }
    }

    private static void OrthonormalizeRows(float[] m, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            var row = m.AsSpan(i * cols, cols);
            for (int prev = 0; prev < i; prev++)
            {
                var prevRow = m.AsSpan(prev * cols, cols);
                float dot = 0f;
                for (int c = 0; c < cols; c++) dot += row[c] * prevRow[c];
                for (int c = 0; c < cols; c++) row[c] -= dot * prevRow[c];
            }
            float norm = 0f;
            for (int c = 0; c < cols; c++) norm += row[c] * row[c];
            norm = MathF.Sqrt(norm);
            for (int c = 0; c < cols; c++) row[c] /= norm;
        }
    }
}
