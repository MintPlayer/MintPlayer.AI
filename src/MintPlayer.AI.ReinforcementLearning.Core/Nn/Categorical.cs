using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Categorical distribution over discrete actions, parameterized by logits [B,N].
/// Log-probs and entropy are built from autograd ops so policy-gradient losses
/// differentiate through them; sampling itself carries no gradient.
/// </summary>
public readonly struct Categorical(Tensor logits)
{
    private readonly Tensor _logProbs = logits.LogSoftmax();

    /// <summary>Samples one action per row from the softmax distribution.</summary>
    public int[] Sample(Xoshiro256StarStar rng)
    {
        int rows = _logProbs.Rows, cols = _logProbs.Cols;
        var actions = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            double roll = rng.NextDouble();
            double cumulative = 0;
            int action = cols - 1;
            for (int c = 0; c < cols; c++)
            {
                cumulative += Math.Exp(_logProbs.Data[r * cols + c]);
                if (roll < cumulative) { action = c; break; }
            }
            actions[r] = action;
        }
        return actions;
    }

    /// <summary>Greedy (argmax) action per row.</summary>
    public int[] Mode()
    {
        int rows = _logProbs.Rows, cols = _logProbs.Cols;
        var actions = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int c = 1; c < cols; c++)
                if (_logProbs.Data[r * cols + c] > _logProbs.Data[r * cols + best]) best = c;
            actions[r] = best;
        }
        return actions;
    }

    /// <summary>log π(a|s) for the given actions → [B].</summary>
    public Tensor LogProb(int[] actions) => _logProbs.Gather(actions);

    /// <summary>Per-row entropy −Σ p·log p → [B].</summary>
    public Tensor Entropy() => _logProbs.Exp().Mul(_logProbs).SumRows().MulScalar(-1f);
}
