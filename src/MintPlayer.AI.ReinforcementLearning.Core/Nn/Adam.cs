using System.Numerics.Tensors;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>Adam optimizer (Kingma &amp; Ba 2015) with bias correction.</summary>
public sealed class Adam
{
    private readonly Tensor[] _parameters;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private readonly float _beta1, _beta2, _epsilon;
    private int _step;

    public Adam(IEnumerable<Tensor> parameters, float learningRate = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    {
        _parameters = parameters.ToArray();
        _m = _parameters.Select(p => new float[p.Length]).ToArray();
        _v = _parameters.Select(p => new float[p.Length]).ToArray();
        LearningRate = learningRate;
        (_beta1, _beta2, _epsilon) = (beta1, beta2, epsilon);
    }

    /// <summary>Mutable so trainers can anneal it from a schedule.</summary>
    public float LearningRate { get; set; }

    // Internal state access for checkpointing (Checkpoints/AdamCheckpoint.cs).
    internal float Beta1 => _beta1;
    internal float Beta2 => _beta2;
    internal float EpsilonValue => _epsilon;
    internal int StepCount { get => _step; set => _step = value; }
    internal float[][] FirstMoments => _m;
    internal float[][] SecondMoments => _v;

    public void Step()
    {
        _step++;
        float correction1 = 1f - MathF.Pow(_beta1, _step);
        float correction2 = 1f - MathF.Pow(_beta2, _step);

        for (int p = 0; p < _parameters.Length; p++)
        {
            var grad = _parameters[p].Grad;
            if (grad is null) continue;
            var data = _parameters[p].Data;
            var m = _m[p];
            var v = _v[p];

            for (int i = 0; i < data.Length; i++)
            {
                m[i] = _beta1 * m[i] + (1f - _beta1) * grad[i];
                v[i] = _beta2 * v[i] + (1f - _beta2) * grad[i] * grad[i];
                float mHat = m[i] / correction1;
                float vHat = v[i] / correction2;
                data[i] -= LearningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
            }
        }
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.ZeroGrad();
    }

    /// <summary>Clips gradients so their global L2 norm is at most <paramref name="maxNorm"/>. Returns the pre-clip norm.</summary>
    public float ClipGradNorm(float maxNorm)
    {
        double sumSquares = 0;
        foreach (var p in _parameters)
            if (p.Grad is not null)
                sumSquares += TensorPrimitives.SumOfSquares<float>(p.Grad);

        float norm = (float)Math.Sqrt(sumSquares);
        if (norm > maxNorm)
        {
            float scale = maxNorm / (norm + 1e-6f);
            foreach (var p in _parameters)
                if (p.Grad is not null)
                    TensorPrimitives.Multiply(p.Grad, scale, p.Grad);
        }
        return norm;
    }
}
