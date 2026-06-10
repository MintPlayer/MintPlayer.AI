using RLNet.Core.Numerics;
using RLNet.Core.Random;

namespace RLNet.Core.Nn;

public interface IModule
{
    Tensor Forward(Tensor input);
    IEnumerable<Tensor> Parameters();
}

public enum Activation { None, Relu, Tanh }

/// <summary>Fully-connected layer: x[B,in] · W[in,out] + b[out].</summary>
public sealed class Linear : IModule
{
    public Linear(int inputSize, int outputSize, Xoshiro256StarStar rng, Activation initFor = Activation.Tanh)
    {
        // He init for ReLU, Xavier/Glorot otherwise.
        float std = initFor == Activation.Relu
            ? MathF.Sqrt(2f / inputSize)
            : MathF.Sqrt(2f / (inputSize + outputSize));
        Weight = new Tensor(Tensor.RandomNormal(rng, 0f, std, inputSize, outputSize).Data, inputSize, outputSize) { RequiresGrad = true };
        Bias = new Tensor(new float[outputSize], outputSize) { RequiresGrad = true };
    }

    public Tensor Weight { get; }
    public Tensor Bias { get; }

    public Tensor Forward(Tensor input) => input.MatMul(Weight).AddBias(Bias);

    public IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
        yield return Bias;
    }
}

/// <summary>Multi-layer perceptron: Linear + activation per hidden layer, linear output head.</summary>
public sealed class Mlp : IModule
{
    private readonly Linear[] _layers;
    private readonly Activation _hidden;

    public Mlp(int[] sizes, Xoshiro256StarStar rng, Activation hidden = Activation.Tanh)
    {
        if (sizes.Length < 2)
            throw new ArgumentException("An MLP needs at least input and output sizes.");
        _hidden = hidden;
        _layers = new Linear[sizes.Length - 1];
        for (int i = 0; i < _layers.Length; i++)
        {
            bool isOutput = i == _layers.Length - 1;
            _layers[i] = new Linear(sizes[i], sizes[i + 1], rng, isOutput ? Activation.None : hidden);
        }
    }

    /// <summary>The constituent layers, e.g. for custom (re-)initialization schemes.</summary>
    public IReadOnlyList<Linear> Layers => _layers;

    public Tensor Forward(Tensor input)
    {
        var x = input;
        for (int i = 0; i < _layers.Length; i++)
        {
            x = _layers[i].Forward(x);
            if (i < _layers.Length - 1)
                x = _hidden switch
                {
                    Activation.Relu => x.Relu(),
                    Activation.Tanh => x.Tanh(),
                    _ => x,
                };
        }
        return x;
    }

    public IEnumerable<Tensor> Parameters() => _layers.SelectMany(l => l.Parameters());

    /// <summary>Copies all parameters from another structurally identical MLP (target-network sync).</summary>
    public void CopyFrom(Mlp source)
    {
        using var mine = Parameters().GetEnumerator();
        using var theirs = source.Parameters().GetEnumerator();
        while (mine.MoveNext() && theirs.MoveNext())
            theirs.Current.Data.CopyTo(mine.Current.Data.AsSpan());
    }
}
