using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Dueling Q-network (Wang et al. 2016): a shared ReLU trunk splits into a scalar <b>state-value</b> stream
/// V(s) and a per-action <b>advantage</b> stream A(s,a), recombined as
/// <c>Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))</c>. Subtracting the advantage mean fixes the identifiability
/// of the V/A decomposition (otherwise a constant could shift freely between the two streams) and is what
/// makes dueling train stably. The benefit: the value of a state is learned once, shared across all actions,
/// so states where the action choice barely matters are evaluated more sample-efficiently than a plain MLP
/// that must learn each Q(s,a) independently.
/// <para>
/// Implements <see cref="IValueNet"/> — the same [B,in]→[B,out] forward / parameter-sync / structural-clone
/// contract as <see cref="Mlp"/> — so it drops into the DQN trainer (and its target-net sync) unchanged.
/// </para>
/// </summary>
public sealed class DuelingQNet : IValueNet
{
    private readonly Linear[] _trunk;
    private readonly Linear _valueHead;   // → [B,1]
    private readonly Linear _advantageHead; // → [B,actions]
    private readonly int _actions;
    private readonly int[] _hidden;
    private readonly Tensor _ones;         // constant [1,actions], broadcasts (V − meanA) across actions

    public DuelingQNet(int inputSize, int[] hidden, int actions, Xoshiro256StarStar rng)
    {
        if (hidden.Length < 1)
            throw new ArgumentException("A dueling Q-net needs at least one shared hidden layer for the two heads.");
        _actions = actions;
        _hidden = [.. hidden];

        _trunk = new Linear[hidden.Length];
        int prev = inputSize;
        for (int i = 0; i < hidden.Length; i++)
        {
            _trunk[i] = new Linear(prev, hidden[i], rng, Activation.Relu);
            prev = hidden[i];
        }
        _valueHead = new Linear(prev, 1, rng, Activation.None);
        _advantageHead = new Linear(prev, actions, rng, Activation.None);

        var ones = new float[actions];
        Array.Fill(ones, 1f);
        _ones = new Tensor(ones, 1, actions); // RequiresGrad = false (a constant)
    }

    public int InputSize => _trunk[0].Weight.Rows;
    public int Actions => _actions;
    public int[] HiddenSizes => [.. _hidden];

    public Tensor Forward(Tensor input)
    {
        var x = input;
        foreach (var layer in _trunk)
            x = layer.Forward(x).Relu();

        var value = _valueHead.Forward(x);        // [B,1]
        var advantage = _advantageHead.Forward(x); // [B,actions]
        // meanA over actions, per row → [B,1].
        var meanAdvantage = advantage.SumRows().MulScalar(1f / _actions).Reshape(input.Rows, 1);
        // Q = A + (V − meanA) broadcast across the action columns (outer product with the ones row).
        return advantage.Add(value.Sub(meanAdvantage).MatMul(_ones));
    }

    public IEnumerable<Tensor> Parameters()
    {
        foreach (var layer in _trunk)
            foreach (var p in layer.Parameters()) yield return p;
        foreach (var p in _valueHead.Parameters()) yield return p;
        foreach (var p in _advantageHead.Parameters()) yield return p;
    }

    public IValueNet CloneStructure() => new DuelingQNet(InputSize, _hidden, _actions, new Xoshiro256StarStar(0));

    /// <summary>Copies every parameter from a structurally identical net (target-network sync).</summary>
    public void CopyFrom(IValueNet source)
    {
        using var mine = Parameters().GetEnumerator();
        using var theirs = source.Parameters().GetEnumerator();
        while (mine.MoveNext() && theirs.MoveNext())
            theirs.Current.Data.CopyTo(mine.Current.Data.AsSpan());
    }
}
