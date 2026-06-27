using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// Two-headed policy/value network for FruitCake <b>planner distillation</b> (PRD FRUITCAKE_IMPROVE lever F6),
/// mirroring <see cref="RubiksCube.CubePolicyNet"/>: a shared ReLU trunk over the 83-dim
/// <see cref="FruitCakeEnv.BuildObservation"/> features, a <see cref="FruitCakeEnv.ColumnCount"/>-way column
/// policy head and a scalar value head predicting the (discounted) <b>remaining game score</b> from the position.
///
/// <para>The reactive DQN plateaued at pineapple because it cannot plan; the depth-3 <see cref="FruitCakeSearch"/>
/// plans and reaches watermelons ~half the time, but is too slow to run deep at serve time. This net is trained to
/// <b>imitate that search</b> — the policy head copies the planner's chosen column, the value head regresses the
/// realized return — so it absorbs the planner's strength into the weights. Two payoffs: the policy head plays
/// strongly with a single forward pass (cheap serving, no search), and the value head is a <b>planning-aware leaf
/// value</b> for <see cref="FruitCakeSearch"/> — a better leaf than the pineapple-capped DQN max-Q, so search on
/// top compounds.</para>
/// </summary>
public sealed class FruitCakePolicyNet
{
    public const string CheckpointKind = "fruitcake-policy";
    private const int Version = 1;

    /// <summary>Value-head target/output scale: the value head learns return-to-go ÷ this, so the loss stays O(1)
    /// while <see cref="Evaluate"/> / <see cref="BoardValue"/> report a raw-score-unit value comparable to the
    /// merge points the search accumulates along a line.</summary>
    public const float ValueScale = 500f;

    private readonly Linear _trunk1, _trunk2, _policyHead, _valueHead;

    public FruitCakePolicyNet(Xoshiro256StarStar rng, int hidden = 256)
    {
        _trunk1 = new Linear(FruitCakeEnv.ObservationSize, hidden, rng, Activation.Relu);
        _trunk2 = new Linear(hidden, hidden, rng, Activation.Relu);
        _policyHead = new Linear(hidden, FruitCakeEnv.ColumnCount, rng, Activation.None);
        _valueHead = new Linear(hidden, 1, rng, Activation.None);
    }

    public IEnumerable<Tensor> Parameters()
        => _trunk1.Parameters().Concat(_trunk2.Parameters())
            .Concat(_policyHead.Parameters()).Concat(_valueHead.Parameters());

    /// <summary>Batched forward pass (autograd-recorded): raw column logits [B,ColumnCount] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations)
    {
        var h = _trunk2.Forward(_trunk1.Forward(observations).Relu()).Relu();
        return (_policyHead.Forward(h), _valueHead.Forward(h));
    }

    /// <summary>
    /// Single-state inference for a board with the given current/next droppable tiers: the raw column logits and
    /// the predicted remaining score (de-normalized to raw points, clamped ≥ 0).
    /// </summary>
    public (float[] Logits, float Value) Evaluate(FruitCakeWorld world, int current, int next)
    {
        var obs = FruitCakeEnv.BuildObservation(world, current, next);
        using (GradMode.NoGrad())
        {
            var (logits, value) = Forward(new Tensor(obs, 1, obs.Length));
            return ((float[])logits.Data.Clone(), MathF.Max(0f, value.Data[0]) * ValueScale);
        }
    }

    /// <summary>The best column to drop <paramref name="current"/> into, by the policy head alone (no search).</summary>
    public int ChooseColumn(FruitCakeWorld world, int current, int next)
    {
        var (logits, _) = Evaluate(world, current, next);
        int best = 0;
        for (int c = 1; c < logits.Length; c++)
            if (logits[c] > logits[best]) best = c;
        return best;
    }

    /// <summary>
    /// Planning-aware board value for use as a <see cref="FruitCakeSearch"/> leaf: the upcoming fruit is unknown at
    /// a leaf, so average the value head over the droppable tiers (board features dominate; the exact next is
    /// second-order — the same marginalization the DQN-max-Q leaf used). Raw-score units.
    /// </summary>
    public double BoardValue(FruitCakeWorld world)
    {
        double sum = 0;
        foreach (var d in FruitCatalog.Droppable)
            sum += Evaluate(world, d.Tier, d.Tier).Value;
        return sum / FruitCatalog.Droppable.Count;
    }

    public void Save(Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, CheckpointKind, Version);
        writer.Write(_trunk1.Weight.Cols); // hidden size
        foreach (var layer in Layers())
        {
            CheckpointFormat.WriteFloats(writer, layer.Weight.Data);
            CheckpointFormat.WriteFloats(writer, layer.Bias.Data);
        }
    }

    public static FruitCakePolicyNet Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, CheckpointKind, Version);
        int hidden = reader.ReadInt32();
        var net = new FruitCakePolicyNet(new Xoshiro256StarStar(0), hidden);
        foreach (var layer in net.Layers())
        {
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Weight.Data.AsSpan());
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Bias.Data.AsSpan());
        }
        return net;
    }

    private IEnumerable<Linear> Layers() => [_trunk1, _trunk2, _policyHead, _valueHead];
}
