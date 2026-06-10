using System.Text;
using RLNet.Core.Checkpoints;
using RLNet.Core.Nn;
using RLNet.Core.Numerics;
using RLNet.Core.Random;

namespace RLNet.Environments.RushHour;

/// <summary>
/// Two-headed policy/value network for Rush Hour imitation learning: a shared ReLU
/// trunk, a 32-way policy head (one logit per masked action) and a scalar value head
/// predicting distance-to-goal (normalized by <see cref="DistanceScale"/>). The value
/// head doubles as the heuristic for policy-guided A* search.
/// </summary>
public sealed class RushHourPolicyNet
{
    public const string CheckpointKind = "rushhour-policy";
    private const int Version = 1;
    public const float DistanceScale = 20f;

    private readonly Linear _trunk1, _trunk2, _policyHead, _valueHead;

    public RushHourPolicyNet(Xoshiro256StarStar rng, int hidden = 384)
    {
        _trunk1 = new Linear(RushHourBoard.ObservationSize, hidden, rng, Activation.Relu);
        _trunk2 = new Linear(hidden, hidden, rng, Activation.Relu);
        _policyHead = new Linear(hidden, RushHourBoard.ActionCount, rng, Activation.None);
        _valueHead = new Linear(hidden, 1, rng, Activation.None);
    }

    public IEnumerable<Tensor> Parameters()
        => _trunk1.Parameters().Concat(_trunk2.Parameters())
            .Concat(_policyHead.Parameters()).Concat(_valueHead.Parameters());

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B,32] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations)
    {
        var h = _trunk2.Forward(_trunk1.Forward(observations).Relu()).Relu();
        return (_policyHead.Forward(h), _valueHead.Forward(h));
    }

    /// <summary>Single-state inference: masked logits (illegal = −∞) and predicted distance-to-goal in MOVES.</summary>
    public (float[] Logits, float Distance) Evaluate(RushHourPuzzle puzzle, ReadOnlySpan<int> positions)
    {
        var obs = new float[RushHourBoard.ObservationSize];
        RushHourBoard.WriteObservation(puzzle, positions, obs);
        var mask = RushHourBoard.ActionMask(puzzle, positions);

        using (GradMode.NoGrad())
        {
            var (logits, value) = Forward(new Tensor(obs, 1, obs.Length));
            var masked = new float[RushHourBoard.ActionCount];
            for (int a = 0; a < masked.Length; a++)
                masked[a] = mask[a] ? logits.Data[a] : float.NegativeInfinity;
            return (masked, MathF.Max(0f, value.Data[0]) * DistanceScale);
        }
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

    public static RushHourPolicyNet Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, CheckpointKind, Version);
        int hidden = reader.ReadInt32();
        var net = new RushHourPolicyNet(new Xoshiro256StarStar(0), hidden);
        foreach (var layer in net.Layers())
        {
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Weight.Data.AsSpan());
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Bias.Data.AsSpan());
        }
        return net;
    }

    private IEnumerable<Linear> Layers() => [_trunk1, _trunk2, _policyHead, _valueHead];
}
