using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Two-headed policy/value network for cube imitation learning (PLAN M16), mirroring
/// <see cref="RushHour.RushHourPolicyNet"/>: a shared ReLU trunk over the one-hot
/// sticker observation, a 12-way quarter-turn policy head and a scalar value head
/// predicting quarter-turn distance-to-solved (normalized by <see cref="DistanceScale"/>).
/// The value head doubles as the heuristic for policy-guided A*.
/// </summary>
public sealed class CubePolicyNet
{
    public const string CheckpointKind = "cube-policy";
    private const int Version = 1;
    public const float DistanceScale = 30f;

    private readonly Linear _trunk1, _trunk2, _policyHead, _valueHead;

    public CubePolicyNet(Xoshiro256StarStar rng, int hidden = 512)
    {
        _trunk1 = new Linear(RubiksCubeEnv.ObservationSize, hidden, rng, Activation.Relu);
        _trunk2 = new Linear(hidden, hidden, rng, Activation.Relu);
        _policyHead = new Linear(hidden, RubiksCubeEnv.ActionCount, rng, Activation.None);
        _valueHead = new Linear(hidden, 1, rng, Activation.None);
    }

    public IEnumerable<Tensor> Parameters()
        => _trunk1.Parameters().Concat(_trunk2.Parameters())
            .Concat(_policyHead.Parameters()).Concat(_valueHead.Parameters());

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B,12] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations)
    {
        var h = _trunk2.Forward(_trunk1.Forward(observations).Relu()).Relu();
        return (_policyHead.Forward(h), _valueHead.Forward(h));
    }

    /// <summary>
    /// Single-state inference: logits (the inverse of <paramref name="lastAction"/> masked
    /// to −∞, −1 = none) and predicted distance-to-solved in quarter-turn MOVES.
    /// </summary>
    public (float[] Logits, float Distance) Evaluate(FaceletCube cube, int lastAction = -1)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);

        using (GradMode.NoGrad())
        {
            var (logits, value) = Forward(new Tensor(obs, 1, obs.Length));
            var masked = new float[RubiksCubeEnv.ActionCount];
            int undo = RubiksCubeEnv.InverseAction(lastAction);
            for (int a = 0; a < masked.Length; a++)
                masked[a] = a == undo ? float.NegativeInfinity : logits.Data[a];
            return (masked, MathF.Max(0f, value.Data[0]) * DistanceScale);
        }
    }

    /// <summary>
    /// The policy path (trunk → trunk → policy head) as a standalone <see cref="Mlp"/>, so a GPU backend
    /// can build a device-resident forward over it (the value head is irrelevant to beam search). Weights
    /// are COPIED, so the result is a frozen snapshot — rebuild it after training to pick up new weights.
    /// </summary>
    public Mlp PolicyAsMlp()
    {
        int hidden = _trunk1.Weight.Cols;
        var mlp = new Mlp([RubiksCubeEnv.ObservationSize, hidden, hidden, RubiksCubeEnv.ActionCount],
            new Xoshiro256StarStar(0), Activation.Relu);
        var source = new[] { _trunk1, _trunk2, _policyHead };
        for (int i = 0; i < source.Length; i++)
        {
            source[i].Weight.Data.CopyTo(mlp.Layers[i].Weight.Data.AsSpan());
            source[i].Bias.Data.CopyTo(mlp.Layers[i].Bias.Data.AsSpan());
        }
        return mlp;
    }

    /// <summary>
    /// The trunk plus a COMBINED head of [12 policy logits ‖ 1 value] as a standalone <see cref="Mlp"/>
    /// (<c>[324, h, h, 13]</c>; output index <see cref="RubiksCubeEnv.ActionCount"/> = the raw value, i.e.
    /// distance-to-solved / <see cref="DistanceScale"/>). This lets a GPU backend serve BOTH the beam policy and
    /// the value heuristic from one device-resident forward — the substrate for value-guided beam search. Weights
    /// are COPIED (a frozen snapshot — rebuild after training). See <see cref="PolicyAsMlp"/> for the policy-only
    /// variant used by pure-policy beam search.
    /// </summary>
    public Mlp PolicyAndValueAsMlp()
    {
        int hidden = _trunk1.Weight.Cols;
        const int outWidth = RubiksCubeEnv.ActionCount + 1; // 12 logits + 1 value
        var mlp = new Mlp([RubiksCubeEnv.ObservationSize, hidden, hidden, outWidth],
            new Xoshiro256StarStar(0), Activation.Relu);
        _trunk1.Weight.Data.CopyTo(mlp.Layers[0].Weight.Data.AsSpan());
        _trunk1.Bias.Data.CopyTo(mlp.Layers[0].Bias.Data.AsSpan());
        _trunk2.Weight.Data.CopyTo(mlp.Layers[1].Weight.Data.AsSpan());
        _trunk2.Bias.Data.CopyTo(mlp.Layers[1].Bias.Data.AsSpan());

        // Output layer weight is [hidden, 13] row-major: interleave the policy head's [hidden, 12] columns with
        // the value head's [hidden, 1] column so index 12 of each row carries the value.
        var w = mlp.Layers[2].Weight.Data;
        var pw = _policyHead.Weight.Data; // [hidden, 12]
        var vw = _valueHead.Weight.Data;  // [hidden, 1]
        for (int r = 0; r < hidden; r++)
        {
            for (int c = 0; c < RubiksCubeEnv.ActionCount; c++)
                w[r * outWidth + c] = pw[r * RubiksCubeEnv.ActionCount + c];
            w[r * outWidth + RubiksCubeEnv.ActionCount] = vw[r];
        }
        _policyHead.Bias.Data.CopyTo(mlp.Layers[2].Bias.Data.AsSpan(0, RubiksCubeEnv.ActionCount));
        mlp.Layers[2].Bias.Data[RubiksCubeEnv.ActionCount] = _valueHead.Bias.Data[0];
        return mlp;
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

    public static CubePolicyNet Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, CheckpointKind, Version);
        int hidden = reader.ReadInt32();
        var net = new CubePolicyNet(new Xoshiro256StarStar(0), hidden);
        foreach (var layer in net.Layers())
        {
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Weight.Data.AsSpan());
            CheckpointFormat.ReadFloats(reader).CopyTo(layer.Bias.Data.AsSpan());
        }
        return net;
    }

    private IEnumerable<Linear> Layers() => [_trunk1, _trunk2, _policyHead, _valueHead];
}
