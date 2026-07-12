using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Two-headed policy/value network for cube imitation / EfficientCube learning: a shared ReLU trunk over the
/// one-hot sticker observation, a 12-way quarter-turn policy head and a scalar value head predicting quarter-turn
/// distance-to-solved. A thin wrapper over the shared <see cref="PolicyValueNet"/> that fixes the cube's
/// observation/action sizes and checkpoint kind; the trunk is variable-depth, so the net can grow wider
/// (<see cref="WidenTo"/>) and deeper (<see cref="Deepen"/>) mid-training (Net2Net).
/// </summary>
public sealed class CubePolicyNet : IGrowableTrunkNet<CubePolicyNet>
{
    public const string CheckpointKind = "cube-policy";
    public const float DistanceScale = 30f;

    private readonly PolicyValueNet _core;

    /// <summary>A fresh net with the classic two-layer trunk of the given width.</summary>
    public CubePolicyNet(Xoshiro256StarStar rng, int hidden = 512)
        : this(new PolicyValueNet(RubiksCubeEnv.ObservationSize, [hidden, hidden], RubiksCubeEnv.ActionCount, rng)) { }

    /// <summary>A fresh net with an explicit trunk shape (e.g. a small stage for a growing run).</summary>
    public CubePolicyNet(Xoshiro256StarStar rng, int[] trunk)
        : this(new PolicyValueNet(RubiksCubeEnv.ObservationSize, trunk, RubiksCubeEnv.ActionCount, rng)) { }

    private CubePolicyNet(PolicyValueNet core) => _core = core;

    /// <summary>The shared-trunk hidden widths (drives growth schedules).</summary>
    public int[] Trunk => _core.Trunk;

    public IEnumerable<Tensor> Parameters() => _core.Parameters();

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B,12] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations) => _core.Forward(observations);

    /// <summary>Per-layer activations for one input row (for the live-network viewer).</summary>
    public float[][] LayerActivations(Tensor observation) => _core.LayerActivations(observation);

    /// <summary>Net2WiderNet: a wider net computing the same function.</summary>
    public CubePolicyNet WidenTo(int[] newHidden, Xoshiro256StarStar rng) => new(_core.WidenTo(newHidden, rng));

    /// <summary>Net2DeeperNet: a deeper net (one extra trunk layer) computing the same function.</summary>
    public CubePolicyNet Deepen(Xoshiro256StarStar rng) => new(_core.Deepen(rng));

    /// <summary>The policy path as a standalone <see cref="Mlp"/> for a GPU-resident beam-search forward.</summary>
    public Mlp PolicyAsMlp() => _core.PolicyAsMlp();

    /// <summary>
    /// Single-state inference: logits (the inverse of <paramref name="lastAction"/> masked to −∞, −1 = none) and
    /// predicted distance-to-solved in quarter-turn MOVES.
    /// </summary>
    public (float[] Logits, float Distance) Evaluate(FaceletCube cube, int lastAction = -1)
    {
        var obs = new float[RubiksCubeEnv.ObservationSize];
        RubiksCubeEnv.WriteObservation(cube, obs);

        using (GradMode.NoGrad())
        {
            var (logits, value) = _core.Forward(new Tensor(obs, 1, obs.Length));
            var masked = new float[RubiksCubeEnv.ActionCount];
            int undo = RubiksCubeEnv.InverseAction(lastAction);
            for (int a = 0; a < masked.Length; a++)
                masked[a] = a == undo ? float.NegativeInfinity : logits.Data[a];
            return (masked, MathF.Max(0f, value.Data[0]) * DistanceScale);
        }
    }

    public void Save(Stream destination) => _core.Save(destination, CheckpointKind);

    public static CubePolicyNet Load(Stream source)
        => new(PolicyValueNet.Load(source, CheckpointKind, RubiksCubeEnv.ObservationSize, RubiksCubeEnv.ActionCount));
}
