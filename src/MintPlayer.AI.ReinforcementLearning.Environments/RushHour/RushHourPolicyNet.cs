using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

/// <summary>
/// Two-headed policy/value network for Rush Hour imitation learning: a shared ReLU trunk, a 32-way policy head
/// (one logit per masked action) and a scalar value head predicting distance-to-goal. A thin wrapper over the
/// shared <see cref="PolicyValueNet"/> fixing Rush Hour's observation/action sizes and checkpoint kind; the trunk
/// is variable-depth, so the net can grow wider (<see cref="WidenTo"/>) and deeper (<see cref="Deepen"/>)
/// mid-training (Net2Net).
/// </summary>
public sealed class RushHourPolicyNet : IGrowableTrunkNet<RushHourPolicyNet>
{
    public const string CheckpointKind = "rushhour-policy";
    public const float DistanceScale = 20f;

    private readonly PolicyValueNet _core;

    /// <summary>A fresh net with the classic two-layer trunk of the given width.</summary>
    public RushHourPolicyNet(Xoshiro256StarStar rng, int hidden = 384)
        : this(new PolicyValueNet(RushHourBoard.ObservationSize, [hidden, hidden], RushHourBoard.ActionCount, rng)) { }

    /// <summary>A fresh net with an explicit trunk shape (e.g. a small stage for a growing run).</summary>
    public RushHourPolicyNet(Xoshiro256StarStar rng, int[] trunk)
        : this(new PolicyValueNet(RushHourBoard.ObservationSize, trunk, RushHourBoard.ActionCount, rng)) { }

    private RushHourPolicyNet(PolicyValueNet core) => _core = core;

    /// <summary>The shared-trunk hidden widths (drives growth schedules).</summary>
    public int[] Trunk => _core.Trunk;

    public IEnumerable<Tensor> Parameters() => _core.Parameters();

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B,32] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations) => _core.Forward(observations);

    /// <summary>Per-layer activations for one input row (for the live-network viewer).</summary>
    public float[][] LayerActivations(Tensor observation) => _core.LayerActivations(observation);

    /// <summary>Net2WiderNet: a wider net computing the same function.</summary>
    public RushHourPolicyNet WidenTo(int[] newHidden, Xoshiro256StarStar rng) => new(_core.WidenTo(newHidden, rng));

    /// <summary>Net2DeeperNet: a deeper net (one extra trunk layer) computing the same function.</summary>
    public RushHourPolicyNet Deepen(Xoshiro256StarStar rng) => new(_core.Deepen(rng));

    /// <summary>Single-state inference: masked logits (illegal = −∞) and predicted distance-to-goal in MOVES.</summary>
    public (float[] Logits, float Distance) Evaluate(RushHourPuzzle puzzle, ReadOnlySpan<int> positions)
    {
        var obs = new float[RushHourBoard.ObservationSize];
        RushHourBoard.WriteObservation(puzzle, positions, obs);
        var mask = RushHourBoard.ActionMask(puzzle, positions);

        using (GradMode.NoGrad())
        {
            var (logits, value) = _core.Forward(new Tensor(obs, 1, obs.Length));
            var masked = new float[RushHourBoard.ActionCount];
            for (int a = 0; a < masked.Length; a++)
                masked[a] = mask[a] ? logits.Data[a] : float.NegativeInfinity;
            return (masked, MathF.Max(0f, value.Data[0]) * DistanceScale);
        }
    }

    public void Save(Stream destination) => _core.Save(destination, CheckpointKind);

    public static RushHourPolicyNet Load(Stream source)
        => new(PolicyValueNet.Load(source, CheckpointKind, RushHourBoard.ObservationSize, RushHourBoard.ActionCount));
}
