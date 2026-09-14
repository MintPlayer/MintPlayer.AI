using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

/// <summary>
/// Two-headed policy/value network for Block Dude imitation learning: a shared ReLU trunk, a 4-way policy head
/// (one logit per action) and a scalar head predicting distance-to-goal. A thin wrapper over the shared
/// <see cref="PolicyValueNet"/> fixing Block Dude's observation/action sizes and checkpoint kind; the trunk is
/// variable-depth, so the net can grow wider and deeper mid-training (Net2Net).
/// </summary>
/// <remarks>
/// <para>The distance head is not decoration — it is the heuristic for policy-guided A* at inference. Because
/// Block Dude is irreversible, a pure-argmax player that errs once can reach a terminally lost state, so the
/// shipped artefact is net-as-heuristic plus search (PRD §8.1a).</para>
///
/// <para><b>Observation width is load-bearing.</b> The egocentric encoding is constant across board sizes, so a
/// checkpoint stays valid as the curriculum grows its boards — but it does NOT survive a change to the encoding
/// itself. <see cref="Load"/> inherits <see cref="PolicyValueNet"/>'s exact-length validation, so a stale
/// checkpoint now throws instead of silently half-loading and leaving the tail of layer 0 at random init.</para>
/// </remarks>
public sealed class BlockDudePolicyNet : IGrowableTrunkNet<BlockDudePolicyNet>
{
    public const string CheckpointKind = "blockdude-policy";

    /// <summary>Distance targets are divided by this before regression and multiplied back on the way out.
    /// Sized for Block Dude's horizons, which are far longer than Rush Hour's — level 1 needs 19 optimal moves
    /// and level 3 needs 94.</summary>
    public const float DistanceScale = 40f;

    private readonly PolicyValueNet _core;

    /// <summary>The trunk a plain run trains. Growth ladders root here, so enabling growth can never start a run
    /// with less capacity than leaving it off.</summary>
    public static int[] DefaultTrunk => [512, 512];

    /// <summary>A fresh net with a two-layer trunk of the given width.</summary>
    public BlockDudePolicyNet(Xoshiro256StarStar rng, int hidden = 512)
        : this(new PolicyValueNet(BlockDudeBoard.ObservationSize, [hidden, hidden], BlockDudeBoard.ActionCount, rng)) { }

    /// <summary>A fresh net with an explicit trunk shape (e.g. a small stage for a growing run).</summary>
    public BlockDudePolicyNet(Xoshiro256StarStar rng, int[] trunk)
        : this(new PolicyValueNet(BlockDudeBoard.ObservationSize, trunk, BlockDudeBoard.ActionCount, rng)) { }

    private BlockDudePolicyNet(PolicyValueNet core) => _core = core;

    /// <summary>The shared-trunk hidden widths (drives growth schedules).</summary>
    public int[] Trunk => _core.Trunk;

    public IEnumerable<Tensor> Parameters() => _core.Parameters();

    /// <summary>Batched forward pass (autograd-recorded): raw policy logits [B,4] + value [B,1].</summary>
    public (Tensor Logits, Tensor Value) Forward(Tensor observations) => _core.Forward(observations);

    /// <summary>Per-layer activations for one input row (for the live-network viewer).</summary>
    public float[][] LayerActivations(Tensor observation) => _core.LayerActivations(observation);

    /// <summary>Net2WiderNet: a wider net computing the same function.</summary>
    public BlockDudePolicyNet WidenTo(int[] newHidden, Xoshiro256StarStar rng) => new(_core.WidenTo(newHidden, rng));

    /// <summary>Net2DeeperNet: a deeper net (one extra trunk layer) computing the same function.</summary>
    public BlockDudePolicyNet Deepen(Xoshiro256StarStar rng) => new(_core.Deepen(rng));

    /// <summary>Single-position inference: masked logits (illegal = −∞) and predicted distance-to-goal in MOVES.</summary>
    public (float[] Logits, float Distance) Evaluate(BlockDudeBoard board)
    {
        var observation = new float[BlockDudeBoard.ObservationSize];
        board.WriteObservation(observation);

        using (GradMode.NoGrad())
        {
            var (logits, value) = _core.Forward(new Tensor(observation, 1, observation.Length));
            var masked = new float[BlockDudeBoard.ActionCount];
            for (int a = 0; a < masked.Length; a++)
                masked[a] = board.IsLegal((BlockDudeAction)a) ? logits.Data[a] : float.NegativeInfinity;
            return (masked, MathF.Max(0f, value.Data[0]) * DistanceScale);
        }
    }

    /// <summary>
    /// Predicted distance-to-goal in MOVES for a whole batch of positions, in one forward pass.
    /// </summary>
    /// <remarks>
    /// Search is the dominant consumer of this net and the net is the dominant cost of search, so the shape of
    /// this call decides how deep a search can reach in a fixed budget. One forward over N·4 successors is far
    /// cheaper than N·4 single-row forwards — the whole reason <c>SolveBatched</c> exists. Values are clamped at
    /// 0 exactly as <see cref="Evaluate"/> clamps: a negative cost-to-go is meaningless and would let A* order a
    /// node ahead of the goal.
    /// </remarks>
    public float[] Distances(IReadOnlyList<BlockDudeBoard> boards)
    {
        int n = boards.Count;
        var result = new float[n];
        if (n == 0) return result;

        var obs = new float[n * BlockDudeBoard.ObservationSize];
        for (int i = 0; i < n; i++)
            boards[i].WriteObservation(obs.AsSpan(i * BlockDudeBoard.ObservationSize, BlockDudeBoard.ObservationSize));

        using (GradMode.NoGrad())
        {
            var (_, value) = _core.Forward(new Tensor(obs, n, BlockDudeBoard.ObservationSize));
            for (int i = 0; i < n; i++) result[i] = MathF.Max(0f, value.Data[i]) * DistanceScale;
        }
        return result;
    }

    /// <summary>
    /// Both heads for a whole batch, in one forward pass: per-action log-probabilities (illegal moves heavily
    /// penalised but FINITE) and distance-to-goal in moves.
    /// </summary>
    /// <param name="illegalLogProb">What an illegal move scores. Deliberately not −∞: a policy-guided search
    /// accumulates these along a path, and −∞ would turn the whole path's cost into NaN, which sorts
    /// unpredictably and silently corrupts the frontier. A large negative number expresses the same preference
    /// and stays arithmetic.</param>
    /// <remarks>
    /// Log-probabilities rather than logits, because a search that accumulates them along a path needs them
    /// comparable ACROSS states — two nodes at different depths are ordered against each other, and unnormalised
    /// logits would let one state's arbitrary offset outweigh a real preference in another.
    /// </remarks>
    public (float[] LogPriors, float[] Distances) EvaluateBatch(IReadOnlyList<BlockDudeBoard> boards,
                                                               float illegalLogProb = -20f)
    {
        int n = boards.Count;
        const int actions = BlockDudeBoard.ActionCount;
        var priors = new float[n * actions];
        var distances = new float[n];
        if (n == 0) return (priors, distances);

        var obs = new float[n * BlockDudeBoard.ObservationSize];
        for (int i = 0; i < n; i++)
            boards[i].WriteObservation(obs.AsSpan(i * BlockDudeBoard.ObservationSize, BlockDudeBoard.ObservationSize));

        using (GradMode.NoGrad())
        {
            var (logits, value) = _core.Forward(new Tensor(obs, n, BlockDudeBoard.ObservationSize));

            for (int i = 0; i < n; i++)
            {
                distances[i] = MathF.Max(0f, value.Data[i]) * DistanceScale;

                // Log-softmax over the legal moves only, computed in the numerically stable form. Masking before
                // normalising matters: probability spent on moves the engine will refuse is probability stolen
                // from the ones the search can actually take.
                int row = i * actions;
                float max = float.NegativeInfinity;
                for (int a = 0; a < actions; a++)
                    if (boards[i].IsLegal((BlockDudeAction)a) && logits.Data[row + a] > max) max = logits.Data[row + a];

                if (float.IsNegativeInfinity(max))
                {
                    // No legal move at all (a dead position). Uniform is the honest prior: there is nothing to
                    // prefer, and the search will discover the successors go nowhere.
                    for (int a = 0; a < actions; a++) priors[row + a] = -MathF.Log(actions);
                    continue;
                }

                float sum = 0f;
                for (int a = 0; a < actions; a++)
                    if (boards[i].IsLegal((BlockDudeAction)a)) sum += MathF.Exp(logits.Data[row + a] - max);

                float logSum = max + MathF.Log(sum);
                for (int a = 0; a < actions; a++)
                    priors[row + a] = boards[i].IsLegal((BlockDudeAction)a)
                        ? logits.Data[row + a] - logSum
                        : illegalLogProb;
            }
        }
        return (priors, distances);
    }

    /// <summary>The highest-scoring legal action, or null when the position has none.</summary>
    public BlockDudeAction? Greedy(BlockDudeBoard board)
    {
        var (logits, _) = Evaluate(board);
        int best = -1;
        for (int a = 0; a < logits.Length; a++)
            if (!float.IsNegativeInfinity(logits[a]) && (best < 0 || logits[a] > logits[best])) best = a;
        return best < 0 ? null : (BlockDudeAction)best;
    }

    public void Save(Stream destination) => _core.Save(destination, CheckpointKind);

    /// <exception cref="InvalidDataException">The checkpoint was trained at a different observation width — for
    /// instance before an encoding change. Previously this corrupted the net silently.</exception>
    public static BlockDudePolicyNet Load(Stream source)
        => new(PolicyValueNet.Load(source, CheckpointKind, BlockDudeBoard.ObservationSize, BlockDudeBoard.ActionCount));
}
