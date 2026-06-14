using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>Hyper-parameters for <see cref="ValueIterationTrainer{TState}"/>.</summary>
public sealed class ValueIterationOptions
{
    public int BatchSize { get; init; } = 128;
    public float LearningRate { get; init; } = 1e-3f;
    /// <summary>Steps between copying the online net into the bootstrapping target net (stability).</summary>
    public int TargetUpdateInterval { get; init; } = 100;
    /// <summary>
    /// DeepCubeA-style ε-loss target sync: when &gt; 0, the periodic target update is gated on the
    /// batch loss having fallen below this threshold, so the bootstrapping target only advances once
    /// the online net has actually converged on the current target — a stability win at depth, where
    /// a target that chases a still-moving net oscillates. 0 (default) = sync every interval, always.
    /// </summary>
    public float TargetUpdateLossThreshold { get; init; } = 0f;
    /// <summary>Cost-to-go is regressed in units of this many moves (keeps the target ~O(1)).</summary>
    public float DistanceScale { get; init; } = 20f;
    public float HuberDelta { get; init; } = 1f;
    /// <summary>Gradient-norm clip (matches the DQN/imitation trainers).</summary>
    public float GradClipNorm { get; init; } = 5f;
}

/// <summary>
/// Approximate value iteration (DAVI, à la DeepCubeA) over an
/// <see cref="IDeterministicModel{TState}"/> — the trainer that lets an agent learn to reach a
/// goal <b>without a teacher</b>, bounded only by the cost objective (fewest moves), so it can
/// surpass an imitation policy capped at its demonstrator. It learns a value net V predicting
/// cost-to-go by bootstrapping each state's target from a one-step lookahead over the model:
/// <c>target(s) = min_a [ 1 + (IsGoal(s') ? 0 : V_target(s')) ]</c>, anchored at <c>V(goal)=0</c>.
/// The signal originates at the goal and propagates outward as training proceeds; a periodically
/// synced target net stabilizes the bootstrap. Inference policy: <see cref="GreedyValuePlanner"/>.
/// <para>
/// Generic over the state type; the two env-specific pieces are injected — a <c>featurize</c>
/// (state → net input) and a state sampler passed to <see cref="Train"/> (e.g. random scrambles
/// from the goal). The DAVI algorithm itself lives here, reusable across goal-directed envs.
/// </para>
/// <para>
/// Distinct from <c>Solvers.ValueIteration</c>, which is EXACT tabular value iteration over a
/// small enumerable MDP (FrozenLake-scale). This is the function-approximation counterpart for
/// state spaces far too large to enumerate (a cube has ~4.3×10¹⁹ states): the net generalizes a
/// cost-to-go it never tabulates.
/// </para>
/// </summary>
public sealed class ValueIterationTrainer<TState>
{
    private readonly IDeterministicModel<TState> _model;
    private readonly Func<TState, float[]> _featurize;
    private readonly IValueNet _net;
    private readonly IValueNet _target;
    private readonly Adam _adam;
    private readonly ValueIterationOptions _options;
    private readonly int _featureSize;
    private readonly ITargetForward _targetForward;
    private readonly IResidentTrainStep? _residentTrain;

    /// <param name="valueNet">Scalar-output MLP (… → 1) predicting scaled cost-to-go.</param>
    /// <param name="optimizer">
    /// The Adam optimizer over <paramref name="valueNet"/>'s parameters. Passed in (not created
    /// internally) so a campaign can persist and restore its moment estimates across a resume —
    /// without them, a resumed run spends its first steps re-estimating gradient statistics.
    /// </param>
    /// <param name="targetForward">
    /// Optional no-grad forward for the bootstrapping target's ActionCount× successor evaluation —
    /// the dominant cost. Lets a campaign inject a device-resident GPU forward (the Ilgpu
    /// <c>DeviceMlp</c>) that keeps weights resident across steps, without coupling this (Core) layer
    /// to any GPU backend. Null → <see cref="AutogradTargetForward"/> (autograd via Backend.Current).
    /// The trainer drives its <see cref="ITargetForward.OnTargetSynced"/> on every target-net sync.
    /// </param>
    /// <param name="residentTrain">
    /// Optional fully device-resident train step (the Ilgpu <c>DeviceResidualTrainer</c>): when provided,
    /// the online net's forward+backward+clip+Adam run on-device and only sync back to the CPU net for
    /// eval/checkpoint/target copy (PLAN M20 Stage 3). Null → the autograd train path via Backend.Current.
    /// </param>
    public ValueIterationTrainer(
        IDeterministicModel<TState> model, Func<TState, float[]> featurize, IValueNet valueNet, Adam optimizer, ValueIterationOptions options,
        ITargetForward? targetForward = null, IResidentTrainStep? residentTrain = null)
    {
        _model = model;
        _featurize = featurize;
        _net = valueNet;
        _adam = optimizer;
        _options = options;
        _featureSize = valueNet.InputSize;
        _residentTrain = residentTrain;

        // Independent bootstrapping target, kept structurally identical and periodically synced.
        // On resume it starts equal to the loaded net (as if just synced) — harmless.
        _target = valueNet.CloneStructure();
        _target.CopyFrom(valueNet);

        _targetForward = targetForward ?? new AutogradTargetForward(_featureSize);
        _targetForward.OnTargetSynced(_target); // initial sync — _target now holds the eval weights
    }

    /// <summary>Estimated cost-to-go from <paramref name="state"/> to the goal, in MOVES (≥ 0).</summary>
    public float Value(TState state)
    {
        using (GradMode.NoGrad())
        {
            var features = _featurize(state);
            var v = _net.Forward(new Tensor(features, 1, _featureSize));
            return MathF.Max(0f, v.Data[0]) * _options.DistanceScale;
        }
    }

    /// <summary>
    /// A greedy solver bound to this trainer's value net. Evaluates each step's successors in a
    /// single batched forward (the eval hot path — far cheaper than one tiny forward per action).
    /// </summary>
    public IReadOnlyList<int>? Solve(TState start, int maxSteps)
        => GreedyValuePlanner.Solve(_model, BatchValue, start, maxSteps);

    /// <summary>Cost-to-go (MOVES, ≥ 0) for a batch of states in ONE forward — used by the greedy solver.</summary>
    private float[] BatchValue(IReadOnlyList<TState> states)
    {
        int n = states.Count;
        var features = new float[n * _featureSize];
        for (int i = 0; i < n; i++)
            _featurize(states[i]).CopyTo(features.AsSpan(i * _featureSize, _featureSize));

        using (GradMode.NoGrad())
        {
            var values = _net.Forward(new Tensor(features, n, _featureSize));
            var result = new float[n];
            for (int i = 0; i < n; i++)
                result[i] = MathF.Max(0f, values.Data[i]) * _options.DistanceScale;
            return result;
        }
    }

    /// <summary>
    /// A weighted-A* solver bound to this trainer's value net — reaches states the greedy policy
    /// gets stuck on, at the cost of expanding a frontier. <paramref name="weight"/> &gt; 1 trades
    /// optimality for depth/speed.
    /// </summary>
    public IReadOnlyList<int>? SolveWithSearch(TState start, int maxExpansions, float weight = 1f)
        => ValueGuidedSearch.Solve(_model, Value, start, maxExpansions, weight);

    /// <summary>
    /// Batch-weighted A* (BWAS) bound to this trainer's value net — the same search as
    /// <see cref="SolveWithSearch"/> but it scores each expansion round's successors in ONE batched
    /// forward (<see cref="BatchValue"/>), so the value net's cost amortizes over the whole frontier
    /// slice. This is the form fast enough for deep solves and the GPU. <paramref name="expandBatch"/>
    /// is how many open nodes are expanded per round; <paramref name="weight"/> = 1 is optimal under an
    /// admissible value, &gt; 1 reaches deeper for possibly-longer solutions.
    /// </summary>
    public IReadOnlyList<int>? SolveWithSearchBatched(TState start, int maxExpansions, float weight = 1f, int expandBatch = 100)
        => ValueGuidedSearch.SolveBatched(_model, BatchValue, start, maxExpansions, weight, expandBatch);

    /// <summary>
    /// Run <paramref name="iterations"/> DAVI updates. <paramref name="sampleState"/> draws a
    /// training state each call (typically a random scramble from the goal). <paramref name="onIteration"/>
    /// receives (iteration, batch-mean Huber loss) for progress logging.
    /// </summary>
    public void Train(Func<TState> sampleState, int iterations, Action<int, float>? onIteration = null)
    {
        int batch = _options.BatchSize, actions = _model.ActionCount;
        // ε-loss target sync: when a loss threshold is set, the target only advances once the online net
        // has converged on it — but with a max-interval fallback so it can never freeze (DeepCubeA-style).
        int itersSinceSync = 0;
        int maxSyncInterval = 5 * _options.TargetUpdateInterval;

        for (int it = 0; it < iterations; it++)
        {
            itersSinceSync++;
            // Sampling stays sequential to keep the RNG draw order deterministic.
            var states = new TState[batch];
            for (int b = 0; b < batch; b++) states[b] = sampleState();

            // Successor features for every (state, action) pair, plus which successors are goals.
            // This Apply+featurize fan-out is pure (Apply clones, featurize writes a disjoint slice),
            // so it parallelizes across the batch with results independent of scheduling — the bottleneck
            // once the GPU forward is resident (PLAN P.7). Determinism is preserved (disjoint writes).
            var successorFeatures = new float[batch * actions * _featureSize];
            var successorIsGoal = new bool[batch * actions];
            Parallel.For(0, batch, b =>
            {
                for (int a = 0; a < actions; a++)
                {
                    var next = _model.Apply(states[b], a);
                    int idx = b * actions + a;
                    successorIsGoal[idx] = _model.IsGoal(next);
                    _featurize(next).CopyTo(successorFeatures.AsSpan(idx * _featureSize, _featureSize));
                }
            });

            // Bootstrapped targets from the (frozen) target net: target = min_a [1 + cost(s')].
            // This ActionCount× batch is the dominant cost — routed through the injected forward
            // (a device-resident GPU path when provided, else the default autograd forward).
            float[] successorValues = _targetForward.Forward(successorFeatures, batch * actions);

            var targets = new float[batch];
            for (int b = 0; b < batch; b++)
            {
                float best = float.PositiveInfinity;
                for (int a = 0; a < actions; a++)
                {
                    int idx = b * actions + a;
                    float costNext = successorIsGoal[idx] ? 0f : MathF.Max(0f, successorValues[idx]) * _options.DistanceScale;
                    best = MathF.Min(best, 1f + costNext);
                }
                targets[b] = best / _options.DistanceScale; // regress in scaled units
            }

            var features = new float[batch * _featureSize];
            Parallel.For(0, batch, b =>
                _featurize(states[b]).CopyTo(features.AsSpan(b * _featureSize, _featureSize)));

            // Train step: resident on-device (Stage 3) when injected, else the autograd path.
            float stepLoss;
            if (_residentTrain is not null)
            {
                stepLoss = _residentTrain.Step(features, targets, batch);
            }
            else
            {
                _adam.ZeroGrad();
                var predicted = _net.Forward(new Tensor(features, batch, _featureSize)).Reshape(batch);
                var loss = predicted.HuberLoss(new Tensor(targets, batch), _options.HuberDelta);
                loss.Backward();
                _adam.ClipGradNorm(_options.GradClipNorm);
                _adam.Step();
                stepLoss = loss.Data[0];
            }

            // Sync the bootstrap target on schedule — and, when an ε-loss threshold is set, only once the
            // online net has converged on it (loss < ε), with a max-interval fallback so it can't freeze.
            if ((it + 1) % _options.TargetUpdateInterval == 0 &&
                (_options.TargetUpdateLossThreshold <= 0f
                 || stepLoss < _options.TargetUpdateLossThreshold
                 || itersSinceSync >= maxSyncInterval))
            {
                itersSinceSync = 0;
                _residentTrain?.SyncToHost();        // resident weights → CPU net before copying to the target
                _target.CopyFrom(_net);
                _targetForward.OnTargetSynced(_target); // re-sync resident target weights (per-sync, not per-step)
            }
            onIteration?.Invoke(it, stepLoss);
        }

        // Leave the CPU net holding the freshest (resident) weights so eval / checkpointing see them.
        _residentTrain?.SyncToHost();
    }
}
