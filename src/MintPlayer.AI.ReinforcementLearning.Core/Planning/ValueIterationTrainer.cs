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
    private readonly Mlp _net;
    private readonly Mlp _target;
    private readonly Adam _adam;
    private readonly ValueIterationOptions _options;
    private readonly int _featureSize;

    /// <param name="valueNet">Scalar-output MLP (… → 1) predicting scaled cost-to-go.</param>
    /// <param name="optimizer">
    /// The Adam optimizer over <paramref name="valueNet"/>'s parameters. Passed in (not created
    /// internally) so a campaign can persist and restore its moment estimates across a resume —
    /// without them, a resumed run spends its first steps re-estimating gradient statistics.
    /// </param>
    public ValueIterationTrainer(
        IDeterministicModel<TState> model, Func<TState, float[]> featurize, Mlp valueNet, Adam optimizer, ValueIterationOptions options)
    {
        _model = model;
        _featurize = featurize;
        _net = valueNet;
        _adam = optimizer;
        _options = options;
        _featureSize = valueNet.Sizes[0];

        // Independent bootstrapping target, kept structurally identical and periodically synced.
        // On resume it starts equal to the loaded net (as if just synced) — harmless.
        _target = new Mlp(valueNet.Sizes, new Xoshiro256StarStar(0), valueNet.HiddenActivation);
        _target.CopyFrom(valueNet);
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

    /// <summary>A greedy solver bound to this trainer's current value net.</summary>
    public IReadOnlyList<int>? Solve(TState start, int maxSteps)
        => GreedyValuePlanner.Solve(_model, Value, start, maxSteps);

    /// <summary>
    /// A weighted-A* solver bound to this trainer's value net — reaches states the greedy policy
    /// gets stuck on, at the cost of expanding a frontier. <paramref name="weight"/> &gt; 1 trades
    /// optimality for depth/speed.
    /// </summary>
    public IReadOnlyList<int>? SolveWithSearch(TState start, int maxExpansions, float weight = 1f)
        => ValueGuidedSearch.Solve(_model, Value, start, maxExpansions, weight);

    /// <summary>
    /// Run <paramref name="iterations"/> DAVI updates. <paramref name="sampleState"/> draws a
    /// training state each call (typically a random scramble from the goal). <paramref name="onIteration"/>
    /// receives (iteration, batch-mean Huber loss) for progress logging.
    /// </summary>
    public void Train(Func<TState> sampleState, int iterations, Action<int, float>? onIteration = null)
    {
        int batch = _options.BatchSize, actions = _model.ActionCount;

        for (int it = 0; it < iterations; it++)
        {
            var states = new TState[batch];
            for (int b = 0; b < batch; b++) states[b] = sampleState();

            // Successor features for every (state, action) pair, plus which successors are goals.
            var successorFeatures = new float[batch * actions * _featureSize];
            var successorIsGoal = new bool[batch * actions];
            for (int b = 0; b < batch; b++)
                for (int a = 0; a < actions; a++)
                {
                    var next = _model.Apply(states[b], a);
                    int idx = b * actions + a;
                    successorIsGoal[idx] = _model.IsGoal(next);
                    _featurize(next).CopyTo(successorFeatures.AsSpan(idx * _featureSize, _featureSize));
                }

            // Bootstrapped targets from the (frozen) target net: target = min_a [1 + cost(s')].
            float[] successorValues;
            using (GradMode.NoGrad())
                successorValues = _target.Forward(new Tensor(successorFeatures, batch * actions, _featureSize)).Data.ToArray();

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
            for (int b = 0; b < batch; b++)
                _featurize(states[b]).CopyTo(features.AsSpan(b * _featureSize, _featureSize));

            _adam.ZeroGrad();
            var predicted = _net.Forward(new Tensor(features, batch, _featureSize)).Reshape(batch);
            var loss = predicted.HuberLoss(new Tensor(targets, batch), _options.HuberDelta);
            loss.Backward();
            _adam.ClipGradNorm(_options.GradClipNorm);
            _adam.Step();

            if ((it + 1) % _options.TargetUpdateInterval == 0) _target.CopyFrom(_net);
            onIteration?.Invoke(it, loss.Data[0]);
        }
    }
}
