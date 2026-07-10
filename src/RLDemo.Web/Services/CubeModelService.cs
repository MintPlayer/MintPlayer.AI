using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rubik's Cube nets: loads the shipped DQN baseline plus the imitation-policy, teacher-free DAVI value,
/// and EfficientCube policy nets from the model store (all trained on a dev machine via the Lab campaigns and
/// committed to <c>models/</c> via Git LFS — the web never trains, PRD §14). The three deep-solver nets hot-reload
/// during a training campaign via <see cref="RefreshingModel{T}"/>; for the two GPU-scored nets the reload hook
/// rebuilds a device-resident forward (weights on device — the host-span path is transfer-bound and barely beats
/// CPU, so the resident forward is what makes the deep solvers fast).
/// </summary>
public sealed class CubeModelService : StartupModelService, IDisposable
{
    public const string EnvironmentId = "cube";
    public const string AlgorithmId = "dqn";
    public const string PolicyAlgorithmId = "policy";
    public const string ValueDaviAlgorithmId = "value-davi-res";
    public const string EfficientPolicyAlgorithmId = "policy-efficient";

    public const int MaxMoves = 20;

    private readonly IModelStore _store;
    private readonly AdaptiveBackend _backend;
    private readonly ILogger<CubeModelService> _logger;

    private GreedyQAgent? _agent;
    private readonly RefreshingModel<CubePolicyNet> _policy;
    private readonly RefreshingModel<ResidualMlp> _value;
    private readonly RefreshingModel<CubePolicyNet> _efficient;
    private DeviceResidualMlp? _residentValue;
    private DeviceMlp? _residentEfficient;

    public CubeModelService(IModelStore store, AdaptiveBackend backend, ILogger<CubeModelService> logger) : base(logger)
    {
        _store = store;
        _backend = backend;
        _logger = logger;
        _policy = new(store, EnvironmentId, PolicyAlgorithmId, CubePolicyNet.Load, logger, "cube policy net");
        _value = new(store, EnvironmentId, ValueDaviAlgorithmId, ResidualMlpCheckpoint.Load, logger, "cube DAVI value net",
            onLoaded: RebuildResidentValue);
        _efficient = new(store, EnvironmentId, EfficientPolicyAlgorithmId, CubePolicyNet.Load, logger, "EfficientCube policy net",
            onLoaded: RebuildResidentEfficient);
    }

    protected override string ModelName => "Rubik's Cube";

    /// <summary>The greedy inference agent, or null while the model is not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    /// <summary>
    /// The Kociemba-imitation policy/value net (preferred over the DQN when present, PLAN M16), or null if the store
    /// has none. Re-read every few minutes so a long-running Lab campaign's checkpoints are picked up live.
    /// </summary>
    public CubePolicyNet? PolicyNet => _policy.Value;

    /// <summary>
    /// The teacher-free DAVI value net (the "self-taught AI" shortest-move solver, PLAN M21), or null if the store
    /// has none. Reading it also keeps <see cref="ResidentValueForward"/> in sync (rebuilt on the GPU on reload).
    /// </summary>
    public ResidualMlp? ValueNet => _value.Value;

    /// <summary>
    /// A device-resident GPU forward over the current value net, or null when no CUDA device is present (the solver
    /// then scores on the CPU). Touching it applies any pending reload first. Thread-safe: the forward serializes
    /// GPU access internally.
    /// </summary>
    public CubeValueSearch.BatchForward? ResidentValueForward
    {
        get { _ = ValueNet; return _residentValue is { } dev ? dev.Forward : null; }
    }

    /// <summary>
    /// The teacher-free EfficientCube policy net (the website's "self-taught AI", trained self-supervised on
    /// scramble reversals — no Kociemba, no value-iteration bootstrap), or null if the store has none. Solved by
    /// beam search; reading it keeps <see cref="ResidentEfficientForward"/> in sync.
    /// </summary>
    public CubePolicyNet? EfficientPolicyNet => _efficient.Value;

    /// <summary>
    /// A device-resident GPU forward over the current EfficientCube policy head (rows × 12 logits), or null when no
    /// CUDA device is present (beam search then scores on the CPU). Touching it applies any pending reload first.
    /// </summary>
    public Func<float[], int, float[]>? ResidentEfficientForward
    {
        get { _ = EfficientPolicyNet; return _residentEfficient is { } dev ? dev.Forward : null; }
    }

    private void RebuildResidentValue(ResidualMlp net)
    {
        // Serialized against Dispose via Lock (both are the only writer/disposer of _residentValue).
        lock (Lock)
        {
            _residentValue?.Dispose();
            _residentValue = _backend.Gpu is { } gpu ? gpu.CreateResidentForward(net) : null;
        }
        _logger.LogInformation("Cube DAVI value net scored on {Mode}.", _residentValue is not null ? "resident GPU forward" : "CPU");
    }

    private void RebuildResidentEfficient(CubePolicyNet net)
    {
        lock (Lock)
        {
            _residentEfficient?.Dispose();
            // Beam search runs the bulk of the forwards; mirror the policy head onto the GPU when a device is present.
            _residentEfficient = _backend.Gpu is { } gpu ? gpu.CreateResidentForward(net.PolicyAsMlp()) : null;
        }
        _logger.LogInformation("EfficientCube policy net scored on {Mode}.", _residentEfficient is not null ? "resident GPU forward" : "CPU");
    }

    public override bool TryLoadFromStore()
    {
        lock (Lock)
        {
            if (_agent is not null) return true;
            using var stream = _store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var network = MlpCheckpoint.Load(stream);
            _agent = new GreedyQAgent(network, RubiksCubeEnv.ActionCount);
            Status = ModelStatus.Ready;
            _logger.LogInformation("Loaded Rubik's Cube model from the store.");
            return true;
        }
    }

    /// <summary>
    /// Greedy rollout on a drawn cube: up to <see cref="MaxMoves"/> quarter-turns under
    /// the same no-undo mask the agent was trained with, reported honestly whether they
    /// solve it or not.
    /// </summary>
    public static (bool Solved, List<string> Moves) Rollout(GreedyQAgent agent, FaceletCube start)
    {
        var cube = start.Clone();
        var moves = new List<string>(MaxMoves);
        var observation = new float[RubiksCubeEnv.ObservationSize];
        int lastAction = -1;

        for (int i = 0; i < MaxMoves && !cube.IsSolved; i++)
        {
            RubiksCubeEnv.WriteObservation(cube, observation);
            int action = agent.Act(observation, RubiksCubeEnv.ActionMask(lastAction), greedy: true);
            cube.ApplyQuarterTurn(action);
            moves.Add(FaceletCube.QuarterTurnMoves[action]);
            lastAction = action;
        }

        return (cube.IsSolved, moves);
    }

    public void Dispose()
    {
        lock (Lock)
        {
            _residentValue?.Dispose();
            _residentValue = null;
            _residentEfficient?.Dispose();
            _residentEfficient = null;
        }
    }
}
