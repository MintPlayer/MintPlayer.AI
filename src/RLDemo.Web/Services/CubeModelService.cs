using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rubik's Cube nets: loads the shipped DQN baseline plus the imitation-policy, teacher-free DAVI value,
/// and EfficientCube policy nets from the model store (all trained on a dev machine via the Lab campaigns and
/// committed to <c>models/</c> via Git LFS — the web never trains, PRD §14). Holds the device-resident GPU
/// forwards for the deep solvers when a CUDA device is present.
/// </summary>
public sealed class CubeModelService(IModelStore store, AdaptiveBackend backend, ILogger<CubeModelService> logger) : IModelStartupService, IDisposable
{
    public const string EnvironmentId = "cube";
    public const string AlgorithmId = "dqn";

    public const int MaxMoves = 20;

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public string? Error { get; private set; }

    /// <summary>The greedy inference agent, or null while the model is not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    public const string PolicyAlgorithmId = "policy";
    private CubePolicyNet? _policyNet;
    private DateTime _policyLoadedUtc = DateTime.MinValue;
    private static readonly TimeSpan PolicyRefresh = TimeSpan.FromMinutes(5);

    /// <summary>
    /// The Kociemba-imitation policy/value net (preferred over the DQN when present, PLAN
    /// M16), or null if the store has none. Re-read every few minutes so a long-running
    /// Lab campaign's improving checkpoints are picked up without restarting the host.
    /// </summary>
    public CubePolicyNet? PolicyNet
    {
        get
        {
            if (DateTime.UtcNow - _policyLoadedUtc < PolicyRefresh) return _policyNet;
            lock (_lock)
            {
                if (DateTime.UtcNow - _policyLoadedUtc < PolicyRefresh) return _policyNet;
                try
                {
                    using var stream = store.TryOpenRead(EnvironmentId, PolicyAlgorithmId);
                    if (stream is not null)
                    {
                        _policyNet = CubePolicyNet.Load(stream);
                        logger.LogInformation("Loaded cube policy net from the store.");
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break solving; keep the previous net.
                    logger.LogWarning(ex, "Failed to (re)load the cube policy net; keeping the previous one.");
                }
                _policyLoadedUtc = DateTime.UtcNow;
                return _policyNet;
            }
        }
    }

    public const string ValueDaviAlgorithmId = "value-davi-res";
    private ResidualMlp? _valueNet;
    private DeviceResidualMlp? _residentValue;
    private DateTime _valueLoadedUtc = DateTime.MinValue;

    /// <summary>
    /// The teacher-free DAVI value net (the "self-taught AI" shortest-move solver, PLAN M21), or null
    /// if the store has none. Re-read on the same cadence as <see cref="PolicyNet"/> so an improving
    /// Lab campaign's checkpoints are picked up without restarting the host. Accessing this also keeps
    /// <see cref="ResidentValueForward"/> in sync (rebuilt on the GPU whenever the net reloads).
    /// </summary>
    public ResidualMlp? ValueNet
    {
        get
        {
            if (DateTime.UtcNow - _valueLoadedUtc < PolicyRefresh) return _valueNet;
            lock (_lock)
            {
                if (DateTime.UtcNow - _valueLoadedUtc < PolicyRefresh) return _valueNet;
                try
                {
                    using var stream = store.TryOpenRead(EnvironmentId, ValueDaviAlgorithmId);
                    if (stream is not null)
                    {
                        _valueNet = ResidualMlpCheckpoint.Load(stream);
                        // Mirror the freshly-loaded weights onto the GPU as a resident forward when a CUDA
                        // device is present — the host-span path is transfer-bound and barely beats CPU, so
                        // the resident forward (weights on device) is what makes the deep solver fast.
                        _residentValue?.Dispose();
                        _residentValue = backend.Gpu is { } gpu ? gpu.CreateResidentForward(_valueNet) : null;
                        logger.LogInformation("Loaded cube DAVI value net from the store ({Mode}).",
                            _residentValue is not null ? "resident GPU forward" : "CPU forward");
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break solving; keep the previous net.
                    logger.LogWarning(ex, "Failed to (re)load the cube DAVI value net; keeping the previous one.");
                }
                _valueLoadedUtc = DateTime.UtcNow;
                return _valueNet;
            }
        }
    }

    /// <summary>
    /// A device-resident GPU forward over the current value net, or null when no CUDA device is present
    /// (the solver then scores on the CPU). Kept in sync by the <see cref="ValueNet"/> getter — read that
    /// first. Thread-safe: the underlying forward serializes GPU access internally.
    /// </summary>
    public CubeValueSearch.BatchForward? ResidentValueForward => _residentValue is { } dev ? dev.Forward : null;

    public const string EfficientPolicyAlgorithmId = "policy-efficient";
    private CubePolicyNet? _efficientNet;
    private DeviceMlp? _residentEfficient;
    private DateTime _efficientLoadedUtc = DateTime.MinValue;

    /// <summary>
    /// The teacher-free EfficientCube policy net (the website's "self-taught AI", trained self-supervised on
    /// scramble reversals — no Kociemba, no value-iteration bootstrap), or null if the store has none. Solved
    /// by beam search (<see cref="CubePolicySearch.BeamSearch(System.Func{float[],int,float[]}, FaceletCube, int, int)"/>).
    /// Re-read on the same cadence as the other nets so an improving Lab campaign's checkpoints are picked up
    /// without restarting the host; accessing this keeps <see cref="ResidentEfficientForward"/> in sync (the
    /// policy head's weights mirrored onto the GPU whenever the net reloads).
    /// </summary>
    public CubePolicyNet? EfficientPolicyNet
    {
        get
        {
            if (DateTime.UtcNow - _efficientLoadedUtc < PolicyRefresh) return _efficientNet;
            lock (_lock)
            {
                if (DateTime.UtcNow - _efficientLoadedUtc < PolicyRefresh) return _efficientNet;
                try
                {
                    using var stream = store.TryOpenRead(EnvironmentId, EfficientPolicyAlgorithmId);
                    if (stream is not null)
                    {
                        _efficientNet = CubePolicyNet.Load(stream);
                        // Beam search runs the bulk of the forwards; mirror the policy head onto the GPU as a
                        // resident forward when a CUDA device is present (host-span is transfer-bound).
                        _residentEfficient?.Dispose();
                        _residentEfficient = backend.Gpu is { } gpu ? gpu.CreateResidentForward(_efficientNet.PolicyAsMlp()) : null;
                        logger.LogInformation("Loaded EfficientCube policy net from the store ({Mode}).",
                            _residentEfficient is not null ? "resident GPU forward" : "CPU forward");
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break solving; keep the previous net.
                    logger.LogWarning(ex, "Failed to (re)load the EfficientCube policy net; keeping the previous one.");
                }
                _efficientLoadedUtc = DateTime.UtcNow;
                return _efficientNet;
            }
        }
    }

    /// <summary>
    /// A device-resident GPU forward over the current EfficientCube policy head (rows × 12 logits), or null when
    /// no CUDA device is present (beam search then scores on the CPU). Kept in sync by the
    /// <see cref="EfficientPolicyNet"/> getter — read that first. Thread-safe: the forward serializes GPU access.
    /// </summary>
    public Func<float[], int, float[]>? ResidentEfficientForward => _residentEfficient is { } dev ? dev.Forward : null;

    public bool TryLoadFromStore()
    {
        lock (_lock)
        {
            if (_agent is not null) return true;
            using var stream = store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var network = MlpCheckpoint.Load(stream);
            _agent = new GreedyQAgent(network, RubiksCubeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded Rubik's Cube model from the store.");
            return true;
        }
    }

    /// <summary>Loads the shipped DQN baseline at startup. The web does not train (PRD §14); the deep solver
    /// nets (policy / DAVI value / EfficientCube) load lazily through their getters.</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (_lock)
        {
            Status = ModelStatus.Failed;
            Error = "No trained Rubik's Cube model in the store.";
        }
        logger.LogWarning("No Rubik's Cube DQN model in the store — train it on a dev machine and commit the checkpoint to models/ (Git LFS).");
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
        lock (_lock)
        {
            _residentValue?.Dispose();
            _residentValue = null;
            _residentEfficient?.Dispose();
            _residentEfficient = null;
        }
    }
}
