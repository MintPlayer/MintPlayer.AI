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
public sealed class CubeModelService : IModelStartupService, IDisposable
{
    public const string EnvironmentId = "cube";
    public const string AlgorithmId = "dqn";
    public const string PolicyAlgorithmId = "policy";
    public const string ValueDaviAlgorithmId = "value-davi-res";
    public const string EfficientPolicyAlgorithmId = "policy-efficient";

    public const int MaxMoves = 20;

    private readonly AdaptiveBackend _backend;
    private readonly ILogger<CubeModelService> _logger;

    // Shared by every refreshing net's reload hook and Dispose, so a resident-GPU rebuild can never race the
    // teardown of the forward it rebuilds.
    private readonly object _lock = new();
    private DeviceResidualMlp? _residentValue;
    private DeviceMlp? _residentEfficient;

    private readonly StartupCheckpoint<GreedyQAgent> _dqn;
    private readonly RefreshingCheckpoint<CubePolicyNet> _policy;
    private readonly RefreshingCheckpoint<ResidualMlp> _value;
    private readonly RefreshingCheckpoint<CubePolicyNet> _efficient;

    public CubeModelService(IModelStore store, AdaptiveBackend backend, ILogger<CubeModelService> logger)
    {
        _backend = backend;
        _logger = logger;

        _dqn = new(store, EnvironmentId, AlgorithmId,
            stream => new GreedyQAgent(MlpCheckpoint.Load(stream), RubiksCubeEnv.ActionCount),
            logger, "Rubik's Cube DQN model");
        _policy = new(store, EnvironmentId, PolicyAlgorithmId, CubePolicyNet.Load, logger, "cube policy net");
        // The value / efficient nets rebuild a resident GPU forward on each reload, under the shared _lock.
        _value = new(store, EnvironmentId, ValueDaviAlgorithmId, ResidualMlpCheckpoint.Load, logger,
            "cube DAVI value net", onReload: RebuildResidentValue, gate: _lock);
        _efficient = new(store, EnvironmentId, EfficientPolicyAlgorithmId, CubePolicyNet.Load, logger,
            "EfficientCube policy net", onReload: RebuildResidentEfficient, gate: _lock);
    }

    public ModelStatus Status => _dqn.Status;
    public string? Error => _dqn.Error;

    /// <summary>The greedy inference agent, or null while the model is not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent => _dqn.Value;

    /// <summary>
    /// The Kociemba-imitation policy/value net (preferred over the DQN when present, PLAN M16), or null if the
    /// store has none. Re-read every few minutes so a long-running Lab campaign's improving checkpoints are picked
    /// up without restarting the host.
    /// </summary>
    public CubePolicyNet? PolicyNet => _policy.Current;

    /// <summary>
    /// The teacher-free DAVI value net (the "self-taught AI" shortest-move solver, PLAN M21), or null if the store
    /// has none. Re-read on the same cadence as <see cref="PolicyNet"/>; accessing it keeps
    /// <see cref="ResidentValueForward"/> in sync (rebuilt on the GPU whenever the net reloads).
    /// </summary>
    public ResidualMlp? ValueNet => _value.Current;

    /// <summary>
    /// A device-resident GPU forward over the current value net, or null when no CUDA device is present (the solver
    /// then scores on the CPU). Kept in sync by the <see cref="ValueNet"/> getter — read that first. Thread-safe:
    /// the underlying forward serializes GPU access internally.
    /// </summary>
    public CubeValueSearch.BatchForward? ResidentValueForward => _residentValue is { } dev ? dev.Forward : null;

    /// <summary>
    /// The teacher-free EfficientCube policy net (the website's "self-taught AI", trained self-supervised on
    /// scramble reversals — no Kociemba, no value-iteration bootstrap), or null if the store has none. Solved by
    /// beam search (<see cref="CubePolicySearch.BeamSearch(System.Func{float[],int,float[]}, FaceletCube, int, int)"/>).
    /// Re-read on the same cadence as the other nets; accessing it keeps <see cref="ResidentEfficientForward"/> in
    /// sync (the policy head's weights mirrored onto the GPU whenever the net reloads).
    /// </summary>
    public CubePolicyNet? EfficientPolicyNet => _efficient.Current;

    /// <summary>
    /// A device-resident GPU forward over the current EfficientCube policy head (rows × 12 logits), or null when no
    /// CUDA device is present (beam search then scores on the CPU). Kept in sync by the
    /// <see cref="EfficientPolicyNet"/> getter — read that first. Thread-safe: the forward serializes GPU access.
    /// </summary>
    public Func<float[], int, float[]>? ResidentEfficientForward => _residentEfficient is { } dev ? dev.Forward : null;

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

    /// <summary>Loads the shipped DQN baseline at startup. The web does not train (PRD §14); the deep solver nets
    /// (policy / DAVI value / EfficientCube) load lazily through their getters.</summary>
    public void Initialize(CancellationToken cancellationToken) => _dqn.Initialize();

    // Mirror the freshly-loaded value-net weights onto the GPU as a resident forward when a CUDA device is present:
    // the host-span path is transfer-bound and barely beats CPU, so the resident forward (weights on device) is what
    // makes the deep solver fast. Runs under _lock (RefreshingCheckpoint holds the shared gate), serialized vs Dispose.
    private void RebuildResidentValue(ResidualMlp net)
    {
        _residentValue?.Dispose();
        _residentValue = _backend.Gpu is { } gpu ? gpu.CreateResidentForward(net) : null;
        _logger.LogInformation("Rebuilt cube DAVI value forward ({Mode}).",
            _residentValue is not null ? "resident GPU" : "CPU");
    }

    // Beam search runs the bulk of the forwards; mirror the policy head onto the GPU when a device is present.
    private void RebuildResidentEfficient(CubePolicyNet net)
    {
        _residentEfficient?.Dispose();
        _residentEfficient = _backend.Gpu is { } gpu ? gpu.CreateResidentForward(net.PolicyAsMlp()) : null;
        _logger.LogInformation("Rebuilt EfficientCube policy forward ({Mode}).",
            _residentEfficient is not null ? "resident GPU" : "CPU");
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
