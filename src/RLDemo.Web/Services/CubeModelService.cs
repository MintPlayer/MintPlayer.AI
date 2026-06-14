using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using MintPlayer.AI.ReinforcementLearning.Ilgpu;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rubik's Cube DQN: loads it from the model store at startup, or trains it
/// once on the shallow-scramble curriculum (depths 1–6, the PRD §11 band — full
/// 20-move scrambles are deliberately out of scope for v1 RL) and saves it. Same
/// lifecycle as <see cref="RushHourModelService"/>.
/// </summary>
public sealed class CubeModelService(IModelStore store, AdaptiveBackend backend, ILogger<CubeModelService> logger) : ITrainableModelService, IDisposable
{
    public const string EnvironmentId = "cube";
    public const string AlgorithmId = "dqn";

    /// <summary>The trained band: scrambles this deep are the gate; deeper is honest failure.</summary>
    public const int MaxScrambleDepth = 6;
    public const int MaxMoves = 20;
    public const ulong TrainingMasterSeed = 42;

    public static DqnOptions TrainingOptions(Action<DqnProgress>? onProgress = null) => new()
    {
        Hidden = [256, 256],
        Gamma = 0.99,
        LearningRate = 5e-4f,
        MaxSteps = 600_000,
        BufferCapacity = 200_000,
        Epsilon = new LinearSchedule(1.0, 0.05, 200_000),
        EvalEvery = 10_000,
        EvalEpisodes = 100,
        // Perfect play on the 1–6 band averages ≈ 97.5 (return = 101 − moves); 90%
        // solved with the rest timing out (−20) sits near 85 — gate the run above that.
        SolveThreshold = 88,
        OnProgress = onProgress,
    };

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int TrainingStep { get; private set; }
    public int TrainingMaxSteps { get; private set; }
    public double LastEvalReturn { get; private set; }
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

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No Rubik's Cube model in the store — training (a few minutes)...");
            lock (_lock) Status = ModelStatus.Training;

            var env = new RubiksCubeEnv(MaxScrambleDepth, MaxMoves);
            var result = DqnTrainer.Train(env, TrainingOptions(p =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                lock (_lock)
                {
                    TrainingStep = p.Step;
                    TrainingMaxSteps = p.MaxSteps;
                    LastEvalReturn = p.EvalMeanReturn;
                }
                logger.LogInformation("Rubik's Cube training: step {Step}/{Max}, eval {Eval:F1}",
                    p.Step, p.MaxSteps, p.EvalMeanReturn);
            }), new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => MlpCheckpoint.Save(result.Network, s));
            _agent = new GreedyQAgent(result.Network, RubiksCubeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Rubik's Cube model trained ({Steps} steps, eval {Eval:F1}) and saved.",
                result.StepsTrained, result.FinalEvalReturn);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "Rubik's Cube model setup failed.");
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
        lock (_lock)
        {
            _residentValue?.Dispose();
            _residentValue = null;
        }
    }
}
