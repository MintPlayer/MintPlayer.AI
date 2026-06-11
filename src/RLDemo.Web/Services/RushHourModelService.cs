using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace RLDemo.Web.Services;

public enum ModelStatus { Loading, Training, Ready, Failed }

/// <summary>A model that can load itself from the store or train once at startup.</summary>
public interface ITrainableModelService
{
    void EnsureModel(CancellationToken cancellationToken);
}

/// <summary>
/// Owns the Rush Hour DQN: loads it from the model store at startup, or trains it once
/// (same recipe that passed the M6 gate) and saves it — so the app never trains again
/// across restarts (PRD §7.5). Thread-safe snapshot of training progress for /api/rushhour/status.
/// </summary>
public sealed class RushHourModelService(IModelStore store, ILogger<RushHourModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "rushhour";
    public const string AlgorithmId = "dqn";
    public const int MaxMoves = 60;

    // Unlike the M6 gate (fixed 30-puzzle set), this model must handle ANYTHING a user
    // draws — so it trains on a procedurally-scaled set: ~2,000 generated puzzles across
    // the whole easy-medium band, sparse layouts (down to 2 vehicles) and both red
    // lengths. A small fixed set gets memorized; variety is what buys generalization.
    public const int PuzzleSetSeed = 99;
    public const ulong TrainingMasterSeed = 42;

    // Band up to 20 covers the official deck's beginner cards (e.g. card 1 = optimal 16);
    // random generation can't reach expert depths (card 40 = 81) — that needs the M11
    // imitation/search work, not a wider band.
    public static List<RushHourPuzzle> TrainingPuzzles()
        => RushHourGenerator.Generate(PuzzleSetSeed, count: 3000, minOptimal: 2, maxOptimal: 20,
            minVehicles: 2, maxVehicles: 9, maxAttempts: 4_000_000, varyRedLength: true);

    public static DqnOptions TrainingOptions(Action<DqnProgress>? onProgress = null) => new()
    {
        Hidden = [256, 256],
        Gamma = 0.98,
        LearningRate = 5e-4f,
        MaxSteps = 600_000,
        BufferCapacity = 100_000,
        Epsilon = new LinearSchedule(1.0, 0.05, 200_000),
        EvalEvery = 10_000,
        EvalEpisodes = 40,
        SolveThreshold = 90, // perfect play on this band averages ≈ 91 (return = 101 − moves)
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
    private RushHourPolicyNet? _policyNet;
    private DateTime _policyLoadedUtc = DateTime.MinValue;
    private static readonly TimeSpan PolicyRefresh = TimeSpan.FromMinutes(5);

    /// <summary>
    /// The imitation-learned policy/value net (preferred over the DQN when present), or
    /// null if the store has none. Re-read every few minutes so a long-running training
    /// campaign's improving checkpoints are picked up without restarting the host.
    /// </summary>
    public RushHourPolicyNet? PolicyNet
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
                        _policyNet = RushHourPolicyNet.Load(stream);
                        logger.LogInformation("Loaded Rush Hour policy net from the store.");
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break solving; keep the previous net.
                    logger.LogWarning(ex, "Failed to (re)load the Rush Hour policy net; keeping the previous one.");
                }
                _policyLoadedUtc = DateTime.UtcNow;
                return _policyNet;
            }
        }
    }

    /// <summary>Loads a stored checkpoint if one exists; safe to call at any time.</summary>
    public bool TryLoadFromStore()
    {
        lock (_lock)
        {
            if (_agent is not null) return true;
            using var stream = store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var network = MlpCheckpoint.Load(stream);
            _agent = new GreedyQAgent(network, RushHourBoard.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded Rush Hour model from the store.");
            return true;
        }
    }

    /// <summary>Loads the model from the store, or trains + saves it. Called once at startup.</summary>
    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No Rush Hour model in the store — training (~1 min)...");
            lock (_lock) Status = ModelStatus.Training;

            var env = new RushHourEnv(TrainingPuzzles(), MaxMoves);
            var result = DqnTrainer.Train(env, TrainingOptions(p =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                lock (_lock)
                {
                    TrainingStep = p.Step;
                    TrainingMaxSteps = p.MaxSteps;
                    LastEvalReturn = p.EvalMeanReturn;
                }
                logger.LogInformation("Rush Hour training: step {Step}/{Max}, eval {Eval:F1}",
                    p.Step, p.MaxSteps, p.EvalMeanReturn);
            }), new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => MlpCheckpoint.Save(result.Network, s));
            _agent = new GreedyQAgent(result.Network, RushHourBoard.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Rush Hour model trained ({Steps} steps, eval {Eval:F1}) and saved.",
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
            logger.LogError(ex, "Rush Hour model setup failed.");
        }
    }
}

/// <summary>Runs every model's load-or-train once, in parallel, off the request path, at startup.</summary>
public sealed class ModelTrainingHostedService(IEnumerable<ITrainableModelService> models) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => Task.WhenAll(models.Select(m => Task.Run(() => m.EnsureModel(stoppingToken), stoppingToken)));
}
