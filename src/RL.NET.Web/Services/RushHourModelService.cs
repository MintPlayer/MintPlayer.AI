using RLNet.Core.Checkpoints;
using RLNet.Core.Nn;
using RLNet.Core.Random;
using RLNet.Core.Schedules;
using RLNet.Core.Training;
using RLNet.Environments.RushHour;

namespace RLNet.Web.Services;

public enum ModelStatus { Loading, Training, Ready, Failed }

/// <summary>
/// Owns the Rush Hour DQN: loads it from the model store at startup, or trains it once
/// (same recipe that passed the M6 gate) and saves it — so the app never trains again
/// across restarts (PRD §7.5). Thread-safe snapshot of training progress for /api/rushhour/status.
/// </summary>
public sealed class RushHourModelService(IModelStore store, ILogger<RushHourModelService> logger)
{
    public const string EnvironmentId = "rushhour";
    public const string AlgorithmId = "dqn";
    public const int MaxMoves = 60;

    // The generated training set: same parameters that passed the M6 gate (100% within 2× optimal).
    public const int PuzzleSetSeed = 99;
    public const ulong TrainingMasterSeed = 42;

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

            var puzzles = RushHourGenerator.Generate(PuzzleSetSeed, count: 30, minOptimal: 4, maxOptimal: 10);
            var env = new RushHourEnv(puzzles, MaxMoves);
            var result = DqnTrainer.Train(env, new DqnOptions
            {
                Hidden = [128, 128],
                Gamma = 0.98,
                LearningRate = 5e-4f,
                MaxSteps = 200_000,
                BufferCapacity = 100_000,
                Epsilon = new LinearSchedule(1.0, 0.05, 60_000),
                EvalEvery = 10_000,
                EvalEpisodes = 20,
                SolveThreshold = 88,
                OnProgress = p =>
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
                },
            }, new SeedSequence(TrainingMasterSeed));

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

/// <summary>Runs the load-or-train once, off the request path, at application startup.</summary>
public sealed class ModelTrainingHostedService(RushHourModelService model) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => Task.Run(() => model.EnsureModel(stoppingToken), stoppingToken);
}
