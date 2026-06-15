using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Snake Q-network: loads it from the model store at startup, or trains one once with a
/// masked Double+Dueling DQN (PLAN M22) and saves it. Same lifecycle as the other game services.
/// The net is a <see cref="DuelingQNet"/> (the value/advantage split fits Snake — position value
/// dominates the near-equivalent direction choice), so it persists via <see cref="DuelingQNetCheckpoint"/>.
/// </summary>
public sealed class SnakeModelService(IModelStore store, ILogger<SnakeModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "snake";
    public const string AlgorithmId = "dqn";
    public const ulong TrainingMasterSeed = 1;

    /// <summary>Train on a small grid (fast, dense food); the compact size-invariant observation lets the
    /// net transfer to the larger demo grid (<see cref="SnakeController"/> serves a full-size <c>SnakeEnv</c>).
    /// Measured: train on 6×6 → eats ~22 food on the 12×12 demo grid (PLAN M22).</summary>
    public const int TrainingGridSize = 6;

    public static DqnOptions TrainingOptions(Action<DqnProgress>? onProgress = null) => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = [128, 128],
        Gamma = 0.99,
        LearningRate = 5e-4f,
        BufferCapacity = 100_000,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        Epsilon = new LinearSchedule(1.0, 0.05, 30_000),
        MaxSteps = 100_000, // the learning curve plateaus by ~30k; 100k is a safe margin
        EvalEvery = 25_000,
        EvalEpisodes = 20,
        OnProgress = onProgress,
    };

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int TrainingStep { get; private set; }
    public int TrainingMaxSteps { get; private set; }
    public double LastEvalReturn { get; private set; }
    public string? Error { get; private set; }

    /// <summary>The greedy inference agent, or null while not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    public bool TryLoadFromStore()
    {
        lock (_lock)
        {
            if (_agent is not null) return true;
            using var stream = store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var net = DuelingQNetCheckpoint.Load(stream);
            _agent = new GreedyQAgent(net, SnakeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded Snake model from the store.");
            return true;
        }
    }

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No Snake model in the store — training (a few minutes)...");
            lock (_lock) Status = ModelStatus.Training;

            var result = DqnTrainer.Train(new SnakeEnv(TrainingGridSize), TrainingOptions(p =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                lock (_lock)
                {
                    TrainingStep = p.Step;
                    TrainingMaxSteps = p.MaxSteps;
                    LastEvalReturn = p.EvalMeanReturn;
                }
                logger.LogInformation("Snake training: step {Step}/{Max}, eval {Eval:F2}", p.Step, p.MaxSteps, p.EvalMeanReturn);
            }), new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => DuelingQNetCheckpoint.Save((DuelingQNet)result.Network, s));
            _agent = new GreedyQAgent(result.Network, SnakeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Snake model trained ({Steps} steps) and saved.", result.StepsTrained);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "Snake model setup failed.");
        }
    }
}
