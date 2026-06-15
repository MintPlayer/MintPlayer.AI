using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the MountainCar policy: loads it from the model store at startup, or trains one once with PPO
/// (PLAN M22) and saves it. MountainCar is a sparse-reward swing-up — DQN's ε-greedy can't reach the goal,
/// so PPO (entropy-held stochastic exploration) is used, trained against an extended episode horizon so a
/// fresh policy ever sees the goal; eval/gate use the standard 200-step env.
/// </summary>
public sealed class MountainCarModelService(IModelStore store, ILogger<MountainCarModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "mountaincar";
    public const string AlgorithmId = "ppo";
    public const ulong TrainingMasterSeed = 1;
    public const int TrainingHorizon = 1_000; // extended during training only

    public static PpoOptions TrainingOptions(Action<PpoProgress>? onProgress = null) => new()
    {
        Hidden = [64, 64],
        NumEnvs = 16,
        RolloutSteps = 256,
        EntropyCoef = 0.01f,
        LearningRate = 3e-4f,
        ParallelEnvs = true,
        TotalSteps = 600_000,
        EvalEpisodes = 50,
        SolveThreshold = -110.0,
        OnProgress = onProgress,
    };

    private readonly object _lock = new();
    private PolicyAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int TrainingStep { get; private set; }
    public int TrainingMaxSteps { get; private set; }
    public double LastEvalReturn { get; private set; }
    public string? Error { get; private set; }

    /// <summary>The greedy policy agent, or null while not ready. Loads lazily from the store.</summary>
    public PolicyAgent? Agent
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
            var actor = MlpCheckpoint.Load(stream);
            _agent = new PolicyAgent(actor, new Xoshiro256StarStar(TrainingMasterSeed));
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded MountainCar policy from the store.");
            return true;
        }
    }

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No MountainCar model in the store — training with PPO (a few minutes)...");
            lock (_lock) Status = ModelStatus.Training;

            var result = PpoTrainer.Train(
                _ => new MountainCarEnv(maxEpisodeSteps: TrainingHorizon, shapeReward: true), // shaping makes the swing-up learnable
                new MountainCarEnv(), // gate on the standard 200-step, unshaped env
                TrainingOptions(p =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    lock (_lock)
                    {
                        TrainingStep = p.EnvSteps;
                        TrainingMaxSteps = p.TotalSteps;
                        LastEvalReturn = p.EvalMeanReturn;
                    }
                    logger.LogInformation("MountainCar training: {Steps}/{Max}, eval {Eval:F1}", p.EnvSteps, p.TotalSteps, p.EvalMeanReturn);
                }),
                new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => MlpCheckpoint.Save(result.Actor, s));
            _agent = result.Agent;
            Status = ModelStatus.Ready;
            logger.LogInformation("MountainCar model trained ({Steps} steps, eval {Eval:F1}) and saved.", result.StepsTrained, result.FinalEvalReturn);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "MountainCar model setup failed.");
        }
    }
}
