using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Pendulum policy: loads it from the model store at startup, or trains one once with SAC (PLAN M23)
/// and saves it. Pendulum is the SDK's continuous-control showcase — a real-valued torque rather than a
/// discrete button — so it trains with SAC (off-policy, squashed-Gaussian actor + twin critics). Only the
/// actor is served (the deterministic tanh(mean) action); the critics and temperature are training-only.
/// </summary>
public sealed class PendulumModelService(IModelStore store, ILogger<PendulumModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "pendulum";
    public const string AlgorithmId = "sac";
    public const ulong TrainingMasterSeed = 1;

    public static SacOptions TrainingOptions(Action<SacProgress>? onProgress = null) => new()
    {
        Hidden = [128, 128],
        BufferCapacity = 200_000,
        BatchSize = 256,
        WarmupSteps = 1_000,
        MaxSteps = 30_000,
        EvalEvery = 5_000,
        EvalEpisodes = 20,
        SolveThreshold = -150.0,
        OnProgress = onProgress,
    };

    private readonly object _lock = new();
    private ContinuousPolicyAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int TrainingStep { get; private set; }
    public int TrainingMaxSteps { get; private set; }
    public double LastEvalReturn { get; private set; }
    public string? Error { get; private set; }

    /// <summary>The greedy policy agent, or null while not ready. Loads lazily from the store.</summary>
    public ContinuousPolicyAgent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    private static ContinuousPolicyAgent WrapActor(Mlp actor)
    {
        var box = (BoxSpace)new PendulumEnv().ActionSpace;
        return new ContinuousPolicyAgent(actor, box.Dimensions, box.Low, box.High, new Xoshiro256StarStar(TrainingMasterSeed));
    }

    public bool TryLoadFromStore()
    {
        lock (_lock)
        {
            if (_agent is not null) return true;
            using var stream = store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            _agent = WrapActor(MlpCheckpoint.Load(stream));
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded Pendulum policy from the store.");
            return true;
        }
    }

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No Pendulum model in the store — training with SAC (a few minutes)...");
            lock (_lock) Status = ModelStatus.Training;

            var result = SacTrainer.Train(new PendulumEnv(), TrainingOptions(p =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                lock (_lock)
                {
                    TrainingStep = p.Step;
                    TrainingMaxSteps = p.MaxSteps;
                    LastEvalReturn = p.EvalMeanReturn;
                }
                logger.LogInformation("Pendulum training: step {Step}/{Max}, eval {Eval:F1}", p.Step, p.MaxSteps, p.EvalMeanReturn);
            }), new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => MlpCheckpoint.Save(result.Actor, s));
            _agent = WrapActor(result.Actor);
            Status = ModelStatus.Ready;
            logger.LogInformation("Pendulum model trained ({Steps} steps, eval {Eval:F1}) and saved.", result.StepsTrained, result.FinalEvalReturn);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "Pendulum model setup failed.");
        }
    }
}
