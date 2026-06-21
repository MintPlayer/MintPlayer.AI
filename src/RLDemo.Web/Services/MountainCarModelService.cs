using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the MountainCar policy: loads the shipped PPO actor from the model store at startup (trained on a dev
/// machine, PLAN M22, and committed to <c>models/</c> via Git LFS — the web never trains, PRD §14).
/// </summary>
public sealed class MountainCarModelService(IModelStore store, ILogger<MountainCarModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "mountaincar";
    public const string AlgorithmId = "ppo";
    public const ulong AgentSeed = 1; // seeds the loaded policy's sampling RNG

    private readonly object _lock = new();
    private PolicyAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
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
            _agent = new PolicyAgent(actor, new Xoshiro256StarStar(AgentSeed));
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded MountainCar policy from the store.");
            return true;
        }
    }

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (_lock)
        {
            Status = ModelStatus.Failed;
            Error = "No trained MountainCar model in the store.";
        }
        logger.LogWarning("No MountainCar model in the store — game unavailable. Train it on a dev machine and commit the checkpoint to models/ (Git LFS).");
    }
}
