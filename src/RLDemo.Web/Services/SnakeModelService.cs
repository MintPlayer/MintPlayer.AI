using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Snake Q-network: loads the shipped <see cref="DuelingQNet"/> checkpoint from the model store at
/// startup (trained on a dev machine via the `--game snake` Lab campaign, PLAN M22, and committed to
/// <c>models/</c> via Git LFS — the web never trains, PRD §14). Persists via <see cref="DuelingQNetCheckpoint"/>.
/// </summary>
public sealed class SnakeModelService(IModelStore store, ILogger<SnakeModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "snake";
    public const string AlgorithmId = "dqn";

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
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

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (_lock)
        {
            Status = ModelStatus.Failed;
            Error = "No trained Snake model in the store.";
        }
        logger.LogWarning("No Snake model in the store — game unavailable. Train it via `--game snake` in the Lab and commit the checkpoint to models/ (Git LFS).");
    }
}
