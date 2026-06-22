using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
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
    private IValueNet? _net;

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

    /// <summary>The loaded value network (or null while not ready) — for building a per-connection
    /// <see cref="SnakeSearchAgent"/>, the net-guided look-ahead that drives the live demo.</summary>
    public IValueNet? Net
    {
        get
        {
            if (_net is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _net;
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
            // Guard against a stale net whose input width no longer matches the observation (e.g. a persistent
            // volume holding a pre-observation-rework checkpoint). Feeding it the current obs would throw deep in
            // the live stream ("stream closed"); instead reject it cleanly so /status reports the game unavailable.
            if (net.InputSize != SnakeEnv.ObservationSize)
            {
                Status = ModelStatus.Failed;
                Error = $"Stored Snake net expects {net.InputSize} inputs but the environment produces {SnakeEnv.ObservationSize}; the checkpoint is stale.";
                logger.LogError("Snake model is incompatible ({Stored} vs {Expected} obs) — refusing to load it.", net.InputSize, SnakeEnv.ObservationSize);
                return false;
            }
            _net = net;
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
