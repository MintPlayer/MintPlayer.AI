using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace RLDemo.Web.Services;

public enum ModelStatus { Loading, Ready, Failed }

/// <summary>
/// A game service with startup work (load its trained checkpoint from the store, or warm a cache). The web does
/// NOT train: models are produced on a dev machine (the Lab campaigns / Console) and committed to <c>models/</c>
/// via Git LFS, so a single training path lives off the request path. A missing checkpoint = the game is
/// unavailable, not a trigger to train in-process.
/// </summary>
public interface IModelStartupService
{
    void Initialize(CancellationToken cancellationToken);
}

/// <summary>
/// Owns the Rush Hour DQN: loads it from the model store at startup (the checkpoint is trained on a dev machine
/// and shipped in <c>models/</c> via Git LFS — the web never trains, PRD §14). Also serves the imitation policy
/// net when present. Thread-safe readiness snapshot for /api/rushhour/status.
/// </summary>
public sealed class RushHourModelService(IModelStore store, ILogger<RushHourModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "rushhour";
    public const string AlgorithmId = "dqn";
    public const int MaxMoves = 60;

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
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

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (_lock)
        {
            Status = ModelStatus.Failed;
            Error = "No trained Rush Hour model in the store.";
        }
        logger.LogWarning("No Rush Hour model in the store — game unavailable. Train it on a dev machine (Lab/Console) and commit the checkpoint to models/ (Git LFS).");
    }
}

/// <summary>Runs every game's startup work (load checkpoint / warm caches) once, in parallel, off the request path.</summary>
public sealed class ModelStartupHostedService(IEnumerable<IModelStartupService> models) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => Task.WhenAll(models.Select(m => Task.Run(() => m.Initialize(stoppingToken), stoppingToken)));
}
