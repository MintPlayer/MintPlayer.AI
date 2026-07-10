using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

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
/// Base for a game's model service: owns the readiness snapshot (<see cref="Status"/>/<see cref="Error"/>) and the
/// standard "load the shipped checkpoint at startup, else mark the game unavailable" <see cref="Initialize"/> flow.
/// Subclasses implement <see cref="TryLoadFromStore"/> — load the primary checkpoint, build the typed inference
/// object, set <see cref="Status"/> to Ready and return true — and expose that object through their own typed
/// getter (the type differs per game, so it stays in the subclass). Secondary nets that hot-reload during a
/// training campaign belong in a <see cref="RefreshingModel{T}"/>.
/// </summary>
public abstract class StartupModelService(ILogger logger) : IModelStartupService
{
    protected readonly object Lock = new();

    public ModelStatus Status { get; protected set; } = ModelStatus.Loading;
    public string? Error { get; protected set; }

    /// <summary>Display name used in the "no model" failure message/log (e.g. "Rush Hour", "2048").</summary>
    protected abstract string ModelName { get; }

    /// <summary>Load the primary checkpoint if present; build the inference object, set <see cref="Status"/>=Ready and return true. Safe to call any time.</summary>
    public abstract bool TryLoadFromStore();

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (Lock)
        {
            Status = ModelStatus.Failed;
            Error = $"No trained {ModelName} model in the store.";
        }
        logger.LogWarning("No {ModelName} model in the store — game unavailable. Train it on a dev machine (Lab/Console) and commit the checkpoint to models/ (Git LFS).", ModelName);
    }
}

/// <summary>
/// A checkpoint that is (re)read from the model store on a fixed cadence, so a long-running training campaign's
/// improving checkpoints are picked up without restarting the host. <see cref="Value"/> re-reads at most once per
/// refresh interval, under a double-checked lock; a mid-write or corrupt checkpoint is swallowed and the previous
/// value kept, so serving never breaks on a bad read. The optional <c>onLoaded</c> hook runs right after each
/// successful (re)load — used to mirror freshly-loaded weights onto the GPU as a resident forward. Returns null
/// until the store first has the checkpoint.
/// </summary>
public sealed class RefreshingModel<T>(
    IModelStore store,
    string environmentId,
    string algorithmId,
    Func<Stream, T> load,
    ILogger logger,
    string name,
    TimeSpan? refresh = null,
    Action<T>? onLoaded = null) where T : class
{
    private readonly TimeSpan _refresh = refresh ?? TimeSpan.FromMinutes(5);
    private readonly object _lock = new();
    private T? _value;
    private DateTime _loadedUtc = DateTime.MinValue;

    public T? Value
    {
        get
        {
            if (DateTime.UtcNow - _loadedUtc < _refresh) return _value;
            lock (_lock)
            {
                if (DateTime.UtcNow - _loadedUtc < _refresh) return _value;
                try
                {
                    using var stream = store.TryOpenRead(environmentId, algorithmId);
                    if (stream is not null)
                    {
                        _value = load(stream);
                        onLoaded?.Invoke(_value);
                        logger.LogInformation("Loaded {Name} from the store.", name);
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break serving; keep the previous value.
                    logger.LogWarning(ex, "Failed to (re)load {Name}; keeping the previous one.", name);
                }
                _loadedUtc = DateTime.UtcNow;
                return _value;
            }
        }
    }
}

/// <summary>Runs every game's startup work (load checkpoint / warm caches) once, in parallel, off the request path.</summary>
public sealed class ModelStartupHostedService(IEnumerable<IModelStartupService> models) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => Task.WhenAll(models.Select(m => Task.Run(() => m.Initialize(stoppingToken), stoppingToken)));
}
