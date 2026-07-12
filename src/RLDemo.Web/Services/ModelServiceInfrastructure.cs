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

/// <summary>Runs every game's startup work (load checkpoint / warm caches) once, in parallel, off the request path.</summary>
public sealed class ModelStartupHostedService(IEnumerable<IModelStartupService> models) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => Task.WhenAll(models.Select(m => Task.Run(() => m.Initialize(stoppingToken), stoppingToken)));
}

/// <summary>
/// A trained model loaded once from the store and then served read-only for a game's lifetime: it owns the
/// readiness snapshot (<see cref="Status"/> / <see cref="Error"/>) behind the <c>/status</c> endpoint, loads
/// lazily on first access (or eagerly at startup via <see cref="Initialize"/>), and turns a missing checkpoint
/// into "unavailable" rather than an exception (the web never trains — the checkpoint ships in <c>models/</c>).
/// Composed into a game's model service, so one service can hold several (e.g. a DQN baseline alongside the
/// refreshing deep-solver nets); the load recipe is the only per-game knowledge, passed as <paramref name="load"/>.
/// </summary>
public sealed class StartupCheckpoint<T>(
    IModelStore store, string environmentId, string algorithmId, Func<Stream, T> load,
    ILogger logger, string modelName) where T : class
{
    private readonly object _lock = new();
    private T? _value;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public string? Error { get; private set; }

    /// <summary>The loaded model, or null while unavailable. Loads lazily from the store on first access.</summary>
    public T? Value
    {
        get
        {
            if (_value is null && Status == ModelStatus.Loading) TryLoad();
            return _value;
        }
    }

    /// <summary>Loads the stored checkpoint if one exists; safe to call any time. Returns whether a model is loaded.</summary>
    public bool TryLoad()
    {
        lock (_lock)
        {
            if (_value is not null) return true;
            using var stream = store.TryOpenRead(environmentId, algorithmId);
            if (stream is null) return false;
            _value = load(stream);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded {ModelName} from the store.", modelName);
            return true;
        }
    }

    /// <summary>Startup: load the shipped checkpoint, or mark the game unavailable and warn how to provide one.</summary>
    public void Initialize()
    {
        if (TryLoad()) return;
        lock (_lock)
        {
            Status = ModelStatus.Failed;
            Error = $"No trained {modelName} in the store.";
        }
        logger.LogWarning(
            "No {ModelName} in the store — game unavailable. Train it on a dev machine and commit the checkpoint to models/ (Git LFS).",
            modelName);
    }
}

/// <summary>
/// A single trained net kept fresh from the model store: the <see cref="Current"/> getter re-reads the checkpoint
/// on a cadence so a long-running Lab campaign's improving weights are picked up without restarting the host,
/// while a corrupt or mid-write read is swallowed and the previous net kept — the caller only ever sees a usable
/// net or null, never an exception (the error is defined out of existence). The re-read is double-checked against
/// a lock so concurrent readers do at most one load per cadence. <paramref name="onReload"/> runs under that lock
/// right after a successful load — Cube uses it to mirror the fresh weights onto a resident GPU forward, and
/// passes its own service lock as <paramref name="gate"/> so the rebuild serializes against <c>Dispose</c>.
/// </summary>
public sealed class RefreshingCheckpoint<T>(
    IModelStore store, string environmentId, string algorithmId, Func<Stream, T> load,
    ILogger logger, string name, TimeSpan? refresh = null, Action<T>? onReload = null, object? gate = null)
    where T : class
{
    private readonly object _lock = gate ?? new object();
    private readonly TimeSpan _refresh = refresh ?? TimeSpan.FromMinutes(5);
    private T? _value;
    private DateTime _loadedUtc = DateTime.MinValue;

    /// <summary>The current net, or null until the store first has one. Re-reads on the cadence; never throws.</summary>
    public T? Current
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
                        onReload?.Invoke(_value);
                        logger.LogInformation("Loaded {Name} from the store.", name);
                    }
                }
                catch (Exception ex)
                {
                    // A mid-write or corrupt checkpoint must not break solving; keep the previous net.
                    logger.LogWarning(ex, "Failed to (re)load {Name}; keeping the previous one.", name);
                }
                _loadedUtc = DateTime.UtcNow;
                return _value;
            }
        }
    }
}
