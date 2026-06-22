using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the 2048 n-tuple agent: loads the shipped checkpoint from the model store at startup (trained on a dev
/// machine via self-play, the M5 recipe, and committed to <c>models/</c> via Git LFS — the web never trains,
/// PRD §14).
/// </summary>
public sealed class Game2048ModelService(IModelStore store, ILogger<Game2048ModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "2048";
    public const string AlgorithmId = "ntuple";

    private readonly object _lock = new();
    private NTuple2048Agent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public string? Error { get; private set; }

    /// <summary>The trained agent, or null while not ready. Loads lazily from the store.</summary>
    public NTuple2048Agent? Agent
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
            _agent = NTuple2048Agent.Load(stream);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded 2048 n-tuple model from the store.");
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
            Error = "No trained 2048 model in the store.";
        }
        logger.LogWarning("No 2048 model in the store — game unavailable. Train it on a dev machine and commit the checkpoint to models/ (Git LFS).");
    }
}
