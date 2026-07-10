using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the 2048 n-tuple agent: loads the shipped checkpoint from the model store at startup (trained on a dev
/// machine via self-play, the M5 recipe, and committed to <c>models/</c> via Git LFS — the web never trains,
/// PRD §14). Startup/status plumbing lives in <see cref="StartupModelService"/>.
/// </summary>
public sealed class Game2048ModelService(IModelStore store, ILogger<Game2048ModelService> logger) : StartupModelService(logger)
{
    public const string EnvironmentId = "2048";
    public const string AlgorithmId = "ntuple";

    private NTuple2048Agent? _agent;

    protected override string ModelName => "2048";

    /// <summary>The trained agent, or null while not ready. Loads lazily from the store.</summary>
    public NTuple2048Agent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    public override bool TryLoadFromStore()
    {
        lock (Lock)
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
}
