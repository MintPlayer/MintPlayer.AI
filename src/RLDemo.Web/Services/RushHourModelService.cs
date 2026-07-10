using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rush Hour nets: loads the DQN baseline from the model store at startup and serves the imitation
/// policy/value net (hot-reloaded during a training campaign) when present. The checkpoints are trained on a dev
/// machine and shipped in <c>models/</c> via Git LFS — the web never trains (PRD §14). Startup/status plumbing
/// lives in <see cref="StartupModelService"/>; the refreshing policy net in <see cref="RefreshingModel{T}"/>.
/// </summary>
public sealed class RushHourModelService : StartupModelService
{
    public const string EnvironmentId = "rushhour";
    public const string AlgorithmId = "dqn";
    public const string PolicyAlgorithmId = "policy";
    public const int MaxMoves = 60;

    private readonly IModelStore _store;
    private readonly ILogger<RushHourModelService> _logger;
    private readonly RefreshingModel<RushHourPolicyNet> _policy;
    private GreedyQAgent? _agent;

    public RushHourModelService(IModelStore store, ILogger<RushHourModelService> logger) : base(logger)
    {
        _store = store;
        _logger = logger;
        _policy = new(store, EnvironmentId, PolicyAlgorithmId, RushHourPolicyNet.Load, logger, "Rush Hour policy net");
    }

    protected override string ModelName => "Rush Hour";

    /// <summary>The greedy inference agent, or null while the model is not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent
    {
        get
        {
            if (_agent is null && Status == ModelStatus.Loading) TryLoadFromStore();
            return _agent;
        }
    }

    /// <summary>
    /// The imitation-learned policy/value net (preferred over the DQN when present), or null if the store has none.
    /// Re-read every few minutes so a long-running training campaign's improving checkpoints are picked up without
    /// restarting the host.
    /// </summary>
    public RushHourPolicyNet? PolicyNet => _policy.Value;

    public override bool TryLoadFromStore()
    {
        lock (Lock)
        {
            if (_agent is not null) return true;
            using var stream = _store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var network = MlpCheckpoint.Load(stream);
            _agent = new GreedyQAgent(network, RushHourBoard.ActionCount);
            Status = ModelStatus.Ready;
            _logger.LogInformation("Loaded Rush Hour model from the store.");
            return true;
        }
    }
}
