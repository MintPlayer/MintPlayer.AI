using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the FruitCake Q-network: loads the shipped <see cref="MintPlayer.AI.ReinforcementLearning.Core.Nn.DuelingQNet"/>
/// checkpoint from the model store at startup (trained on a dev machine via the `--game fruitcake` Lab campaign and
/// committed to <c>models/</c> via Git LFS — the web never trains, PRD §14). When no checkpoint is present the
/// controller falls back to the greedy heuristic, so the "Watch AI" demo always works; a loaded net simply upgrades
/// the play.
/// </summary>
public sealed class FruitCakeModelService(IModelStore store, ILogger<FruitCakeModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "fruitcake";
    public const string AlgorithmId = "dqn";

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public string? Error { get; private set; }

    /// <summary>The greedy inference agent, or null when no (valid) checkpoint is available. Loads lazily from the store.</summary>
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
            // Reject a stale net whose input width no longer matches the observation (e.g. a persistent volume
            // holding a pre-rework checkpoint) — feeding it the current obs would throw mid-stream.
            if (net.InputSize != FruitCakeEnv.ObservationSize)
            {
                Status = ModelStatus.Failed;
                Error = $"Stored FruitCake net expects {net.InputSize} inputs but the environment produces {FruitCakeEnv.ObservationSize}; the checkpoint is stale.";
                logger.LogError("FruitCake model is incompatible ({Stored} vs {Expected} obs) — refusing to load it.", net.InputSize, FruitCakeEnv.ObservationSize);
                return false;
            }
            _agent = new GreedyQAgent(net, FruitCakeEnv.ColumnCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded FruitCake model from the store.");
            return true;
        }
    }

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken)
    {
        if (TryLoadFromStore()) return;
        lock (_lock)
        {
            // Not fatal: the controller falls back to the heuristic, so the demo still plays.
            Status = ModelStatus.Failed;
            Error = "No trained FruitCake model in the store; using the heuristic baseline.";
        }
        logger.LogInformation("No FruitCake model in the store — serving the heuristic baseline. Train via `--game fruitcake` in the Lab and commit the checkpoint to models/ (Git LFS).");
    }
}
