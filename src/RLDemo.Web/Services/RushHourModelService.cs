using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rush Hour DQN: loads it from the model store at startup (the checkpoint is trained on a dev machine
/// and shipped in <c>models/</c> via Git LFS — the web never trains, PRD §14). Also serves the imitation policy
/// net when present. Thread-safe readiness snapshot for /api/rushhour/status.
/// </summary>
public sealed class RushHourModelService(IModelStore store, ILogger<RushHourModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "rushhour";
    public const string AlgorithmId = "dqn";
    public const string PolicyAlgorithmId = "policy";
    public const int MaxMoves = 60;

    private readonly StartupCheckpoint<GreedyQAgent> _dqn = new(
        store, EnvironmentId, AlgorithmId,
        stream => new GreedyQAgent(MlpCheckpoint.Load(stream), RushHourBoard.ActionCount),
        logger, "Rush Hour model");

    private readonly RefreshingCheckpoint<RushHourPolicyNet> _policy = new(
        store, EnvironmentId, PolicyAlgorithmId, RushHourPolicyNet.Load, logger, "Rush Hour policy net");

    public ModelStatus Status => _dqn.Status;
    public string? Error => _dqn.Error;

    /// <summary>The greedy inference agent, or null while the model is not ready. Loads lazily from the store.</summary>
    public GreedyQAgent? Agent => _dqn.Value;

    /// <summary>
    /// The imitation-learned policy/value net (preferred over the DQN when present), or null if the store has
    /// none. Re-read every few minutes so a long-running training campaign's improving checkpoints are picked up
    /// without restarting the host.
    /// </summary>
    public RushHourPolicyNet? PolicyNet => _policy.Current;

    /// <summary>Loads the shipped checkpoint at startup. The web does not train (PRD §14).</summary>
    public void Initialize(CancellationToken cancellationToken) => _dqn.Initialize();
}
