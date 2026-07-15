using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.SourceGenerators.Attributes;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the 2048 n-tuple agent: loads the shipped checkpoint from the model store at startup (trained on a dev
/// machine via self-play, the M5 recipe, and committed to <c>models/</c> via Git LFS — the web never trains,
/// PRD §14).
/// </summary>
[Register(ServiceLifetime.Singleton, "RLDemoWebModelServices")]
public sealed class Game2048ModelService(IModelStore store, ILogger<Game2048ModelService> logger) : IModelStartupService
{
    public const string EnvironmentId = "2048";
    public const string AlgorithmId = "ntuple";

    private readonly StartupCheckpoint<NTuple2048Agent> _model =
        new(store, EnvironmentId, AlgorithmId, NTuple2048Agent.Load, logger, "2048 n-tuple model");

    public ModelStatus Status => _model.Status;
    public string? Error => _model.Error;

    /// <summary>The trained agent, or null while not ready. Loads lazily from the store.</summary>
    public NTuple2048Agent? Agent => _model.Value;

    public void Initialize(CancellationToken cancellationToken) => _model.Initialize();
}
