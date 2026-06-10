using RLNet.Core.Checkpoints;
using RLNet.Core.Random;
using RLNet.Environments.Game2048;

namespace RLNet.Web.Services;

/// <summary>
/// Owns the 2048 n-tuple agent: loads it from the model store, or trains it once via
/// self-play (the M5 recipe: afterstate TD(0), 100k games ≈ 3 minutes, 84% win rate)
/// and saves it. Same lifecycle as <see cref="RushHourModelService"/>.
/// </summary>
public sealed class Game2048ModelService(IModelStore store, ILogger<Game2048ModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "2048";
    public const string AlgorithmId = "ntuple";
    public const int TrainingGames = 100_000;
    public const ulong TrainingSeed = 42;

    private readonly object _lock = new();
    private NTuple2048Agent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int GamesPlayed { get; private set; }
    public double LastEvalScore { get; private set; }
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

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No 2048 model in the store — training {Games} self-play games (~3 min)...", TrainingGames);
            lock (_lock) Status = ModelStatus.Training;

            var agent = new NTuple2048Agent();
            var rng = new Xoshiro256StarStar(new SeedSequence(TrainingSeed).Derive(RngStreams.Policy));
            double recentScores = 0;
            for (int game = 1; game <= TrainingGames; game++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var (score, _) = agent.PlayGame(rng, learn: true);
                recentScores += score;
                if (game % 10_000 == 0)
                {
                    lock (_lock)
                    {
                        GamesPlayed = game;
                        LastEvalScore = recentScores / 10_000;
                    }
                    logger.LogInformation("2048 training: {Game}/{Total} games, avg score (last 10k) {Score:F0}",
                        game, TrainingGames, recentScores / 10_000);
                    recentScores = 0;
                }
            }

            store.Save(EnvironmentId, AlgorithmId, s => agent.Save(s));
            _agent = agent;
            Status = ModelStatus.Ready;
            logger.LogInformation("2048 model trained and saved.");
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "2048 model setup failed.");
        }
    }
}
