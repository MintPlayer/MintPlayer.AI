using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Rubik's Cube DQN: loads it from the model store at startup, or trains it
/// once on the shallow-scramble curriculum (depths 1–6, the PRD §11 band — full
/// 20-move scrambles are deliberately out of scope for v1 RL) and saves it. Same
/// lifecycle as <see cref="RushHourModelService"/>.
/// </summary>
public sealed class CubeModelService(IModelStore store, ILogger<CubeModelService> logger) : ITrainableModelService
{
    public const string EnvironmentId = "cube";
    public const string AlgorithmId = "dqn";

    /// <summary>The trained band: scrambles this deep are the gate; deeper is honest failure.</summary>
    public const int MaxScrambleDepth = 6;
    public const int MaxMoves = 20;
    public const ulong TrainingMasterSeed = 42;

    public static DqnOptions TrainingOptions(Action<DqnProgress>? onProgress = null) => new()
    {
        Hidden = [256, 256],
        Gamma = 0.99,
        LearningRate = 5e-4f,
        MaxSteps = 600_000,
        BufferCapacity = 200_000,
        Epsilon = new LinearSchedule(1.0, 0.05, 200_000),
        EvalEvery = 10_000,
        EvalEpisodes = 100,
        // Perfect play on the 1–6 band averages ≈ 97.5 (return = 101 − moves); 90%
        // solved with the rest timing out (−20) sits near 85 — gate the run above that.
        SolveThreshold = 88,
        OnProgress = onProgress,
    };

    private readonly object _lock = new();
    private GreedyQAgent? _agent;

    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public int TrainingStep { get; private set; }
    public int TrainingMaxSteps { get; private set; }
    public double LastEvalReturn { get; private set; }
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

    public bool TryLoadFromStore()
    {
        lock (_lock)
        {
            if (_agent is not null) return true;
            using var stream = store.TryOpenRead(EnvironmentId, AlgorithmId);
            if (stream is null) return false;
            var network = MlpCheckpoint.Load(stream);
            _agent = new GreedyQAgent(network, RubiksCubeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Loaded Rubik's Cube model from the store.");
            return true;
        }
    }

    public void EnsureModel(CancellationToken cancellationToken)
    {
        try
        {
            if (TryLoadFromStore()) return;

            logger.LogInformation("No Rubik's Cube model in the store — training (a few minutes)...");
            lock (_lock) Status = ModelStatus.Training;

            var env = new RubiksCubeEnv(MaxScrambleDepth, MaxMoves);
            var result = DqnTrainer.Train(env, TrainingOptions(p =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                lock (_lock)
                {
                    TrainingStep = p.Step;
                    TrainingMaxSteps = p.MaxSteps;
                    LastEvalReturn = p.EvalMeanReturn;
                }
                logger.LogInformation("Rubik's Cube training: step {Step}/{Max}, eval {Eval:F1}",
                    p.Step, p.MaxSteps, p.EvalMeanReturn);
            }), new SeedSequence(TrainingMasterSeed));

            store.Save(EnvironmentId, AlgorithmId, s => MlpCheckpoint.Save(result.Network, s));
            _agent = new GreedyQAgent(result.Network, RubiksCubeEnv.ActionCount);
            Status = ModelStatus.Ready;
            logger.LogInformation("Rubik's Cube model trained ({Steps} steps, eval {Eval:F1}) and saved.",
                result.StepsTrained, result.FinalEvalReturn);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Status = ModelStatus.Failed;
            Error = ex.Message;
            logger.LogError(ex, "Rubik's Cube model setup failed.");
        }
    }

    /// <summary>
    /// Greedy rollout on a drawn cube: up to <see cref="MaxMoves"/> quarter-turns under
    /// the same no-undo mask the agent was trained with, reported honestly whether they
    /// solve it or not.
    /// </summary>
    public static (bool Solved, List<string> Moves) Rollout(GreedyQAgent agent, FaceletCube start)
    {
        var cube = start.Clone();
        var moves = new List<string>(MaxMoves);
        var observation = new float[RubiksCubeEnv.ObservationSize];
        int lastAction = -1;

        for (int i = 0; i < MaxMoves && !cube.IsSolved; i++)
        {
            RubiksCubeEnv.WriteObservation(cube, observation);
            int action = agent.Act(observation, RubiksCubeEnv.ActionMask(lastAction), greedy: true);
            cube.ApplyQuarterTurn(action);
            moves.Add(FaceletCube.QuarterTurnMoves[action]);
            lastAction = action;
        }

        return (cube.IsSolved, moves);
    }
}
