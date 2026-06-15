using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// The complete mutable state of a SAC training run — actor, twin critics + their targets, the actor and
/// (shared) critic optimizers, the entropy temperature (log-α) and its optimizer, the replay buffer, RNG
/// streams, current observation and (for <see cref="Environments.IStatefulEnvironment"/> envs) the
/// environment snapshot. Saving this and resuming via <c>SacTrainer.Train(..., resume: state)</c> continues
/// bitwise-identically to a run that was never interrupted. Mirrors <see cref="DqnTrainingState"/>.
/// </summary>
public sealed class SacTrainingState
{
    public const string Kind = "sac-state";
    private const int Version = 1;

    public required Mlp Actor { get; init; }
    public required Mlp Critic1 { get; init; }
    public required Mlp Critic2 { get; init; }
    public required Mlp Target1 { get; init; }
    public required Mlp Target2 { get; init; }

    public required Adam ActorOptimizer { get; init; }
    public required Adam CriticOptimizer { get; init; } // bound to Critic1.Parameters() ++ Critic2.Parameters()

    /// <summary>Entropy temperature in log-space: a 1-element grad-carrying leaf. α = exp(LogAlpha).</summary>
    public required Tensor LogAlpha { get; init; }
    /// <summary>Optimizer for <see cref="LogAlpha"/>; null when the temperature is fixed.</summary>
    public Adam? AlphaOptimizer { get; init; }

    public required ContinuousReplayBuffer Buffer { get; init; }
    public required Xoshiro256StarStar PolicyRng { get; init; }
    public required Xoshiro256StarStar BufferRng { get; init; }

    public float[] CurrentObs { get; set; } = [];
    public int StepsCompleted { get; set; }
    public float LastCriticLoss { get; set; }
    public float LastActorLoss { get; set; }
    public double LastEval { get; set; } = double.NegativeInfinity;

    /// <summary>Opaque environment snapshot; null when the env is not IStatefulEnvironment.</summary>
    public byte[]? EnvState { get; set; }

    public void Save(Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, Kind, Version);

        MlpCheckpoint.Write(Actor, writer);
        MlpCheckpoint.Write(Critic1, writer);
        MlpCheckpoint.Write(Critic2, writer);
        MlpCheckpoint.Write(Target1, writer);
        MlpCheckpoint.Write(Target2, writer);

        AdamCheckpoint.Write(ActorOptimizer, writer);
        AdamCheckpoint.Write(CriticOptimizer, writer);

        writer.Write(AlphaOptimizer is not null);
        writer.Write(LogAlpha.Data[0]);
        if (AlphaOptimizer is not null) AdamCheckpoint.Write(AlphaOptimizer, writer);

        ContinuousReplayBufferCheckpoint.Write(Buffer, writer);
        CheckpointFormat.WriteRngState(writer, PolicyRng);
        CheckpointFormat.WriteRngState(writer, BufferRng);
        CheckpointFormat.WriteFloats(writer, CurrentObs);
        writer.Write(StepsCompleted);
        writer.Write(LastCriticLoss);
        writer.Write(LastActorLoss);
        writer.Write(LastEval);

        int envStateLength = EnvState?.Length ?? -1;
        writer.Write(envStateLength);
        if (EnvState is not null) writer.Write(EnvState);
    }

    public static SacTrainingState Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, Kind, Version);

        var actor = MlpCheckpoint.Read(reader);
        var critic1 = MlpCheckpoint.Read(reader);
        var critic2 = MlpCheckpoint.Read(reader);
        var target1 = MlpCheckpoint.Read(reader);
        var target2 = MlpCheckpoint.Read(reader);

        var actorOptimizer = AdamCheckpoint.Read(actor.Parameters(), reader);
        var criticOptimizer = AdamCheckpoint.Read([.. critic1.Parameters(), .. critic2.Parameters()], reader);

        bool autoAlpha = reader.ReadBoolean();
        var logAlpha = new Tensor([reader.ReadSingle()], 1) { RequiresGrad = true };
        Adam? alphaOptimizer = autoAlpha ? AdamCheckpoint.Read([logAlpha], reader) : null;

        var buffer = ContinuousReplayBufferCheckpoint.Read(reader);

        var state = new SacTrainingState
        {
            Actor = actor,
            Critic1 = critic1,
            Critic2 = critic2,
            Target1 = target1,
            Target2 = target2,
            ActorOptimizer = actorOptimizer,
            CriticOptimizer = criticOptimizer,
            LogAlpha = logAlpha,
            AlphaOptimizer = alphaOptimizer,
            Buffer = buffer,
            PolicyRng = CheckpointFormat.ReadRngState(reader),
            BufferRng = CheckpointFormat.ReadRngState(reader),
            CurrentObs = CheckpointFormat.ReadFloats(reader),
            StepsCompleted = reader.ReadInt32(),
            LastCriticLoss = reader.ReadSingle(),
            LastActorLoss = reader.ReadSingle(),
            LastEval = reader.ReadDouble(),
        };

        int envStateLength = reader.ReadInt32();
        if (envStateLength >= 0) state.EnvState = reader.ReadBytes(envStateLength);
        return state;
    }
}
