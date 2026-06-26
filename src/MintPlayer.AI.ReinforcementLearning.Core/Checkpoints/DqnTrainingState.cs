using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// The complete mutable state of a DQN training run — networks, optimizer, replay
/// buffer, RNG streams, current observation and (for <see cref="Environments.IStatefulEnvironment"/>
/// envs) the environment snapshot. Saving this and resuming via
/// <c>DqnTrainer.Train(..., resume: state)</c> continues bitwise-identically to a run
/// that was never interrupted.
/// </summary>
public sealed class DqnTrainingState
{
    public const string Kind = "dqn-state";
    // v2: networks are type-tagged (Mlp | DuelingQNet) via QNetCheckpoint.
    // v3: adds NoiseRng (NoisyNets). v2 states load with a default NoiseRng — harmless when not noisy.
    private const int Version = 3;

    public required IValueNet Online { get; init; }
    public required IValueNet Target { get; init; }
    public required Adam Optimizer { get; init; }
    public required ReplayBuffer Buffer { get; init; }
    public required Xoshiro256StarStar PolicyRng { get; init; }
    public required Xoshiro256StarStar BufferRng { get; init; }

    /// <summary>Exploration-noise RNG for NoisyNets resampling; unused (but serialized) when not noisy.</summary>
    public Xoshiro256StarStar NoiseRng { get; init; } = new(0);

    public float[] CurrentObs { get; set; } = [];
    public int StepsCompleted { get; set; }
    public float LastLoss { get; set; }
    public double LastEval { get; set; } = double.NegativeInfinity;

    /// <summary>Opaque environment snapshot; null when the env is not IStatefulEnvironment (resume then re-Resets, losing bitwise equality).</summary>
    public byte[]? EnvState { get; set; }

    public void Save(Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, Kind, Version);

        QNetCheckpoint.Write(Online, writer);
        QNetCheckpoint.Write(Target, writer);
        AdamCheckpoint.Write(Optimizer, writer);

        // Algorithm-agnostic transition payload (same bytes as the former inline block).
        ReplayBufferCheckpoint.Write(Buffer, writer);

        CheckpointFormat.WriteRngState(writer, PolicyRng);
        CheckpointFormat.WriteRngState(writer, BufferRng);
        CheckpointFormat.WriteRngState(writer, NoiseRng);
        CheckpointFormat.WriteFloats(writer, CurrentObs);
        writer.Write(StepsCompleted);
        writer.Write(LastLoss);
        writer.Write(LastEval);

        int envStateLength = EnvState?.Length ?? -1;
        writer.Write(envStateLength);
        if (EnvState is not null) writer.Write(EnvState);
    }

    public static DqnTrainingState Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        int version = CheckpointFormat.ReadHeader(reader, Kind, Version);

        var online = QNetCheckpoint.Read(reader);
        var target = QNetCheckpoint.Read(reader);
        var optimizer = AdamCheckpoint.Read(online.Parameters(), reader);
        var buffer = ReplayBufferCheckpoint.Read(reader);

        var state = new DqnTrainingState
        {
            Online = online,
            Target = target,
            Optimizer = optimizer,
            Buffer = buffer,
            PolicyRng = CheckpointFormat.ReadRngState(reader),
            BufferRng = CheckpointFormat.ReadRngState(reader),
            NoiseRng = version >= 3 ? CheckpointFormat.ReadRngState(reader) : new Xoshiro256StarStar(0),
            CurrentObs = CheckpointFormat.ReadFloats(reader),
            StepsCompleted = reader.ReadInt32(),
            LastLoss = reader.ReadSingle(),
            LastEval = reader.ReadDouble(),
        };

        int envStateLength = reader.ReadInt32();
        if (envStateLength >= 0) state.EnvState = reader.ReadBytes(envStateLength);
        return state;
    }
}
