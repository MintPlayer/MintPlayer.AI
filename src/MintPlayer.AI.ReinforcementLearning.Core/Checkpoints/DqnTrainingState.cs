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
    private const int Version = 1;

    public required Mlp Online { get; init; }
    public required Mlp Target { get; init; }
    public required Adam Optimizer { get; init; }
    public required ReplayBuffer Buffer { get; init; }
    public required Xoshiro256StarStar PolicyRng { get; init; }
    public required Xoshiro256StarStar BufferRng { get; init; }

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

        MlpCheckpoint.Write(Online, writer);
        MlpCheckpoint.Write(Target, writer);
        AdamCheckpoint.Write(Optimizer, writer);

        // Replay buffer: only the filled prefix (entries 0..Count-1 are the live ones;
        // NextIndex preserves the circular write position).
        writer.Write(Buffer.Capacity);
        writer.Write(Buffer.ObsDim);
        writer.Write(Buffer.ActionCount);
        writer.Write(Buffer.Count);
        writer.Write(Buffer.NextIndex);
        CheckpointFormat.WriteFloats(writer, Buffer.ObsData.AsSpan(0, Buffer.Count * Buffer.ObsDim));
        CheckpointFormat.WriteFloats(writer, Buffer.NextObsData.AsSpan(0, Buffer.Count * Buffer.ObsDim));
        CheckpointFormat.WriteInts(writer, Buffer.ActionsData.AsSpan(0, Buffer.Count));
        CheckpointFormat.WriteFloats(writer, Buffer.RewardsData.AsSpan(0, Buffer.Count));
        CheckpointFormat.WriteBools(writer, Buffer.TerminatedData.AsSpan(0, Buffer.Count));
        CheckpointFormat.WriteBools(writer, Buffer.NextMaskData.AsSpan(0, Buffer.Count * Buffer.ActionCount));

        CheckpointFormat.WriteRngState(writer, PolicyRng);
        CheckpointFormat.WriteRngState(writer, BufferRng);
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
        CheckpointFormat.ReadHeader(reader, Kind, Version);

        var online = MlpCheckpoint.Read(reader);
        var target = MlpCheckpoint.Read(reader);
        var optimizer = AdamCheckpoint.Read(online.Parameters(), reader);

        int capacity = reader.ReadInt32();
        int obsDim = reader.ReadInt32();
        int actionCount = reader.ReadInt32();
        int count = reader.ReadInt32();
        int nextIndex = reader.ReadInt32();
        var buffer = new ReplayBuffer(capacity, obsDim, actionCount) { Count = count, NextIndex = nextIndex };
        ReadInto(CheckpointFormat.ReadFloats(reader), buffer.ObsData);
        ReadInto(CheckpointFormat.ReadFloats(reader), buffer.NextObsData);
        CheckpointFormat.ReadInts(reader).CopyTo(buffer.ActionsData.AsSpan(0, count));
        ReadInto(CheckpointFormat.ReadFloats(reader), buffer.RewardsData);
        CheckpointFormat.ReadBools(reader).CopyTo(buffer.TerminatedData.AsSpan(0, count));
        CheckpointFormat.ReadBools(reader).CopyTo(buffer.NextMaskData.AsSpan(0, count * actionCount));

        var state = new DqnTrainingState
        {
            Online = online,
            Target = target,
            Optimizer = optimizer,
            Buffer = buffer,
            PolicyRng = CheckpointFormat.ReadRngState(reader),
            BufferRng = CheckpointFormat.ReadRngState(reader),
            CurrentObs = CheckpointFormat.ReadFloats(reader),
            StepsCompleted = reader.ReadInt32(),
            LastLoss = reader.ReadSingle(),
            LastEval = reader.ReadDouble(),
        };

        int envStateLength = reader.ReadInt32();
        if (envStateLength >= 0) state.EnvState = reader.ReadBytes(envStateLength);
        return state;

        static void ReadInto(float[] stored, float[] target) => stored.CopyTo(target.AsSpan(0, stored.Length));
    }
}
