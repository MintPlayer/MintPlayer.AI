using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Serializes a <see cref="ReplayBuffer"/>'s transitions in the SDK's versioned binary
/// form. Factored out of <see cref="DqnTrainingState"/> so the transition data is
/// <b>algorithm-agnostic</b>: the tuple <c>(s, a, r, s′, terminated, next-mask)</c> belongs
/// to no single learner, so any off-policy algorithm (DQN, and later SAC/DDPG) — or an
/// oracle producing demonstrations — can persist and reload the same buffer (PRD goal 8,
/// "switch algorithm, keep the work").
/// <para>
/// <see cref="Write"/>/<see cref="Read"/> emit the field-level payload with NO header, so
/// <see cref="DqnTrainingState"/> embeds them inline byte-for-byte as before (shipped
/// <c>dqn-state</c> checkpoints still load). <see cref="Save"/>/<see cref="Load"/> wrap that
/// payload with a header for a self-describing standalone file. Only the live prefix
/// (entries <c>0..Count-1</c>) is written; <c>NextIndex</c> preserves the circular write
/// position so a resumed buffer keeps overwriting where it left off.
/// </para>
/// </summary>
public static class ReplayBufferCheckpoint
{
    public const string Kind = "replay-buffer";
    private const int Version = 1;

    /// <summary>Standalone, self-describing buffer file (header + payload).</summary>
    public static void Save(ReplayBuffer buffer, Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        Write(buffer, writer);
    }

    /// <summary>Reads a buffer written by <see cref="Save"/>.</summary>
    public static ReplayBuffer Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, Kind, Version);
        return Read(reader);
    }

    /// <summary>Header-less payload, for embedding in a larger checkpoint stream.</summary>
    public static void Write(ReplayBuffer buffer, BinaryWriter writer)
    {
        writer.Write(buffer.Capacity);
        writer.Write(buffer.ObsDim);
        writer.Write(buffer.ActionCount);
        writer.Write(buffer.Count);
        writer.Write(buffer.NextIndex);
        CheckpointFormat.WriteFloats(writer, buffer.ObsData.AsSpan(0, buffer.Count * buffer.ObsDim));
        CheckpointFormat.WriteFloats(writer, buffer.NextObsData.AsSpan(0, buffer.Count * buffer.ObsDim));
        CheckpointFormat.WriteInts(writer, buffer.ActionsData.AsSpan(0, buffer.Count));
        CheckpointFormat.WriteFloats(writer, buffer.RewardsData.AsSpan(0, buffer.Count));
        CheckpointFormat.WriteBools(writer, buffer.TerminatedData.AsSpan(0, buffer.Count));
        CheckpointFormat.WriteBools(writer, buffer.NextMaskData.AsSpan(0, buffer.Count * buffer.ActionCount));
    }

    /// <summary>Reads a header-less payload written by <see cref="Write"/>.</summary>
    public static ReplayBuffer Read(BinaryReader reader)
    {
        int capacity = reader.ReadInt32();
        int obsDim = reader.ReadInt32();
        int actionCount = reader.ReadInt32();
        int count = reader.ReadInt32();
        int nextIndex = reader.ReadInt32();

        var buffer = new ReplayBuffer(capacity, obsDim, actionCount) { Count = count, NextIndex = nextIndex };
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.ObsData.AsSpan(0, count * obsDim));
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.NextObsData.AsSpan(0, count * obsDim));
        CheckpointFormat.ReadInts(reader).CopyTo(buffer.ActionsData.AsSpan(0, count));
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.RewardsData.AsSpan(0, count));
        CheckpointFormat.ReadBools(reader).CopyTo(buffer.TerminatedData.AsSpan(0, count));
        CheckpointFormat.ReadBools(reader).CopyTo(buffer.NextMaskData.AsSpan(0, count * actionCount));
        return buffer;
    }
}
