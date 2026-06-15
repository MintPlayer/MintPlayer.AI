using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Serializes a <see cref="ContinuousReplayBuffer"/> in the SDK's versioned binary form — the continuous
/// counterpart of <see cref="ReplayBufferCheckpoint"/> (float action vectors, no mask). <see cref="Write"/>/
/// <see cref="Read"/> emit a header-less payload for embedding in <see cref="SacTrainingState"/>; only the
/// live prefix (entries 0..Count-1) is written, with <c>NextIndex</c> preserving the circular write position.
/// </summary>
public static class ContinuousReplayBufferCheckpoint
{
    public const string Kind = "continuous-replay-buffer";
    private const int Version = 1;

    /// <summary>Standalone, self-describing buffer file (header + payload).</summary>
    public static void Save(ContinuousReplayBuffer buffer, Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        Write(buffer, writer);
    }

    /// <summary>Reads a buffer written by <see cref="Save"/>.</summary>
    public static ContinuousReplayBuffer Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, Kind, Version);
        return Read(reader);
    }

    /// <summary>Header-less payload, for embedding in a larger checkpoint stream.</summary>
    public static void Write(ContinuousReplayBuffer buffer, BinaryWriter writer)
    {
        writer.Write(buffer.Capacity);
        writer.Write(buffer.ObsDim);
        writer.Write(buffer.ActionDim);
        writer.Write(buffer.Count);
        writer.Write(buffer.NextIndex);
        CheckpointFormat.WriteFloats(writer, buffer.ObsData.AsSpan(0, buffer.Count * buffer.ObsDim));
        CheckpointFormat.WriteFloats(writer, buffer.NextObsData.AsSpan(0, buffer.Count * buffer.ObsDim));
        CheckpointFormat.WriteFloats(writer, buffer.ActionsData.AsSpan(0, buffer.Count * buffer.ActionDim));
        CheckpointFormat.WriteFloats(writer, buffer.RewardsData.AsSpan(0, buffer.Count));
        CheckpointFormat.WriteBools(writer, buffer.TerminatedData.AsSpan(0, buffer.Count));
    }

    /// <summary>Reads a header-less payload written by <see cref="Write"/>.</summary>
    public static ContinuousReplayBuffer Read(BinaryReader reader)
    {
        int capacity = reader.ReadInt32();
        int obsDim = reader.ReadInt32();
        int actionDim = reader.ReadInt32();
        int count = reader.ReadInt32();
        int nextIndex = reader.ReadInt32();

        var buffer = new ContinuousReplayBuffer(capacity, obsDim, actionDim) { Count = count, NextIndex = nextIndex };
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.ObsData.AsSpan(0, count * obsDim));
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.NextObsData.AsSpan(0, count * obsDim));
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.ActionsData.AsSpan(0, count * actionDim));
        CheckpointFormat.ReadFloats(reader).CopyTo(buffer.RewardsData.AsSpan(0, count));
        CheckpointFormat.ReadBools(reader).CopyTo(buffer.TerminatedData.AsSpan(0, count));
        return buffer;
    }
}
