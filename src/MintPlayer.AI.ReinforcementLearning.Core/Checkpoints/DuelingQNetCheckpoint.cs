using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Saves/loads a <see cref="DuelingQNet"/>'s architecture (input size, shared hidden sizes, action count)
/// + weights. Parameters are streamed in <see cref="DuelingQNet.Parameters"/> order, so a freshly
/// constructed net of the stored shape can be filled back in one pass.
/// </summary>
public static class DuelingQNetCheckpoint
{
    public const string Kind = "dueling-q";
    // v2 adds the bool Noisy flag (after Actions, before the parameters). v1 files have no flag and
    // load as plain nets — every already-shipped checkpoint keeps loading unchanged.
    private const int Version = 2;

    public static void Save(DuelingQNet network, Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        Write(network, writer);
    }

    public static DuelingQNet Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        return Read(reader);
    }

    public static void Write(DuelingQNet network, BinaryWriter writer)
    {
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        writer.Write(network.InputSize);
        CheckpointFormat.WriteInts(writer, network.HiddenSizes);
        writer.Write(network.Actions);
        writer.Write(network.Noisy);
        foreach (var p in network.Parameters())
            CheckpointFormat.WriteFloats(writer, p.Data);
    }

    public static DuelingQNet Read(BinaryReader reader)
    {
        int version = CheckpointFormat.ReadHeader(reader, Kind, Version);
        int inputSize = reader.ReadInt32();
        int[] hidden = CheckpointFormat.ReadInts(reader);
        int actions = reader.ReadInt32();
        bool noisy = version >= 2 && reader.ReadBoolean(); // v1 had no flag → plain net

        var network = new DuelingQNet(inputSize, hidden, actions, new Xoshiro256StarStar(0), noisy);
        foreach (var p in network.Parameters())
        {
            var stored = CheckpointFormat.ReadFloats(reader);
            if (stored.Length != p.Data.Length)
                throw new InvalidDataException($"Checkpoint parameter length {stored.Length} does not match network ({p.Data.Length}).");
            stored.CopyTo(p.Data.AsSpan());
        }
        return network;
    }
}

/// <summary>
/// Type-tagged save/load for a DQN Q-network (<see cref="IValueNet"/>), so a training state round-trips
/// whichever architecture was used — plain <see cref="Mlp"/> or <see cref="DuelingQNet"/>. The store's
/// standalone net files keep their own un-tagged formats (<see cref="MlpCheckpoint"/> etc.); this tag only
/// lives inside the embedded training-state stream.
/// </summary>
public static class QNetCheckpoint
{
    private const byte TagMlp = 0, TagDueling = 1;

    public static void Write(IValueNet network, BinaryWriter writer)
    {
        switch (network)
        {
            case Mlp mlp:
                writer.Write(TagMlp);
                MlpCheckpoint.Write(mlp, writer);
                break;
            case DuelingQNet dueling:
                writer.Write(TagDueling);
                DuelingQNetCheckpoint.Write(dueling, writer);
                break;
            default:
                throw new NotSupportedException($"No checkpoint support for Q-network type '{network.GetType().Name}'.");
        }
    }

    public static IValueNet Read(BinaryReader reader)
    {
        byte tag = reader.ReadByte();
        return tag switch
        {
            TagMlp => MlpCheckpoint.Read(reader),
            TagDueling => DuelingQNetCheckpoint.Read(reader),
            _ => throw new InvalidDataException($"Unknown Q-network type tag {tag}."),
        };
    }
}
