using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Saves/loads a <see cref="ResidualMlp"/>'s architecture (input size, width, block count) + weights.
/// The parameter stream follows <see cref="ResidualMlp.Parameters"/> order exactly, so a freshly
/// constructed net of the stored shape can be filled back in one pass (optimizer state lives in
/// <see cref="AdamCheckpoint"/>).
/// </summary>
public static class ResidualMlpCheckpoint
{
    public const string Kind = "resmlp";
    private const int Version = 1;

    public static void Save(ResidualMlp network, Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        Write(network, writer);
    }

    public static ResidualMlp Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        return Read(reader);
    }

    public static void Write(ResidualMlp network, BinaryWriter writer)
    {
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        writer.Write(network.InputSize);
        writer.Write(network.Width);
        writer.Write(network.Blocks);
        foreach (var p in network.Parameters())
            CheckpointFormat.WriteFloats(writer, p.Data);
    }

    public static ResidualMlp Read(BinaryReader reader)
    {
        CheckpointFormat.ReadHeader(reader, Kind, Version);
        int inputSize = reader.ReadInt32();
        int width = reader.ReadInt32();
        int blocks = reader.ReadInt32();

        var network = new ResidualMlp(inputSize, width, blocks, new Xoshiro256StarStar(0));
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
