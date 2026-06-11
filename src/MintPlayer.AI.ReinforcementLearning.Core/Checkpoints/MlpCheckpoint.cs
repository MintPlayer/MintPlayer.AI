using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Saves/loads an <see cref="Mlp"/>'s architecture + weights — enough to run inference
/// or warm-start training (optimizer state lives in <see cref="AdamCheckpoint"/>).
/// </summary>
public static class MlpCheckpoint
{
    public const string Kind = "mlp";
    private const int Version = 1;

    public static void Save(Mlp network, Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        Write(network, writer);
    }

    public static Mlp Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        return Read(reader);
    }

    public static void Write(Mlp network, BinaryWriter writer)
    {
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        CheckpointFormat.WriteInts(writer, network.Sizes);
        writer.Write((byte)network.HiddenActivation);
        foreach (var layer in network.Layers)
        {
            CheckpointFormat.WriteFloats(writer, layer.Weight.Data);
            CheckpointFormat.WriteFloats(writer, layer.Bias.Data);
        }
    }

    public static Mlp Read(BinaryReader reader)
    {
        CheckpointFormat.ReadHeader(reader, Kind, Version);
        int[] sizes = CheckpointFormat.ReadInts(reader);
        var activation = (Activation)reader.ReadByte();

        // Construct with a throwaway init, then overwrite every parameter.
        var network = new Mlp(sizes, new Xoshiro256StarStar(0), activation);
        foreach (var layer in network.Layers)
        {
            CopyExact(CheckpointFormat.ReadFloats(reader), layer.Weight.Data, "weight");
            CopyExact(CheckpointFormat.ReadFloats(reader), layer.Bias.Data, "bias");
        }
        return network;
    }

    private static void CopyExact(float[] stored, float[] target, string what)
    {
        if (stored.Length != target.Length)
            throw new InvalidDataException($"Checkpoint {what} length {stored.Length} does not match network ({target.Length}).");
        stored.CopyTo(target.AsSpan());
    }
}
