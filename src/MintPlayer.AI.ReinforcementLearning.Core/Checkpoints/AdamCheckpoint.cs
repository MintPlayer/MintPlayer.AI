using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Saves/loads <see cref="Adam"/> state (hyperparameters, step count, first/second
/// moments) so resumed training continues bitwise-identically instead of restarting
/// the moment estimates from zero.
/// </summary>
public static class AdamCheckpoint
{
    public const string Kind = "adam";
    private const int Version = 1;

    public static void Write(Adam optimizer, BinaryWriter writer)
    {
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        writer.Write(optimizer.LearningRate);
        writer.Write(optimizer.Beta1);
        writer.Write(optimizer.Beta2);
        writer.Write(optimizer.EpsilonValue);
        writer.Write(optimizer.StepCount);
        writer.Write(optimizer.FirstMoments.Length);
        foreach (var m in optimizer.FirstMoments) CheckpointFormat.WriteFloats(writer, m);
        foreach (var v in optimizer.SecondMoments) CheckpointFormat.WriteFloats(writer, v);
    }

    /// <summary>Reconstructs an Adam bound to <paramref name="parameters"/> (same order/shapes as at save time).</summary>
    public static Adam Read(IEnumerable<Tensor> parameters, BinaryReader reader)
    {
        CheckpointFormat.ReadHeader(reader, Kind, Version);
        float learningRate = reader.ReadSingle();
        float beta1 = reader.ReadSingle();
        float beta2 = reader.ReadSingle();
        float epsilon = reader.ReadSingle();
        int step = reader.ReadInt32();
        int count = reader.ReadInt32();

        var optimizer = new Adam(parameters, learningRate, beta1, beta2, epsilon) { StepCount = step };
        if (optimizer.FirstMoments.Length != count)
            throw new InvalidDataException($"Checkpoint has {count} parameter moments, optimizer has {optimizer.FirstMoments.Length} parameters.");
        Restore(optimizer.FirstMoments, reader, "first");
        Restore(optimizer.SecondMoments, reader, "second");
        return optimizer;
    }

    private static void Restore(float[][] moments, BinaryReader reader, string which)
    {
        foreach (var target in moments)
        {
            var stored = CheckpointFormat.ReadFloats(reader);
            if (stored.Length != target.Length)
                throw new InvalidDataException($"Checkpoint {which}-moment length {stored.Length} does not match parameter ({target.Length}).");
            stored.CopyTo(target.AsSpan());
        }
    }
}
