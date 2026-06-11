using System.Runtime.InteropServices;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

/// <summary>
/// Shared conventions for MintPlayer.AI.ReinforcementLearning's versioned binary checkpoints: a 4-byte magic,
/// a kind string and a format version, then kind-specific payload. All multi-byte
/// values are little-endian (BinaryWriter's contract; array payloads are memcpy'd,
/// which matches on every platform .NET supports).
/// </summary>
public static class CheckpointFormat
{
    private const uint Magic = 0x434E4C52; // "RLNC"

    public static void WriteHeader(BinaryWriter writer, string kind, int version)
    {
        writer.Write(Magic);
        writer.Write(kind);
        writer.Write(version);
    }

    /// <summary>Validates magic + kind; returns the stored version (1..maxSupportedVersion).</summary>
    public static int ReadHeader(BinaryReader reader, string kind, int maxSupportedVersion)
    {
        if (reader.ReadUInt32() != Magic)
            throw new InvalidDataException("Not an MintPlayer.AI.ReinforcementLearning checkpoint (bad magic).");
        string actual = reader.ReadString();
        if (actual != kind)
            throw new InvalidDataException($"Checkpoint kind mismatch: expected '{kind}', found '{actual}'.");
        int version = reader.ReadInt32();
        if (version < 1 || version > maxSupportedVersion)
            throw new InvalidDataException($"Unsupported '{kind}' checkpoint version {version} (max supported: {maxSupportedVersion}).");
        return version;
    }

    public static void WriteFloats(BinaryWriter writer, ReadOnlySpan<float> values)
    {
        writer.Write(values.Length);
        writer.Write(MemoryMarshal.AsBytes(values));
    }

    public static float[] ReadFloats(BinaryReader reader)
    {
        var values = new float[reader.ReadInt32()];
        reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    public static void WriteInts(BinaryWriter writer, ReadOnlySpan<int> values)
    {
        writer.Write(values.Length);
        writer.Write(MemoryMarshal.AsBytes(values));
    }

    public static int[] ReadInts(BinaryReader reader)
    {
        var values = new int[reader.ReadInt32()];
        reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    public static void WriteBools(BinaryWriter writer, ReadOnlySpan<bool> values)
    {
        writer.Write(values.Length);
        writer.Write(MemoryMarshal.AsBytes(values));
    }

    public static bool[] ReadBools(BinaryReader reader)
    {
        var values = new bool[reader.ReadInt32()];
        reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    public static void WriteRngState(BinaryWriter writer, Xoshiro256StarStar rng)
    {
        var (s0, s1, s2, s3) = rng.GetState();
        writer.Write(s0);
        writer.Write(s1);
        writer.Write(s2);
        writer.Write(s3);
    }

    /// <summary>Reads an RNG state and returns an RNG that will continue that exact sequence.</summary>
    public static Xoshiro256StarStar ReadRngState(BinaryReader reader)
    {
        var rng = new Xoshiro256StarStar(0);
        rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        return rng;
    }
}
