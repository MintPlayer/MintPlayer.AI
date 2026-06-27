namespace MintPlayer.AI.ReinforcementLearning.Core.Random;

/// <summary>
/// Fans a single master seed out into independent named RNG streams
/// (environment, policy, network init, buffer sampling, ...), so one seed
/// reproduces an entire training run.
/// </summary>
public sealed class SeedSequence(ulong masterSeed)
{
    public ulong MasterSeed { get; } = masterSeed;

    /// <summary>Derives a deterministic child seed for the given stream index.</summary>
    public ulong Derive(int streamIndex)
    {
        ulong state = MasterSeed;
        ulong mixed = SplitMix64.Next(ref state) ^ ((ulong)streamIndex * 0xA24BAED4963EE407UL);
        return SplitMix64.Next(ref mixed);
    }

    public Xoshiro256StarStar CreateRng(int streamIndex) => new(Derive(streamIndex));
}

/// <summary>Conventional stream indices, so all trainers fan out the same way.</summary>
public static class RngStreams
{
    public const int Environment = 0;
    public const int Policy = 1;
    public const int Init = 2;
    public const int Buffer = 3;
    public const int Evaluation = 4;
    public const int Noise = 5; // NoisyNets exploration-noise resampling
}
