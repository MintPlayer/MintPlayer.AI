using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Everything a Block Dude run needs to continue exactly where it stopped — and the evidence that it is the
/// same run at all.
/// </summary>
/// <remarks>
/// <para>Curriculum progression is a pure function of persisted state, so this file <i>is</i> the reproducibility
/// guarantee. Without it a restart silently drops to stage 0 and replays data it has already trained on, which
/// is precisely the latent bug found in the other imitation campaigns.</para>
///
/// <para><b>Board generation RNG is deliberately absent.</b> Each candidate board derives its RNG from the
/// attempt counter, the way <c>DeterministicParallel</c> derives per-item seeds from the item index, so replaying
/// generation needs only <see cref="BoardAttempts"/> — and the stream cannot drift when rejections change.</para>
/// </remarks>
public sealed class BlockDudeTrainingState
{
    public const string Kind = "blockdude-imitation-state";
    private const int Version = 1;

    public int ObservationSize { get; set; }
    public int ActionCount { get; set; }

    /// <summary>Hash over seed, observation shape, curriculum version and the core hyper-parameters. A mismatch
    /// means this checkpoint belongs to a different run, and resuming it would silently blend two schedules.</summary>
    public ulong RunFingerprint { get; set; }

    public int Stage { get; set; }
    public long StageSamples { get; set; }
    public long TotalSamples { get; set; }
    public long SamplesAtLastGate { get; set; }

    /// <summary>Most recent greedy solve rate per rung; -1 where a rung has not been gated yet.</summary>
    public double[] GateRates { get; set; } = [];

    /// <summary>Candidate boards DRAWN, accepted or not. This is the generator's stream position.</summary>
    public long BoardAttempts { get; set; }

    public long AcceptedBoards { get; set; }
    public long TruncatedBoards { get; set; }
    public long RejectedBoards { get; set; }

    public Xoshiro256StarStar ShuffleRng { get; set; } = new(0);
    public Xoshiro256StarStar GrowRng { get; set; } = new(0);

    /// <summary>Fingerprint of the run's identity-defining settings.</summary>
    public static ulong Fingerprint(ulong seed, int observationSize, int actionCount, float learningRate, int phase)
    {
        ulong h = 1469598103934665603UL;                       // FNV-1a, 64-bit
        void Mix(ulong v)
        {
            h ^= v;
            h *= 1099511628211UL;
        }
        Mix(seed);
        Mix((ulong)observationSize);
        Mix((ulong)actionCount);
        Mix((ulong)BlockDudeCurriculum.Version);
        Mix((ulong)BlockDudeCurriculum.Stages.Length);
        Mix(BitConverter.DoubleToUInt64Bits(learningRate));
        Mix((ulong)phase);
        return h;
    }

    public void Save(Stream destination)
    {
        using var writer = new BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, Kind, Version);
        writer.Write(ObservationSize);
        writer.Write(ActionCount);
        writer.Write(RunFingerprint);
        writer.Write(Stage);
        writer.Write(StageSamples);
        writer.Write(TotalSamples);
        writer.Write(SamplesAtLastGate);
        writer.Write(GateRates.Length);
        foreach (double rate in GateRates) writer.Write(rate);
        writer.Write(BoardAttempts);
        writer.Write(AcceptedBoards);
        writer.Write(TruncatedBoards);
        writer.Write(RejectedBoards);
        CheckpointFormat.WriteRngState(writer, ShuffleRng);
        CheckpointFormat.WriteRngState(writer, GrowRng);
    }

    /// <summary>Reads the sidecar, or null when it is absent, unreadable, or belongs to a different run.</summary>
    public static BlockDudeTrainingState? TryLoad(
        IModelStore store, string environmentId, string algorithmId, ulong expectedFingerprint,
        int observationSize, int actionCount, Action<string>? log = null)
    {
        using var stream = store.TryOpenRead(environmentId, algorithmId);
        if (stream is null) return null;

        try
        {
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            CheckpointFormat.ReadHeader(reader, Kind, Version);

            var state = new BlockDudeTrainingState
            {
                ObservationSize = reader.ReadInt32(),
                ActionCount = reader.ReadInt32(),
                RunFingerprint = reader.ReadUInt64(),
                Stage = reader.ReadInt32(),
                StageSamples = reader.ReadInt64(),
                TotalSamples = reader.ReadInt64(),
                SamplesAtLastGate = reader.ReadInt64(),
            };

            int rates = reader.ReadInt32();
            state.GateRates = new double[rates];
            for (int i = 0; i < rates; i++) state.GateRates[i] = reader.ReadDouble();

            state.BoardAttempts = reader.ReadInt64();
            state.AcceptedBoards = reader.ReadInt64();
            state.TruncatedBoards = reader.ReadInt64();
            state.RejectedBoards = reader.ReadInt64();
            state.ShuffleRng = CheckpointFormat.ReadRngState(reader);
            state.GrowRng = CheckpointFormat.ReadRngState(reader);

            // Refuse loudly rather than blending two runs. A silent resume into a different observation shape or
            // a different curriculum is the failure this whole file exists to make impossible.
            if (state.ObservationSize != observationSize || state.ActionCount != actionCount)
            {
                log?.Invoke($"STALE progress checkpoint: trained at observation {state.ObservationSize}/" +
                            $"{state.ActionCount} actions, this build supplies {observationSize}/{actionCount} — " +
                            "ignoring it and starting the curriculum from scratch.");
                return null;
            }
            if (state.RunFingerprint != expectedFingerprint)
            {
                log?.Invoke("progress checkpoint belongs to a DIFFERENT run (seed, learning rate, phase or " +
                            "curriculum version changed) — ignoring it and starting the curriculum from scratch.");
                return null;
            }
            return state;
        }
        catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException)
        {
            log?.Invoke($"progress checkpoint unreadable ({ex.Message}) — starting the curriculum from scratch.");
            return null;
        }
    }
}
