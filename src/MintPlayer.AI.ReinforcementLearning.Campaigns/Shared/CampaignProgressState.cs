using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>Training progress that must survive a restart: counters, the most recent gate metric, and the
/// owner-thread RNG states.</summary>
/// <param name="Samples">Total training samples consumed across the whole run.</param>
/// <param name="Units">Campaign-specific work unit — labelled configs, rounds, or self-play games.</param>
/// <param name="LastMetric">Most recent eval metric that drives progression (0 when none yet).</param>
/// <param name="Rngs">Owner-thread RNG states, in the order the campaign declares them.</param>
public sealed record CampaignProgress(long Samples, long Units, double LastMetric, Xoshiro256StarStar[] Rngs);

/// <summary>
/// A tiny sidecar checkpoint for campaign progress, stored under its own algorithm id alongside the net and
/// Adam moments.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> The imitation and self-play campaigns persisted their net and Adam state but
/// none of their <i>progress</i>: <c>_totalSamples</c>, <c>_totalConfigs</c>, <c>_round</c>, <c>_totalGames</c>
/// and every owner-thread RNG were re-initialised from the seed on every <c>Resume</c>. A restarted run
/// therefore replayed the same generated data from the beginning and — worse — reset
/// <see cref="PolicyGrowth"/>'s stage target, because that keys off the sample counter, so a long run
/// interrupted repeatedly would keep re-growing its trunk from the first stage. Only the DQN family got this
/// right, via <c>DqnTrainingState</c>.</para>
///
/// <para>The file is <b>additive and optional</b>: a run whose store predates it simply starts its counters at
/// zero, which is exactly the old behaviour, so existing checkpoints keep working.</para>
///
/// <para>RNG states round-trip through <see cref="CheckpointFormat.WriteRngState"/>, which already existed and
/// was simply never wired up here. Only <b>owner-thread</b> streams belong in this file — work distributed
/// through <c>DeterministicParallel</c> derives its seeds from the item index, so replaying it needs the
/// counter and nothing else.</para>
/// </remarks>
public static class CampaignProgressState
{
    private const int Version = 1;

    /// <summary>Reads progress, or null when absent or shaped for a different campaign (both meaning "start from zero").</summary>
    /// <param name="rngCount">Number of RNG streams the caller expects, in its declared order.</param>
    public static CampaignProgress? TryLoad(
        IModelStore store, string environmentId, string algorithmId, string kind, int rngCount, Action<string>? log = null)
    {
        using var stream = store.TryOpenRead(environmentId, algorithmId);
        if (stream is null) return null;

        try
        {
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            CheckpointFormat.ReadHeader(reader, kind, Version);

            long samples = reader.ReadInt64();
            long units = reader.ReadInt64();
            double lastMetric = reader.ReadDouble();
            int storedRngs = reader.ReadInt32();
            if (storedRngs != rngCount)
            {
                log?.Invoke($"progress checkpoint '{environmentId}.{algorithmId}' declares {storedRngs} RNG streams " +
                            $"but this campaign has {rngCount} — ignoring it and restarting counters at zero.");
                return null;
            }

            var rngs = new Xoshiro256StarStar[rngCount];
            for (int i = 0; i < rngCount; i++) rngs[i] = CheckpointFormat.ReadRngState(reader);
            return new CampaignProgress(samples, units, lastMetric, rngs);
        }
        catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException)
        {
            // Progress is an optimisation, never a correctness requirement: a damaged sidecar costs a replay of
            // already-seen data, so degrade to a fresh start rather than failing the run.
            log?.Invoke($"progress checkpoint '{environmentId}.{algorithmId}' is unreadable ({ex.Message}) — restarting counters at zero.");
            return null;
        }
    }

    /// <summary>Copies a restored RNG's state into a live stream, so campaigns can keep their RNG fields
    /// <c>readonly</c> instead of reassigning them on resume.</summary>
    public static void RestoreInto(Xoshiro256StarStar restored, Xoshiro256StarStar live)
    {
        var (s0, s1, s2, s3) = restored.GetState();
        live.SetState(s0, s1, s2, s3);
    }

    /// <summary>Writes progress. Call from <c>Checkpoint</c>, next to the net and Adam saves.</summary>
    public static void Save(
        IModelStore store, string environmentId, string algorithmId, string kind,
        long samples, long units, double lastMetric, params Xoshiro256StarStar[] rngs)
    {
        store.Save(environmentId, algorithmId, stream =>
        {
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            CheckpointFormat.WriteHeader(writer, kind, Version);
            writer.Write(samples);
            writer.Write(units);
            writer.Write(lastMetric);
            writer.Write(rngs.Length);
            foreach (var rng in rngs) CheckpointFormat.WriteRngState(writer, rng);
        });
    }
}
