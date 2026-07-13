using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Generates a batch of independent items — one per index in <c>[0, count)</c> — with a determinism guarantee:
/// the result array is <b>bitwise-identical whether produced in parallel or sequentially, and at any degree of
/// parallelism.</b> This is the reusable form of the data-generation loop the cube campaigns hand-roll and that
/// self-play needs: episodes/games/labeled-samples are embarrassingly parallel, but a seeded run must stay
/// reproducible regardless of how many cores happen to run it.
/// <para>
/// It holds the same way Core's other parallel primitives do (<see cref="Environments.VectorEnv"/>: per-unit RNG;
/// the backend GEMM: disjoint output rows) — each item's randomness is a pure function of its <i>global index</i>
/// (never execution order), and each item writes only its own result slot, so there is no shared mutable state and
/// no reduction. The caller supplies a <see cref="SeedSequence"/> + a stream index so generation stays on its own
/// RNG stream, disjoint from the trainer's other streams (init, eval, …), exactly as the rest of the codebase fans
/// seeds out.
/// </para>
/// </summary>
public static class DeterministicParallel
{
    /// <summary>
    /// Runs <paramref name="makeItem"/> for each local index in <c>[0, count)</c>, each with its OWN RNG derived
    /// from <c>(<paramref name="seeds"/>, <paramref name="stream"/>, <paramref name="baseIndex"/> + localIndex)</c>,
    /// and returns the results in ascending index order. Because a given global index always derives the same RNG
    /// and writes the same slot, the output does not depend on <paramref name="parallel"/> or on the worker count —
    /// callers get free multi-core scaling with zero effect on a seeded run's outcome.
    /// </summary>
    /// <param name="count">Number of items to generate (0 ⇒ empty array).</param>
    /// <param name="seeds">The run's master seed fan-out; the per-item base seed is <c>seeds.Derive(stream)</c>.</param>
    /// <param name="stream">RNG stream index for this generation phase (see <see cref="RngStreams"/>), keeping it
    /// disjoint from other streams.</param>
    /// <param name="baseIndex">Global index of the first item — advance it across calls (e.g. total games so far)
    /// so successive batches draw non-overlapping, non-repeating RNGs.</param>
    /// <param name="makeItem">Produces item <c>i</c> from its derived RNG. MUST be pure w.r.t. that RNG and any
    /// read-only captured state (a shared read-only model snapshot is fine); it must not touch shared mutable state,
    /// or the determinism guarantee is lost.</param>
    /// <param name="parallel">Spread work across cores (bitwise-identical to the sequential path).</param>
    /// <param name="maxDop">Optional cap on the degree of parallelism (defaults to the runtime's choice).</param>
    public static TItem[] Generate<TItem>(
        int count, SeedSequence seeds, int stream, long baseIndex,
        Func<int, Xoshiro256StarStar, TItem> makeItem, bool parallel, int? maxDop = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentNullException.ThrowIfNull(seeds);
        ArgumentNullException.ThrowIfNull(makeItem);

        var results = new TItem[count];
        ulong baseSeed = seeds.Derive(stream);

        if (parallel && count > 1)
        {
            var options = new ParallelOptions();
            if (maxDop is int dop) options.MaxDegreeOfParallelism = dop;
            Parallel.For(0, count, options, i => results[i] = makeItem(i, DeriveRng(baseSeed, baseIndex + i)));
        }
        else
        {
            for (int i = 0; i < count; i++)
                results[i] = makeItem(i, DeriveRng(baseSeed, baseIndex + i));
        }
        return results;
    }

    /// <summary>Per-item RNG: the golden-ratio index stride the codebase uses everywhere (<c>VectorEnv.Reset</c>),
    /// whose xoshiro seeding runs each seed through SplitMix64 — so adjacent indices give decorrelated streams.</summary>
    private static Xoshiro256StarStar DeriveRng(ulong baseSeed, long globalIndex)
        => new(unchecked(baseSeed + (ulong)globalIndex * 0x9E3779B97F4A7C15UL));
}
