namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// The sequence of trunk shapes a growing net climbs, each rung exactly one function-preserving Net2Net step —
/// one widen, or one extra layer — from the one below it.
/// </summary>
/// <remarks>
/// <para><b>Rung 0 is the net you would have trained anyway.</b> This is the invariant the type exists to
/// enforce, and it is enforced because violating it silently produced the opposite of the intended effect: the
/// original shared schedule ran <c>[16] → … → [128,128,128]</c>, sized for small DQN nets, while the games using
/// it had defaults of <c>[384,384]</c> (Rush Hour), <c>[512,512]</c> (Cube, Block Dude). Enabling growth on that
/// ladder started those runs 24–32× narrower and finished them SMALLER than not enabling it at all — so a
/// "capacity experiment" measured less capacity, and its result pointed the wrong way.</para>
///
/// <para>Prefer <see cref="FromTrunk"/> over hand-written rungs: it takes the net's own default shape as rung 0,
/// so a ladder cannot be below the net it grows by construction.</para>
/// </remarks>
public sealed class GrowthLadder
{
    /// <summary>Trunk shape per rung, rung 0 first.</summary>
    public IReadOnlyList<int[]> Rungs { get; }

    /// <summary>Index of the last rung; a net here has no capacity left to add.</summary>
    public int Top => Rungs.Count - 1;

    /// <summary>The trunk for a rung, clamped into range.</summary>
    public int[] TrunkFor(int rung) => Rungs[Math.Clamp(rung, 0, Top)];

    /// <summary>
    /// Builds a ladder from explicit rungs, validating that each is exactly one widen or one deepen from the
    /// previous — Net2Net can do one or the other in a single function-preserving step, not both at once.
    /// </summary>
    /// <exception cref="ArgumentException">A rung is empty, or is not a single step from its predecessor.</exception>
    public GrowthLadder(params int[][] rungs)
    {
        ArgumentNullException.ThrowIfNull(rungs);
        if (rungs.Length == 0) throw new ArgumentException("A ladder needs at least one rung.", nameof(rungs));

        for (int i = 0; i < rungs.Length; i++)
        {
            if (rungs[i] is null || rungs[i].Length == 0)
                throw new ArgumentException($"Rung {i} is empty.", nameof(rungs));
            if (rungs[i].Any(w => w <= 0))
                throw new ArgumentException($"Rung {i} has a non-positive width.", nameof(rungs));
            if (i == 0) continue;

            int[] previous = rungs[i - 1];
            int[] current = rungs[i];

            // A deepen appends ONE layer and leaves the existing widths alone. Checking only the length would
            // accept [64,64] → [96,96,96], which widens and deepens at once — two Net2Net steps masquerading as
            // one, and not reachable without a loss spike.
            bool deepened = current.Length == previous.Length + 1
                            && current.Take(previous.Length).SequenceEqual(previous);

            bool widened = current.Length == previous.Length
                           && current.Zip(previous).All(p => p.First >= p.Second)
                           && current.Zip(previous).Any(p => p.First > p.Second);

            if (!(deepened ^ widened))
                throw new ArgumentException(
                    $"Rung {i} ([{string.Join(",", current)}]) is neither exactly one widen nor exactly one deepen " +
                    $"from rung {i - 1} ([{string.Join(",", previous)}]).", nameof(rungs));
        }

        Rungs = rungs;
    }

    /// <summary>
    /// A ladder rooted at a net's OWN default trunk, alternating widen → deepen upward. Rung 0 is
    /// <paramref name="start"/> exactly, so growth can never start a run smaller than not growing would.
    /// </summary>
    /// <param name="start">The net's default trunk — rung 0.</param>
    /// <param name="steps">How many rungs to add above rung 0.</param>
    /// <param name="widenFactor">Multiplier applied on a widen step.</param>
    public static GrowthLadder FromTrunk(int[] start, int steps, double widenFactor = 1.5)
    {
        ArgumentNullException.ThrowIfNull(start);
        if (start.Length == 0) throw new ArgumentException("The starting trunk is empty.", nameof(start));
        ArgumentOutOfRangeException.ThrowIfNegative(steps);
        if (widenFactor <= 1) throw new ArgumentOutOfRangeException(nameof(widenFactor), "A widen must widen.");

        var rungs = new List<int[]> { start };
        for (int i = 0; i < steps; i++)
        {
            int[] previous = rungs[^1];
            // Alternate so capacity grows both ways rather than running off in one dimension: a trunk that only
            // widens hits diminishing returns per parameter, one that only deepens gets hard to optimise.
            rungs.Add(i % 2 == 0
                ? [.. previous.Select(w => (int)Math.Round(w * widenFactor))]
                : [.. previous, previous[^1]]);
        }
        return new GrowthLadder([.. rungs]);
    }

    public override string ToString()
        => string.Join(" → ", Rungs.Select(r => $"[{string.Join(",", r)}]"));
}
