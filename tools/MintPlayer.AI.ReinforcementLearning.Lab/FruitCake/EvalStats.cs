/// <summary>
/// The shared statistics behind the FruitCake evaluation verdicts (M63.5). <c>Std</c>, <c>Report</c> and
/// <c>Max</c> previously existed as byte-identical private copies in <see cref="FruitCakeAb"/> and
/// <see cref="FruitCakeSearchEval"/>, so the ship/no-ship rule was duplicated and untestable.
/// <para>
/// Formatting returns strings rather than writing to the console: <c>Console.Out</c> is process-global, so a
/// test that captured it would force the whole suite to run serially (PRD §12.3).
/// </para>
/// </summary>
internal static class EvalStats
{
    /// <summary>
    /// The <b>sample</b> standard deviation (Bessel-corrected, divisor <c>n−1</c>).
    /// </summary>
    /// <remarks>
    /// M63.5 bug fix: both copies previously divided by <c>n</c> (the POPULATION divisor) and the caller then
    /// formed <c>SE = Std/√n</c>. That understates the standard error by √((n−1)/n), which biases the
    /// "SIGNIFICANTLY BETTER → ship it" verdict towards shipping. The effect is small at the default episode
    /// counts (~0.5% at n=100) but grows as n falls, and it was wrong in the direction that matters.
    /// Returns 0 for n &lt; 2, where the sample SD is undefined — never NaN, which would poison every
    /// downstream comparison silently.
    /// </remarks>
    internal static double Std(IReadOnlyList<double> xs, double mean)
    {
        if (xs.Count < 2) return 0;
        double s = 0;
        foreach (var x in xs) s += (x - mean) * (x - mean);
        return Math.Sqrt(s / (xs.Count - 1));
    }

    /// <summary>Standard error of the mean: the sample SD over √n. 0 for n &lt; 2.</summary>
    internal static double StandardError(IReadOnlyList<double> xs)
    {
        if (xs.Count < 2) return 0;
        double mean = 0;
        foreach (var x in xs) mean += x;
        mean /= xs.Count;
        return Std(xs, mean) / Math.Sqrt(xs.Count);
    }

    /// <summary>
    /// The median, averaging the two middle values for an even count.
    /// </summary>
    /// <remarks>
    /// M63.5 bug fix: both copies used <c>sorted[Length / 2]</c>, which is the UPPER middle value for even
    /// n, not the median. It biased the reported median upward on every even-sized run.
    /// </remarks>
    internal static double Median(IReadOnlyList<double> xs)
    {
        if (xs.Count == 0) return double.NaN;
        var sorted = xs.ToArray();
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        return (sorted.Length & 1) == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    /// <summary>Largest value, or 0 for an empty span (matching the previous local helper).</summary>
    internal static double Max(IReadOnlyList<double> xs)
    {
        double m = 0;
        foreach (var x in xs) if (x > m) m = x;
        return m;
    }

    /// <summary>
    /// The ship/no-ship rule for a paired A/B run: significant at roughly 2 standard errors either way.
    /// </summary>
    /// <remarks>
    /// A tie deliberately keeps the baseline — shipping a candidate that is merely not-worse spends the
    /// review and retraining cost for nothing. <paramref name="se"/> of 0 (n &lt; 2) can never be
    /// significant, because <c>meanDiff &gt; 0</c> would otherwise declare a single episode decisive.
    /// </remarks>
    internal static string Verdict(double meanDiff, double se)
    {
        if (se <= 0) return "NO significant difference → keep the baseline (don't ship a tie)";
        return meanDiff > 2 * se ? "candidate is SIGNIFICANTLY BETTER → ship it"
            : meanDiff < -2 * se ? "candidate is SIGNIFICANTLY WORSE → keep the baseline"
            : "NO significant difference → keep the baseline (don't ship a tie)";
    }

    /// <summary>The per-arm summary line: mean ± SD, median, mean tier and the tier histogram.</summary>
    internal static string ReportLine(string label, double[] score, int[] tier)
    {
        double mean = score.Average();
        double sd = Std(score, mean);
        double median = Median(score);
        var hist = tier.GroupBy(t => t).OrderByDescending(g => g.Key).Select(g => $"t{g.Key}:{g.Count()}");
        return $"  {label}: mean {mean,7:F1} ± {sd,5:F0} (SD)  median {median,6:F0}  meanTier {tier.Average():F2}  [{string.Join(" ", hist)}]";
    }
}
