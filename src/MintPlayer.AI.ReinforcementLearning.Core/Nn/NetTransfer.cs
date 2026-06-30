namespace MintPlayer.AI.ReinforcementLearning.Core.Nn;

/// <summary>
/// Generic, function-preserving weight transfers shared across <see cref="IValueNet"/> architectures — the single
/// home for the transfer mechanics that are NOT tied to a specific net's structure. The structure-specific
/// transforms deliberately stay on their nets, where the structural knowledge lives: hidden-width growth on
/// <see cref="ResidualMlp.WidenTo"/> (Net2WiderNet), plain→noisy head promotion on <see cref="DuelingQNet.ToNoisy"/>,
/// and trunk extraction on <c>CubePolicyNet.PolicyAsMlp</c>. This class holds the two that are generic: an exact
/// parameter copy (identical-shape target sync) and input-dimension growth (<see cref="IValueNet.GrowInput"/>).
///
/// <para><b>Invariant every <see cref="IValueNet"/> obeys:</b> the first parameter yielded by
/// <see cref="IModule.Parameters"/> is the input-consuming weight, shaped [<see cref="IValueNet.InputSize"/>,
/// firstHidden], and the second is its bias. <see cref="TransferGrownInput"/> depends on this and asserts it.</para>
/// </summary>
public static class NetTransfer
{
    /// <summary>
    /// Copies every parameter from <paramref name="src"/> into the structurally identical <paramref name="dst"/> —
    /// the target-network sync behind every <see cref="IValueNet.CopyFrom"/>. Parameters are matched in enumeration
    /// order; each pair is the same length by construction (both nets share a structure).
    /// </summary>
    public static void CopyParameters(IValueNet dst, IValueNet src)
    {
        using var mine = dst.Parameters().GetEnumerator();
        using var theirs = src.Parameters().GetEnumerator();
        while (mine.MoveNext() && theirs.MoveNext())
            theirs.Current.Data.CopyTo(mine.Current.Data.AsSpan());
    }

    /// <summary>
    /// Carries <paramref name="original"/>'s learned parameters into the freshly-constructed, wider-input
    /// <paramref name="grown"/> (same config, larger <see cref="IValueNet.InputSize"/>) so the result computes the
    /// IDENTICAL function on the original features: the input weight's existing rows are copied into the prefix and
    /// the new rows are left ZERO — since the input weight is row-major [in, out], the old weight is exactly the
    /// first <c>oldIn × out</c> entries, and zero rows for the new inputs contribute nothing to any output until
    /// trained. Every other parameter keeps its shape (only the input grew) and is copied exactly.
    /// </summary>
    public static void TransferGrownInput(IValueNet grown, IValueNet original)
    {
        if (grown.InputSize <= original.InputSize)
            throw new ArgumentException(
                $"GrowInput expects a larger input than the current {original.InputSize}, got {grown.InputSize}.");

        using var dst = grown.Parameters().GetEnumerator();
        using var src = original.Parameters().GetEnumerator();

        // First parameter = the input-consuming weight [InputSize, firstHidden]: copy the old input rows into the
        // prefix, leave the new rows zero (function-preserving). The invariant is asserted, not assumed.
        if (!dst.MoveNext() || !src.MoveNext())
            throw new InvalidOperationException("GrowInput: the value net yielded no parameters.");
        var newWeight = dst.Current;
        var oldWeight = src.Current;
        if (newWeight.Rows != grown.InputSize || oldWeight.Rows != original.InputSize || newWeight.Cols != oldWeight.Cols)
            throw new InvalidOperationException(
                "GrowInput contract violated: a value net's first parameter must be its input weight [InputSize, firstHidden].");
        Array.Clear(newWeight.Data);
        oldWeight.Data.CopyTo(newWeight.Data.AsSpan());

        // Every remaining parameter is shape-unchanged by an input grow → exact copy.
        while (dst.MoveNext() && src.MoveNext())
        {
            if (dst.Current.Data.Length != src.Current.Data.Length)
                throw new InvalidOperationException("GrowInput: non-input parameters must match in length.");
            src.Current.Data.CopyTo(dst.Current.Data.AsSpan());
        }
    }
}
