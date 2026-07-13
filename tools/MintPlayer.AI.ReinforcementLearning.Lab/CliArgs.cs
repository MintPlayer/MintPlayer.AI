using System.Globalization;

/// <summary>
/// A tiny reader over a Lab's <c>string[] args</c>: each accessor finds a <c>--flag</c> and parses the token after
/// it, hiding the three things every hand-rolled flag loop repeated — the <c>i + 1 &lt; args.Length</c> bounds
/// guard, stepping past the consumed value, and (the part that was applied inconsistently) parsing with
/// <see cref="CultureInfo.InvariantCulture"/> so a comma-decimal machine locale can't reinterpret <c>5e-4</c> or a
/// thousands-grouped integer. A flag's LAST occurrence wins, matching the old forward-loop-overwrites behaviour; a
/// missing flag — or one with no following value — yields the supplied default. Boolean switches use <see cref="Has"/>.
/// </summary>
internal readonly struct CliArgs(string[] args)
{
    /// <summary>True when the switch is present anywhere (e.g. <c>--grow</c>, <c>--eval-only</c>).</summary>
    public bool Has(string flag) => Array.IndexOf(args, flag) >= 0;

    /// <summary>The token following the flag's LAST occurrence, or null if the flag is absent / has no value.</summary>
    private string? Value(string flag)
    {
        int i = Array.LastIndexOf(args, flag);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    public string Str(string flag, string @default) => Value(flag) ?? @default;
    public double Dbl(string flag, double @default) => Value(flag) is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : @default;
    public float Flt(string flag, float @default) => Value(flag) is { } v ? float.Parse(v, CultureInfo.InvariantCulture) : @default;
    public int Int(string flag, int @default) => Value(flag) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : @default;
    public long Long(string flag, long @default) => Value(flag) is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : @default;
    public ulong ULong(string flag, ulong @default) => Value(flag) is { } v ? ulong.Parse(v, CultureInfo.InvariantCulture) : @default;

    /// <summary>A comma-separated int list (e.g. <c>--hidden 256,256</c>), or the default when the flag is absent.</summary>
    public int[] Ints(string flag, int[] @default)
        => Value(flag) is { } v ? [.. v.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture))] : @default;
}
