extern alias Lab; // the Lab exe's internal CLI helper (global namespace, aliased like the campaigns)

using System.Globalization;
using CliArgs = Lab::CliArgs;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Unit tests for the Lab's <c>CliArgs</c> flag reader (M38 B3): the parsing semantics every game Lab now shares —
/// defaults, typed parsing, last-occurrence-wins, missing-value tolerance, and (the bug B3 fixes) culture-invariant
/// numeric parsing.
/// </summary>
public class CliArgsTests
{
    [Fact]
    public void Missing_flags_return_their_defaults()
    {
        var a = new CliArgs([]);
        Assert.Equal(9.0, a.Dbl("--hours", 9));
        Assert.Equal("data", a.Str("--data", "data"));
        Assert.Equal(1UL, a.ULong("--seed", 1));
        Assert.Equal(2048, a.Int("--grow-every", 2048));
        Assert.False(a.Has("--grow"));
        Assert.Equal([128, 128], a.Ints("--hidden", [128, 128]));
    }

    [Fact]
    public void Parses_the_value_after_each_flag()
    {
        var a = new CliArgs(["--hours", "2.5", "--seed", "42", "--steps", "100000", "--lr", "5e-4",
                             "--data", "./d", "--grow", "--hidden", "64,64,64"]);
        Assert.Equal(2.5, a.Dbl("--hours", 1));
        Assert.Equal(42UL, a.ULong("--seed", 1));
        Assert.Equal(100000L, a.Long("--steps", 0));
        Assert.Equal(5e-4f, a.Flt("--lr", 1e-3f));
        Assert.Equal("./d", a.Str("--data", "data"));
        Assert.True(a.Has("--grow"));
        Assert.Equal([64, 64, 64], a.Ints("--hidden", [1]));
    }

    [Fact]
    public void Last_occurrence_of_a_flag_wins()
    {
        // Matches the old forward-loop behaviour (each match reassigned the variable).
        var a = new CliArgs(["--seed", "1", "--seed", "7"]);
        Assert.Equal(7UL, a.ULong("--seed", 0));
    }

    [Fact]
    public void A_flag_with_no_following_value_falls_back_to_the_default()
    {
        var a = new CliArgs(["--lr"]); // trailing flag, no value token
        Assert.Equal(3e-4f, a.Flt("--lr", 3e-4f));
    }

    [Fact]
    public void An_exact_flag_name_is_not_confused_with_a_longer_one()
    {
        var a = new CliArgs(["--grow-every", "5000"]); // must NOT register "--grow"
        Assert.False(a.Has("--grow"));
        Assert.Equal(5000, a.Int("--grow-every", 0));
    }

    [Fact]
    public void Numeric_parsing_is_culture_invariant()
    {
        // The bug B3 fixes: on a comma-decimal machine locale, a dot-decimal CLI value like "0.997" must still
        // parse to 0.997 — not 997 (dot read as a thousands group) — because CliArgs pins InvariantCulture.
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("nl-BE"); // ',' decimal, '.' grouping
            var a = new CliArgs(["--lr", "0.0005", "--gamma", "0.997"]);
            Assert.Equal(0.0005f, a.Flt("--lr", 1f));
            Assert.Equal(0.997, a.Dbl("--gamma", 0));
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }
}
