namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

// Behaviour mirror of: src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-fruits.ts
// — keep the two ports in sync (PRD docs/prd/FRUITCAKE_AI_PRD.md §4.8).

/// <summary>One link in the merge chain. Tier is 1-based (1 = cherry … 11 = watermelon).</summary>
public readonly record struct FruitDef(int Tier, float RadiusPx, bool Droppable, int MergePoints);

/// <summary>
/// The canonical 11-fruit Suika chain (radii, scores) and the merge rule over it — a C# port of the
/// web game's <c>fruit-cake-fruits.ts</c> (training drops the colors/themes; only geometry + scoring
/// matter). Only tiers 1–5 are player-droppable; 6–11 exist only as merge products.
/// </summary>
public static class FruitCatalog
{
    /// <summary>Highest tier a player may drop; tiers above this only appear as merge products.</summary>
    public const int MaxDroppableTier = 5;

    public static readonly IReadOnlyList<FruitDef> Fruits =
    [
        new(1, 24f, true, 1),
        new(2, 32f, true, 3),
        new(3, 40f, true, 6),
        new(4, 56f, true, 10),
        new(5, 64f, true, 15),
        new(6, 72f, false, 21),
        new(7, 84f, false, 28),
        new(8, 96f, false, 36),
        new(9, 128f, false, 45),
        new(10, 160f, false, 55),
        new(11, 192f, false, 66),
    ];

    /// <summary>Top tier (watermelon); a pair of these vanishes instead of producing a new fruit.</summary>
    public static int TopTier => Fruits.Count;

    public static FruitDef ByTier(int tier) => Fruits[tier - 1];

    /// <summary>The tiers a player may drop (1..<see cref="MaxDroppableTier"/>).</summary>
    public static readonly IReadOnlyList<FruitDef> Droppable = Fruits.Where(f => f.Droppable).ToArray();

    /// <summary>The tier two fruit of <paramref name="tier"/> merge into, or null when a top-tier pair vanishes.</summary>
    public static int? MergeResultTier(int tier) => tier >= TopTier ? null : tier + 1;
}
