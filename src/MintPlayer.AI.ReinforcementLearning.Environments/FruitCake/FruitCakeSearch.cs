namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// Serving-side forward-model search for FruitCake (PRD FRUITCAKE_IMPROVE lever A / F1). The physics is fully
/// deterministic and the current + next fruit are both known, so this is exact maximization over the next one
/// or two drops — <b>no chance node</b>, unlike 2048 expectimax (cleaner). For each column it clones the board,
/// drops the current fruit, settles to rest (the exact loop the live game runs), prunes lines that lose, and at
/// depth&#160;2 recurses on the <i>next</i> fruit over the top-K most promising first plies. A line's value is
/// its realized merge points plus a board-value estimate of the leaf — the trained net (its learned sense of a
/// good board), or a heuristic when no net is loaded. This <b>amplifies the shipped net without retraining</b>:
/// the reactive net plateaued at pineapple because it can't plan; a 1–2 drop lookahead does the planning the
/// literature shows is what actually beats humans at Suika.
///
/// <para>The leaf board value is injected (<see cref="FruitCakeSearch(Func{FruitCakeWorld, double})"/>) so this
/// type stays free of any net/Core dependency: the serving layer passes the net's max-Q; tests/eval pass either.
/// Planning runs on rotation-off clones (cheaper, deterministic; merges don't depend on orientation), so it
/// works against a live rotation-on world too.</para>
/// </summary>
public sealed class FruitCakeSearch(Func<FruitCakeWorld, double> leafBoardValue)
{
    /// <summary>1 = greedy one-drop lookahead; 2 = plan the current + next drop (default).</summary>
    public int MaxDepth { get; set; } = 2;

    /// <summary>At depth&#160;2, only the top-K first-ply columns (by their one-drop value) are expanded — the
    /// rest keep their one-drop value. Caps cost to ~ColumnCount + K·ColumnCount settles per decision.</summary>
    public int TopK { get; set; } = 5;

    /// <summary>Weight on realized merge points relative to the leaf board value (mirrors the heuristic's merge bias).</summary>
    public double MergeWeight { get; set; } = 1.0;

    // Large enough to dominate any board-value/merge difference, so a losing line is chosen only if every line loses.
    private const double LosePenalty = 1e9;

    /// <summary>Default leaf value when no net is available: prefer a lower pile (mirrors the greedy heuristic).</summary>
    public static double HeuristicBoardValue(FruitCakeWorld world) => -world.PileHeight();

    /// <summary>Pick the best column (0..ColumnCount-1) to drop <paramref name="current"/> into, planning ahead with <paramref name="next"/>.</summary>
    public int ChooseColumn(FruitCakeWorld world, int current, int next)
    {
        int columns = FruitCakeEnv.ColumnCount;

        // First ply: drop `current` in every column, settle, score one-drop value.
        var first = new PlyResult[columns];
        for (int col = 0; col < columns; col++)
            first[col] = DropAndScore(world, current, col);

        if (MaxDepth <= 1)
            return ArgMax(first, static p => p.OneDropValue);

        // Depth 2: refine the top-K non-losing first plies with the best `next` drop; others keep one-drop value.
        var topK = SelectTopK(first, TopK);
        double bestValue = double.NegativeInfinity;
        int bestCol = columns / 2;
        for (int col = 0; col < columns; col++)
        {
            var f = first[col];
            double value;
            if (f.Lost)
                value = MergeWeight * f.Points - LosePenalty; // losing first drop: never recurse
            else if (topK[col])
            {
                double bestNext = double.NegativeInfinity;
                for (int col2 = 0; col2 < columns; col2++)
                    bestNext = Math.Max(bestNext, DropAndScore(f.World, next, col2).OneDropValue);
                value = MergeWeight * f.Points + bestNext; // realized first-drop merges + best second-drop line
            }
            else
                value = f.OneDropValue;

            if (value > bestValue) { bestValue = value; bestCol = col; }
        }
        return bestCol;
    }

    private readonly record struct PlyResult(FruitCakeWorld World, int Points, bool Lost, double OneDropValue);

    private PlyResult DropAndScore(FruitCakeWorld world, int tier, int col)
    {
        var sim = world.Clone(enableRotation: false);
        sim.SpawnFruit(tier, FruitCakeEnv.ColumnX(col, tier), FruitCakeEnv.HeldY(tier));
        int points = sim.SettleAfterDrop(FruitCakeEnv.SettleSpeedPx, FruitCakeEnv.MinSettleSubsteps, FruitCakeEnv.MaxSubsteps);
        bool lost = sim.AnyEjected() || sim.AnyRestingAboveDangerLine(FruitCakeEnv.RestSpeedPx);
        double value = lost
            ? MergeWeight * points - LosePenalty
            : MergeWeight * points + leafBoardValue(sim);
        return new PlyResult(sim, points, lost, value);
    }

    // Mark the K highest-one-drop-value non-losing columns for depth-2 expansion (all of them if fewer than K survive).
    private static bool[] SelectTopK(PlyResult[] plies, int k)
    {
        var keep = new bool[plies.Length];
        var order = Enumerable.Range(0, plies.Length)
            .Where(i => !plies[i].Lost)
            .OrderByDescending(i => plies[i].OneDropValue)
            .Take(Math.Max(1, k));
        foreach (int i in order) keep[i] = true;
        return keep;
    }

    private static int ArgMax(PlyResult[] plies, Func<PlyResult, double> value)
    {
        double best = double.NegativeInfinity;
        int bestCol = plies.Length / 2;
        for (int i = 0; i < plies.Length; i++)
        {
            double v = value(plies[i]);
            if (v > best) { best = v; bestCol = i; }
        }
        return bestCol;
    }
}
