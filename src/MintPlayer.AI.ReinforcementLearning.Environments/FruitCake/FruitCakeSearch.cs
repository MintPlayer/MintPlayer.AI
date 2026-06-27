namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// Serving-side forward-model search for FruitCake (PRD FRUITCAKE_IMPROVE lever A / F1). For each column it
/// clones the board, drops a fruit, settles to rest (the exact loop the live game runs), prunes lines that lose,
/// and recurses. The current + next fruit are <b>known</b> (deterministic maximization); a third look-ahead ply
/// (<see cref="MaxDepth"/>&#160;= 3) is an <b>expectimax chance node</b> over the unknown upcoming fruit (average
/// over the droppable tiers). A line's value is its realized merge points plus a board-value estimate of the leaf
/// — the trained net (its learned sense of a good board), or a heuristic when no net is loaded. This
/// <b>amplifies the shipped net without retraining</b>: the reactive net plateaued at pineapple because it can't
/// plan; a 2–3 drop lookahead does the planning the literature shows is what actually beats humans at Suika.
///
/// <para>The leaf board value is injected (<see cref="FruitCakeSearch(Func{FruitCakeWorld, double})"/>) so this
/// type stays free of any net/Core dependency: the serving layer passes the net's max-Q; tests/eval pass either.
/// Planning runs on rotation-off clones (cheaper, deterministic; merges don't depend on orientation), so it
/// works against a live rotation-on world too.</para>
/// </summary>
public sealed class FruitCakeSearch(Func<FruitCakeWorld, double> leafBoardValue)
{
    /// <summary>1 = greedy one-drop lookahead; 2 = plan current + next (default); 3 = + an expectimax ply over the unknown 3rd fruit.</summary>
    public int MaxDepth { get; set; } = 2;

    /// <summary>First-ply expansion width: only the top-K columns (by one-drop value) are recursed; the rest keep
    /// their one-drop value. Default 10 is the empirical depth-2 sweet spot (200-game eval: 30% watermelon).</summary>
    public int TopK { get; set; } = 10;

    /// <summary>Expansion width at the deeper known plies (depth&#160;≥&#160;3), keeping the chance-node cost bounded.</summary>
    public int TopK2 { get; set; } = 3;

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

        // Refine the top-K non-losing first plies with the best continuation (next drop is known); others keep
        // their one-drop value. MaxDepth-1 more drops are simulated, starting with the known `next` fruit.
        var topK = SelectTopK(first, TopK);
        double bestValue = double.NegativeInfinity;
        int bestCol = columns / 2;
        for (int col = 0; col < columns; col++)
        {
            var f = first[col];
            double value = f.Lost
                ? MergeWeight * f.Points - LosePenalty            // losing first drop: never recurse
                : topK[col]
                    ? MergeWeight * f.Points + BestContinuation(f.World, next, MaxDepth - 1)
                    : f.OneDropValue;
            if (value > bestValue) { bestValue = value; bestCol = col; }
        }
        return bestCol;
    }

    /// <summary>
    /// Best achievable value (weighted merge points + leaf board value) from <paramref name="world"/> over
    /// <paramref name="pliesLeft"/> more drops. The immediate drop's fruit is <paramref name="fruit"/> when known;
    /// when null it's an expectimax chance node — average over the droppable tiers (nature reveals the fruit, then
    /// we pick the best column). The fruit after each known drop is unknown, so it recurses into a chance node.
    /// </summary>
    private double BestContinuation(FruitCakeWorld world, int? fruit, int pliesLeft)
    {
        if (pliesLeft == 0)
            return leafBoardValue(world);

        if (fruit is not int f)
        {
            // Chance node: the upcoming fruit is unknown — average optimal play over the droppable tiers.
            double sum = 0;
            foreach (var d in FruitCatalog.Droppable)
                sum += BestContinuation(world, d.Tier, pliesLeft);
            return sum / FruitCatalog.Droppable.Count;
        }

        int columns = FruitCakeEnv.ColumnCount;
        var plies = new PlyResult[columns];
        for (int c = 0; c < columns; c++)
            plies[c] = DropAndScore(world, f, c);

        // Last simulated drop: the one-drop value (merge pts + leaf) is already the full value — take the max.
        if (pliesLeft == 1)
        {
            double best = double.NegativeInfinity;
            for (int c = 0; c < columns; c++) best = Math.Max(best, plies[c].OneDropValue);
            return best;
        }

        // Deeper: expand only the top-K columns into the next (unknown) fruit; the rest keep their one-drop value.
        var keep = SelectTopK(plies, TopK2);
        double bestDeep = double.NegativeInfinity;
        for (int c = 0; c < columns; c++)
        {
            var p = plies[c];
            double v = p.Lost
                ? MergeWeight * p.Points - LosePenalty
                : keep[c]
                    ? MergeWeight * p.Points + BestContinuation(p.World, null, pliesLeft - 1)
                    : p.OneDropValue;
            bestDeep = Math.Max(bestDeep, v);
        }
        return bestDeep;
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
