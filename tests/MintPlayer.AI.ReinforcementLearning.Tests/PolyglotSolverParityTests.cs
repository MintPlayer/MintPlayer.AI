using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// PG1 gate for the single-source FruitCake solver (docs/prd/POLYGLOT_FRUITCAKE_PRD.md). The physics is
/// transpiled once from fruitcake_solver.pg into the internal <c>PgFruitCakeWorld</c>; the public
/// <see cref="FruitCakeWorld"/> is a thin facade over it. These pin (a) the facade delegates to the generated
/// core faithfully — same outcomes through the public API as the raw core — and (b) the solver is deterministic.
/// The generated core is internal (global namespace), visible here via InternalsVisibleTo.
/// </summary>
public class PolyglotSolverParityTests
{
    // (tier, dropX). Settled with the env's real params (settleSpeed 30, min 8, max 600 substeps).
    private static readonly (int Tier, double X)[] Script =
    [
        (1, 305), (1, 312), (1, 308), (1, 315), (2, 310), (1, 250), (1, 256),
        (1, 300), (3, 310), (1, 260), (2, 258), (1, 400), (1, 406),
    ];

    private static (int Score, int Count) RunCore()
    {
        var w = new PgFruitCakeWorld();
        int score = 0;
        foreach (var (tier, x) in Script)
        {
            w.spawnFruit(tier, x, 90.0);
            score += w.settleAfterDrop(30.0, 8, 600);
        }
        return (score, w.count);
    }

    private static (int Score, int Count) RunFacade()
    {
        var w = new FruitCakeWorld();
        int score = 0;
        foreach (var (tier, x) in Script)
        {
            w.SpawnFruit(tier, (float)x, 90f);
            score += w.SettleAfterDrop(30f, 8, 600);
        }
        return (score, w.Count);
    }

    [Fact]
    public void PublicFacade_DelegatesFaithfullyTo_GeneratedCore()
    {
        Assert.Equal(RunCore(), RunFacade());
    }

    [Fact]
    public void GeneratedSolver_IsDeterministic()
    {
        Assert.Equal(RunFacade(), RunFacade());
    }

    // CS0 (docs/prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md): the search's world-queries now live in the .pg (single
    // source, so the browser search reuses them). These pin the generated core clone/query behaviour the
    // C# facade delegates to (and the TS adapter gets byte-identically from the same .pg).

    [Fact]
    public void CoreClone_IsIndependentDeepCopy()
    {
        var w = new PgFruitCakeWorld();
        foreach (var (tier, x) in Script)
        {
            w.spawnFruit(tier, x, 90.0);
            w.settleAfterDrop(30.0, 8, 600);
        }

        var copy = w.clone(false);
        Assert.Equal(w.count, copy.count);

        int countBefore = w.count;
        double sumXBefore = 0; foreach (var b in w.bodies) sumXBefore += b.x;

        // Mutate the clone (a tier-11 has no merge partner, so it just adds one body); the original must be untouched.
        copy.spawnFruit(11, 300.0, 300.0);
        copy.settleAfterDrop(30.0, 8, 600);

        double sumXAfter = 0; foreach (var b in w.bodies) sumXAfter += b.x;
        double sumXCopy = 0; foreach (var b in copy.bodies) sumXCopy += b.x;
        Assert.Equal(countBefore, w.count);          // original count unchanged
        Assert.Equal(sumXBefore, sumXAfter);         // original positions unchanged
        Assert.NotEqual(sumXBefore, sumXCopy);       // the mutation actually landed on the clone
    }

    [Fact]
    public void FacadeQueries_DelegateToCore()
    {
        // Seed the core with the SAME float-cast inputs the facade uses, so both sims are bit-identical and the
        // continuous pile-height matches exactly — this isolates the query delegation, not input precision.
        var core = new PgFruitCakeWorld();
        var facade = new FruitCakeWorld();
        foreach (var (tier, x) in Script)
        {
            core.spawnFruit(tier, (float)x, 90f);
            core.settleAfterDrop(30f, 8, 600);
            facade.SpawnFruit(tier, (float)x, 90f);
            facade.SettleAfterDrop(30f, 8, 600);
        }

        Assert.Equal(core.anyEjected(), facade.AnyEjected());
        Assert.Equal(core.anyRestingAboveDangerLine(40.0), facade.AnyRestingAboveDangerLine(40f));
        Assert.Equal((float)core.pileHeight(), facade.PileHeight());
    }

    // CS1 (docs/prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md): BuildObservation now lives in the .pg (f64), and the C#
    // env delegates (core f64 -> float). This pins that the single-source obs reproduces the legacy float32
    // observation the net was trained on, within f32/f64 rounding — so the argmax-relevant features are unchanged
    // and no retrain is needed. (Also validates the cast-free column-overlap reformulation on real boards.)
    [Fact]
    public void CoreObservation_MatchesLegacyFloat32_WithinTolerance()
    {
        var w = new FruitCakeWorld();
        foreach (var (tier, x) in Script) { w.SpawnFruit(tier, (float)x, 90f); w.SettleAfterDrop(30f, 8, 600); }

        foreach (var (cur, next) in new[] { (1, 2), (3, 5), (5, 1) })
        {
            var actual = FruitCakeEnv.BuildObservation(w, cur, next); // single-source (.pg f64 -> float)
            var legacy = LegacyObservationFloat32(w, cur, next);
            Assert.Equal(legacy.Length, actual.Length);
            for (int i = 0; i < legacy.Length; i++)
                Assert.True(Math.Abs(legacy[i] - actual[i]) < 1e-5f,
                    $"obs[{i}] (cur={cur},next={next}) legacy={legacy[i]} actual={actual[i]}");
        }
    }

    // The pre-.pg float32 observation, kept here as the one-time equivalence reference for the port above.
    private static float[] LegacyObservationFloat32(FruitCakeWorld world, int current, int next)
    {
        const int ColumnCount = FruitCakeEnv.ColumnCount;
        const float W = FruitCakeWorld.Width, H = FruitCakeWorld.Height, DangerY = FruitCakeWorld.DangerLineY;
        float binW = W / ColumnCount;

        var topY = new float[ColumnCount];
        var topTier = new int[ColumnCount];
        for (int c = 0; c < ColumnCount; c++) topY[c] = H;

        float fillArea = 0f;
        foreach (var b in world.Bodies)
        {
            fillArea += MathF.PI * b.R * b.R;
            int c0 = Math.Clamp((int)((b.X - b.R) / binW), 0, ColumnCount - 1);
            int c1 = Math.Clamp((int)((b.X + b.R) / binW), 0, ColumnCount - 1);
            float t = b.Y - b.R;
            for (int c = c0; c <= c1; c++)
                if (t < topY[c]) { topY[c] = t; topTier[c] = b.Tier; }
        }

        var obs = new float[FruitCakeEnv.ObservationSize];
        int i = 0;
        float minTop = H;
        for (int c = 0; c < ColumnCount; c++)
        {
            obs[i++] = Math.Clamp((H - topY[c]) / H, 0f, 1f);
            obs[i++] = Math.Clamp(topTier[c] / 11f, 0f, 1f);
            if (topY[c] < minTop) minTop = topY[c];
        }
        for (int c = 0; c < ColumnCount; c++)
            obs[i++] = Math.Clamp((topY[c] - DangerY) / (H - DangerY), 0f, 1f);
        for (int c = 0; c < ColumnCount; c++)
            obs[i++] = topTier[c] == current ? 1f : 0f;
        for (int c = 0; c < ColumnCount; c++)
        {
            bool pair = topTier[c] > 0 &&
                        ((c > 0 && topTier[c - 1] == topTier[c]) ||
                         (c < ColumnCount - 1 && topTier[c + 1] == topTier[c]));
            obs[i++] = pair ? 1f : 0f;
        }
        for (int t = 1; t <= FruitCatalog.MaxDroppableTier; t++) obs[i++] = current == t ? 1f : 0f;
        for (int t = 1; t <= FruitCatalog.MaxDroppableTier; t++) obs[i++] = next == t ? 1f : 0f;
        obs[i++] = Math.Clamp(world.Count / 100f, 0f, 1f);
        obs[i++] = Math.Clamp(fillArea / (W * H), 0f, 1f);
        obs[i++] = Math.Clamp(minTop / H, 0f, 1f);

        // Block E — tier-occupancy grid (ColumnCount × GridRows): max overlapping tier per cell ÷ 11 (mirrors the .pg).
        const int GridRows = FruitCakeEnv.GridRows;
        float binH = H / GridRows;
        var grid = new int[ColumnCount * GridRows];
        foreach (var b in world.Bodies)
        {
            float bl = b.X - b.R, br = b.X + b.R, bt = b.Y - b.R, bb = b.Y + b.R;
            for (int c = 0; c < ColumnCount; c++)
            {
                if (!(br > binW * c && bl < binW * (c + 1))) continue;
                for (int row = 0; row < GridRows; row++)
                    if (bb > binH * row && bt < binH * (row + 1))
                    {
                        int idx = row * ColumnCount + c;
                        if (b.Tier > grid[idx]) grid[idx] = b.Tier;
                    }
            }
        }
        for (int k = 0; k < ColumnCount * GridRows; k++)
            obs[i++] = Math.Clamp(grid[k] / 11f, 0f, 1f);

        FruitBody? big1 = null, big2 = null;
        foreach (var b in world.Bodies)
        {
            if (big1 is null || IsLarger(b, big1)) (big1, big2) = (b, big1);
            else if (big2 is null || IsLarger(b, big2)) big2 = b;
        }
        WriteBig(obs, ref i, big1);
        WriteBig(obs, ref i, big2);
        return obs;

        static bool IsLarger(FruitBody cand, FruitBody cur)
        {
            if (cand.Tier != cur.Tier) return cand.Tier > cur.Tier;
            if (cand.Y != cur.Y) return cand.Y > cur.Y;
            return cand.X < cur.X;
        }
        static void WriteBig(float[] o, ref int idx, FruitBody? b)
        {
            if (b is not null)
            {
                o[idx++] = Math.Clamp(b.X / FruitCakeWorld.Width, 0f, 1f);
                o[idx++] = Math.Clamp(b.Y / FruitCakeWorld.Height, 0f, 1f);
                o[idx++] = Math.Clamp(b.Tier / 11f, 0f, 1f);
            }
            else { o[idx++] = 0.5f; o[idx++] = 1f; o[idx++] = 0f; }
        }
    }
}
