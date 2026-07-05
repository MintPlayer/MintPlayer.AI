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
}
