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
}
