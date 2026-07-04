using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// PG1 fidelity gate for the single-source FruitCake solver (see docs/prd/POLYGLOT_FRUITCAKE_PRD.md):
/// the transpiled <c>PgFruitCakeWorld</c> (generated from fruitcake_solver.pg, f64/double) must reproduce
/// the hand-written <see cref="FruitCakeWorld"/> (float32). It runs identical scripted drops through both
/// and compares the integer outcomes (merge score + body count). The generated type is internal
/// (global namespace) — visible here via InternalsVisibleTo.
/// </summary>
public class PolyglotSolverParityTests
{
    // (tier, dropX). Settled with the env's real params (settleSpeed 30, min 8, max 600 substeps).
    private static readonly (int Tier, double X)[] Script =
    [
        (1, 305), (1, 312), (1, 308), (1, 315), (2, 310), (1, 250), (1, 256),
        (1, 300), (3, 310), (1, 260), (2, 258), (1, 400), (1, 406),
    ];

    private static (int Score, int Count) RunGenerated()
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

    private static (int Score, int Count) RunHandWritten()
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
    public void GeneratedSolver_IsDeterministic()
    {
        Assert.Equal(RunGenerated(), RunGenerated());
    }

    [Fact]
    public void GeneratedSolver_MatchesHandWrittenSolver_OnScriptedCascade()
    {
        // Faithfulness of the .pg port to the current physics. The generated solver is f64 and the
        // hand-written one is float32, so this asserts the merge OUTCOMES agree on a well-separated
        // cascade (not bit-exact float state — that gap is the PG3 f64-everywhere upgrade).
        Assert.Equal(RunHandWritten(), RunGenerated());
    }
}
