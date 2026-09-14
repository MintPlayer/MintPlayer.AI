using MintPlayer.AI.ReinforcementLearning.Campaigns;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The reverse curriculum's one real decision: how far into a level the net is asked to solve next.
/// </summary>
/// <remarks>
/// Everything the phase depends on rests here. Move the frontier out too eagerly and the level sets tasks the
/// net cannot do, which produces no training samples — and a level producing no samples cannot improve, so it
/// stays stuck for the rest of the run. That failure is silent: the run keeps logging healthy loss and accuracy
/// from the OTHER levels while one of them has quietly stopped contributing anything.
/// </remarks>
public class BlockDudeFrontierTests
{
    private static readonly BlockDudeExpertIterationOptions Options = new();

    private static int Next(int depth, int solved, int attempts = 4, int cap = 1_000)
        => BlockDudeExpertIterationCampaign.NextFrontier(depth, solved, attempts, cap, Options);

    [Fact]
    public void AClearMajorityMovesTheFrontierOutward()
    {
        Assert.True(Next(depth: 100, solved: 4) > 100);
        Assert.True(Next(depth: 100, solved: 3) > 100);   // exactly the 0.75 threshold
    }

    [Fact]
    public void AMinorityHoldsTheFrontierWhereItIs()
    {
        // Solving some but not most is the frontier sitting right at the edge of the net's ability, which is
        // where it belongs — neither direction is an improvement.
        Assert.Equal(100, Next(depth: 100, solved: 2));
        Assert.Equal(100, Next(depth: 100, solved: 1));
    }

    [Fact]
    public void SolvingNothingRetreats()
    {
        // The case that was missing. Growth is a 1.5x jump, so a level can be thrown past what the net can do.
        Assert.True(Next(depth: 150, solved: 0) < 150);
    }

    [Fact]
    public void RetreatIsGentlerThanGrowthSoTheFrontierSettlesRatherThanOscillates()
    {
        // If a retreat undid more than a growth added, a level at its limit would swing between two depths
        // forever instead of converging on the edge of the net's ability.
        const int depth = 200;
        int advanced = Next(depth, solved: 4);
        int retreated = Next(advanced, solved: 0);

        Assert.True(advanced > depth);
        Assert.True(retreated > depth, $"retreat from {advanced} fell to {retreated}, below the original {depth}");
    }

    [Fact]
    public void ARetreatAlwaysActuallyMoves()
    {
        // A multiplier alone rounds to a no-op at small depths (int)(2 * 0.8) == 1 is fine, but (int)(1 * 0.8)
        // is 0 and (int)(5 * 0.8) == 4 only by luck. A retreat that silently changed nothing would leave the
        // level just as stuck as having no retreat at all.
        for (int depth = 2; depth < 40; depth++)
            Assert.True(Next(depth, solved: 0) < depth, $"depth {depth} did not retreat");
    }

    [Fact]
    public void TheFrontierNeverRetreatsBelowASingleMove()
    {
        // Depth 0 is not a task — ReverseCurriculumStarts rejects it outright.
        Assert.Equal(1, Next(depth: 1, solved: 0));
        Assert.True(Next(depth: 2, solved: 0) >= 1);
    }

    [Fact]
    public void TheFrontierIsCappedAtTheWholeLevel()
    {
        // Past the solution's length there is no more level to ask for, and an uncapped frontier would silently
        // request start states that do not exist.
        Assert.Equal(94, Next(depth: 90, solved: 4, cap: 94));
        Assert.Equal(94, Next(depth: 94, solved: 4, cap: 94));
    }

    [Fact]
    public void GrowthAlwaysAdvancesAtLeastOneMove()
    {
        // At small depths the 1.5x multiplier truncates back to the same integer, which would freeze a level
        // that is succeeding — the opposite of the stall the retreat fixes, from the same rounding.
        for (int depth = 1; depth < 40; depth++)
            Assert.True(Next(depth, solved: 4) > depth, $"depth {depth} did not advance");
    }
}
