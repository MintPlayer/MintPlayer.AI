using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The training-board generator. Its contract matters twice over: the boards must be labellable by the exact
/// oracle, and generation must be a pure function of the RNG so a run replays from a blank slate (PRD §7.1).
/// </summary>
public class BlockDudeGeneratorTests
{
    // A small early-curriculum rung, well inside the measured oracle frontier (≤6-7 blocks / ≤150 free cells).
    private static BlockDudeStageSpec EarlyStage() => new(
        MinWidth: 9, MaxWidth: 12,
        MinHeight: 6, MaxHeight: 8,
        MinBlocks: 1, MaxBlocks: 3,
        MinOptimal: 4, MaxOptimal: 60,
        MaxFreeCells: 60,
        OracleMaxStates: 60_000);

    [Fact]
    public void GenerationIsDeterministic_ForAGivenSeed()
    {
        // The blank-slate guarantee rests on this: the same seed must produce the same boards, in the same order.
        var first = Draw(new Xoshiro256StarStar(58), 6);
        var second = Draw(new Xoshiro256StarStar(58), 6);

        Assert.NotEmpty(first);
        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
            Assert.Equal(first[i], second[i]);
    }

    [Fact]
    public void DifferentSeedsProduceDifferentBoards()
    {
        var a = Draw(new Xoshiro256StarStar(1), 4);
        var b = Draw(new Xoshiro256StarStar(2), 4);
        Assert.NotEmpty(a);
        Assert.NotEmpty(b);
        Assert.NotEqual(a[0], b[0]);
    }

    [Fact]
    public void AcceptedBoards_SatisfyTheStageSpec()
    {
        var spec = EarlyStage();
        var rng = new Xoshiro256StarStar(7);

        int checkedBoards = 0;
        for (int i = 0; i < 40 && checkedBoards < 8; i++)
        {
            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            if (outcome != BlockDudeGenerationOutcome.Accepted) continue;
            Assert.NotNull(board);
            checkedBoards++;

            Assert.InRange(board!.Width, spec.MinWidth, spec.MaxWidth);
            Assert.InRange(board.Height, spec.MinHeight, spec.MaxHeight);
            Assert.InRange(board.BlockCells.Count, spec.MinBlocks, spec.MaxBlocks);

            var grid = board.ToGrid();
            Assert.True(grid.Sum(r => r.Count(c => c != 'W')) <= spec.MaxFreeCells);

            var oracle = new BlockDudeOracle(board, spec.OracleMaxStates);
            Assert.False(oracle.Truncated);
            Assert.InRange(oracle.OptimalFromStart, spec.MinOptimal, spec.MaxOptimal);
        }

        Assert.True(checkedBoards > 0, "the early stage produced no acceptable board in 40 attempts");
    }

    [Fact]
    public void AcceptedBoards_ActuallyRequireCarryingABlock()
    {
        // The point of the game. A board solvable without ever touching a block is scenery, and the generator
        // rejects it — so every accepted board must be unsolvable once the blocks are stripped.
        var spec = EarlyStage();
        var rng = new Xoshiro256StarStar(11);

        int verified = 0;
        for (int i = 0; i < 40 && verified < 5; i++)
        {
            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            if (outcome != BlockDudeGenerationOutcome.Accepted) continue;

            var blockless = BlockDudeBoard.FromGrid([.. board!.ToGrid().Select(r => r.Replace('B', '.'))]);
            var oracle = new BlockDudeOracle(blockless, spec.OracleMaxStates);
            Assert.True(oracle.Truncated || oracle.OptimalFromStart < 0,
                "an accepted board was solvable without using any block");
            verified++;
        }

        Assert.True(verified > 0);
    }

    [Fact]
    public void TheGeneratorProducesOverhangs()
    {
        // Real Block Dude terrain is not a skyline: every original level has overhangs. A generator that cannot
        // produce them would train the net on terrain that never appears in the content it must play.
        var spec = EarlyStage();
        var rng = new Xoshiro256StarStar(3);

        bool sawOverhang = false;
        for (int i = 0; i < 120 && !sawOverhang; i++)
        {
            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            if (outcome != BlockDudeGenerationOutcome.Accepted) continue;

            var grid = board!.ToGrid();
            for (int y = 1; y < grid.Length - 1 && !sawOverhang; y++)
                for (int x = 1; x < grid[y].Length - 1; x++)
                    if (grid[y][x] == 'W' && grid[y + 1][x] == '.') { sawOverhang = true; break; }
        }

        Assert.True(sawOverhang, "no accepted board had an overhang");
    }

    [Fact]
    public void RejectionsAreTallied_SoTheAcceptRateIsVisible()
    {
        // A silently-falling accept rate biases the label set toward shallow puzzles, so the campaign must be
        // able to report WHY boards were thrown away, not just how many survived.
        var tally = new Dictionary<BlockDudeGenerationOutcome, int>();
        BlockDudeGenerator.Generate(new Xoshiro256StarStar(23), EarlyStage(), maxAttempts: 25, tally);

        Assert.NotEmpty(tally);
        Assert.True(tally.Values.Sum() > 0);
    }

    [Fact]
    public void GeneratedBoardsNeverCollideWithAShippedLevel()
    {
        // The 11 originals are the held-out gate set.
        var shipped = BlockDudeLevels.All.Select(l => string.Join("\n", l.Grid)).ToHashSet();
        foreach (var grid in Draw(new Xoshiro256StarStar(99), 10))
            Assert.DoesNotContain(grid, shipped);
    }

    private static List<string> Draw(Xoshiro256StarStar rng, int wanted)
    {
        var spec = EarlyStage();
        var boards = new List<string>();
        for (int i = 0; i < 60 && boards.Count < wanted; i++)
        {
            var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
            if (outcome == BlockDudeGenerationOutcome.Accepted)
                boards.Add(string.Join("\n", board!.ToGrid()));
        }
        return boards;
    }
}
