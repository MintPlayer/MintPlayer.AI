using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Guards the F1 forward-model search: it returns a legal column and is deterministic (serving must be
/// reproducible). Behavioural strength (the watermelon lift) is validated by the headless search-eval, not
/// unit tests — physics-constructed "this column loses" boards are too fragile to assert on here.
/// </summary>
public class FruitCakeSearchTests
{
    private static FruitCakeWorld PlayedBoard(ulong seed, int drops)
    {
        // Drive the env a few drops to get a non-trivial board, then search over its world.
        var env = new FruitCakeEnv();
        env.Reset(seed);
        for (int i = 0; i < drops; i++)
        {
            var step = env.Step(i % FruitCakeEnv.ColumnCount);
            if (step.Done) { env.Reset(seed); }
        }
        return env.World;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)] // depth 3 adds an expectimax chance node over the unknown 3rd fruit
    public void ChooseColumn_returns_a_legal_column(int depth)
    {
        var search = new FruitCakeSearch(FruitCakeSearch.HeuristicBoardValue) { MaxDepth = depth, TopK = 5, TopK2 = 2 };
        int col = search.ChooseColumn(PlayedBoard(7, 12), current: 1, next: 2);
        Assert.InRange(col, 0, FruitCakeEnv.ColumnCount - 1);
    }

    [Fact]
    public void ChooseColumn_is_deterministic()
    {
        var world = PlayedBoard(11, 15);
        var search = new FruitCakeSearch(FruitCakeSearch.HeuristicBoardValue) { MaxDepth = 2, TopK = 5 };
        int a = search.ChooseColumn(world, 3, 4);
        int b = search.ChooseColumn(world, 3, 4);
        Assert.Equal(a, b);
    }
}
