using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Guards the planner-distillation net (<see cref="FruitCakePolicyNet"/>): it produces a legal column, and a
/// save/load round-trip reproduces its outputs bitwise (serving must reload the trained net exactly). Strength
/// (does the distilled policy lift the watermelon rate?) is validated by the headless A/B, not a unit test.
/// </summary>
public class FruitCakePolicyNetTests
{
    private static FruitCakeWorld PlayedBoard(ulong seed, int drops)
    {
        var env = new FruitCakeEnv();
        env.Reset(seed);
        for (int i = 0; i < drops; i++)
        {
            var step = env.Step(i % FruitCakeEnv.ColumnCount);
            if (step.Done) env.Reset(seed);
        }
        return env.World;
    }

    [Fact]
    public void ChooseColumn_returns_a_legal_column()
    {
        var net = new FruitCakePolicyNet(new Xoshiro256StarStar(42));
        int col = net.ChooseColumn(PlayedBoard(7, 12), current: 1, next: 2);
        Assert.InRange(col, 0, FruitCakeEnv.ColumnCount - 1);
    }

    [Fact]
    public void SaveLoad_round_trips_outputs()
    {
        var net = new FruitCakePolicyNet(new Xoshiro256StarStar(123));
        var world = PlayedBoard(11, 15);

        using var ms = new MemoryStream();
        net.Save(ms);
        ms.Position = 0;
        var reloaded = FruitCakePolicyNet.Load(ms);

        var (logitsA, valueA) = net.Evaluate(world, 3, 4);
        var (logitsB, valueB) = reloaded.Evaluate(world, 3, 4);
        Assert.Equal(logitsA, logitsB);
        Assert.Equal(valueA, valueB);
    }
}
