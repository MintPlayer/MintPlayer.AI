using System.Reflection;
using MintPlayer.AI.ReinforcementLearning.Environments.Snake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Safety-cycle mode (M48.1). The mode's whole promise is structural — the snake can never die because the next
/// cycle cell is always legal — so the tests assert the structure: the generated cycle is a genuine Hamiltonian
/// cycle aligned with the fresh body, the tail→head cycle ordering (the safety invariant) survives every move the
/// chooser makes, and full games end in a board-full win, never a death.
/// </summary>
public class SnakeCycleTests
{
    private static PgSnakeEnv Core(SnakeEnv e) =>
        (global::PgSnakeEnv)typeof(SnakeEnv).GetField("_core", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(e)!;

    // An all-zero-weight net (Q ≡ 0 for every action): the chooser degrades to the pure furthest-safe-shortcut
    // rule, so the structural guarantees are tested without depending on the trained checkpoint file.
    private static PgSnakeNet ZeroNet()
    {
        static List<double> Zeros(int n) => [.. new double[n]];
        const int obs = SnakeEnv.ObservationSize, actions = SnakeEnv.ActionCount;
        return new PgSnakeNet(obs, actions, [], Zeros(0), Zeros(0), Zeros(obs), Zeros(1), Zeros(obs * actions), Zeros(actions));
    }

    private static int RelDist(PgSnakeEnv core, int fromCell, int toCell)
    {
        int d = core.cycleIndex[toCell] - core.cycleIndex[fromCell];
        return d < 0 ? d + core.cycle.Count : d;
    }

    // The safety invariant chooseActionCycle relies on: body cells sit at strictly increasing cycle positions
    // from tail to head, all within one lap — so the cycle's free arc ahead of the head contains no body cell.
    private static void AssertBodyOrderedOnCycle(SnakeEnv env)
    {
        var core = Core(env);
        int tail = core.body[0];
        int prev = 0;
        for (int i = 1; i < core.body.Count; i++)
        {
            int d = RelDist(core, tail, core.body[i]);
            Assert.True(d > prev, $"body[{i}] at cycle distance {d} from the tail is not ahead of body[{i - 1}] at {prev}");
            prev = d;
        }
    }

    [Fact]
    public void InitCycle_IsAHamiltonianCycle_AlignedWithTheFreshBody()
    {
        foreach (int size in new[] { 6, 8, 12 })
        {
            var env = new SnakeEnv(size);
            env.Reset(1);
            var core = Core(env);
            core.initCycle();

            Assert.Equal(env.Cells, core.cycle.Count);
            Assert.Equal(env.Cells, core.cycle.Distinct().Count()); // every cell exactly once
            for (int i = 0; i < core.cycle.Count; i++)
            {
                int a = core.cycle[i], b = core.cycle[(i + 1) % core.cycle.Count];
                int manhattan = Math.Abs(a / size - b / size) + Math.Abs(a % size - b % size);
                Assert.True(manhattan == 1, $"cycle positions {i}→{i + 1} are not adjacent cells ({a}→{b})");
            }
            AssertBodyOrderedOnCycle(env);
        }
    }

    [Fact]
    public void CycleMode_KeepsTheOrderingInvariant_EveryMoveOfAFullGame()
    {
        var env = new SnakeEnv(8);
        env.Reset(2);
        var core = Core(env);
        var net = ZeroNet();

        bool done = false;
        while (!done)
        {
            int action = core.chooseActionCycle(net, 0, 1_000, 4);
            var step = env.Step(action);
            done = step.Terminated || step.Truncated;
            if (!done)
                AssertBodyOrderedOnCycle(env);
        }
    }

    [Fact]
    public void CycleMode_FullGames_EndBoardFull_NeverDead()
    {
        var net = ZeroNet();
        for (ulong seed = 1; seed <= 3; seed++)
        {
            var env = new SnakeEnv(8);
            env.Reset(seed);
            var core = Core(env);

            bool terminated = false, truncated = false;
            while (!terminated && !truncated)
            {
                var step = env.Step(core.chooseActionCycle(net, 0, 1_000, 4));
                terminated = step.Terminated;
                truncated = step.Truncated;
            }
            // Terminated == board-full win. A death also terminates, but leaves Length < Cells.
            Assert.True(terminated, $"seed {seed}: episode hit the step ceiling instead of winning");
            Assert.Equal(env.Cells, env.Length);
        }
    }

    [Fact]
    public void ChooseActionCycle_OddBoard_Throws()
    {
        var env = new SnakeEnv(7);
        env.Reset(1);
        Assert.Throws<InvalidOperationException>(() => env.ChooseActionCycle(new SnakeCycleConfig()));
    }
}
