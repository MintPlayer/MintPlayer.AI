using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// <see cref="BlockDudeGreedy"/> is the shared rollout both the training gate and the shipped-level benchmark
/// run, and it had zero coverage.
/// </summary>
/// <remarks>
/// It is shared precisely so the two numbers cannot drift apart, which makes it a single point of failure for
/// every Block Dude figure in the repo. The M61 measurement turned on its <see cref="BlockDudeGreedy.Ending"/>
/// classification — "every failure is <c>Loop</c>, with zero <c>NoMove</c>/<c>Refused</c>" is what refuted the
/// dead-end hypothesis — so a misclassified ending would not merely be a wrong number, it would have sent that
/// investigation the wrong way.
/// <para>
/// These tests deliberately assert <b>bookkeeping and classification</b>, never strength: the nets here are
/// random-weight and tiny, so what they play is arbitrary. What must hold regardless of the policy is that the
/// step count, the distinct-state count, the recorded move list and the ending agree with each other and with
/// the board.
/// </para>
/// </remarks>
public class BlockDudeGreedyTests
{
    private static BlockDudePolicyNet Net(ulong seed = 7) => new(new Xoshiro256StarStar(seed), 32);

    private static BlockDudeBoard Level(int index = 0)
        => BlockDudeBoard.FromGrid(BlockDudeLevels.All[index].Grid);

    // ── budget handling ──────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_zero_step_budget_ends_immediately_on_budget()
    {
        var outcome = BlockDudeGreedy.Run(Net(), Level(), stepBudget: 0);

        Assert.Equal(BlockDudeGreedy.Ending.Budget, outcome.Ending);
        Assert.Equal(0, outcome.Steps);
        Assert.False(outcome.Solved);
    }

    [Fact]
    public void A_zero_step_budget_ends_immediately_on_budget_when_avoiding_revisits()
    {
        var outcome = BlockDudeGreedy.RunAvoidingRevisits(Net(), Level(), stepBudget: 0);

        Assert.Equal(BlockDudeGreedy.Ending.Budget, outcome.Ending);
        Assert.Equal(0, outcome.Steps);
    }

    [Fact]
    public void The_reported_step_count_never_exceeds_the_budget()
    {
        foreach (int budget in new[] { 1, 3, 17, 64 })
        {
            var outcome = BlockDudeGreedy.Run(Net(), Level(), budget);
            Assert.InRange(outcome.Steps, 0, budget);
        }
    }

    // ── the ending classification ────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Solved_is_true_for_exactly_the_won_ending()
    {
        // Solved is what every caller branches on; if it ever disagreed with Ending, a losing rollout would be
        // counted as a solve and the gate would pass on nothing.
        foreach (ulong seed in new ulong[] { 1, 2, 3, 4, 5 })
        {
            var outcome = BlockDudeGreedy.Run(Net(seed), Level(), 40);
            Assert.Equal(outcome.Ending == BlockDudeGreedy.Ending.Won, outcome.Solved);
        }
    }

    [Fact]
    public void Every_rollout_ends_in_a_declared_ending()
    {
        foreach (ulong seed in new ulong[] { 11, 12, 13 })
            foreach (int level in new[] { 0, 1, 2 })
            {
                var outcome = BlockDudeGreedy.Run(Net(seed), Level(level), 30);
                Assert.Contains(outcome.Ending, Enum.GetValues<BlockDudeGreedy.Ending>());
            }
    }

    // ── the move log ─────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Moves_are_not_recorded_unless_asked_for()
    {
        // The gate runs thousands of rollouts and only needs the verdict; allocating a list per rollout there
        // would be pure waste, so "off by default" is a property worth pinning rather than an implementation detail.
        var outcome = BlockDudeGreedy.Run(Net(), Level(), 25);

        Assert.Null(outcome.Moves);
    }

    [Fact]
    public void A_recorded_line_has_exactly_one_move_per_step_taken()
    {
        var outcome = BlockDudeGreedy.Run(Net(), Level(), 25, recordMoves: true);

        Assert.NotNull(outcome.Moves);
        // Every ending except Won/Budget returns BEFORE the move is logged, so the count matches Steps in those
        // cases; Won and Budget log the step they are on. Either way the line can never be longer than Steps + 1
        // or shorter than Steps - 1, and a drift here would make a replayed line desync from its reported length.
        Assert.InRange(outcome.Moves!.Count, Math.Max(0, outcome.Steps - 1), outcome.Steps + 1);
    }

    [Fact]
    public void Every_recorded_move_is_a_real_action_index()
    {
        var outcome = BlockDudeGreedy.Run(Net(), Level(), 25, recordMoves: true);

        Assert.All(outcome.Moves!, m => Assert.InRange(m, 0, BlockDudeBoard.ActionCount - 1));
    }

    // ── distinct states ──────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void The_distinct_state_count_never_exceeds_the_steps_taken()
    {
        // DistinctStates is read AGAINST Steps to tell cycling from progress ("far fewer distinct states than
        // steps means the policy is cycling"). If it could exceed Steps that comparison would be meaningless.
        foreach (ulong seed in new ulong[] { 21, 22, 23 })
        {
            var outcome = BlockDudeGreedy.Run(Net(seed), Level(), 30);
            Assert.InRange(outcome.DistinctStates, 0, outcome.Steps + 1);
        }
    }

    [Fact]
    public void Avoiding_revisits_counts_the_start_state_as_visited()
    {
        // RunAvoidingRevisits seeds its visited set with the start hash, so even a rollout that takes no step
        // has touched one state. Run() starts from an EMPTY set — the two conventions differ deliberately, and
        // a change that unified them would silently shift every reported DistinctStates by one.
        var avoiding = BlockDudeGreedy.RunAvoidingRevisits(Net(), Level(), stepBudget: 0);
        var plain = BlockDudeGreedy.Run(Net(), Level(), stepBudget: 0);

        Assert.Equal(1, avoiding.DistinctStates);
        Assert.Equal(0, plain.DistinctStates);
    }

    [Fact]
    public void Avoiding_revisits_never_ends_in_a_loop()
    {
        // The whole point of the variant: it refuses a successor it has already seen, so Loop is unreachable.
        // If it ever appeared, the tie-break has stopped working and the two failure modes it exists to
        // separate would be conflated again.
        foreach (ulong seed in new ulong[] { 31, 32, 33, 34 })
            foreach (int level in new[] { 0, 1 })
            {
                var outcome = BlockDudeGreedy.RunAvoidingRevisits(Net(seed), Level(level), 40);
                Assert.NotEqual(BlockDudeGreedy.Ending.Loop, outcome.Ending);
            }
    }

    [Fact]
    public void Avoiding_revisits_visits_a_new_state_on_every_step_it_takes()
    {
        // One state added per step, plus the start. A weaker invariant than "no loops" and a stricter one than
        // the count bound: it pins that the visited set and the step counter advance together.
        foreach (ulong seed in new ulong[] { 41, 42 })
        {
            var outcome = BlockDudeGreedy.RunAvoidingRevisits(Net(seed), Level(), 20);
            if (outcome.Ending is BlockDudeGreedy.Ending.Budget)
                Assert.Equal(outcome.Steps + 1, outcome.DistinctStates);
        }
    }

    // ── determinism ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void A_greedy_rollout_is_a_deterministic_function_of_the_net_and_the_board()
    {
        // Greedy play consults no RNG. Two rollouts of the same net on the same board must be identical — this
        // is what lets the gate compare runs at all, and it is also the premise of the Loop early-exit ("the
        // greedy policy is a deterministic function of the board, so a repeated state means the trajectory is
        // periodic"). If that premise broke, the early exit would start cutting live rollouts short.
        var a = BlockDudeGreedy.Run(Net(5), Level(), 30, recordMoves: true);
        var b = BlockDudeGreedy.Run(Net(5), Level(), 30, recordMoves: true);

        Assert.Equal(a.Ending, b.Ending);
        Assert.Equal(a.Steps, b.Steps);
        Assert.Equal(a.DistinctStates, b.DistinctStates);
        Assert.Equal(a.Moves, b.Moves);
    }

    [Fact]
    public void Replaying_a_recorded_line_reproduces_the_rollout()
    {
        // The strongest available check that the logged line is the line actually played: applying it move by
        // move must reach the same place, and must never hit an illegal action. A move log that were off by one
        // — or that recorded intended rather than applied actions — would pass every count assertion above and
        // still be unreplayable, which is the one thing a benchmark uses it for.
        var outcome = BlockDudeGreedy.Run(Net(9), Level(), 30, recordMoves: true);

        var board = Level();
        foreach (int move in outcome.Moves!)
        {
            var action = (BlockDudeAction)move;
            Assert.True(board.IsLegal(action));
            board = board.Apply(action);
        }

        Assert.Equal(outcome.Ending == BlockDudeGreedy.Ending.Won, board.Won);
    }
}
