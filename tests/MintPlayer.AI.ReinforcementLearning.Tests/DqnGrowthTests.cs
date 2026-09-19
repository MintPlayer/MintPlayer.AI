using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69 (COVERAGE_90_PRD §19.5 item 4) — <see cref="DqnGrowth.Maybe"/>, the mid-run Net2Net step every DQN campaign
/// (Snake, FruitCake, CrazyFruits, Tetris) reaches once per chunk. Only its early-out was covered.
/// </summary>
/// <remarks>
/// <para><b>Three silent failures, none of which throws.</b></para>
/// <list type="bullet">
/// <item><b>The growth must be function-preserving.</b> Net2WiderNet/Net2DeeperNet add capacity without changing
/// the function computed; if a widen were mis-wired the run would simply take a loss spike mid-training that
/// reads as "DQN being noisy" and costs the previous chunk's progress.</item>
/// <item><b>The target net must be re-synced.</b> <c>GrowTo</c> clones the grown online net's STRUCTURE and then
/// copies its weights. Drop the <c>CopyFrom</c> and the run continues against a randomly-initialised target:
/// nothing crashes, the loss stays finite, training just quietly stops improving.</item>
/// <item><b>The climb is one stage per call.</b> The step count picks a target stage and the loop walks toward it;
/// jumping straight to the top would break the single-Net2Net-step guarantee, and treating an off-schedule trunk
/// as anything but stage 0 changes which shape a resumed run grows into. The off-schedule case is asserted
/// explicitly because it is the same recovered-from-shape guess that once made Rush Hour and Cube nets SMALLER
/// when growth was enabled.</item>
/// </list>
/// </remarks>
public class DqnGrowthTests
{
    private const int ObsSize = 4;
    private const int Actions = 2;

    private static DqnTrainingState State(int[] hidden, int steps)
    {
        var online = new DuelingQNet(ObsSize, hidden, Actions, new Xoshiro256StarStar(1));
        var target = (DuelingQNet)online.CloneStructure();
        target.CopyFrom(online);
        return new DqnTrainingState
        {
            Online = online,
            Target = target,
            Optimizer = new Adam(online.Parameters(), 1e-3f),
            Buffer = new ReplayBuffer(16, ObsSize, Actions),
            PolicyRng = new Xoshiro256StarStar(2),
            BufferRng = new Xoshiro256StarStar(3),
            StepsCompleted = steps,
        };
    }

    private static Tensor Obs() => new Tensor([0.25f, -0.5f, 1f, 0.75f], 1, ObsSize);

    private static float[] QValues(IValueNet net) => [.. net.Forward(Obs()).Data];

    private static DqnTrainingState Grow(DqnTrainingState state, int growEvery = 100)
        => DqnGrowth.Maybe(state, grow: true, growEvery, learningRate: 1e-3f, new Xoshiro256StarStar(9), _ => { });

    [Fact]
    public void Growth_leaves_the_function_the_net_computes_unchanged()
    {
        var state = State(DqnGrowth.Start, steps: 100);
        var before = QValues(state.Online);

        var grown = Grow(state);
        var after = QValues(grown.Online);

        Assert.NotSame(state.Online, grown.Online);
        Assert.Equal(before.Length, after.Length);
        for (int a = 0; a < before.Length; a++) Assert.Equal(before[a], after[a], 3);
    }

    [Fact]
    public void Growth_advances_exactly_one_stage_not_straight_to_the_top()
    {
        // StepsCompleted / growEvery == 1, so stage 1 is the target -- a net that landed on Stages[^1] here would
        // have skipped every intermediate function-preserving step.
        var grown = Grow(State(DqnGrowth.Start, steps: 100));

        Assert.Equal(DqnGrowth.Stages[1], ((DuelingQNet)grown.Online).Trunk);
    }

    [Fact]
    public void The_target_net_is_resynced_to_the_grown_online_net()
    {
        // The failure mode with no symptom: an un-synced target is a random net, so every bootstrap target is
        // noise and the run silently stops learning while still logging a finite loss.
        var grown = Grow(State(DqnGrowth.Start, steps: 100));

        var online = QValues(grown.Online);
        var target = QValues(grown.Target);

        Assert.Equal(((DuelingQNet)grown.Online).Trunk, ((DuelingQNet)grown.Target).Trunk);
        for (int a = 0; a < online.Length; a++) Assert.Equal(online[a], target[a], 5);
    }

    [Fact]
    public void Growth_carries_the_replay_buffer_step_count_and_rng_streams_forward()
    {
        // WithNetwork swaps only the net + optimizer; losing the buffer would throw away the run's whole sample
        // history at the exact moment capacity was added.
        var state = State(DqnGrowth.Start, steps: 100);
        var grown = Grow(state);

        Assert.Same(state.Buffer, grown.Buffer);
        Assert.Same(state.PolicyRng, grown.PolicyRng);
        Assert.Same(state.BufferRng, grown.BufferRng);
        Assert.Equal(state.StepsCompleted, grown.StepsCompleted);
        Assert.NotSame(state.Optimizer, grown.Optimizer);   // Adam's moments are keyed to the parameter set
    }

    [Fact]
    public void A_net_already_at_its_target_stage_is_left_alone()
    {
        var state = State(DqnGrowth.Stages[1], steps: 100);

        Assert.Same(state, Grow(state));
    }

    [Fact]
    public void An_off_schedule_trunk_is_treated_as_stage_zero()
    {
        // A trunk the schedule never describes (a non-grow resume) is indistinguishable from the bottom rung, so
        // it is walked from stage 0 -- here to Stages[1], NOT left alone and NOT taken to the top. This is a
        // recorded hazard rather than a nicety: the same shape-recovery guess is what once downgraded nets that
        // borrowed this schedule while defaulting to a far wider trunk.
        var grown = Grow(State([24], steps: 100));

        Assert.Equal(DqnGrowth.Stages[1], ((DuelingQNet)grown.Online).Trunk);
    }

    [Fact]
    public void Growth_is_off_unless_it_was_asked_for()
    {
        var state = State(DqnGrowth.Start, steps: 100);

        Assert.Same(state, DqnGrowth.Maybe(state, grow: false, 100, 1e-3f, new Xoshiro256StarStar(9), _ => { }));
    }

    [Fact]
    public void A_noisy_net_is_never_grown()
    {
        // NoisyNets carry per-parameter noise tensors that Net2Net has no transfer rule for, so the schedule
        // skips them outright rather than producing a net whose exploration is quietly broken.
        var online = new DuelingQNet(ObsSize, DqnGrowth.Start, Actions, new Xoshiro256StarStar(1), noisy: true);
        var target = (DuelingQNet)online.CloneStructure();
        var state = new DqnTrainingState
        {
            Online = online,
            Target = target,
            Optimizer = new Adam(online.Parameters(), 1e-3f),
            Buffer = new ReplayBuffer(16, ObsSize, Actions),
            PolicyRng = new Xoshiro256StarStar(2),
            BufferRng = new Xoshiro256StarStar(3),
            StepsCompleted = 100,
        };

        Assert.Same(state, Grow(state));
    }

    [Fact]
    public void The_growth_log_names_the_shape_it_grew_into()
    {
        // The live network viewer is driven off this line, and a run's log is the only record of when capacity
        // was added -- a silent grow is indistinguishable from none.
        var lines = new List<string>();
        DqnGrowth.Maybe(State(DqnGrowth.Start, steps: 100), grow: true, 100, 1e-3f,
            new Xoshiro256StarStar(9), lines.Add);

        Assert.Single(lines);
        Assert.Contains(string.Join(",", DqnGrowth.Stages[1]), lines[0]);
    }
}
