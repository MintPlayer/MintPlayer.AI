using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Contract tests for n-step return folding (PRD FRUITCAKE_IMPROVE §F4 gate): n=1 is exactly single-step DQN,
/// n-step rewards are the correct discounted sums, episode ends are handled, and the in-flight window round-trips
/// through serialization (so a resumed run stays bitwise-identical).
/// </summary>
public class NStepAccumulatorTests
{
    private static float[] S(float v) => [v];

    // n = 1 must emit exactly one transition per push, unchanged — the bitwise-identical single-step default.
    [Fact]
    public void SingleStep_EmitsEveryPushUnchanged()
    {
        var acc = new NStepAccumulator(n: 1, gamma: 0.99, obsDim: 1, actionCount: 1);

        var mid = acc.Push(S(0), action: 7, reward: 1.5, S(1), terminated: false, truncated: false, default);
        Assert.Single(mid);
        Assert.Equal(0f, mid[0].Obs[0]);
        Assert.Equal(7, mid[0].Action);
        Assert.Equal(1.5f, mid[0].Reward);
        Assert.Equal(1f, mid[0].NextObs[0]);
        Assert.False(mid[0].Terminated);

        var end = acc.Push(S(1), action: 3, reward: 2.0, S(2), terminated: true, truncated: false, default);
        Assert.Single(end);
        Assert.Equal(2f, end[0].Reward);
        Assert.True(end[0].Terminated);
    }

    // Steady state: a 3-step transition completes on the 3rd push with the discounted-sum reward and s_{t+3}.
    [Fact]
    public void NStep_AccumulatesDiscountedRewardAndBootstrapState()
    {
        var acc = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);

        Assert.Empty(acc.Push(S(0), 0, 1, S(1), false, false, default)); // window 1 < 3
        Assert.Empty(acc.Push(S(1), 1, 2, S(2), false, false, default)); // window 2 < 3

        var emit = acc.Push(S(2), 2, 4, S(3), false, false, default);    // window 3 == 3 → emit head
        Assert.Single(emit);
        Assert.Equal(0f, emit[0].Obs[0]);                 // head is s0
        Assert.Equal(0, emit[0].Action);
        Assert.Equal(1f + 0.5f * 2f + 0.25f * 4f, emit[0].Reward); // 1 + 1 + 1 = 3
        Assert.Equal(3f, emit[0].NextObs[0]);             // bootstrap from s3
        Assert.False(emit[0].Terminated);
    }

    // A terminal flushes the whole window; each remaining head accumulates to the terminal with terminated=true.
    [Fact]
    public void Terminal_FlushesWindowToTheEnd()
    {
        var acc = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);
        acc.Push(S(0), 0, 1, S(1), false, false, default); // no emit
        acc.Push(S(1), 1, 2, S(2), false, false, default); // no emit
        acc.Push(S(2), 2, 4, S(3), false, false, default); // emits s0; window now [s1,s2]

        var flush = acc.Push(S(3), 3, 8, S(4), terminated: true, truncated: false, default);
        Assert.Equal(3, flush.Count); // s1, s2, s3
        Assert.Equal(2f + 0.5f * 4f + 0.25f * 8f, flush[0].Reward); // s1: 2+2+2 = 6
        Assert.Equal(4f + 0.5f * 8f, flush[1].Reward);              // s2: 4+4 = 8
        Assert.Equal(8f, flush[2].Reward);                          // s3: 8
        Assert.All(flush, e => Assert.True(e.Terminated));
    }

    // Truncation keeps only the one full-length window and drops the partial tail (no per-transition discount).
    [Fact]
    public void Truncation_KeepsFullWindowDropsPartials()
    {
        var full = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);
        full.Push(S(0), 0, 1, S(1), false, false, default);
        full.Push(S(1), 1, 2, S(2), false, false, default);
        var emit = full.Push(S(2), 2, 4, S(3), terminated: false, truncated: true, default);
        Assert.Single(emit);
        Assert.Equal(1f + 0.5f * 2f + 0.25f * 4f, emit[0].Reward);
        Assert.False(emit[0].Terminated);

        var partial = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);
        partial.Push(S(0), 0, 1, S(1), false, false, default);
        Assert.Empty(partial.Push(S(1), 1, 2, S(2), terminated: false, truncated: true, default)); // < n → dropped
    }

    // The in-flight window survives Save/Load so a resumed run produces identical subsequent transitions.
    [Fact]
    public void SaveLoad_PreservesInFlightWindow()
    {
        var control = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);
        control.Push(S(0), 0, 1, S(1), false, false, default);
        control.Push(S(1), 1, 2, S(2), false, false, default);

        var resumed = new NStepAccumulator(n: 3, gamma: 0.5, obsDim: 1, actionCount: 1);
        resumed.Push(S(0), 0, 1, S(1), false, false, default);
        resumed.Push(S(1), 1, 2, S(2), false, false, default);

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            resumed.Save(writer);
        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        var loaded = NStepAccumulator.Load(reader, obsDim: 1, actionCount: 1);

        Assert.Equal(control.N, loaded.N);
        Assert.Equal(control.Gamma, loaded.Gamma);

        var fromControl = control.Push(S(2), 2, 4, S(3), false, false, default);
        var fromLoaded = loaded.Push(S(2), 2, 4, S(3), false, false, default);
        Assert.Single(fromLoaded);
        Assert.Equal(fromControl[0].Reward, fromLoaded[0].Reward);
        Assert.Equal(fromControl[0].Obs[0], fromLoaded[0].Obs[0]);
        Assert.Equal(fromControl[0].NextObs[0], fromLoaded[0].NextObs[0]);
    }
}
