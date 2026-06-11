using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class VectorEnvTests
{
    [Fact]
    public void ParallelMode_ProducesIdenticalResultsToSequential()
    {
        // Each env owns its RNG, so parallel stepping must be bitwise-identical,
        // not just statistically equivalent.
        var sequential = new VectorEnv(_ => new CartPoleEnv(), count: 4, parallel: false);
        var parallel = new VectorEnv(_ => new CartPoleEnv(), count: 4, parallel: true);

        var obsA = sequential.Reset(99);
        var obsB = parallel.Reset(99);
        Assert.Equal(obsA, obsB);

        var rng = new Xoshiro256StarStar(5);
        for (int step = 0; step < 200; step++)
        {
            var actions = Enumerable.Range(0, 4).Select(_ => rng.NextInt(2)).ToArray();
            var a = sequential.Step(actions);
            var b = parallel.Step(actions);

            Assert.Equal(a.Obs, b.Obs);
            Assert.Equal(a.Rewards, b.Rewards);
            Assert.Equal(a.Terminated, b.Terminated);
            Assert.Equal(a.Truncated, b.Truncated);
            Assert.Equal(a.FinalObs, b.FinalObs);
        }
    }

    [Fact]
    public void Autoreset_ReturnsFreshObs_AndPreservesFinalObservation()
    {
        var vec = new VectorEnv(_ => new CartPoleEnv(), count: 2);
        vec.Reset(1);

        // Push env 0 to the left until it falls; env 1 balances with a PD-ish policy.
        var lastObs = new float[2 * vec.ObsDim];
        while (true)
        {
            var step = vec.Step([0, 0]);
            if (step.Done[0])
            {
                // FinalObs carries the terminal state (out of bounds)…
                float finalX = step.FinalObs[0];
                float finalTheta = step.FinalObs[2];
                Assert.True(Math.Abs(finalX) > CartPoleEnv.XThreshold
                    || Math.Abs(finalTheta) > CartPoleEnv.ThetaThresholdRadians);

                // …while Obs is already the new episode's initial state (within ±0.05).
                for (int i = 0; i < vec.ObsDim; i++)
                    Assert.InRange(step.Obs[i], -0.05f, 0.05f);

                // The untouched env's slot is unaffected.
                Assert.False(step.Done[1]);
                return;
            }
            step.Obs.CopyTo(lastObs.AsSpan());
        }
    }

    [Fact]
    public void Reset_GivesEachEnvADistinctSeed()
    {
        var vec = new VectorEnv(_ => new CartPoleEnv(), count: 4);
        var obs = vec.Reset(7);

        // All four initial observations must differ (per-env derived seeds).
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
            {
                bool identical = true;
                for (int d = 0; d < vec.ObsDim; d++)
                    if (obs[i * vec.ObsDim + d] != obs[j * vec.ObsDim + d]) { identical = false; break; }
                Assert.False(identical, $"envs {i} and {j} started in identical states");
            }
    }
}
