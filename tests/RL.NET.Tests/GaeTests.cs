using RLNet.Core.Training;

namespace RLNet.Tests;

/// <summary>
/// Hand-computed 3-step GAE (PLAN.md M4 gate). γ = 0.5, λ = 0.5, single env;
/// rewards [1,1,1], values [0.5, 0.4, 0.3], bootstrap V = 0.2.
/// </summary>
public class GaeTests
{
    private static readonly float[] Rewards = [1f, 1f, 1f];
    private static readonly float[] Values = [0.5f, 0.4f, 0.3f];
    private static readonly float[] Bootstrap = [0.2f];
    private const double Gamma = 0.5, Lambda = 0.5;

    private static float[] Run(bool[] terminated, bool[] done, float[] finalValues)
        => Gae.Compute(Rewards, Values, terminated, done, finalValues, Bootstrap, steps: 3, numEnvs: 1, Gamma, Lambda);

    [Fact]
    public void NoEpisodeBoundaries_ChainsThroughBootstrap()
    {
        var adv = Run(new bool[3], new bool[3], new float[3]);

        // δ2 = 1 + 0.5·0.2 − 0.3 = 0.8           → A2 = 0.8
        // δ1 = 1 + 0.5·0.3 − 0.4 = 0.75          → A1 = 0.75 + 0.25·0.8  = 0.95
        // δ0 = 1 + 0.5·0.4 − 0.5 = 0.7           → A0 = 0.7  + 0.25·0.95 = 0.9375
        Assert.Equal(0.9375f, adv[0], 1e-6f);
        Assert.Equal(0.95f, adv[1], 1e-6f);
        Assert.Equal(0.8f, adv[2], 1e-6f);
    }

    [Fact]
    public void Termination_ZeroesBothTheBootstrapAndTheRecursion()
    {
        var adv = Run(
            terminated: [false, true, false],
            done: [false, true, false],
            finalValues: new float[3]);

        // δ1 = 1 + 0 − 0.4 = 0.6 (no bootstrap)   → A1 = 0.6 (recursion cut by done)
        // δ0 = 1 + 0.5·0.4 − 0.5 = 0.7            → A0 = 0.7 + 0.25·0.6 = 0.85
        // t=2 is a fresh episode                   → A2 = δ2 = 0.8
        Assert.Equal(0.85f, adv[0], 1e-6f);
        Assert.Equal(0.6f, adv[1], 1e-6f);
        Assert.Equal(0.8f, adv[2], 1e-6f);
    }

    [Fact]
    public void Truncation_StillBootstraps_ButCutsTheRecursion()
    {
        // Same episode boundary at t=1, but truncated: δ uses V(final_observation)=0.6,
        // while the advantage recursion is still cut — the two masks differ. This is the
        // distinction that silently breaks most from-scratch PPO implementations.
        var adv = Run(
            terminated: [false, false, false],
            done: [false, true, false],
            finalValues: [0f, 0.6f, 0f]);

        // δ1 = 1 + 0.5·0.6 − 0.4 = 0.9            → A1 = 0.9 (recursion cut by done)
        // δ0 = 0.7                                 → A0 = 0.7 + 0.25·0.9 = 0.925
        Assert.Equal(0.925f, adv[0], 1e-6f);
        Assert.Equal(0.9f, adv[1], 1e-6f);
        Assert.Equal(0.8f, adv[2], 1e-6f);
    }

    [Fact]
    public void MultiEnv_ColumnsAreIndependent()
    {
        // Env 0: no boundaries. Env 1: terminates at t=1. Interleaved [T,N] layout.
        var adv = Gae.Compute(
            rewards: [1f, 1f, 1f, 1f, 1f, 1f],
            values: [0.5f, 0.5f, 0.4f, 0.4f, 0.3f, 0.3f],
            terminated: [false, false, false, true, false, false],
            done: [false, false, false, true, false, false],
            finalValues: new float[6],
            bootstrapValues: [0.2f, 0.2f],
            steps: 3, numEnvs: 2, Gamma, Lambda);

        Assert.Equal(0.9375f, adv[0], 1e-6f); // env 0 = the no-boundary scenario
        Assert.Equal(0.95f, adv[2], 1e-6f);
        Assert.Equal(0.8f, adv[4], 1e-6f);
        Assert.Equal(0.85f, adv[1], 1e-6f);   // env 1 = the termination scenario
        Assert.Equal(0.6f, adv[3], 1e-6f);
        Assert.Equal(0.8f, adv[5], 1e-6f);
    }
}
