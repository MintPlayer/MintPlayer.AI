namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Generalized Advantage Estimation (Schulman et al. 2016) over a [T,N] rollout
/// (row-major: index = t*N + i).
/// <para>
/// The two masks are deliberately distinct (the classic off-by-one trap):
/// inside δ, the next-state value is masked by TERMINATED only — a truncated episode
/// still bootstraps, using the value of <c>final_observation</c> (passed via
/// <c>finalValues</c>) rather than the autoreset observation. The recursive
/// advantage term is masked by DONE (terminated OR truncated) — advantages never leak
/// across episode boundaries, even at time limits.
/// </para>
/// </summary>
public static class Gae
{
    /// <param name="rewards">r_t, [T*N].</param>
    /// <param name="values">V(s_t) recorded BEFORE stepping, [T*N].</param>
    /// <param name="terminated">true terminal at t, [T*N].</param>
    /// <param name="done">terminated || truncated at t, [T*N].</param>
    /// <param name="finalValues">V(final_observation) where done (0 is fine where terminated), [T*N].</param>
    /// <param name="bootstrapValues">V of the observation after the last rollout step, [N].</param>
    public static float[] Compute(
        float[] rewards, float[] values, bool[] terminated, bool[] done, float[] finalValues,
        float[] bootstrapValues, int steps, int numEnvs, double gamma, double lambda)
    {
        var advantages = new float[steps * numEnvs];

        for (int i = 0; i < numEnvs; i++)
        {
            double nextAdvantage = 0;
            for (int t = steps - 1; t >= 0; t--)
            {
                int idx = t * numEnvs + i;

                double nextValue;
                if (done[idx])
                    nextValue = terminated[idx] ? 0 : finalValues[idx];   // truncated: bootstrap from final_observation
                else
                    nextValue = t == steps - 1 ? bootstrapValues[i] : values[idx + numEnvs];

                double delta = rewards[idx] + gamma * nextValue - values[idx];
                nextAdvantage = delta + gamma * lambda * (done[idx] ? 0 : nextAdvantage);
                advantages[idx] = (float)nextAdvantage;
            }
        }

        return advantages;
    }
}
