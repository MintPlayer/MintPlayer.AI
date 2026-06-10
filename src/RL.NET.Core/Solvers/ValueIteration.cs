using RLNet.Core.Environments;

namespace RLNet.Core.Solvers;

/// <summary>
/// Exact dynamic-programming solver for finite MDPs with a known model.
/// Serves as the correctness oracle for tabular learning algorithms.
/// </summary>
public static class ValueIteration
{
    public static ValueIterationResult Solve(
        ITabularEnvironment env, double gamma, double tolerance = 1e-12, int maxIterations = 100_000)
    {
        var values = new double[env.StateCount];
        var q = new double[env.StateCount, env.ActionCount];

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            double maxDelta = 0;
            for (int s = 0; s < env.StateCount; s++)
            {
                if (env.IsTerminal(s)) continue;

                double bestValue = double.NegativeInfinity;
                for (int a = 0; a < env.ActionCount; a++)
                {
                    double qValue = 0;
                    foreach (var t in env.Model(s, a))
                        qValue += t.Probability * (t.Reward + (t.Terminated ? 0 : gamma * values[t.NextState]));
                    q[s, a] = qValue;
                    bestValue = Math.Max(bestValue, qValue);
                }

                maxDelta = Math.Max(maxDelta, Math.Abs(bestValue - values[s]));
                values[s] = bestValue;
            }

            if (maxDelta < tolerance)
                return new ValueIterationResult(values, q, iteration + 1);
        }

        throw new InvalidOperationException($"Value iteration did not converge within {maxIterations} iterations.");
    }
}

public sealed record ValueIterationResult(double[] Values, double[,] Q, int Iterations)
{
    /// <summary>True if <paramref name="action"/> is (one of) the optimal action(s) in <paramref name="state"/>.</summary>
    public bool IsOptimalAction(int state, int action, double tolerance = 1e-9)
    {
        double best = double.NegativeInfinity;
        for (int a = 0; a < Q.GetLength(1); a++)
            best = Math.Max(best, Q[state, a]);
        return Q[state, action] >= best - tolerance;
    }
}
