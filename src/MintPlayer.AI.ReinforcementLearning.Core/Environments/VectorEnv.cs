namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>
/// Steps N independent environments as one batched unit, with autoreset:
/// when an episode ends, the env is reset immediately and <see cref="VecStep.Obs"/>
/// carries the NEW episode's first observation. The true s_{t+1} of the finished episode
/// is preserved in <see cref="VecStep.FinalObs"/> — value bootstrapping at truncation must
/// use that, never the autoreset observation (the classic cross-episode-garbage bug).
/// <para>
/// Each env owns its RNG, so parallel and sequential stepping produce identical results;
/// parallel mode just spreads the work across cores.
/// </para>
/// </summary>
public sealed class VectorEnv
{
    private readonly IEnvironment<float[], int>[] _envs;

    public VectorEnv(Func<int, IEnvironment<float[], int>> factory, int count, bool parallel = false)
    {
        _envs = [.. Enumerable.Range(0, count).Select(factory)];
        Parallel = parallel;
        ObsDim = ((BoxSpace)_envs[0].ObservationSpace).Dimensions;
        ActionCount = ((DiscreteSpace)_envs[0].ActionSpace).N;
    }

    public int Count => _envs.Length;
    public int ObsDim { get; }
    public int ActionCount { get; }
    public bool Parallel { get; }

    /// <summary>True when the sub-envs expose invalid-action masks (<see cref="IActionMaskProvider"/>).</summary>
    public bool Masked => _envs[0] is IActionMaskProvider;

    /// <summary>
    /// Each env's current legality mask, flattened row-major to <see cref="Count"/>·<see cref="ActionCount"/>,
    /// or null when the envs are not maskable. Reflects each env's <i>current</i> state — call it on the
    /// observation about to be acted on (after any autoreset inside the previous <see cref="Step"/>).
    /// </summary>
    public bool[]? CurrentActionMasks()
    {
        if (!Masked) return null;
        var masks = new bool[Count * ActionCount];
        for (int i = 0; i < Count; i++)
            ((IActionMaskProvider)_envs[i]).CurrentActionMask().CopyTo(masks.AsSpan(i * ActionCount, ActionCount));
        return masks;
    }

    /// <summary>Resets all envs, deriving a distinct deterministic seed per env.</summary>
    public float[] Reset(ulong baseSeed)
    {
        var obs = new float[Count * ObsDim];
        for (int i = 0; i < Count; i++)
        {
            var (o, _) = _envs[i].Reset(baseSeed + (ulong)i * 0x9E3779B97F4A7C15UL);
            o.CopyTo(obs.AsSpan(i * ObsDim, ObsDim));
        }
        return obs;
    }

    public VecStep Step(int[] actions)
    {
        var step = new VecStep(
            new float[Count * ObsDim], new double[Count], new bool[Count],
            new bool[Count], new float[Count * ObsDim], new bool[Count]);

        if (Parallel)
            System.Threading.Tasks.Parallel.For(0, Count, i => StepOne(i, actions[i], step));
        else
            for (int i = 0; i < Count; i++)
                StepOne(i, actions[i], step);

        return step;
    }

    private void StepOne(int i, int action, VecStep step)
    {
        var result = _envs[i].Step(action);
        step.Rewards[i] = result.Reward;
        step.Terminated[i] = result.Terminated;
        step.Truncated[i] = result.Truncated;
        step.Done[i] = result.Done;

        if (result.Done)
        {
            result.Observation.CopyTo(step.FinalObs.AsSpan(i * ObsDim, ObsDim));
            var (fresh, _) = _envs[i].Reset();
            fresh.CopyTo(step.Obs.AsSpan(i * ObsDim, ObsDim));
        }
        else
        {
            result.Observation.CopyTo(step.Obs.AsSpan(i * ObsDim, ObsDim));
        }
    }
}

/// <summary>Batched step result; <c>FinalObs</c> rows are only meaningful where <c>Done</c>.</summary>
public sealed record VecStep(
    float[] Obs, double[] Rewards, bool[] Terminated, bool[] Truncated, float[] FinalObs, bool[] Done);
