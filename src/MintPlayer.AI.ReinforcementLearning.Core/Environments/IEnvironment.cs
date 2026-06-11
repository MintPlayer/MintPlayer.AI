namespace MintPlayer.AI.ReinforcementLearning.Core.Environments;

/// <summary>
/// Gymnasium-faithful environment contract.
/// <para>
/// <c>Terminated</c> means the MDP reached a true terminal state (bootstrap value 0);
/// <c>Truncated</c> means the episode was cut off externally (time limit) and value
/// targets MUST still bootstrap from the final observation. Conflating the two is the
/// most common silent correctness bug in RL implementations.
/// </para>
/// </summary>
public interface IEnvironment<TObs, TAct>
{
    Space<TObs> ObservationSpace { get; }
    Space<TAct> ActionSpace { get; }

    /// <summary>Starts a new episode. A seed reseeds the environment's RNG stream.</summary>
    (TObs Observation, EnvInfo Info) Reset(ulong? seed = null);

    StepResult<TObs> Step(TAct action);

    /// <summary>Human-readable rendering of the current state (console-oriented in v1).</summary>
    string RenderString();
}

public readonly record struct StepResult<TObs>(
    TObs Observation,
    double Reward,
    bool Terminated,
    bool Truncated,
    EnvInfo Info)
{
    public bool Done => Terminated || Truncated;
}

/// <summary>
/// Side-channel for auxiliary step data (e.g. <c>final_observation</c> on autoreset
/// in vectorized environments). Kept allocation-free for the common empty case.
/// </summary>
public sealed class EnvInfo
{
    public static EnvInfo Empty { get; } = new();

    private Dictionary<string, object>? _values;

    public object? this[string key]
    {
        get => _values?.GetValueOrDefault(key);
        set
        {
            if (ReferenceEquals(this, Empty))
                throw new InvalidOperationException("EnvInfo.Empty is immutable; create a new EnvInfo instead.");
            (_values ??= [])[key] = value!;
        }
    }

    public bool TryGet<T>(string key, out T value)
    {
        if (_values is not null && _values.TryGetValue(key, out var raw) && raw is T typed)
        {
            value = typed;
            return true;
        }
        value = default!;
        return false;
    }
}
