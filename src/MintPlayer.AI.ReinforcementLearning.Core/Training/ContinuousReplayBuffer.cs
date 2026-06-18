using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Circular experience-replay buffer for off-policy <b>continuous</b>-action learning (SAC). The sibling of
/// <see cref="ReplayBuffer"/>: it stores a real-valued action vector per transition instead of a discrete
/// index, and carries no action mask (continuous policies aren't masked). Like the discrete buffer it stores
/// the <c>terminated</c> flag ONLY — a time-limit <c>truncated</c> transition must still bootstrap from its
/// next state (conflating the two is the classic silent value-capping bug, PRD §8).
/// </summary>
public sealed class ContinuousReplayBuffer
{
    private readonly int _obsDim;
    private readonly int _actionDim;
    private readonly float[] _obs;
    private readonly float[] _nextObs;
    private readonly float[] _actions;
    private readonly float[] _rewards;
    private readonly bool[] _terminated;
    private int _next;

    public ContinuousReplayBuffer(int capacity, int obsDim, int actionDim)
    {
        Capacity = capacity;
        _obsDim = obsDim;
        _actionDim = actionDim;
        _obs = new float[capacity * obsDim];
        _nextObs = new float[capacity * obsDim];
        _actions = new float[capacity * actionDim];
        _rewards = new float[capacity];
        _terminated = new bool[capacity];
    }

    public int Capacity { get; }
    public int Count { get; internal set; }

    // Internal state access for checkpointing (Checkpoints/ContinuousReplayBufferCheckpoint.cs).
    internal int ObsDim => _obsDim;
    internal int ActionDim => _actionDim;
    internal float[] ObsData => _obs;
    internal float[] NextObsData => _nextObs;
    internal float[] ActionsData => _actions;
    internal float[] RewardsData => _rewards;
    internal bool[] TerminatedData => _terminated;
    internal int NextIndex { get => _next; set => _next = value; }

    public void Add(ReadOnlySpan<float> obs, ReadOnlySpan<float> action, double reward,
        ReadOnlySpan<float> nextObs, bool terminated)
    {
        obs.CopyTo(_obs.AsSpan(_next * _obsDim, _obsDim));
        nextObs.CopyTo(_nextObs.AsSpan(_next * _obsDim, _obsDim));
        action.CopyTo(_actions.AsSpan(_next * _actionDim, _actionDim));
        _rewards[_next] = (float)reward;
        _terminated[_next] = terminated;

        _next = (_next + 1) % Capacity;
        Count = Math.Min(Count + 1, Capacity);
    }

    public Batch Sample(int batchSize, Xoshiro256StarStar rng)
    {
        var batch = new Batch(batchSize, _obsDim, _actionDim);
        for (int i = 0; i < batchSize; i++)
        {
            int index = rng.NextInt(Count);
            _obs.AsSpan(index * _obsDim, _obsDim).CopyTo(batch.Obs.AsSpan(i * _obsDim, _obsDim));
            _nextObs.AsSpan(index * _obsDim, _obsDim).CopyTo(batch.NextObs.AsSpan(i * _obsDim, _obsDim));
            _actions.AsSpan(index * _actionDim, _actionDim).CopyTo(batch.Actions.AsSpan(i * _actionDim, _actionDim));
            batch.Rewards[i] = _rewards[index];
            batch.Terminated[i] = _terminated[index];
        }
        return batch;
    }

    public sealed class Batch(int size, int obsDim, int actionDim)
    {
        public int Size { get; } = size;
        public int ObsDim { get; } = obsDim;
        public int ActionDim { get; } = actionDim;
        public float[] Obs { get; } = new float[size * obsDim];
        public float[] NextObs { get; } = new float[size * obsDim];
        public float[] Actions { get; } = new float[size * actionDim];
        public float[] Rewards { get; } = new float[size];
        public bool[] Terminated { get; } = new bool[size];
    }
}
