using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Circular experience-replay buffer for off-policy learning.
/// Stores the <c>terminated</c> flag ONLY (never <c>truncated</c>): a transition cut by a
/// time limit must still bootstrap from its next state, and storing the combined done flag
/// is the classic silent DQN bug that caps learned values (PRD §8).
/// </summary>
public sealed class ReplayBuffer
{
    private readonly int _obsDim;
    private readonly int _actionCount;
    private readonly float[] _obs;
    private readonly float[] _nextObs;
    private readonly int[] _actions;
    private readonly float[] _rewards;
    private readonly bool[] _terminated;
    private readonly bool[] _nextMask;
    private int _next;

    public ReplayBuffer(int capacity, int obsDim, int actionCount)
    {
        Capacity = capacity;
        _obsDim = obsDim;
        _actionCount = actionCount;
        _obs = new float[capacity * obsDim];
        _nextObs = new float[capacity * obsDim];
        _actions = new int[capacity];
        _rewards = new float[capacity];
        _terminated = new bool[capacity];
        _nextMask = new bool[capacity * actionCount];
    }

    public int Capacity { get; }
    public int Count { get; internal set; }

    // Internal state access for checkpointing (Checkpoints/DqnTrainingState.cs).
    internal int ObsDim => _obsDim;
    internal int ActionCount => _actionCount;
    internal float[] ObsData => _obs;
    internal float[] NextObsData => _nextObs;
    internal int[] ActionsData => _actions;
    internal float[] RewardsData => _rewards;
    internal bool[] TerminatedData => _terminated;
    internal bool[] NextMaskData => _nextMask;
    internal int NextIndex { get => _next; set => _next = value; }

    /// <param name="nextActionMask">
    /// Legal actions in the NEXT state (for masked TD-target max); empty span = all legal.
    /// </param>
    public void Add(ReadOnlySpan<float> obs, int action, double reward, ReadOnlySpan<float> nextObs,
        bool terminated, ReadOnlySpan<bool> nextActionMask = default)
    {
        obs.CopyTo(_obs.AsSpan(_next * _obsDim, _obsDim));
        nextObs.CopyTo(_nextObs.AsSpan(_next * _obsDim, _obsDim));
        _actions[_next] = action;
        _rewards[_next] = (float)reward;
        _terminated[_next] = terminated;
        if (nextActionMask.IsEmpty)
            _nextMask.AsSpan(_next * _actionCount, _actionCount).Fill(true);
        else
            nextActionMask.CopyTo(_nextMask.AsSpan(_next * _actionCount, _actionCount));

        _next = (_next + 1) % Capacity;
        Count = Math.Min(Count + 1, Capacity);
    }

    public Batch Sample(int batchSize, Xoshiro256StarStar rng)
    {
        var batch = new Batch(batchSize, _obsDim, _actionCount);
        for (int i = 0; i < batchSize; i++)
        {
            int index = rng.NextInt(Count);
            _obs.AsSpan(index * _obsDim, _obsDim).CopyTo(batch.Obs.AsSpan(i * _obsDim, _obsDim));
            _nextObs.AsSpan(index * _obsDim, _obsDim).CopyTo(batch.NextObs.AsSpan(i * _obsDim, _obsDim));
            batch.Actions[i] = _actions[index];
            batch.Rewards[i] = _rewards[index];
            batch.Terminated[i] = _terminated[index];
            _nextMask.AsSpan(index * _actionCount, _actionCount).CopyTo(batch.NextMasks.AsSpan(i * _actionCount, _actionCount));
        }
        return batch;
    }

    public sealed class Batch(int size, int obsDim, int actionCount)
    {
        public int Size { get; } = size;
        public int ObsDim { get; } = obsDim;
        public int ActionCount { get; } = actionCount;
        public float[] Obs { get; } = new float[size * obsDim];
        public float[] NextObs { get; } = new float[size * obsDim];
        public int[] Actions { get; } = new int[size];
        public float[] Rewards { get; } = new float[size];
        public bool[] Terminated { get; } = new bool[size];
        public bool[] NextMasks { get; } = new bool[size * actionCount];
    }
}
