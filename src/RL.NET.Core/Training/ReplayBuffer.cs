using RLNet.Core.Random;

namespace RLNet.Core.Training;

/// <summary>
/// Circular experience-replay buffer for off-policy learning.
/// Stores the <c>terminated</c> flag ONLY (never <c>truncated</c>): a transition cut by a
/// time limit must still bootstrap from its next state, and storing the combined done flag
/// is the classic silent DQN bug that caps learned values (PRD §8).
/// </summary>
public sealed class ReplayBuffer(int capacity, int obsDim)
{
    private readonly float[] _obs = new float[capacity * obsDim];
    private readonly float[] _nextObs = new float[capacity * obsDim];
    private readonly int[] _actions = new int[capacity];
    private readonly float[] _rewards = new float[capacity];
    private readonly bool[] _terminated = new bool[capacity];
    private int _next;

    public int Capacity => capacity;
    public int Count { get; private set; }

    public void Add(ReadOnlySpan<float> obs, int action, double reward, ReadOnlySpan<float> nextObs, bool terminated)
    {
        obs.CopyTo(_obs.AsSpan(_next * obsDim, obsDim));
        nextObs.CopyTo(_nextObs.AsSpan(_next * obsDim, obsDim));
        _actions[_next] = action;
        _rewards[_next] = (float)reward;
        _terminated[_next] = terminated;

        _next = (_next + 1) % capacity;
        Count = Math.Min(Count + 1, capacity);
    }

    public Batch Sample(int batchSize, Xoshiro256StarStar rng)
    {
        var batch = new Batch(batchSize, obsDim);
        for (int i = 0; i < batchSize; i++)
        {
            int index = rng.NextInt(Count);
            _obs.AsSpan(index * obsDim, obsDim).CopyTo(batch.Obs.AsSpan(i * obsDim, obsDim));
            _nextObs.AsSpan(index * obsDim, obsDim).CopyTo(batch.NextObs.AsSpan(i * obsDim, obsDim));
            batch.Actions[i] = _actions[index];
            batch.Rewards[i] = _rewards[index];
            batch.Terminated[i] = _terminated[index];
        }
        return batch;
    }

    public sealed class Batch(int size, int obsDim)
    {
        public int Size { get; } = size;
        public int ObsDim { get; } = obsDim;
        public float[] Obs { get; } = new float[size * obsDim];
        public float[] NextObs { get; } = new float[size * obsDim];
        public int[] Actions { get; } = new int[size];
        public float[] Rewards { get; } = new float[size];
        public bool[] Terminated { get; } = new bool[size];
    }
}
