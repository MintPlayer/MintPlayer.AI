using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Classic Snake on a configurable <see cref="Size"/>×<see cref="Size"/> grid, as an RL environment
/// (PLAN M22). The observation is <b>compact and grid-size-invariant</b> — 12 engineered features (danger
/// one step in each direction, food direction relative to the head, current heading one-hot) — so a net
/// trained on a small grid transfers directly to a larger one (a raw grid is a poor fit for a dense MLP and
/// learns to survive but not to hunt; these features fix that and enable a grid-size curriculum). Four
/// absolute-direction actions; the single illegal action (the 180° reversal onto the neck) is masked via
/// <see cref="IActionMaskProvider"/>. Walls and the snake's own body are NOT masked — death stays a
/// learnable outcome.
/// <para>
/// Gate (pre-registered): a competent agent eats <b>≥ 5 food / episode</b> over 100 greedy episodes.
/// </para>
/// </summary>
public sealed class SnakeEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ActionCount = 4;
    public const int ObservationSize = 12; // danger×4, food-direction×4, heading×4 — independent of grid size
    public const int MaxEpisodeSteps = 1000;

    public const float FoodReward = 1f;
    public const float StepPenalty = -0.01f;
    public const float DeathReward = -1f;

    // Action = absolute direction; (dRow, dCol) per action index.
    private static readonly (int Dr, int Dc)[] Deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // Up, Down, Left, Right

    private Xoshiro256StarStar _rng = new(0);
    private readonly LinkedList<int> _body = new(); // head = First, tail = Last (cell indices)
    private readonly HashSet<int> _occupied = [];
    private int _food;
    private int _foodEaten;
    private int _heading;
    private int _elapsedSteps;
    private bool _done = true;

    public SnakeEnv(int size = 12)
    {
        if (size < 5)
            throw new ArgumentOutOfRangeException(nameof(size), "Grid must be at least 5×5 (the length-3 start needs room).");
        Size = size;
        Cells = size * size;
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ActionCount);
    }

    public int Size { get; }
    public int Cells { get; }
    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    /// <summary>Snake cells, head first — for rendering / WS frame serialization.</summary>
    public IReadOnlyCollection<int> Body => _body;
    public int Food => _food;
    public int FoodEaten => _foodEaten;
    public int Length => _body.Count;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);

        _body.Clear();
        _occupied.Clear();
        // Length-3 snake, horizontal, centred, head pointing Right; body cells to its left.
        int row = Size / 2, headCol = Size / 2;
        for (int c = headCol; c >= headCol - 2; c--)
        {
            int cell = row * Size + c;
            _body.AddLast(cell);
            _occupied.Add(cell);
        }
        _heading = 3; // Right
        _foodEaten = 0;
        _elapsedSteps = 0;
        _done = false;
        SpawnFood();
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));
        if (!CurrentActionMask()[action])
            throw new ArgumentException($"Illegal action {action} (180° reversal); consult CurrentActionMask().", nameof(action));

        int head = _body.First!.Value;
        int headRow = head / Size, headCol = head % Size;
        var (dr, dc) = Deltas[action];
        int newRow = headRow + dr, newCol = headCol + dc;

        if (newRow < 0 || newRow >= Size || newCol < 0 || newCol >= Size)
            return Die();

        int newHead = newRow * Size + newCol;
        bool eating = newHead == _food;
        int tail = _body.Last!.Value;
        if (_occupied.Contains(newHead) && !(newHead == tail && !eating))
            return Die();

        _body.AddFirst(newHead);
        _occupied.Add(newHead);
        _heading = action;
        float reward;
        bool terminated = false;
        if (eating)
        {
            _foodEaten++;
            reward = FoodReward;
            if (_occupied.Count == Cells)
                terminated = true; // board full — a win
            else
                SpawnFood();
        }
        else
        {
            _occupied.Remove(tail);
            _body.RemoveLast();
            reward = StepPenalty;
        }

        _elapsedSteps++;
        bool truncated = !terminated && _elapsedSteps >= MaxEpisodeSteps;
        _done = terminated || truncated;
        return new StepResult<float[]>(Observation(), reward, terminated, truncated, EnvInfo.Empty);

        StepResult<float[]> Die()
        {
            _done = true;
            return new StepResult<float[]>(Observation(), DeathReward, true, false, EnvInfo.Empty);
        }
    }

    /// <summary>Legal moves: every direction except the one stepping onto the neck (a 180° reversal).</summary>
    public bool[] CurrentActionMask()
    {
        var mask = new[] { true, true, true, true };
        if (_body.Count < 2) return mask;
        int head = _body.First!.Value, neck = _body.First!.Next!.Value;
        int headRow = head / Size, headCol = head % Size;
        for (int a = 0; a < 4; a++)
        {
            var (dr, dc) = Deltas[a];
            int r = headRow + dr, c = headCol + dc;
            if (r >= 0 && r < Size && c >= 0 && c < Size && r * Size + c == neck)
                mask[a] = false;
        }
        return mask;
    }

    public float[] CurrentObservation() => Observation();

    private float[] Observation()
    {
        int head = _body.First!.Value;
        int hr = head / Size, hc = head % Size;
        int tail = _body.Last!.Value;

        bool Danger(int dir)
        {
            var (dr, dc) = Deltas[dir];
            int r = hr + dr, c = hc + dc;
            if (r < 0 || r >= Size || c < 0 || c >= Size) return true; // wall
            int cell = r * Size + c;
            return _occupied.Contains(cell) && cell != tail; // body (the tail cell will vacate)
        }

        int fr = _food / Size, fc = _food % Size;
        return
        [
            Danger(0) ? 1f : 0f, Danger(1) ? 1f : 0f, Danger(2) ? 1f : 0f, Danger(3) ? 1f : 0f,
            fr < hr ? 1f : 0f,  // food up
            fr > hr ? 1f : 0f,  // food down
            fc < hc ? 1f : 0f,  // food left
            fc > hc ? 1f : 0f,  // food right
            _heading == 0 ? 1f : 0f, _heading == 1 ? 1f : 0f, _heading == 2 ? 1f : 0f, _heading == 3 ? 1f : 0f,
        ];
    }

    private void SpawnFood()
    {
        int free = Cells - _occupied.Count;
        int pick = _rng.NextInt(free);
        for (int cell = 0; cell < Cells; cell++)
        {
            if (_occupied.Contains(cell)) continue;
            if (pick-- == 0) { _food = cell; return; }
        }
    }

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_body.Count);
        foreach (int cell in _body) writer.Write(cell); // head first
        writer.Write(_food);
        writer.Write(_heading);
        writer.Write(_foodEaten);
        writer.Write(_elapsedSteps);
        writer.Write(_done);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _body.Clear();
        _occupied.Clear();
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            int cell = reader.ReadInt32();
            _body.AddLast(cell);
            _occupied.Add(cell);
        }
        _food = reader.ReadInt32();
        _heading = reader.ReadInt32();
        _foodEaten = reader.ReadInt32();
        _elapsedSteps = reader.ReadInt32();
        _done = reader.ReadBoolean();
    }

    public string RenderString()
    {
        int head = _body.First!.Value;
        var sb = new StringBuilder();
        for (int r = 0; r < Size; r++)
        {
            for (int c = 0; c < Size; c++)
            {
                int cell = r * Size + c;
                char glyph = cell == head ? '@' : _occupied.Contains(cell) ? 'o' : cell == _food ? '*' : '.';
                sb.Append(glyph);
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
