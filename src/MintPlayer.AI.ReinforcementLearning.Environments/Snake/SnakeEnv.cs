using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Classic Snake on a configurable <see cref="Size"/>×<see cref="Size"/> grid, as an RL environment
/// (PLAN M22; observation reworked in M27). The observation is <b>grid-size-invariant</b>: 8-direction ray-cast
/// vision (food-on-ray, 1/distance to body, 1/distance to wall) for long-range awareness, plus a flood-fill of the
/// open space reachable through each move (the anti-self-trap signal — a long snake can't see a coil beyond a fixed
/// window, but it can be told which moves keep space reachable), food &amp; tail bearing + distance, heading, and
/// length. Four absolute-direction actions; the single illegal action (the 180° reversal onto the neck) is masked
/// via <see cref="IActionMaskProvider"/>. Walls and the snake's own body are NOT masked — death stays learnable.
/// <para>
/// Episodes end on death, a board-full win, a starvation timeout (<see cref="StarveLimit"/> steps without food),
/// or an absolute safety ceiling — never a flat step cap, so high-score (long-snake) games are reachable.
/// </para>
/// </summary>
public sealed class SnakeEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ActionCount = 4;

    // Grid-size-invariant observation (PLAN M27). 8-direction ray-cast vision (CodeBullet-style): per ray, whether
    // food lies on it, 1/distance to the nearest body segment, and 1/distance to the wall — long-range awareness a
    // fixed window can't give. Plus the flood-fill of open space reachable through each of the 4 neighbours (the
    // anti-self-trap signal), food & tail bearing + distance, heading one-hot, and normalized length. Distances are
    // 1/cells so the whole vector is size-invariant — a net trained on one grid transfers to another.
    public const int RayDirections = 8;
    public const int RayChannels = 3;                                   // food-on-ray, 1/dist-to-body, 1/dist-to-wall
    public const int RayFeatures = RayDirections * RayChannels;         // 24
    public const int ObservationSize = RayFeatures + 4 + 3 + 3 + 4 + 1; // 39: rays + flood(4) + food(3) + tail(3) + heading(4) + len(1)

    public const float FoodReward = 1f;
    public const float StepPenalty = -0.01f;
    public const float DeathReward = -1f;

    // Episode limits scale with the board so high-score games are possible: the snake dies only after this many
    // steps WITHOUT eating (a starvation timeout, not a flat cap that would hard-limit food); a generous absolute
    // ceiling guards a true infinite loop that keeps eating just often enough to never starve. The starvation
    // window is a multiple of the board (default 2): training keeps it tight to discourage aimless wandering, but
    // a long snake under look-ahead inference legitimately needs long, food-less detours, so inference widens it.
    private int StarveLimit => _starveLimitCells * Cells;
    private int MaxEpisodeSteps => 100 * Cells;

    // Action = absolute direction; (dRow, dCol) per action index.
    private static readonly (int Dr, int Dc)[] Deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // Up, Down, Left, Right

    // The 8 ray-cast directions for the vision features (4 cardinal + 4 diagonal), clockwise from North.
    private static readonly (int Dr, int Dc)[] RayDeltas =
        [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

    private Xoshiro256StarStar _rng = new(0);
    private readonly LinkedList<int> _body = new(); // head = First, tail = Last (cell indices)
    private readonly HashSet<int> _occupied = [];
    private int _food;
    private int _foodEaten;
    private int _heading;
    private int _elapsedSteps;
    private int _stepsSinceFood;
    private bool _done = true;
    private readonly float _stepPenalty;
    private readonly bool _safeMask;
    private readonly int _starveLimitCells;

    /// <param name="stepPenalty">
    /// Per-non-eating-step reward (training only — inference is greedy and ignores reward). Defaults to
    /// <see cref="StepPenalty"/>; pass ~0 to remove the efficiency pressure that makes pursuing distant food
    /// (unavoidable for a long snake) barely net-positive and so encourages safe starvation.
    /// </param>
    /// <param name="safeMask">
    /// When true, <see cref="CurrentActionMask"/> also forbids moves that would seal the snake into a region too
    /// small to hold its body — a reactive flood-fill "shield" against self-trapping. Off by default (preserves
    /// the reversal-only mask). Can be enabled for both training and inference.
    /// </param>
    /// <param name="starveLimitCells">
    /// Starvation window as a multiple of the board area: the episode truncates after this many steps without
    /// eating. Default 2 keeps training episodes from wandering aimlessly; raise it for look-ahead inference, where
    /// a long snake must legitimately take board-spanning, food-less detours to stay safe.
    /// </param>
    public SnakeEnv(int size = 12, float stepPenalty = StepPenalty, bool safeMask = false, int starveLimitCells = 2)
    {
        if (size < 5)
            throw new ArgumentOutOfRangeException(nameof(size), "Grid must be at least 5×5 (the length-3 start needs room).");
        if (starveLimitCells < 1)
            throw new ArgumentOutOfRangeException(nameof(starveLimitCells), "Starvation window must be at least one board.");
        Size = size;
        Cells = size * size;
        _stepPenalty = stepPenalty;
        _safeMask = safeMask;
        _starveLimitCells = starveLimitCells;
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
        _stepsSinceFood = 0;
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
            _stepsSinceFood = 0;
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
            _stepsSinceFood++;
            reward = _stepPenalty;
        }

        _elapsedSteps++;
        // Starvation timeout (board-scaled) ends a snake that's wandering without progress; the absolute ceiling
        // is just a safety net. A flat cap is deliberately avoided — it would hard-limit how much food is reachable.
        bool truncated = !terminated && (_stepsSinceFood >= StarveLimit || _elapsedSteps >= MaxEpisodeSteps);
        _done = terminated || truncated;
        return new StepResult<float[]>(Observation(), reward, terminated, truncated, EnvInfo.Empty);

        StepResult<float[]> Die()
        {
            _done = true;
            return new StepResult<float[]>(Observation(), DeathReward, true, false, EnvInfo.Empty);
        }
    }

    /// <summary>
    /// Legal moves: every direction except the 180° reversal onto the neck. When <see cref="_safeMask"/> is on,
    /// ALSO forbids any move that would seal the snake into a region smaller than its own body (a flood-fill from
    /// the entered cell reaching fewer than <see cref="Length"/> free cells = a guaranteed trap) — a reactive
    /// 1-ply "shield" that eliminates the self-trapping deaths which otherwise cap a long snake. If that would
    /// leave no legal move, the shield is dropped for this step (pick the least-bad move rather than no move).
    /// </summary>
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

        if (!_safeMask) return mask;

        int tail = _body.Last!.Value, length = _body.Count;
        var safe = (bool[])mask.Clone();
        bool any = false;
        for (int a = 0; a < 4; a++)
        {
            if (!mask[a]) continue;
            var (dr, dc) = Deltas[a];
            // A move is "safe" only if the space reachable after entering it can still hold the whole body.
            if (ReachableFreeSpace(headRow + dr, headCol + dc, tail) >= length) any = true;
            else safe[a] = false;
        }
        return any ? safe : mask; // never return an all-false mask
    }

    public float[] CurrentObservation() => Observation();

    /// <summary>
    /// The largest open region the head can move into next: max over the four neighbours of the free cells reachable
    /// from that neighbour (BFS, tail counts as free). A position where this is below <see cref="Length"/> is a
    /// guaranteed self-trap — the snake can no longer fit its own body. Exposed for look-ahead search, which scores a
    /// simulated leaf position by survivability (the signal a reactive value net is worst at). 0 means already boxed in.
    /// </summary>
    public int FreeSpaceAhead()
    {
        int head = _body.First!.Value, tail = _body.Last!.Value;
        int hr = head / Size, hc = head % Size, best = 0;
        foreach (var (dr, dc) in Deltas)
            best = Math.Max(best, ReachableFreeSpace(hr + dr, hc + dc, tail));
        return best;
    }

    private float[] Observation()
    {
        int head = _body.First!.Value;
        int hr = head / Size, hc = head % Size;
        int tail = _body.Last!.Value;
        int fr = _food / Size, fc = _food % Size;

        var obs = new float[ObservationSize];
        int s = 0;

        // ── 8-direction ray-cast vision: per ray, food-on-line, 1/dist to nearest body, 1/dist to wall ──
        foreach (var (dr, dc) in RayDeltas)
        {
            int r = hr + dr, c = hc + dc, dist = 1;
            float food = 0f, body = 0f;
            while (r >= 0 && r < Size && c >= 0 && c < Size)
            {
                int cell = r * Size + c;
                if (food == 0f && cell == _food) food = 1f;
                if (body == 0f && _occupied.Contains(cell)) body = 1f / dist; // nearest body segment along the ray
                r += dr; c += dc; dist++;
            }
            obs[s++] = food;
            obs[s++] = body;
            obs[s++] = 1f / dist; // wall: closer wall ⇒ larger
        }

        // ── flood-fill of open space reachable through each neighbour (the anti-self-trap signal) ──
        foreach (var (dr, dc) in Deltas)
            obs[s++] = ReachableFreeSpace(hr + dr, hc + dc, tail) / (float)Cells;

        // ── food bearing + L1 distance (precise, unlike the binary ray-food) ──
        obs[s++] = (fc - hc) / (float)Size;
        obs[s++] = (fr - hr) / (float)Size;
        obs[s++] = (Math.Abs(fr - hr) + Math.Abs(fc - hc)) / (2f * Size);

        // ── tail bearing + distance: chasing the tail (whose cell always vacates) is the canonical survival move ──
        int tr = tail / Size, tc = tail % Size;
        obs[s++] = (tc - hc) / (float)Size;
        obs[s++] = (tr - hr) / (float)Size;
        obs[s++] = (Math.Abs(tr - hr) + Math.Abs(tc - hc)) / (2f * Size);

        // ── heading one-hot + normalized length ──
        obs[s++] = _heading == 0 ? 1f : 0f;
        obs[s++] = _heading == 1 ? 1f : 0f;
        obs[s++] = _heading == 2 ? 1f : 0f;
        obs[s++] = _heading == 3 ? 1f : 0f;
        obs[s++] = _body.Count / (float)Cells;
        return obs;
    }

    /// <summary>
    /// BFS flood-fill from <paramref name="r"/>,<paramref name="c"/> over cells that are on the board and not
    /// occupied by a non-vacating body segment; returns the count of reachable free cells (0 if the start itself
    /// is blocked). The tail cell counts as free (it vacates). Bounds the work at <see cref="Cells"/>.
    /// </summary>
    private int ReachableFreeSpace(int r, int c, int tail)
    {
        bool Free(int rr, int cc)
        {
            if (rr < 0 || rr >= Size || cc < 0 || cc >= Size) return false;
            int cell = rr * Size + cc;
            return !_occupied.Contains(cell) || cell == tail;
        }
        if (!Free(r, c)) return 0;

        var seen = new bool[Cells];
        var queue = new Queue<int>();
        int start = r * Size + c;
        seen[start] = true;
        queue.Enqueue(start);
        int count = 0;
        while (queue.Count > 0)
        {
            int cell = queue.Dequeue();
            count++;
            int cr = cell / Size, cc2 = cell % Size;
            foreach (var (dr, dc) in Deltas)
            {
                int nr = cr + dr, nc = cc2 + dc;
                if (!Free(nr, nc)) continue;
                int n = nr * Size + nc;
                if (seen[n]) continue;
                seen[n] = true;
                queue.Enqueue(n);
            }
        }
        return count;
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
        writer.Write(_stepsSinceFood);
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
        _stepsSinceFood = reader.ReadInt32();
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
