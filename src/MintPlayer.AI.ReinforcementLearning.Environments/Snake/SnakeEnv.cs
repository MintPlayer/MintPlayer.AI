using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Classic Snake on a configurable <see cref="Size"/>×<see cref="Size"/> grid as an RL environment (PLAN M22;
/// egocentric observation M27). This is a <b>public facade</b> over the single-source transpiled core
/// (<c>polyglot/snake_solver.pg</c> → <c>PgSnakeEnv</c>): the dynamics, the 177-dim egocentric observation, and the
/// action mask (incl. the anti-self-trap flood-fill shield) live <b>once</b> in the <c>.pg</c>, shared bit-for-bit
/// with the browser's <c>snake_solver.ts</c> (M33). The facade re-adds host concerns the shared core leaves out: the
/// <see cref="IEnvironment{T,U}"/>/<see cref="IStatefulEnvironment"/> API, the food RNG (a seeded
/// <see cref="Xoshiro256StarStar"/> — kept out of the single source so C# stays deterministic and the browser can use
/// its own RNG), state (de)serialization, and the throw-on-illegal-action contract.
/// <para>Four absolute-direction actions; the 180° reversal onto the neck is masked via
/// <see cref="IActionMaskProvider"/>. Walls/body are NOT masked — death stays learnable. Episodes end on death, a
/// board-full win, a starvation timeout, or an absolute ceiling (all in the core).</para>
/// </summary>
public sealed class SnakeEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ActionCount = 4;
    public const int PatchRadius = 4;
    public const int PatchSide = 2 * PatchRadius + 1;                       // 9
    public const int PatchChannels = 2;                                     // 0 = obstacle, 1 = food
    public const int PatchSize = PatchSide * PatchSide * PatchChannels;     // 162
    public const int ScalarFeatures = 15;                                   // foodΔ(2)+dist(1)+heading(4)+len(1)+free(4)+tailΔ(2)+tailDist(1)
    public const int ObservationSize = PatchSize + ScalarFeatures;          // 177

    /// <summary>Plain-language name for each of the 4 actions (absolute directions), index = action id.</summary>
    public static readonly IReadOnlyList<string> ActionLabels = ["Move Up", "Move Down", "Move Left", "Move Right"];

    /// <summary>Plain-language name for each of the <see cref="ObservationSize"/> observation features, in index
    /// order (mirrors <c>snake_solver.pg</c> <c>buildObservation</c>): a 9×9 egocentric obstacle plane, a 9×9 food
    /// plane, then the scalar block. Lets a visualizer say what each input neuron represents.</summary>
    public static readonly IReadOnlyList<string> ObservationLabels = BuildObservationLabels();

    private static string[] BuildObservationLabels()
    {
        const int plane = PatchSide * PatchSide; // 81
        var labels = new string[ObservationSize];
        for (int i = 0; i < plane; i++)
        {
            int dr = i / PatchSide - PatchRadius, dc = i % PatchSide - PatchRadius;
            string where = PatchWhere(dr, dc);
            labels[i] = $"Vision: is a wall/body {where}? (egocentric, head-relative)";
            labels[plane + i] = $"Vision: is the food {where}? (egocentric, head-relative)";
        }
        int s = PatchSize;
        labels[s++] = "Food offset: columns right of the head (− = left)";
        labels[s++] = "Food offset: rows below the head (− = above)";
        labels[s++] = "Distance to the food (normalized)";
        labels[s++] = "Current heading is Up";
        labels[s++] = "Current heading is Down";
        labels[s++] = "Current heading is Left";
        labels[s++] = "Current heading is Right";
        labels[s++] = "Snake length (÷ board cells)";
        labels[s++] = "Reachable free space if it moves Up (÷ cells)";
        labels[s++] = "Reachable free space if it moves Down (÷ cells)";
        labels[s++] = "Reachable free space if it moves Left (÷ cells)";
        labels[s++] = "Reachable free space if it moves Right (÷ cells)";
        labels[s++] = "Tail offset: columns right of the head (− = left)";
        labels[s++] = "Tail offset: rows below the head (− = above)";
        labels[s++] = "Distance to the tail (normalized)";
        return labels;
    }

    // Describe an egocentric patch cell's offset from the head, e.g. (-3, 2) → "3 up, 2 right".
    private static string PatchWhere(int dr, int dc)
    {
        if (dr == 0 && dc == 0) return "at the head";
        string? v = dr < 0 ? $"{-dr} up" : dr > 0 ? $"{dr} down" : null;
        string? h = dc < 0 ? $"{-dc} left" : dc > 0 ? $"{dc} right" : null;
        return v is null ? h! : h is null ? v : $"{v}, {h}";
    }

    public const float FoodReward = 1f;
    public const float StepPenalty = -0.01f;
    public const float DeathReward = -1f;

    private Xoshiro256StarStar _rng = new(0);
    private readonly PgSnakeEnv _core;

    /// <param name="stepPenalty">Per-non-eating-step reward (training only; inference is greedy). ~0 removes the
    /// efficiency pressure that can encourage safe starvation.</param>
    /// <param name="safeMask">When true, <see cref="CurrentActionMask"/> also forbids moves that would seal the snake
    /// into a region too small for its body (a reactive flood-fill shield). The deployed net was trained with it on.</param>
    public SnakeEnv(int size = 12, float stepPenalty = StepPenalty, bool safeMask = false)
    {
        if (size < 5)
            throw new ArgumentOutOfRangeException(nameof(size), "Grid must be at least 5×5 (the length-3 start needs room).");
        _core = new PgSnakeEnv(size, stepPenalty, safeMask);
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ActionCount);
    }

    public int Size => _core.size;
    public int Cells => _core.cells;
    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    /// <summary>Snake cells, head first — for rendering / serialization. (The core stores head-at-end; this reverses.)</summary>
    public IReadOnlyList<int> Body
    {
        get
        {
            int n = _core.body.Count;
            var b = new int[n];
            for (int k = 0; k < n; k++) b[k] = _core.body[n - 1 - k];
            return b;
        }
    }
    public int Food => _core.food;
    public int FoodEaten => _core.foodEaten;
    public int Length => _core.body.Count;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _core.reset();
        _core.spawnFood(_rng.NextInt(_core.freeCount()));
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_core.done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));
        if (!_core.currentActionMask()[action])
            throw new ArgumentException($"Illegal action {action} (180° reversal); consult CurrentActionMask().", nameof(action));

        _core.step(action);
        // The core signals when a move ate and left room; the facade owns the RNG and places the food (matching the
        // exact free-count the core sees), so the Xoshiro sequence — and thus determinism — is preserved.
        if (_core.needsFood)
            _core.spawnFood(_rng.NextInt(_core.freeCount()));

        return new StepResult<float[]>(Observation(), (float)_core.lastReward, _core.lastTerminated, _core.lastTruncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => [.. _core.currentActionMask()];

    public float[] CurrentObservation() => Observation();

    private PgSnakeNet? _searchNet;

    /// <summary>Loads the trained dueling-Q checkpoint the look-ahead planner uses as its leaf evaluator — the same
    /// RLNC/"dueling-q" bytes the browser's <c>snake-net.ts</c> parses. Call once before <see cref="ChooseActionSearch"/>.</summary>
    public void LoadSearchNet(Stream checkpoint) => _searchNet = SnakeNetIo.Parse(checkpoint);

    /// <summary>
    /// Net-guided multi-ply look-ahead move for the current state (M34): simulates every legal line to
    /// <see cref="SnakeSearchConfig.MaxDepth"/> plies on cloned envs and plays the first move of the best-scoring
    /// one. The planner supersedes the reactive 1-ply shield, so construct this env with <c>safeMask: false</c> —
    /// the returned move is always a non-reversal, hence always legal under the reversal-only mask.
    /// </summary>
    /// <exception cref="InvalidOperationException">No net loaded — call <see cref="LoadSearchNet"/> first.</exception>
    public int ChooseActionSearch(SnakeSearchConfig config)
    {
        if (_searchNet is null)
            throw new InvalidOperationException("Call LoadSearchNet(...) before ChooseActionSearch(...).");
        return _core.chooseActionSearch(
            _searchNet, config.MaxDepth, config.BeamWidth, config.FoodWeight, config.TrapPenalty,
            config.NetWeight, config.SpaceWeight, config.FoodDistWeight, config.SpaceRatioWeight);
    }

    /// <summary>
    /// Safety-cycle move for the current state (M48): the snake keeps a Hamiltonian cycle it provably cannot die
    /// on and the trained net picks among the cycle-safe shortcuts toward the food. Must drive the episode from
    /// <see cref="Reset"/> onward (the cycle aligns to the fresh body and its safety argument needs every move to
    /// come from this chooser). Requires an even <see cref="Size"/> — odd×odd boards admit no Hamiltonian cycle
    /// (checkerboard parity) — and a loaded net (<see cref="LoadSearchNet"/>).
    /// </summary>
    /// <exception cref="InvalidOperationException">No net loaded, or the board size is odd.</exception>
    public int ChooseActionCycle(SnakeCycleConfig config)
    {
        if (Size % 2 != 0)
            throw new InvalidOperationException("Cycle mode needs an even board size (an odd×odd grid has no Hamiltonian cycle).");
        if (_searchNet is null)
            throw new InvalidOperationException("Call LoadSearchNet(...) before ChooseActionCycle(...).");
        return _core.chooseActionCycle(_searchNet, config.NetWeight, config.ProgressWeight, config.Margin);
    }

    private float[] Observation()
    {
        var core = _core.buildObservation();
        var obs = new float[core.Count];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)core[i];
        return obs;
    }

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        var body = Body; // head first
        writer.Write(body.Count);
        foreach (int cell in body) writer.Write(cell);
        writer.Write(_core.food);
        writer.Write(_core.heading);
        writer.Write(_core.foodEaten);
        writer.Write(_core.elapsedSteps);
        writer.Write(_core.stepsSinceFood);
        writer.Write(_core.done);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _core.body.Clear();
        for (int i = 0; i < _core.occupied.Count; i++) _core.occupied[i] = false;
        int count = reader.ReadInt32();
        var headFirst = new int[count];
        for (int i = 0; i < count; i++) headFirst[i] = reader.ReadInt32();
        // Core stores head-at-end, so add tail → head (reverse of the head-first buffer).
        for (int i = count - 1; i >= 0; i--)
        {
            int cell = headFirst[i];
            _core.body.Add(cell);
            _core.occupied[cell] = true;
        }
        _core.food = reader.ReadInt32();
        _core.heading = reader.ReadInt32();
        _core.foodEaten = reader.ReadInt32();
        _core.elapsedSteps = reader.ReadInt32();
        _core.stepsSinceFood = reader.ReadInt32();
        _core.done = reader.ReadBoolean();
    }

    public string RenderString()
    {
        int head = _core.headCell();
        var sb = new StringBuilder();
        for (int r = 0; r < Size; r++)
        {
            for (int c = 0; c < Size; c++)
            {
                int cell = r * Size + c;
                char glyph = cell == head ? '@' : _core.occupied[cell] ? 'o' : cell == _core.food ? '*' : '.';
                sb.Append(glyph);
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
