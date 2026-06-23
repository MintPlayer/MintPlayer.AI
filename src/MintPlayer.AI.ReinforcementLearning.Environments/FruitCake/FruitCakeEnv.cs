using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

// Game rules mirror the web game's src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-game.ts
// (training subset: no drop cooldown / effects / audio). Keep in sync (PRD docs/prd/FRUITCAKE_AI_PRD.md §4.8).

/// <summary>
/// FruitCake (Suika merge game) as an RL environment. One <see cref="Step"/> = <b>one drop</b>: place
/// the current fruit in the chosen column, then simulate the physics to rest <b>in pure compute</b>
/// (no wall-clock pacing) with early-settle, and return the new board observation + reward.
///
/// <para>Action = which of <see cref="ColumnCount"/> columns to drop in (the current/next droppable
/// tiers, 1–5, are part of the observation). Observation = a fixed feature vector: per-column surface
/// height + top tier, current/next tier one-hots, and a few globals (PRD §4.3). Reward = normalized
/// merge points scored by the drop, with a small terminal penalty on game-over (a fruit ejected over
/// the rim, or settled above the danger line). Training runs rotation-off physics; merges don't depend
/// on orientation, so the policy transfers to the rotation-on live game.</para>
/// </summary>
public sealed class FruitCakeEnv : IEnvironment<float[], int>, IStatefulEnvironment
{
    public const int ColumnCount = 14;
    public const int ObservationSize = ColumnCount * 2 + 5 + 5 + 3; // 41

    private const float RewardScale = 10f;       // normalize merge points
    private const float TerminalPenalty = -1f;
    public const float SettleSpeedPx = 30f;      // early-settle: below this (and no merge) the drop is at rest
    public const float RestSpeedPx = 40f;        // a fruit slower than this counts as "resting" for game-over
    public const int MinSettleSubsteps = 8;      // let a just-dropped fruit accelerate before testing for rest
    public const int MaxSubsteps = 600;          // safety cap (~10 s sim) — early-settle ends most drops far sooner
    private const int MaxDrops = 500;            // truncation cap
    private const float WallInset = 6f;

    private Xoshiro256StarStar _rng = new(0);
    private FruitCakeWorld _world = new(enableRotation: false);
    private int _current, _next;
    private int _score;
    private int _drops;
    private bool _done = true;

    public FruitCakeEnv()
    {
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ColumnCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public int Score => _score;
    public int Drops => _drops;
    public int CurrentTier => _current;
    public int NextTier => _next;
    public FruitCakeWorld World => _world;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _world = new FruitCakeWorld(enableRotation: false);
        _score = 0;
        _drops = 0;
        _done = false;
        _current = RandomDroppable();
        _next = RandomDroppable();
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));

        _world.SpawnFruit(_current, ColumnX(action, _current), HeldY(_current));

        // Simulate to rest in pure compute — no real-time waiting (PRD §4.2). Shared with the heuristic.
        int points = _world.SettleAfterDrop(SettleSpeedPx, MinSettleSubsteps, MaxSubsteps);

        _score += points;
        _drops++;
        _current = _next;
        _next = RandomDroppable();

        bool terminated = _world.AnyEjected() || _world.AnyRestingAboveDangerLine(RestSpeedPx);
        bool truncated = !terminated && _drops >= MaxDrops;
        _done = terminated || truncated;

        float reward = points / RewardScale + (terminated ? TerminalPenalty : 0f);
        return new StepResult<float[]>(Observation(), reward, terminated, truncated, EnvInfo.Empty);
    }

    public float[] CurrentObservation() => Observation();

    private float[] Observation()
    {
        const float W = FruitCakeWorld.Width, H = FruitCakeWorld.Height;
        float binW = W / ColumnCount;

        // Per-column surface: the highest fruit top (smallest y) overlapping each column's x-extent.
        var topY = new float[ColumnCount];
        var topTier = new int[ColumnCount];
        for (int c = 0; c < ColumnCount; c++) topY[c] = H; // empty column => surface at the floor

        float fillArea = 0f;
        foreach (var b in _world.Bodies)
        {
            fillArea += MathF.PI * b.R * b.R;
            int c0 = Math.Clamp((int)((b.X - b.R) / binW), 0, ColumnCount - 1);
            int c1 = Math.Clamp((int)((b.X + b.R) / binW), 0, ColumnCount - 1);
            float t = b.Y - b.R;
            for (int c = c0; c <= c1; c++)
                if (t < topY[c]) { topY[c] = t; topTier[c] = b.Tier; }
        }

        var obs = new float[ObservationSize];
        int i = 0;
        float minTop = H;
        for (int c = 0; c < ColumnCount; c++)
        {
            obs[i++] = Math.Clamp((H - topY[c]) / H, 0f, 1f);      // surface height (0 floor … 1 top)
            obs[i++] = Math.Clamp(topTier[c] / 11f, 0f, 1f);       // top-fruit tier
            if (topY[c] < minTop) minTop = topY[c];
        }

        for (int t = 1; t <= FruitCatalog.MaxDroppableTier; t++) obs[i++] = _current == t ? 1f : 0f;
        for (int t = 1; t <= FruitCatalog.MaxDroppableTier; t++) obs[i++] = _next == t ? 1f : 0f;

        obs[i++] = Math.Clamp(_world.Count / 100f, 0f, 1f);        // normalized fruit count
        obs[i++] = Math.Clamp(fillArea / (W * H), 0f, 1f);          // board fill ratio
        obs[i++] = Math.Clamp(minTop / H, 0f, 1f);                  // highest surface (0 = pile at the very top)
        return obs;
    }

    private int RandomDroppable() => FruitCatalog.Droppable[_rng.NextInt(FruitCatalog.Droppable.Count)].Tier;

    /// <summary>The clamped drop-x (px) for a column action and the current fruit's radius. Shared with the heuristic + serving.</summary>
    public static float ColumnX(int action, int currentTier)
    {
        float r = FruitCatalog.ByTier(currentTier).RadiusPx;
        float nominalX = (action + 0.5f) * (FruitCakeWorld.Width / ColumnCount);
        return Math.Clamp(nominalX, r + WallInset, FruitCakeWorld.Width - r - WallInset);
    }

    /// <summary>The held drop height (px), just above the danger line, for the current fruit.</summary>
    public static float HeldY(int currentTier) => FruitCakeWorld.DangerLineY - FruitCatalog.ByTier(currentTier).RadiusPx - 4f;

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_current);
        writer.Write(_next);
        writer.Write(_score);
        writer.Write(_drops);
        writer.Write(_done);
        writer.Write(_world.Count);
        foreach (var b in _world.Bodies)
        {
            writer.Write(b.Tier);
            writer.Write(b.X); writer.Write(b.Y);
            writer.Write(b.Vx); writer.Write(b.Vy);
            writer.Write(b.Angle); writer.Write(b.AngularVel);
        }
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _current = reader.ReadInt32();
        _next = reader.ReadInt32();
        _score = reader.ReadInt32();
        _drops = reader.ReadInt32();
        _done = reader.ReadBoolean();
        int count = reader.ReadInt32();
        _world.Clear();
        for (int k = 0; k < count; k++)
        {
            int tier = reader.ReadInt32();
            float x = reader.ReadSingle(), y = reader.ReadSingle();
            float vx = reader.ReadSingle(), vy = reader.ReadSingle();
            float angle = reader.ReadSingle(), angularVel = reader.ReadSingle();
            _world.LoadBody(tier, x, y, vx, vy, angle, angularVel);
        }
    }

    public string RenderString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"score={_score} drops={_drops} fruit={_world.Count} current={_current} next={_next} done={_done}");
        const float W = FruitCakeWorld.Width, H = FruitCakeWorld.Height;
        float binW = W / ColumnCount;
        var topY = new float[ColumnCount];
        for (int c = 0; c < ColumnCount; c++) topY[c] = H;
        foreach (var b in _world.Bodies)
        {
            int c0 = Math.Clamp((int)((b.X - b.R) / binW), 0, ColumnCount - 1);
            int c1 = Math.Clamp((int)((b.X + b.R) / binW), 0, ColumnCount - 1);
            for (int c = c0; c <= c1; c++) topY[c] = MathF.Min(topY[c], b.Y - b.R);
        }
        for (int c = 0; c < ColumnCount; c++)
        {
            int h = (int)Math.Clamp((H - topY[c]) / H * 10f, 0f, 10f);
            sb.Append(h.ToString("X"));
        }
        sb.AppendLine();
        return sb.ToString();
    }
}
