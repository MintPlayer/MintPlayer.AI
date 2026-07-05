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
/// height + top tier + three relational blocks (danger margin, merge-with-current, adjacent-equal-pair),
/// current/next tier one-hots, and a few globals (PRD §4.3 / §4.B). Reward = normalized
/// merge points scored by the drop, with a small terminal penalty on game-over (a fruit ejected over
/// the rim, or settled above the danger line). Training runs rotation-off physics; merges don't depend
/// on orientation, so the policy transfers to the rotation-on live game.</para>
/// </summary>
public sealed class FruitCakeEnv : IEnvironment<float[], int>, IStatefulEnvironment
{
    public const int ColumnCount = 14;
    // Per column (×ColumnCount): surface height, top tier, danger margin, merge-with-current, adjacent-equal-pair.
    // Plus current one-hot (5) + next one-hot (5) + 3 globals + the two biggest fruit's (x, y, tier) = 6. The 3
    // relational per-column blocks (danger margin, merge-with-current, adjacent-equal-pair) are the B1+B2 fix for
    // the pineapple plateau (PRD §4.B); the big-fruit block is FRUITCAKE_BIGFRUIT_INPUTS_PRD §4.A — absolute
    // positions the bare skyline collapses, so the net can locate where the (possibly buried) biggest fruit sits.
    public const int ObservationSize = ColumnCount * 5 + 5 + 5 + 3 + 6; // 89

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
    private int _episodeMaxTier;     // highest tier seen this episode (for the one-time tier-reached bonus)
    private bool _done = true;

    public FruitCakeEnv()
    {
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ColumnCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    // ── Reward shaping (PRD §4.C, training-only; OFF by default so eval/serving see the bare game score) ──────
    // The raw reward (merge points − death penalty) is myopic: it's maximized by a steady drip of cheap low-tier
    // merges, so the policy plateaus at pineapple. These shape the *training* signal toward tier progression and
    // chain-setup WITHOUT changing what the game scores (the A/B + serving still judge real merge points).

    /// <summary>Enable the reward shaping below. Default off — a plain env is byte-for-byte the original game.</summary>
    public bool ShapeRewards { get; set; }

    /// <summary>Discount used by the potential-based term; set to match the learner's γ for policy-invariance.</summary>
    public double ShapingGamma { get; set; } = 0.99;

    /// <summary>One-time bonus the first time tier 6 is reached in an episode; doubles per tier (geometric) up to 11.</summary>
    public float TierBonusBase { get; set; } = 0.5f;

    /// <summary>Potential weight on same-tier near-pairs (tier-weighted): rewards keeping a mergeable layout.</summary>
    public float AdjacencyWeight { get; set; } = 0.02f;

    /// <summary>Potential penalty on normalized pile height: nudges toward survival without a hard, hackable penalty.</summary>
    public float HeightWeight { get; set; } = 0.5f;

    /// <summary>Two same-tier fruit count as a near-pair when their centers are within this multiple of (rA+rB).</summary>
    public float AdjacencyProximity { get; set; } = 1.25f;

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
        _episodeMaxTier = FruitCatalog.MaxDroppableTier; // droppable tiers don't count; bonus starts at the first 6
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

        float potentialBefore = ShapeRewards ? Potential(_world) : 0f;

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
        if (ShapeRewards)
            reward += TierReachedBonus() + ShapingTerm(potentialBefore, terminated);
        return new StepResult<float[]>(Observation(), reward, terminated, truncated, EnvInfo.Empty);
    }

    public float[] CurrentObservation() => Observation();

    private float[] Observation() => BuildObservation(_world, _current, _next);

    /// <summary>
    /// Builds the 89-dim observation from a board + the current/next droppable tiers. Static so the live
    /// "Watch AI" serving handler (which drives a <see cref="FruitCakeWorld"/> directly, not a full env) can
    /// feed the trained net the <b>exact same observation</b> the policy was trained on.
    /// <para>Beyond the bare skyline (surface height + top tier per column), it carries three relational
    /// per-column blocks (PRD §4.B, the plateau fix): <b>danger margin</b> (how much room below the danger
    /// line), <b>merge-with-current</b> (dropping the current fruit here scores an immediate merge), and
    /// <b>adjacent-equal-pair</b> (this surface fruit already has a same-tier neighbour — a half-built merge).
    /// The skyline alone exposed none of mergeable adjacency or survival margin.</para>
    /// <para>Finally, the <b>(x, y, tier) of the two biggest fruit</b> (FRUITCAKE_BIGFRUIT_INPUTS_PRD §4.A):
    /// absolute positions the per-column skyline collapses, so the policy (and the search leaf) can reason
    /// about where the largest fruit — possibly buried below the surface — actually sits.</para>
    /// </summary>
    public static float[] BuildObservation(FruitCakeWorld world, int current, int next)
        => world.BuildObservation(current, next);

    // One-time, geometrically-scaled bonus the first time each new highest tier (6→11) appears this episode.
    // One-time ⇒ unfarmable; directly rewards the goal the cumulative-score objective hides (PRD §4.C C1).
    private float TierReachedBonus()
    {
        int boardMax = 0;
        foreach (var b in _world.Bodies) if (b.Tier > boardMax) boardMax = b.Tier;

        float bonus = 0f;
        while (_episodeMaxTier < boardMax)
        {
            _episodeMaxTier++;
            bonus += TierBonusBase * (1 << (_episodeMaxTier - (FruitCatalog.MaxDroppableTier + 1))); // 6→×1, 7→×2, …
        }
        return bonus;
    }

    // Potential-based shaping term γ·Φ(s′) − Φ(s) (Ng et al. 1999). Policy-invariant for ANY Φ ⇒ it can only help
    // or be neutral, never reward-hack. Φ(terminal) ≡ 0 so no value leaks into the death state (PRD §4.C C2/C3).
    private float ShapingTerm(float potentialBefore, bool terminated)
    {
        float potentialAfter = terminated ? 0f : Potential(_world);
        return (float)(ShapingGamma * potentialAfter - potentialBefore);
    }

    // Φ = (tier-weighted count of same-tier near-pairs) − (normalized pile height). The first term rewards a
    // mergeable, tier-sorted layout (the perception-side mirror of the adjacency observation); the second is a
    // soft survival nudge. Same-tier touching fruit auto-merge, so a "near-pair" is two equal-tier fruit just
    // short of contact — a half-built merge a future drop can complete.
    private float Potential(FruitCakeWorld world)
    {
        var bodies = world.Bodies;
        float adjacency = 0f;
        for (int i = 0; i < bodies.Count; i++)
            for (int j = i + 1; j < bodies.Count; j++)
            {
                var a = bodies[i];
                var b = bodies[j];
                if (a.Tier != b.Tier) continue;
                float dx = b.X - a.X, dy = b.Y - a.Y;
                float reach = (a.R + b.R) * AdjacencyProximity;
                if (dx * dx + dy * dy <= reach * reach) adjacency += a.Tier;
            }
        return AdjacencyWeight * adjacency - HeightWeight * (world.PileHeight() / FruitCakeWorld.Height);
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
        writer.Write(_episodeMaxTier);
        writer.Write(_done);
        _world.WriteBodies(writer); // full double precision (see FruitCakeWorld.WriteBodies)
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
        _episodeMaxTier = reader.ReadInt32();
        _done = reader.ReadBoolean();
        _world.ReadBodies(reader);
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
