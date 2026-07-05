namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

// PUBLIC FACADE over the single-source transpiled solver (polyglot/fruitcake_solver.pg → PgFruitCakeWorld).
// The physics lives ONCE in the .pg (shared with the web client's TypeScript); this facade adapts the generated
// internal, camelCase, f64 core to the public API the env/controller/search/Lab already consume (PascalCase,
// float, plus host-only helpers — danger/eject queries, PileHeight, Clone, LoadBody — that aren't part of the
// shared solver). See docs/prd/POLYGLOT_FRUITCAKE_PRD.md (PG1).

/// <summary>A fruit body — a read-only float view over the generated <c>PgFruitBody</c> (positions/velocities
/// in pixels, angle in radians). The solver mutates the underlying body in place.</summary>
public sealed class FruitBody
{
    internal readonly PgFruitBody Inner;
    internal FruitBody(PgFruitBody inner) => Inner = inner;

    public float X => (float)Inner.x;
    public float Y => (float)Inner.y;
    public float Vx => (float)Inner.vx;
    public float Vy => (float)Inner.vy;
    public float Angle => (float)Inner.angle;
    public float AngularVel => (float)Inner.angularVel;
    public float R => (float)Inner.r;
    public int Tier => Inner.tier;
    public float Speed => (float)global::System.Math.Sqrt(Inner.vx * Inner.vx + Inner.vy * Inner.vy);
}

/// <summary>
/// The FruitCake container as a headless physics simulation. This is a thin adapter; the sequential-impulse
/// circle solver (with rigid-body rotation) is transpiled from <c>polyglot/fruitcake_solver.pg</c> — the single
/// source of truth shared bit-for-bit with the web client's <c>fruit-cake-physics.ts</c>.
///
/// <para>Reusable by both the training environment (step-to-rest, rotation off for speed/determinism) and the
/// live-serving handler (tick-to-snapshot, rotation on for visual roll). Pure compute — no wall-clock pacing.</para>
/// </summary>
public sealed class FruitCakeWorld
{
    public const float Width = 620f;
    public const float Height = 850f;
    public const float DangerLineY = 150f;

    private readonly PgFruitCakeWorld _core;

    /// <param name="enableRotation">
    /// Off (default) for training: angular state stays zero and the angular terms drop out — cheaper and fully
    /// deterministic; merges depend only on positions/tiers, so the policy is unaffected. On for serving so fruit
    /// visibly roll.
    /// </param>
    public FruitCakeWorld(bool enableRotation = false) => _core = new PgFruitCakeWorld(enableRotation);

    private FruitCakeWorld(PgFruitCakeWorld core) => _core = core;

    public IReadOnlyList<FruitBody> Bodies
    {
        get
        {
            var list = new List<FruitBody>(_core.bodies.Count);
            foreach (var b in _core.bodies) list.Add(new FruitBody(b));
            return list;
        }
    }

    public int Count => _core.count;

    /// <summary>Spawn a fruit of <paramref name="tier"/> centered at the given pixel position.</summary>
    public FruitBody SpawnFruit(int tier, float xPx, float yPx) => new(_core.spawnFruit(tier, xPx, yPx));

    public void Clear() => _core.clear();

    /// <summary>Re-add a fruit with full saved state (for env Restore).</summary>
    public void LoadBody(int tier, float x, float y, float vx, float vy, float angle, float angularVel)
    {
        var b = _core.spawnFruit(tier, x, y);
        b.vx = vx;
        b.vy = vy;
        b.angle = angle;
        b.angularVel = angularVel;
    }

    /// <summary>Advance one fixed sub-step; returns the merge points scored this sub-step.</summary>
    public int Step(float dt) => _core.step(dt);

    public float MaxSpeed() => (float)_core.maxSpeed();

    /// <summary>
    /// Advance the sim until the just-dropped fruit (and any cascade) comes to rest, in pure compute. Returns
    /// the merge points scored. Early-settle once the pile is quiet and nothing merged, after
    /// <paramref name="minSubsteps"/>; <paramref name="maxSubsteps"/> is the safety cap.
    /// </summary>
    public int SettleAfterDrop(float settleSpeedPx, int minSubsteps, int maxSubsteps, double dt = 1.0 / 60.0)
        => _core.settleAfterDrop(settleSpeedPx, minSubsteps, maxSubsteps, dt);

    /// <summary>True if any fruit has been pushed up over the rim (center above the top).</summary>
    public bool AnyEjected() => _core.anyEjected();

    /// <summary>True if any fruit is settled (speed &lt; <paramref name="restSpeedPx"/>) above the danger line.</summary>
    public bool AnyRestingAboveDangerLine(float restSpeedPx) => _core.anyRestingAboveDangerLine(restSpeedPx);

    /// <summary>True if any fruit's center is above the danger line (regardless of speed) — for the visual pulse.</summary>
    public bool AnyAboveDangerLine()
    {
        foreach (var b in _core.bodies) if (b.y < DangerLineY) return true;
        return false;
    }

    /// <summary>Height of the tallest point of the pile above the floor (0 = empty board).</summary>
    public float PileHeight() => (float)_core.pileHeight();

    /// <summary>The 89-dim RL observation for the given held/next tiers — the single-source core vector (f64)
    /// cast to float for the float32 net. See <see cref="FruitCakeEnv.BuildObservation"/> for the field spec.</summary>
    public float[] BuildObservation(int current, int next)
    {
        var core = _core.buildObservation(current, next);
        var obs = new float[core.Count];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)core[i];
        return obs;
    }

    /// <summary>The raw f64 observation (not cast to float) — for the single-source f64 inference core, which
    /// consumes it directly so C# serving and the browser feed the net identical inputs.</summary>
    public IReadOnlyList<double> BuildObservationF64(int current, int next) => _core.buildObservation(current, next);

    /// <summary>Serialize every body's full state at native (double) precision — for env Save. Writes the count
    /// then, per body: tier, x, y, vx, vy, angle, angularVel. Double (not the facade's float view) so
    /// <see cref="ReadBodies"/> round-trips the solver's exact state and a resumed sim stays bitwise-identical.</summary>
    public void WriteBodies(BinaryWriter writer)
    {
        writer.Write(_core.bodies.Count);
        foreach (var b in _core.bodies)
        {
            writer.Write(b.tier);
            writer.Write(b.x); writer.Write(b.y);
            writer.Write(b.vx); writer.Write(b.vy);
            writer.Write(b.angle); writer.Write(b.angularVel);
        }
    }

    /// <summary>Restore the body set written by <see cref="WriteBodies"/> (clears the world first).</summary>
    public void ReadBodies(BinaryReader reader)
    {
        _core.clear();
        int count = reader.ReadInt32();
        for (int k = 0; k < count; k++)
        {
            int tier = reader.ReadInt32();
            double x = reader.ReadDouble(), y = reader.ReadDouble();
            double vx = reader.ReadDouble(), vy = reader.ReadDouble();
            double angle = reader.ReadDouble(), angularVel = reader.ReadDouble();
            var b = _core.spawnFruit(tier, x, y);
            b.vx = vx; b.vy = vy; b.angle = angle; b.angularVel = angularVel;
        }
    }

    /// <summary>
    /// Deep copy for what-if planning (the search tries each column on a clone). <paramref name="enableRotation"/>
    /// overrides the copy's rotation mode — planning runs rotation off (cheaper, deterministic) even when the live
    /// world has it on.
    /// </summary>
    public FruitCakeWorld Clone(bool? enableRotation = null)
        => new(_core.clone(enableRotation ?? _core.rotation));
}
