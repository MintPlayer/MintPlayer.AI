namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>A fruit body in the simulation. Mutable; positions/velocities in pixels, angle in radians.</summary>
public sealed class FruitBody
{
    public float X, Y, Vx, Vy, Angle, AngularVel, R;
    public int Tier;
    public float InvMass, InvI;
    public bool PendingMerge, Removed;

    public float Speed => MathF.Sqrt(Vx * Vx + Vy * Vy);
}

/// <summary>
/// The FruitCake container as a headless physics simulation — a faithful C# port of the web game's
/// <c>fruit-cake-physics.ts</c> sequential-impulse circle solver (with rigid-body rotation). Every
/// collider is a circle and the walls are static. Same-tier touching fruit merge into the next tier
/// using the deferred pattern (record touching pairs while resolving, mutate the set only after).
///
/// <para>Reusable by both the training environment (step-to-rest, rotation off for speed/determinism)
/// and a future live-serving handler (tick-to-snapshot, rotation on for visual parity).</para>
///
/// Pure compute — no wall-clock pacing. Callers advance it with <see cref="Step"/> as fast as they like.
/// </summary>
public sealed class FruitCakeWorld
{
    public const float Width = 620f;
    public const float Height = 850f;
    public const float DangerLineY = 150f;

    private const float PixelsPerMeter = 64f;
    private const float Gravity = 9.8f * PixelsPerMeter; // px/s², +Y down
    private const float Restitution = 0.1f;
    private const float Friction = 0.3f;
    private const float RestitutionThresholdPx = 64f;

    private const int VelocityIterations = 12;
    private const int PositionIterations = 4;
    private const float Slop = 0.5f;
    private const float CorrectionPercent = 0.8f;
    private const float AngularDamping = 0.995f;

    private readonly bool _rotation;
    private readonly List<FruitBody> _bodies = [];
    private readonly List<(FruitBody A, FruitBody B)> _mergeQueue = [];

    /// <param name="enableRotation">
    /// Off (default) for training: angular state stays zero and the angular terms drop out, which is
    /// cheaper and fully deterministic; merges depend only on positions/tiers, so the policy is
    /// unaffected. On for serving so fruit visibly roll.
    /// </param>
    public FruitCakeWorld(bool enableRotation = false) => _rotation = enableRotation;

    public IReadOnlyList<FruitBody> Bodies => _bodies;
    public int Count => _bodies.Count;

    /// <summary>Spawn a fruit of <paramref name="tier"/> centered at the given pixel position.</summary>
    public FruitBody SpawnFruit(int tier, float xPx, float yPx)
    {
        var def = FruitCatalog.ByTier(tier);
        float r = def.RadiusPx;
        float mass = MathF.PI * r * r; // uniform density => area-proportional mass
        float inertia = 0.5f * mass * r * r; // solid disc: I = ½·m·r²
        var body = new FruitBody
        {
            X = xPx,
            Y = yPx,
            R = r,
            Tier = tier,
            InvMass = 1f / mass,
            InvI = _rotation ? 1f / inertia : 0f,
        };
        _bodies.Add(body);
        return body;
    }

    public void Clear()
    {
        _bodies.Clear();
        _mergeQueue.Clear();
    }

    /// <summary>Re-add a fruit with full saved state (for env Restore).</summary>
    public void LoadBody(int tier, float x, float y, float vx, float vy, float angle, float angularVel)
    {
        var body = SpawnFruit(tier, x, y);
        body.Vx = vx;
        body.Vy = vy;
        body.Angle = angle;
        body.AngularVel = angularVel;
    }

    /// <summary>Advance one fixed sub-step; returns the merge points scored this sub-step.</summary>
    public int Step(float dt)
    {
        foreach (var b in _bodies) b.Vy += Gravity * dt;

        BuildContacts(detect: true);
        for (int it = 0; it < VelocityIterations; it++)
            for (int i = 0; i < _contacts.Count; i++) ResolveVelocity(_contacts[i]);

        foreach (var b in _bodies)
        {
            b.X += b.Vx * dt;
            b.Y += b.Vy * dt;
            if (_rotation)
            {
                b.Angle += b.AngularVel * dt;
                b.AngularVel *= AngularDamping;
            }
        }

        for (int it = 0; it < PositionIterations; it++)
        {
            BuildContacts(detect: false);
            for (int i = 0; i < _contacts.Count; i++) CorrectPosition(_contacts[i]);
        }

        return FlushMerges();
    }

    public float MaxSpeed()
    {
        float max = 0f;
        foreach (var b in _bodies) max = MathF.Max(max, b.Speed);
        return max;
    }

    /// <summary>True if any fruit has been pushed up over the rim (center above the top).</summary>
    public bool AnyEjected()
    {
        foreach (var b in _bodies) if (b.Y < 0f) return true;
        return false;
    }

    /// <summary>True if any fruit is settled (speed &lt; <paramref name="restSpeedPx"/>) above the danger line.</summary>
    public bool AnyRestingAboveDangerLine(float restSpeedPx)
    {
        foreach (var b in _bodies)
            if (b.Y < DangerLineY && b.Speed < restSpeedPx) return true;
        return false;
    }

    /// <summary>True if any fruit's center is above the danger line (regardless of speed) — for the visual pulse.</summary>
    public bool AnyAboveDangerLine()
    {
        foreach (var b in _bodies) if (b.Y < DangerLineY) return true;
        return false;
    }

    /// <summary>Height of the tallest point of the pile above the floor (0 = empty board).</summary>
    public float PileHeight()
    {
        float minTop = Height;
        foreach (var b in _bodies) minTop = MathF.Min(minTop, b.Y - b.R);
        return Height - minTop;
    }

    /// <summary>
    /// Advance the sim until the just-dropped fruit (and any cascade) comes to rest, in pure compute (no
    /// wall-clock). Returns the merge points scored. Early-settle: stop the instant the pile is quiet and
    /// nothing merged this sub-step — but only after <paramref name="minSubsteps"/> so a fresh fruit has
    /// time to accelerate and fall; <paramref name="maxSubsteps"/> is a safety cap. Shared by the training
    /// env and the heuristic so they settle identically.
    /// </summary>
    public int SettleAfterDrop(float settleSpeedPx, int minSubsteps, int maxSubsteps, float dt = 1f / 60f)
    {
        int points = 0;
        for (int sub = 0; sub < maxSubsteps; sub++)
        {
            int gained = Step(dt);
            points += gained;
            if (sub >= minSubsteps && gained == 0 && MaxSpeed() < settleSpeedPx) break;
        }
        return points;
    }

    /// <summary>
    /// Deep copy for what-if planning (the heuristic tries each column on a clone). When
    /// <paramref name="enableRotation"/> is given it overrides the copy's rotation mode — the heuristic
    /// plans with rotation off (cheaper, deterministic; merges don't depend on orientation) even when the
    /// live world has it on.
    /// </summary>
    public FruitCakeWorld Clone(bool? enableRotation = null)
    {
        bool rot = enableRotation ?? _rotation;
        var copy = new FruitCakeWorld(rot);
        foreach (var b in _bodies)
            copy._bodies.Add(new FruitBody
            {
                X = b.X, Y = b.Y, Vx = b.Vx, Vy = b.Vy,
                Angle = rot ? b.Angle : 0f,
                AngularVel = rot ? b.AngularVel : 0f,
                R = b.R, Tier = b.Tier,
                InvMass = b.InvMass,
                InvI = rot ? b.InvI : 0f,
            });
        return copy;
    }

    // ── solver internals (ported 1:1 from fruit-cake-physics.ts) ───────────────────────────────

    private readonly struct Contact(FruitBody a, FruitBody? b, float nx, float ny, float pen)
    {
        public readonly FruitBody A = a;
        public readonly FruitBody? B = b; // null => a static wall
        public readonly float Nx = nx, Ny = ny, Pen = pen; // unit normal A→B (or fruit→wall), penetration
    }

    private readonly List<Contact> _contacts = [];

    private void BuildContacts(bool detect)
    {
        _contacts.Clear();

        foreach (var b in _bodies)
        {
            // Walls: left (x=0), right (x=W), floor (y=H). No ceiling — fruit may leave over the rim.
            float left = b.R - b.X;
            if (left > 0) _contacts.Add(new Contact(b, null, -1f, 0f, left));
            float right = b.X + b.R - Width;
            if (right > 0) _contacts.Add(new Contact(b, null, 1f, 0f, right));
            float floor = b.Y + b.R - Height;
            if (floor > 0) _contacts.Add(new Contact(b, null, 0f, 1f, floor));
        }

        for (int i = 0; i < _bodies.Count; i++)
        {
            var a = _bodies[i];
            for (int j = i + 1; j < _bodies.Count; j++)
            {
                var b = _bodies[j];
                float dx = b.X - a.X, dy = b.Y - a.Y;
                float rsum = a.R + b.R;
                float d2 = dx * dx + dy * dy;
                if (d2 >= rsum * rsum) continue;
                float d = MathF.Sqrt(d2);
                if (d == 0f) d = 0.0001f;

                if (detect && a.Tier == b.Tier && !a.PendingMerge && !b.PendingMerge)
                {
                    a.PendingMerge = b.PendingMerge = true;
                    _mergeQueue.Add((a, b)); // the merge covers this contact — no physical shove
                    continue;
                }

                _contacts.Add(new Contact(a, b, dx / d, dy / d, rsum - d));
            }
        }
    }

    private static void ResolveVelocity(in Contact c)
    {
        var a = c.A;
        var b = c.B;
        float nx = c.Nx, ny = c.Ny;
        float invMa = a.InvMass, invMb = b?.InvMass ?? 0f;
        float invIa = a.InvI, invIb = b?.InvI ?? 0f;
        if (invMa + invMb == 0f) return;

        // Lever arms from each center to the contact point (a circle touches along its own normal).
        float rax = a.R * nx, ray = a.R * ny;
        float rbx = b is null ? 0f : -b.R * nx, rby = b is null ? 0f : -b.R * ny;

        // Relative velocity AT the contact point: v + ω × r  (2D: ω × r = (−ω·r.y, ω·r.x)).
        float wa = a.AngularVel, wb = b?.AngularVel ?? 0f;
        float rvx = (b is null ? 0f : b.Vx - wb * rby) - (a.Vx - wa * ray);
        float rvy = (b is null ? 0f : b.Vy + wb * rbx) - (a.Vy + wa * rax);

        float relN = rvx * nx + rvy * ny;
        if (relN > 0f) return; // separating

        float e = relN < -RestitutionThresholdPx ? Restitution : 0f;

        float rnA = rax * ny - ray * nx;
        float rnB = rbx * ny - rby * nx;
        float kn = invMa + invMb + invIa * rnA * rnA + invIb * rnB * rnB;
        if (kn <= 0f) return;
        float jn = -(1f + e) * relN / kn;
        ApplyImpulse(a, b, rax, ray, rbx, rby, jn * nx, jn * ny);

        // Recompute the contact-point relative velocity, then Coulomb friction along the tangent.
        float wa2 = a.AngularVel, wb2 = b?.AngularVel ?? 0f;
        rvx = (b is null ? 0f : b.Vx - wb2 * rby) - (a.Vx - wa2 * ray);
        rvy = (b is null ? 0f : b.Vy + wb2 * rbx) - (a.Vy + wa2 * rax);
        float rvn = rvx * nx + rvy * ny;
        float tx = rvx - rvn * nx, ty = rvy - rvn * ny;
        float tlen = MathF.Sqrt(tx * tx + ty * ty);
        if (tlen < 1e-6f) return;
        tx /= tlen; ty /= tlen;
        float rtA = rax * ty - ray * tx;
        float rtB = rbx * ty - rby * tx;
        float kt = invMa + invMb + invIa * rtA * rtA + invIb * rtB * rtB;
        if (kt <= 0f) return;
        float jt = -(rvx * tx + rvy * ty) / kt;
        float max = Friction * jn;
        jt = MathF.Min(max, MathF.Max(-max, jt));
        ApplyImpulse(a, b, rax, ray, rbx, rby, jt * tx, jt * ty);
    }

    private static void ApplyImpulse(FruitBody a, FruitBody? b, float rax, float ray, float rbx, float rby, float jx, float jy)
    {
        a.Vx -= a.InvMass * jx;
        a.Vy -= a.InvMass * jy;
        a.AngularVel -= a.InvI * (rax * jy - ray * jx); // ω -= invI · cross(r, J)
        if (b is not null)
        {
            b.Vx += b.InvMass * jx;
            b.Vy += b.InvMass * jy;
            b.AngularVel += b.InvI * (rbx * jy - rby * jx);
        }
    }

    private static void CorrectPosition(in Contact c)
    {
        var a = c.A;
        var b = c.B;
        float invSum = a.InvMass + (b?.InvMass ?? 0f);
        if (invSum == 0f) return;
        float corr = MathF.Max(c.Pen - Slop, 0f) / invSum * CorrectionPercent;
        a.X -= a.InvMass * corr * c.Nx;
        a.Y -= a.InvMass * corr * c.Ny;
        if (b is not null)
        {
            b.X += b.InvMass * corr * c.Nx;
            b.Y += b.InvMass * corr * c.Ny;
        }
    }

    private int FlushMerges()
    {
        int points = 0;
        bool removed = false;
        foreach (var (a, b) in _mergeQueue)
        {
            if (a.Removed || b.Removed) continue;
            a.Removed = b.Removed = removed = true;

            int tier = a.Tier;
            float x = (a.X + b.X) * 0.5f;
            float y = (a.Y + b.Y) * 0.5f;
            int? resultTier = FruitCatalog.MergeResultTier(tier);
            if (resultTier is int rt) SpawnFruit(rt, x, y);
            // else: a top-tier pair vanishes.
            points += FruitCatalog.ByTier(tier).MergePoints;
        }
        _mergeQueue.Clear();
        if (removed) _bodies.RemoveAll(b => b.Removed);
        return points;
    }
}
