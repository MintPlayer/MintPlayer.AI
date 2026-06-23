import { byTier, mergeResultTier } from './fruit-cake-fruits';

/** A fruit in flight/at rest: center (px), tier, speed (px/s), age, and whether it was born from a merge. */
export interface FruitView {
  xPx: number;
  yPx: number;
  tier: number;
  speedPx: number;
  ageSeconds: number;
  mergeBorn: boolean;
}

/** Reported when a merge resolves. `resultTier` is null for a vanished top-tier pair. */
export interface MergeEvent {
  sourceTier: number;
  resultTier: number | null;
  xPx: number;
  yPx: number;
  points: number;
}

interface Body {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  tier: number;
  invMass: number;
  spawnTime: number;
  mergeBorn: boolean;
  pendingMerge: boolean;
  hasLanded: boolean;
  removed: boolean;
}

interface Contact {
  a: Body;
  b: Body | null; // null => a static wall
  nx: number; // unit normal, from A toward B (or toward the wall)
  ny: number;
  pen: number; // penetration depth (px)
}

/**
 * The fruit container as a physics simulation. The original game leaned on Aether.Physics2D;
 * the web port carries its own compact sequential-impulse solver instead — every collider here
 * is a circle and the walls are static, so a purpose-built solver is smaller and leaner than a
 * general engine while reproducing the qualities the game depends on: low-restitution settling,
 * stable stacks, and contact-driven merges.
 *
 * Same-tier merges use the original's deferred pattern — record touching pairs while resolving,
 * then add/remove bodies only after the solve — so a merge never mutates the set mid-iteration.
 *
 * Works entirely in pixels; mass is the disc area (uniform density), which keeps mass ratios
 * smooth and stacks stable.
 */
export class FruitWorld {
  static readonly ContainerWidthPx = 620;
  static readonly ContainerHeightPx = 850;
  static readonly DangerLineYPx = 150;

  private static readonly PixelsPerMeter = 64;
  private static readonly Gravity = 9.8 * FruitWorld.PixelsPerMeter; // px/s², +Y down
  private static readonly Restitution = 0.1; // very low bounce; fruit settles fast
  private static readonly Friction = 0.006;
  private static readonly RestitutionThresholdPx = 64; // below this approach speed, treat as inelastic (no jitter)

  private static readonly VelocityIterations = 8;
  private static readonly PositionIterations = 4;
  private static readonly Slop = 0.5; // allowed penetration before correcting
  private static readonly CorrectionPercent = 0.8;

  private static readonly LandThresholdPx = 130; // min impact speed to count as a "landing" thud
  private static readonly LandMaxPx = 900; // impact speed mapped to full volume

  private readonly bodies: Body[] = [];
  private readonly mergeQueue: Array<[Body, Body]> = [];
  private readonly landQueue: number[] = [];
  private time = 0;

  /** Raised once per resolved merge, after the body set is safely mutated. */
  onMerged: ((e: MergeEvent) => void) | null = null;
  /** Raised when a falling fruit first lands hard; argument is impact 0..1. */
  onLanded: ((impact: number) => void) | null = null;

  get fruitCount(): number {
    return this.bodies.length;
  }

  /** Spawn a fruit of `tier` centered at the given pixel position. */
  spawnFruit(tier: number, xPx: number, yPx: number, mergeBorn = false): void {
    const def = byTier(tier);
    const mass = Math.PI * def.radiusPx * def.radiusPx; // uniform density => area-proportional mass
    this.bodies.push({
      x: xPx,
      y: yPx,
      vx: 0,
      vy: 0,
      r: def.radiusPx,
      tier,
      invMass: 1 / mass,
      spawnTime: this.time,
      mergeBorn,
      pendingMerge: false,
      hasLanded: false,
      removed: false,
    });
  }

  /** Advance one fixed step, then resolve any merges that the step's contacts queued. */
  step(dt: number): void {
    this.time += dt;

    for (const b of this.bodies) b.vy += FruitWorld.Gravity * dt;

    // Build contacts once with side effects (queue merges, detect landings).
    const contacts = this.buildContacts(true);

    for (let it = 0; it < FruitWorld.VelocityIterations; it++)
      for (const c of contacts) this.resolveVelocity(c);

    for (const b of this.bodies) {
      b.x += b.vx * dt;
      b.y += b.vy * dt;
    }

    // Non-linear position correction: recompute penetration each pass and project bodies apart.
    for (let it = 0; it < FruitWorld.PositionIterations; it++) {
      const pcs = this.buildContacts(false);
      for (const c of pcs) this.correctPosition(c);
    }

    this.flushMerges();
    while (this.landQueue.length > 0) this.onLanded?.(this.landQueue.shift()!);
  }

  private buildContacts(detect: boolean): Contact[] {
    const contacts: Contact[] = [];
    const W = FruitWorld.ContainerWidthPx;
    const H = FruitWorld.ContainerHeightPx;

    for (const b of this.bodies) {
      // Walls: left (x=0), right (x=W), floor (y=H). No ceiling — fruit may leave over the rim.
      const left = b.r - b.x;
      if (left > 0) {
        contacts.push({ a: b, b: null, nx: -1, ny: 0, pen: left });
        if (detect) this.tryLand(b);
      }
      const right = b.x + b.r - W;
      if (right > 0) {
        contacts.push({ a: b, b: null, nx: 1, ny: 0, pen: right });
        if (detect) this.tryLand(b);
      }
      const floor = b.y + b.r - H;
      if (floor > 0) {
        contacts.push({ a: b, b: null, nx: 0, ny: 1, pen: floor });
        if (detect) this.tryLand(b);
      }
    }

    for (let i = 0; i < this.bodies.length; i++) {
      const a = this.bodies[i];
      for (let j = i + 1; j < this.bodies.length; j++) {
        const b = this.bodies[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const rsum = a.r + b.r;
        const d2 = dx * dx + dy * dy;
        if (d2 >= rsum * rsum) continue;
        const d = Math.sqrt(d2) || 0.0001;

        if (detect && a.tier === b.tier && !a.pendingMerge && !b.pendingMerge) {
          a.pendingMerge = b.pendingMerge = true;
          this.mergeQueue.push([a, b]); // the merge pop covers this contact — no physical shove
          continue;
        }

        contacts.push({ a, b, nx: dx / d, ny: dy / d, pen: rsum - d });
        if (detect) {
          this.tryLand(a);
          this.tryLand(b);
        }
      }
    }
    return contacts;
  }

  // A fruit's first hard impact → a one-time landing thud.
  private tryLand(body: Body): void {
    if (body.hasLanded) return;
    const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy);
    if (speed < FruitWorld.LandThresholdPx) return;
    body.hasLanded = true;
    this.landQueue.push(Math.min(1, Math.max(0, speed / FruitWorld.LandMaxPx)));
  }

  private resolveVelocity(c: Contact): void {
    const { a, b, nx, ny } = c;
    const invB = b ? b.invMass : 0;
    const invSum = a.invMass + invB;
    if (invSum === 0) return;

    const bvx = b ? b.vx : 0;
    const bvy = b ? b.vy : 0;
    let relN = (bvx - a.vx) * nx + (bvy - a.vy) * ny;
    if (relN > 0) return; // separating

    const e = relN < -FruitWorld.RestitutionThresholdPx ? FruitWorld.Restitution : 0;
    const jn = (-(1 + e) * relN) / invSum;
    const inx = jn * nx;
    const iny = jn * ny;
    a.vx -= a.invMass * inx;
    a.vy -= a.invMass * iny;
    if (b) {
      b.vx += b.invMass * inx;
      b.vy += b.invMass * iny;
    }

    // Friction along the tangent, Coulomb-clamped to the normal impulse.
    const rvx = (b ? b.vx : 0) - a.vx;
    const rvy = (b ? b.vy : 0) - a.vy;
    const rvn = rvx * nx + rvy * ny;
    let tx = rvx - rvn * nx;
    let ty = rvy - rvn * ny;
    const tlen = Math.sqrt(tx * tx + ty * ty);
    if (tlen < 1e-4) return;
    tx /= tlen;
    ty /= tlen;
    let jt = (-(rvx * tx + rvy * ty)) / invSum;
    const max = FruitWorld.Friction * jn;
    jt = Math.min(max, Math.max(-max, jt));
    const ftx = jt * tx;
    const fty = jt * ty;
    a.vx -= a.invMass * ftx;
    a.vy -= a.invMass * fty;
    if (b) {
      b.vx += b.invMass * ftx;
      b.vy += b.invMass * fty;
    }
  }

  private correctPosition(c: Contact): void {
    const { a, b, nx, ny } = c;
    const invB = b ? b.invMass : 0;
    const invSum = a.invMass + invB;
    if (invSum === 0) return;
    const corr = (Math.max(c.pen - FruitWorld.Slop, 0) / invSum) * FruitWorld.CorrectionPercent;
    a.x -= a.invMass * corr * nx;
    a.y -= a.invMass * corr * ny;
    if (b) {
      b.x += b.invMass * corr * nx;
      b.y += b.invMass * corr * ny;
    }
  }

  private flushMerges(): void {
    let removed = false;
    while (this.mergeQueue.length > 0) {
      const [a, b] = this.mergeQueue.shift()!;
      if (a.removed || b.removed) continue;
      a.removed = b.removed = removed = true;

      const tier = a.tier;
      const xPx = (a.x + b.x) * 0.5;
      const yPx = (a.y + b.y) * 0.5;
      const resultTier = mergeResultTier(tier);
      if (resultTier !== null) this.spawnFruit(resultTier, xPx, yPx, true);
      // else: a top-tier pair vanishes.

      this.onMerged?.({ sourceTier: tier, resultTier, xPx, yPx, points: byTier(tier).mergePoints });
    }
    if (removed) {
      for (let i = this.bodies.length - 1; i >= 0; i--) if (this.bodies[i].removed) this.bodies.splice(i, 1);
    }
  }

  /** Snapshot of every fruit for rendering. */
  get fruits(): FruitView[] {
    return this.bodies.map(b => ({
      xPx: b.x,
      yPx: b.y,
      tier: b.tier,
      speedPx: Math.sqrt(b.vx * b.vx + b.vy * b.vy),
      ageSeconds: this.time - b.spawnTime,
      mergeBorn: b.mergeBorn,
    }));
  }
}
