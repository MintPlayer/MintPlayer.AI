// Thin host adapter over the SINGLE-SOURCE solver: the sequential-impulse circle physics now lives once in
// src/MintPlayer.AI.ReinforcementLearning.Environments/FruitCake/polyglot/fruitcake_solver.pg, transpiled to
// this project's ./fruitcake_solver.ts (committed) and to C# for training/serving. Edit the .pg, not the physics.
//
// The generated core (PgFruitCakeWorld) is pure physics. This adapter re-creates the host-only surface the game
// needs — merge/landing events for audio+effects and per-fruit age/merge-born for the pop animation — that the
// shared solver deliberately doesn't model: onMerged is exact (from core.lastMerges); mergeBorn/age come from a
// side-table keyed by the core body; onLanded is a host-side approximation (fires when a fast fruit's speed is
// damped by an impact). Rotation is ON here so fruit visibly roll (the AI trains rotation-off; merges match).
import { PgFruitCakeWorld, PgFruitBody } from './fruitcake_solver';

/** A fruit in flight/at rest: center (px), orientation (rad), tier, speed (px/s), age, merge-born. */
export interface FruitView {
  xPx: number;
  yPx: number;
  angle: number;
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

interface HostMeta {
  spawnTime: number;
  mergeBorn: boolean;
  hasLanded: boolean;
}

/**
 * The fruit container as a physics simulation — a thin facade over the transpiled {@link PgFruitCakeWorld}.
 * Preserves the API the game/render depend on so the single-source cutover needs no consumer changes.
 */
export class FruitWorld {
  static readonly ContainerWidthPx = 620;
  static readonly ContainerHeightPx = 850;
  static readonly DangerLineYPx = 150;

  private static readonly LandThresholdPx = 130; // min impact speed to count as a "landing" thud
  private static readonly LandMaxPx = 900; // impact speed mapped to full volume

  private readonly core = new PgFruitCakeWorld(true); // rotation on: fruit roll during human play
  private readonly meta = new Map<PgFruitBody, HostMeta>();
  private time = 0;

  /** Raised once per resolved merge, after the body set is safely mutated. */
  onMerged: ((e: MergeEvent) => void) | null = null;
  /** Raised when a falling fruit first lands hard; argument is impact 0..1. */
  onLanded: ((impact: number) => void) | null = null;

  get fruitCount(): number {
    return this.core.count;
  }

  /** Spawn a fruit of `tier` centered at the given pixel position. */
  spawnFruit(tier: number, xPx: number, yPx: number): void {
    const body = this.core.spawnFruit(tier, xPx, yPx);
    this.meta.set(body, { spawnTime: this.time, mergeBorn: false, hasLanded: false });
  }

  /** Advance one fixed step, then surface the merges/landings the step produced. */
  step(dt: number): void {
    this.time += dt;

    // Pre-step approach speed of each not-yet-landed fruit (for the landing thud, below).
    const preSpeed = new Map<PgFruitBody, number>();
    for (const b of this.core.bodies) {
      const m = this.meta.get(b);
      if (m && !m.hasLanded) preSpeed.set(b, Math.hypot(b.vx, b.vy));
    }

    this.core.step(dt);

    // Merges: exact, straight from the solver.
    if (this.onMerged) {
      for (const e of this.core.lastMerges)
        this.onMerged({ sourceTier: e.sourceTier, resultTier: e.resultTier, xPx: e.x, yPx: e.y, points: e.points });
    }

    // Reconcile the side-table: drop merged-away bodies; any body the solver created this step (a merge
    // product) is new here → merge-born, and doesn't thud.
    const live = new Set(this.core.bodies);
    for (const b of [...this.meta.keys()]) if (!live.has(b)) this.meta.delete(b);
    for (const b of this.core.bodies)
      if (!this.meta.has(b)) this.meta.set(b, { spawnTime: this.time, mergeBorn: true, hasLanded: true });

    // Landings (host-side approximation): a not-yet-landed fruit that came in fast and whose speed an impact
    // just damped. Free fall (gravity still accelerating it) has now >= pre, so it won't false-fire.
    if (this.onLanded) {
      for (const b of this.core.bodies) {
        const m = this.meta.get(b)!;
        if (m.hasLanded) continue;
        const pre = preSpeed.get(b);
        if (pre === undefined || pre < FruitWorld.LandThresholdPx) continue;
        if (Math.hypot(b.vx, b.vy) < pre) {
          m.hasLanded = true;
          this.onLanded(Math.min(1, Math.max(0, pre / FruitWorld.LandMaxPx)));
        }
      }
    }
  }

  /** Snapshot of every fruit for rendering. */
  get fruits(): FruitView[] {
    return this.core.bodies.map(b => {
      const m = this.meta.get(b);
      return {
        xPx: b.x,
        yPx: b.y,
        angle: b.angle,
        tier: b.tier,
        speedPx: Math.hypot(b.vx, b.vy),
        ageSeconds: this.time - (m ? m.spawnTime : this.time),
        mergeBorn: m ? m.mergeBorn : false,
      };
    });
  }
}
