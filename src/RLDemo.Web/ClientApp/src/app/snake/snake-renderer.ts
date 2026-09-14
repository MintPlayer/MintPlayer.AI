// Pure VIEW layer for the Snake board (M35). Draws the snake as a single-colour, uniform-width, round-cornered
// tube on a <canvas>, gliding between the coarse game ticks — with ZERO game logic here. It consumes exactly what
// the component already produces each tick: a head-first `body: number[]` of grid-cell indices, the `food` cell,
// and the `eaten` count. Nothing in snake-logic.ts / snake-director.ts / snake_solver.ts changes.
//
// How the tube is drawn (see docs/prd/SNAKE_RENDER_PRD.md):
//   • a polyline through the (interpolated) cell centres with ONLY the corners rounded (arcTo) → straight runs
//     stay perfectly straight, turns get a rounded elbow;
//   • one round-capped stroke of constant width, one colour — no taper, outline, or shading;
//   • requestAnimationFrame interpolation across each tick — the loop only READS the latest snapshot, so the game
//     keeps ticking on its own setInterval. On a discontinuity (new game / teleport) we snap instead of gliding.

// The cycle overlay (M60, see docs/prd/SNAKE_CYCLE_OVERLAY_PRD.md) adds two flat hairline strokes UNDER the
// tube, drawn only in the Hamiltonian mode:
//   • the whole safety cycle, closed (the wrap segment cycle[n-1] → cycle[0] is implicit in the data and must
//     be stroked explicitly, or the loop reads as an open path);
//   • the arc from the head forward to the food, overdrawn brighter. That arc is a CORRIDOR, not a route — the
//     policy shortcuts forward within it on most ticks — but it is a real invariant: shortcuts can never jump
//     past the food, so the snake provably stays inside it.
// Under the tube is correct rather than merely cheap: by the cycle invariant the body is a contiguous run
// BEHIND the head, so the head→food arc always lies on non-body cells and is never occluded, while the rest of
// the loop is progressively swallowed by the tube — which is exactly the part already covered.

interface Pt { x: number; y: number; }

interface Snapshot {
  prev: number[]; // head-first body at the start of the current tick
  next: number[]; // head-first body at the end of the current tick
  food: number;
  cycle: number[] | null; // the safety cycle in travel order; null in search/human mode or when hidden
  cycleEpoch: number;
  t0: number; // performance.now() when this tick began
}

const BODY_COLOR = '#4caf82';
const FOOD_COLOR = '#ff6b6b';

// Flat fills only, per the house rule in lunar-lockout-render.ts: no gradients, no shadows, no glow.
const CYCLE_REST_COLOR = '#3a4154';        // the `hair` token — the repo's established hairline-hint colour
const CYCLE_ROUTE_RGB = '255, 212, 121';   // #ffd479, already in snake.scss as .banner.training
const CYCLE_ROUTE_ALPHA = 0.45;            // resting alpha; solid gold across a ~140-cell arc out-shouts the tube
const CYCLE_STROKE_PX = 5;                 // owner's call: a hairline plan, not a second snake. Drawing units
                                           // are CSS px (M60.2), so this is exactly 5px at every viewport width.
const FLASH_MS = 300;                      // one-shot brightness decay when the cycle changes

const lerp = (a: Pt, b: Pt, t: number): Pt => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

export class SnakeTubeRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly canvas: HTMLCanvasElement;
  // Not readonly since M60.2: the board is responsive (`width: min(480px, 100%)` in snake.scss), so the
  // backing store follows the element's real CSS width — the same pattern every other responsive canvas in
  // this repo uses (tetris.ts:147-154, crazy-fruits.ts:106-113). Drawing units are therefore CSS pixels.
  private cell: number;
  private boardPx: number;

  private snap: Snapshot | null = null;
  private tickMs = 120;
  private running = false;
  private raf = 0;
  private resizeObserver: ResizeObserver | null = null;

  // Cycle overlay state (M60.3). The full-loop path is expensive (144 rounded corners) and changes only when
  // the cycle is rebuilt — but it is built in board coordinates, so the cache MUST also key on `cell`, or a
  // viewport change leaves the loop drawn at the old scale.
  private loopPath: Path2D | null = null;
  private loopKey = '';
  private flashT0 = 0;

  constructor(canvas: HTMLCanvasElement, private readonly size: number, fallbackPx: number) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    // clientWidth is 0 before layout; fall back to the design size so the first frame is never degenerate.
    this.boardPx = canvas.clientWidth || fallbackPx;
    this.cell = this.boardPx / size;
    this.syncSize();
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => { this.syncSize(); this.kick(); });
      this.resizeObserver.observe(canvas);
    }
  }

  /**
   * Match the backing store to the element's current CSS size. Cheap and idempotent — it early-outs unless the
   * device-pixel dimensions actually changed, so it is safe to call at the top of every draw.
   */
  private syncSize(): boolean {
    const cssW = this.canvas.clientWidth || this.boardPx;
    const dpr = window.devicePixelRatio || 1;
    const backing = Math.round(cssW * dpr);
    if (this.canvas.width === backing && this.canvas.height === backing) return false;
    this.canvas.width = backing;
    this.canvas.height = backing;
    // Assigning canvas.width RESETS the 2D context state, including any earlier ctx.scale(dpr, dpr). Re-apply
    // it explicitly — leaving it out renders the board at 1x on hi-DPI displays after the first resize only,
    // which is exactly the kind of bug that survives a desktop check and shows up on a phone.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.boardPx = cssW;
    this.cell = cssW / this.size;
    return true;
  }

  /** Start a fresh game's animation. `tickMs` is the game's tick period (so glide fills exactly one tick). */
  begin(tickMs: number): void {
    this.tickMs = tickMs;
    this.snap = null;
    this.running = true;
    this.clear();
  }

  /** Feed one tick's state. Snaps (no glide) across a discontinuity — a new game teleports the body. */
  push(body: number[], food: number, _eaten: number, cycle: number[] | null = null, cycleEpoch = 0): void {
    const prev = this.snap;
    const continuous =
      prev != null && prev.next.length > 0 && body.length > 1 &&
      this.adjacent(body[0], prev.next[0]) && body[1] === prev.next[0];
    // The cycle changed ⇒ flash the route arc once. This marks "the plan changed", NOT "it ate": a failed
    // rebuild is retried every tick, so in the late game a successful rebuild lands well after the food.
    if (cycle !== null && prev != null && cycleEpoch !== prev.cycleEpoch) this.flashT0 = performance.now();
    this.snap = {
      prev: continuous ? prev!.next : body, // discontinuity → prev == next ⇒ p is irrelevant, we draw it static
      next: body,
      food,
      cycle,
      cycleEpoch,
      t0: performance.now(),
    };
    this.running = true;
    this.kick();
  }

  /** Stop the animation and clear the board (call from the component's stop()). */
  stop(): void {
    this.running = false;
    if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
    this.snap = null;
    this.clear();
  }

  /** Release the resize observer. The renderer outlives individual games, so this is destroy-time only. */
  destroy(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
  }

  // --- internals ---------------------------------------------------------

  private kick(): void {
    if (this.running && !this.raf) this.raf = requestAnimationFrame(this.loop);
  }

  // rAF loop: interpolate the current tick and park itself once the tick has fully played out (p ≥ 1) AND any
  // rebuild flash has decayed. push() re-kicks it on the next tick, so a resting board costs no frames beyond
  // the bounded, self-terminating flash — there is no permanent animation loop.
  private loop = (): void => {
    this.raf = 0;
    if (!this.running || !this.snap) return;
    const now = performance.now();
    const p = Math.min(1, (now - this.snap.t0) / this.tickMs);
    const flashing = now - this.flashT0 < FLASH_MS;
    this.draw(this.buildPoints(this.snap.next, this.snap.prev, p), this.snap.food, this.snap, now);
    if (p < 1 || flashing) this.raf = requestAnimationFrame(this.loop);
  };

  private center(i: number): Pt {
    return { x: (i % this.size + 0.5) * this.cell, y: (Math.floor(i / this.size) + 0.5) * this.cell };
  }

  private adjacent(a: number, b: number): boolean {
    const ar = Math.floor(a / this.size), ac = a % this.size;
    const br = Math.floor(b / this.size), bc = b % this.size;
    return Math.abs(ar - br) + Math.abs(ac - bc) === 1;
  }

  /** Head→tail centreline points for fractional tick position p, built from two consecutive states. */
  private buildPoints(next: number[], prev: number[], p: number): Pt[] {
    const pts: Pt[] = [];
    // Head emerges from the old head cell toward the new one (prev[0] === next[1] on a normal move ⇒ continuous).
    pts.push(lerp(this.center(prev[0] ?? next[0]), this.center(next[0]), p));
    for (let i = 1; i < next.length; i++) pts.push(this.center(next[i]));
    // Tail recedes: only when a cell actually vacated (not on the growth tick, where next is one longer than prev).
    const growing = next.length > prev.length;
    if (!growing && prev.length > 1) {
      pts.push(lerp(this.center(prev[prev.length - 1]), this.center(next[next.length - 1]), p));
    }
    return pts;
  }

  private draw(pts: Pt[], food: number, snap: Snapshot | null = null, now = 0): void {
    const ctx = this.ctx;
    this.syncSize(); // cheap no-op unless the element actually changed size (mirrors tetris.ts:153)
    this.clear();
    if (snap?.cycle) this.drawCycle(snap.cycle, snap.cycleEpoch, pts.length > 0 ? snap.next[0] : -1, food, now);
    if (food >= 0) this.drawFood(this.center(food));
    if (pts.length === 0) return;

    const w = this.cell * 0.72;
    ctx.fillStyle = BODY_COLOR;

    if (pts.length === 1) {
      // Degenerate (length-1 body): a single rounded dot.
      ctx.beginPath(); ctx.arc(pts[0].x, pts[0].y, w / 2, 0, Math.PI * 2); ctx.fill();
      return;
    }

    // One uniform-width, round-capped stroke — straight runs stay straight, only the corners are rounded.
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = BODY_COLOR;
    ctx.lineWidth = w;
    ctx.beginPath();
    this.tubePath(ctx, pts);
    ctx.stroke();

    // Eyes on the (uniform-width) rounded head cap, oriented to the direction of travel.
    this.drawEyes(pts[0], { x: pts[0].x - pts[1].x, y: pts[0].y - pts[1].y }, w / 2);
  }

  private drawEyes(head: Pt, dir: Pt, r: number): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(head.x, head.y);
    ctx.rotate(Math.atan2(dir.y, dir.x) || 0); // +x now points where the snake is going
    for (const side of [-1, 1] as const) {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(r * 0.1, side * r * 0.42, r * 0.3, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#12202b';
      ctx.beginPath(); ctx.arc(r * 0.22, side * r * 0.42, r * 0.15, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  /**
   * The cycle overlay: the whole loop as a hairline, then the head→food arc overdrawn brighter. Never throws —
   * a head or food that is somehow off-cycle just skips the arc (it cannot happen under the full-coverage
   * invariant, but the overlay must not be able to break the board if it ever does).
   */
  private drawCycle(cycle: number[], epoch: number, head: number, food: number, now: number): void {
    if (cycle.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = CYCLE_STROKE_PX;

    // Layer 1 — the closed loop, cached. Keyed on `cell` as well as the epoch: the path is in board
    // coordinates, so a resize must invalidate it or it is drawn at the previous scale.
    const key = `${epoch}:${this.cell}`;
    if (this.loopPath === null || this.loopKey !== key) {
      const path = new Path2D();
      const pts = cycle.map(c => this.center(c));
      pts.push(pts[0], pts[1]); // close the loop: the wrap segment is implicit in the data, and arcTo needs a
                                // successor to round the final corner against
      this.tubePath(path, pts, this.cell * 0.22);
      this.loopPath = path;
      this.loopKey = key;
    }
    ctx.strokeStyle = CYCLE_REST_COLOR;
    ctx.stroke(this.loopPath);

    // Layer 2 — head → food along the cycle, rebuilt each tick (the head moves every tick).
    const hIdx = cycle.indexOf(head);
    const fIdx = cycle.indexOf(food);
    if (hIdx >= 0 && fIdx >= 0) {
      const arc: Pt[] = [];
      for (let k = 0; ; k++) {
        const i = (hIdx + k) % cycle.length;
        arc.push(this.center(cycle[i]));
        if (i === fIdx || k > cycle.length) break;
      }
      if (arc.length >= 2) {
        const age = now - this.flashT0;
        // One-shot ease-out from full brightness back to the resting alpha. Flat stroke, not a gradient.
        const flash = age < FLASH_MS ? (1 - age / FLASH_MS) ** 2 : 0;
        const alpha = CYCLE_ROUTE_ALPHA + (1 - CYCLE_ROUTE_ALPHA) * flash;
        ctx.strokeStyle = `rgba(${CYCLE_ROUTE_RGB}, ${alpha})`;
        const path = new Path2D();
        this.tubePath(path, arc, this.cell * 0.22);
        ctx.stroke(path);
      }
    }
    ctx.restore();
  }

  /** Emit a polyline through `pts` with only the corners rounded (arcTo). Collinear points ⇒ dead-straight. */
  // `CanvasPath` rather than the context type so the same corner-rounding serves both the live tube stroke and
  // the cached overlay Path2D.
  private tubePath(sink: CanvasPath, pts: Pt[], radius = this.cell * 0.5): void {
    const n = pts.length;
    const R = radius; // corner radius — how rounded a 90° elbow is
    sink.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < n - 1; i++) {
      // Clamp the radius to half of each adjacent segment so tight zig-zags (short interpolated head/tail spans)
      // never overshoot; arcTo draws a straight line when the three points are collinear.
      const r = Math.min(R, this.dist(pts[i - 1], pts[i]) / 2, this.dist(pts[i], pts[i + 1]) / 2);
      sink.arcTo(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, r);
    }
    sink.lineTo(pts[n - 1].x, pts[n - 1].y);
  }

  private dist(a: Pt, b: Pt): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  private drawFood(c: Pt): void {
    const ctx = this.ctx;
    ctx.fillStyle = FOOD_COLOR;
    ctx.beginPath();
    ctx.arc(c.x, c.y, this.cell * 0.3, 0, Math.PI * 2);
    ctx.fill();
  }

  private clear(): void {
    const ctx = this.ctx;
    ctx.fillStyle = '#1c2230';
    ctx.fillRect(0, 0, this.boardPx, this.boardPx);
    // Faint grid so the board still reads as a lattice without competing with the snake.
    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < this.size; i++) {
      const x = i * this.cell;
      ctx.moveTo(x, 0); ctx.lineTo(x, this.boardPx);
      ctx.moveTo(0, x); ctx.lineTo(this.boardPx, x);
    }
    ctx.stroke();
  }
}
