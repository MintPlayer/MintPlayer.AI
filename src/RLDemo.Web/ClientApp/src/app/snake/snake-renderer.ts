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

interface Pt { x: number; y: number; }

interface Snapshot {
  prev: number[]; // head-first body at the start of the current tick
  next: number[]; // head-first body at the end of the current tick
  food: number;
  t0: number; // performance.now() when this tick began
}

const BODY_COLOR = '#4caf82';
const FOOD_COLOR = '#ff6b6b';

const lerp = (a: Pt, b: Pt, t: number): Pt => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

export class SnakeTubeRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly cell: number;
  private readonly boardPx: number;

  private snap: Snapshot | null = null;
  private tickMs = 120;
  private running = false;
  private raf = 0;

  constructor(canvas: HTMLCanvasElement, private readonly size: number, boardPx: number) {
    this.boardPx = boardPx;
    // Scale the backing store by devicePixelRatio so the tube stays crisp on hi-DPI / fractional-scale displays.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(boardPx * dpr);
    canvas.height = Math.round(boardPx * dpr);
    canvas.style.width = `${boardPx}px`;
    canvas.style.height = `${boardPx}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.ctx.scale(dpr, dpr);
    this.cell = boardPx / size;
  }

  /** Start a fresh game's animation. `tickMs` is the game's tick period (so glide fills exactly one tick). */
  begin(tickMs: number): void {
    this.tickMs = tickMs;
    this.snap = null;
    this.running = true;
    this.clear();
  }

  /** Feed one tick's state. Snaps (no glide) across a discontinuity — a new game teleports the body. */
  push(body: number[], food: number, _eaten: number): void {
    const prev = this.snap;
    const continuous =
      prev != null && prev.next.length > 0 && body.length > 1 &&
      this.adjacent(body[0], prev.next[0]) && body[1] === prev.next[0];
    this.snap = {
      prev: continuous ? prev!.next : body, // discontinuity → prev == next ⇒ p is irrelevant, we draw it static
      next: body,
      food,
      t0: performance.now(),
    };
    this.running = true;
    this.kick();
  }

  /** Stop the animation and clear the board (call from the component's stop()/destroy). */
  stop(): void {
    this.running = false;
    if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
    this.snap = null;
    this.clear();
  }

  // --- internals ---------------------------------------------------------

  private kick(): void {
    if (this.running && !this.raf) this.raf = requestAnimationFrame(this.loop);
  }

  // rAF loop: interpolate the current tick and park itself once the tick has fully played out (p ≥ 1). push()
  // re-kicks it on the next tick, so a resting board costs zero frames.
  private loop = (): void => {
    this.raf = 0;
    if (!this.running || !this.snap) return;
    const p = Math.min(1, (performance.now() - this.snap.t0) / this.tickMs);
    this.draw(this.buildPoints(this.snap.next, this.snap.prev, p), this.snap.food);
    if (p < 1) this.raf = requestAnimationFrame(this.loop);
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

  private draw(pts: Pt[], food: number): void {
    const ctx = this.ctx;
    this.clear();
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

  /** Emit a polyline through `pts` with only the corners rounded (arcTo). Collinear points ⇒ dead-straight. */
  private tubePath(ctx: CanvasRenderingContext2D, pts: Pt[]): void {
    const n = pts.length;
    const R = this.cell * 0.5; // corner radius — how rounded a 90° elbow is
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < n - 1; i++) {
      // Clamp the radius to half of each adjacent segment so tight zig-zags (short interpolated head/tail spans)
      // never overshoot; arcTo draws a straight line when the three points are collinear.
      const r = Math.min(R, this.dist(pts[i - 1], pts[i]) / 2, this.dist(pts[i], pts[i + 1]) / 2);
      ctx.arcTo(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, r);
    }
    ctx.lineTo(pts[n - 1].x, pts[n - 1].y);
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
