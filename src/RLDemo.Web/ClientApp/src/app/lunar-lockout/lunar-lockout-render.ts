/**
 * Lunar Lockout board renderer — Canvas 2D, deliberately sober (PRD §10.4, §12.3).
 *
 * Flat fills, no gradients, no shadows, no glow. Five palette tokens, each carrying meaning: nothing, hairline,
 * helper robot, target robot, goal. Identity comes from POSITION, not hue — Rush Hour's 16-colour vehicle ramp is
 * explicitly not adopted.
 *
 * Robots are rockets that point where they are aimed. The rest pose is 45 degrees; aiming rotates to an axis;
 * an ILLEGAL aim does not rotate at all, so a rocket never looks launchable in a direction it cannot fly.
 *
 * TWO ENTRY POINTS, on purpose:
 *   - `push(snapshot, slide)` — a board CHANGE. Immutable snapshot, as before.
 *   - `hover(state)`          — transient AIM state. Mutates three fields and kicks the loop. It must never go
 *                               through `push`, which clones the robot list and would allocate a snapshot per
 *                               pointermove (up to 120/sec) for a board that is otherwise free.
 *
 * The loop parks whenever nothing is animating, so a resting board costs zero frames. Every animation here has a
 * hard end time for exactly that reason — no springs, no decay integrators.
 */

/** Robot positions as cell indices (`row * 5 + col`); index 0 is the target robot. */
export interface LunarSnapshot {
  robots: number[];
  /** Robot selected by KEYBOARD, or -1. Draws the outer ring and the four-direction hints. */
  keyboardSelected: number;
  /** Landing cells for the keyboard selection's legal slides. Empty when aiming with a pointer. */
  keyboardHints: number[];
  solved: boolean;
  moves: number;
}

/** Transient aim state: which robot is engaged, where it is pointing, and whether that direction can fire. */
export interface LunarHover {
  robot: number;
  /** 0=up, 1=right, 2=down, 3=left, or -1 for none (rest pose). */
  direction: number;
  legal: boolean;
  /** Landing cell for a legal aim, else -1. */
  landing: number;
}

interface LunarPalette {
  void_: string;
  hair: string;
  robot: string;
  target: string;
  goal: string;
  text: string;
}

const DARK: LunarPalette = {
  void_: '#14171f', hair: '#3a4154', robot: '#aab2c5', target: '#6ea8fe', goal: '#4caf82', text: '#e6e8ee',
};

/**
 * Kept for when the app gains a theme, but NOT selected from `prefers-color-scheme`: the playground is dark-only
 * today — no `data-theme`, no toggle, every page hard-codes its palette. Following the OS painted a light board
 * onto a dark page for anyone whose system was set to light.
 */
export const LUNAR_LIGHT: LunarPalette = {
  void_: '#f4f6fa', hair: '#d3d9e4', robot: '#5b6378', target: '#2563eb', goal: '#2f8f66', text: '#14171f',
};

export const LUNAR_SIZE = 5;
const CELL = 96;
const LOGICAL = LUNAR_SIZE * CELL;

/** Linear, unfussy. A slide of n cells takes 55ms each, capped so a long slide never drags. */
const MS_PER_CELL = 55;
const MAX_SLIDE_MS = 220;
const VEIL_MS = 200;

/**
 * Rotation is front-loaded easing, deliberately unlike the linear slide: a slide is a RESULT, a rotation is a
 * RESPONSE TO THE HAND and must feel immediate. 140ms sits under the ~180ms where lag becomes perceptible and
 * above the ~100ms where motion reads as a jump.
 */
const ROTATE_MS = 140;
const REST_ANGLE = -Math.PI / 4;   // nose up-right
const NUDGE_MS = 120;

/** Direction code (0=up,1=right,2=down,3=left) to the angle whose nose points that way. */
const DIRECTION_ANGLE = [0, Math.PI / 2, Math.PI, -Math.PI / 2];

export class LunarLockoutRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private palette: LunarPalette = DARK;
  private snapshot: LunarSnapshot | null = null;
  private hovered: LunarHover | null = null;

  /** Per-robot rotation, tweened independently: several rockets can be returning to rest while one aims. */
  private angle: number[] = [];
  private angleFrom: number[] = [];
  private angleTo: number[] = [];
  private angleAt: number[] = [];

  private moving = -1;
  private fromCell = -1;
  private startedAt = 0;
  private durationMs = 0;
  private frame = 0;
  private solvedAt = 0;
  private nudgeRobot = -1;
  private nudgeDirection = 0;
  private nudgeAt = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D is unavailable.');
    this.ctx = ctx;
  }

  /** Honours the viewer's reduced-motion preference by collapsing every duration to zero. */
  private get animated(): boolean {
    return !matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** Draws a new position. `slide` animates one robot from its previous cell. */
  push(snapshot: LunarSnapshot, slide?: { robot: number; from: number }): void {
    const wasSolved = this.snapshot?.solved ?? false;
    this.snapshot = snapshot;

    // Angles survive a board change: robots keep their index, so a rocket mid-rotation is not reset by a move.
    while (this.angle.length < snapshot.robots.length) {
      this.angle.push(REST_ANGLE);
      this.angleFrom.push(REST_ANGLE);
      this.angleTo.push(REST_ANGLE);
      this.angleAt.push(0);
    }

    if (slide && this.animated) {
      const distance = cellDistance(slide.from, snapshot.robots[slide.robot]);
      this.moving = slide.robot;
      this.fromCell = slide.from;
      this.startedAt = performance.now();
      this.durationMs = Math.min(distance * MS_PER_CELL, MAX_SLIDE_MS);
    } else {
      this.moving = -1;
    }

    if (snapshot.solved && !wasSolved) this.solvedAt = performance.now();
    this.kick();
  }

  /** Resets every rocket to rest with no animation — for a level change or a restart. */
  resetAngles(): void {
    for (let i = 0; i < this.angle.length; i++) {
      this.angle[i] = REST_ANGLE;
      this.angleFrom[i] = REST_ANGLE;
      this.angleTo[i] = REST_ANGLE;
      this.angleAt[i] = 0;
    }
    this.hovered = null;
    this.kick();
  }

  /**
   * Transient aim state. Cheap by design: no snapshot, no allocation beyond the caller's small object.
   * An illegal aim leaves the rocket at rest — refusing to point is the primary signal that it cannot fire.
   */
  hover(state: LunarHover | null): void {
    const previous = this.hovered;
    this.hovered = state;

    const wanted = (robot: number) =>
      state && state.robot === robot && state.direction >= 0 && state.legal
        ? DIRECTION_ANGLE[state.direction]
        : REST_ANGLE;

    // Re-target only what changed, so a pointermove that does not alter the aim starts no tween.
    const touched = new Set<number>();
    if (state) touched.add(state.robot);
    if (previous) touched.add(previous.robot);
    for (const robot of touched) {
      if (robot < 0 || robot >= this.angle.length) continue;
      this.retarget(robot, wanted(robot));
    }
    this.kick();
  }

  /** A refused launch: a short nudge along the attempted axis, no rotation and no colour change. */
  refuse(robot: number, direction: number): void {
    if (!this.animated) return;         // the status line carries the message; motion is pure decoration here
    this.nudgeRobot = robot;
    this.nudgeDirection = direction;
    this.nudgeAt = performance.now();
    this.kick();
  }

  private retarget(robot: number, target: number): void {
    if (Math.abs(normalise(this.angleTo[robot] - target)) < 1e-6) return;
    if (!this.animated) {
      this.angle[robot] = target;
      this.angleFrom[robot] = target;
      this.angleTo[robot] = target;
      this.angleAt[robot] = 0;
      return;
    }
    this.angleFrom[robot] = this.angle[robot];
    this.angleTo[robot] = target;
    this.angleAt[robot] = performance.now();
  }

  private kick(): void {
    if (this.frame) return;
    const step = () => {
      this.frame = 0;
      const busy = this.draw();
      if (busy) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  dispose(): void {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
  }

  /** Returns true while something is still moving, so the loop knows to schedule another frame. */
  private draw(): boolean {
    const snapshot = this.snapshot;
    if (!snapshot) return false;

    const { ctx } = this;
    const now = performance.now();

    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const cssSize = this.canvas.clientWidth || LOGICAL;
    if (this.canvas.width !== Math.round(cssSize * dpr)) {
      this.canvas.width = Math.round(cssSize * dpr);
      this.canvas.height = Math.round(cssSize * dpr);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale((cssSize * dpr) / LOGICAL, (cssSize * dpr) / LOGICAL);

    ctx.fillStyle = this.palette.void_;
    ctx.fillRect(0, 0, LOGICAL, LOGICAL);

    this.drawLattice();
    this.drawGoal();

    // ── advance the rotation tweens ────────────────────────────────────────────────────────────────────────
    let busy = false;
    for (let i = 0; i < this.angle.length; i++) {
      if (!this.angleAt[i]) continue;
      const t = Math.min(1, (now - this.angleAt[i]) / ROTATE_MS);
      const eased = 1 - (1 - t) * (1 - t);                 // easeOutQuad
      this.angle[i] = this.angleFrom[i] + normalise(this.angleTo[i] - this.angleFrom[i]) * eased;
      if (t < 1) busy = true;
      else {
        this.angle[i] = this.angleTo[i];
        this.angleAt[i] = 0;
      }
    }

    let slideProgress = 1;
    if (this.moving >= 0) {
      slideProgress = this.durationMs <= 0 ? 1 : Math.min(1, (now - this.startedAt) / this.durationMs);
      if (slideProgress < 1) busy = true;
      else this.moving = -1;
    }

    // Hints are hidden only while a robot is BETWEEN cells — not merely because a frame is scheduled, or every
    // rotation would blink them off.
    if (this.moving < 0) busy = this.drawHints(snapshot) || busy;

    for (let i = 0; i < snapshot.robots.length; i++) {
      let cx = colOf(snapshot.robots[i]) * CELL + CELL / 2;
      let cy = rowOf(snapshot.robots[i]) * CELL + CELL / 2;
      if (i === this.moving) {
        const fx = colOf(this.fromCell) * CELL + CELL / 2;
        const fy = rowOf(this.fromCell) * CELL + CELL / 2;
        cx = fx + (cx - fx) * slideProgress;              // linear: no easing theatrics, no overshoot
        cy = fy + (cy - fy) * slideProgress;
      }

      if (i === this.nudgeRobot) {
        const t = (now - this.nudgeAt) / NUDGE_MS;
        if (t >= 1) this.nudgeRobot = -1;
        else {
          const swing = Math.sin(t * Math.PI) * 0.06 * CELL;
          if (this.nudgeDirection === 0) cy -= swing;
          else if (this.nudgeDirection === 1) cx += swing;
          else if (this.nudgeDirection === 2) cy += swing;
          else cx -= swing;
          busy = true;
        }
      }

      const engaged = this.hovered?.robot === i || snapshot.keyboardSelected === i;
      this.drawRocket(cx, cy, this.angle[i] ?? REST_ANGLE, i === 0, engaged);
    }

    if (snapshot.solved) busy = this.drawSolvedVeil(snapshot, now) || busy;
    return busy;
  }

  private drawLattice(): void {
    const { ctx } = this;
    ctx.strokeStyle = this.palette.hair;
    ctx.lineWidth = 1;
    for (let i = 0; i <= LUNAR_SIZE; i++) {
      const at = Math.round(i * CELL) + 0.5;
      ctx.beginPath();
      ctx.moveTo(at, 0);
      ctx.lineTo(at, LOGICAL);
      ctx.moveTo(0, at);
      ctx.lineTo(LOGICAL, at);
      ctx.stroke();
    }
  }

  /** The centre goal: hollow and crossed, never filled, so a robot sitting on it stays fully visible. */
  private drawGoal(): void {
    const { ctx } = this;
    const x = 2 * CELL;
    const y = 2 * CELL;
    ctx.strokeStyle = this.palette.goal;
    ctx.lineWidth = 0.05 * CELL;
    ctx.strokeRect(x + 0.16 * CELL, y + 0.16 * CELL, CELL - 0.32 * CELL, CELL - 0.32 * CELL);

    ctx.lineWidth = 0.04 * CELL;
    const cx = x + CELL / 2;
    const cy = y + CELL / 2;
    ctx.beginPath();
    ctx.moveTo(cx - 0.11 * CELL, cy);
    ctx.lineTo(cx + 0.11 * CELL, cy);
    ctx.moveTo(cx, cy - 0.11 * CELL);
    ctx.lineTo(cx, cy + 0.11 * CELL);
    ctx.stroke();
  }

  /**
   * Pointer aim draws ONE hint — the aimed direction. With a rotating rocket, four simultaneous dashes plus a
   * turning glyph are two competing signals. Keyboard selection keeps all four, because without a pointer there
   * is no aimed direction.
   */
  private drawHints(snapshot: LunarSnapshot): boolean {
    const { ctx } = this;
    const aim = this.hovered;

    if (aim && aim.direction >= 0 && !aim.legal) {
      this.drawRefusalStub(snapshot.robots[aim.robot], aim.direction);
      return false;
    }

    const from = aim && aim.landing >= 0 ? snapshot.robots[aim.robot] : snapshot.robots[snapshot.keyboardSelected];
    const landings = aim && aim.landing >= 0 ? [aim.landing] : snapshot.keyboardHints;
    if (from === undefined || landings.length === 0) return false;

    ctx.save();
    ctx.setLineDash([5, 7]);
    ctx.strokeStyle = this.palette.hair;
    ctx.lineWidth = 1;
    for (const landing of landings) {
      ctx.beginPath();
      ctx.moveTo(colOf(from) * CELL + CELL / 2, rowOf(from) * CELL + CELL / 2);
      ctx.lineTo(colOf(landing) * CELL + CELL / 2, rowOf(landing) * CELL + CELL / 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 0.03 * CELL;
    for (const landing of landings) {
      ctx.beginPath();
      ctx.arc(colOf(landing) * CELL + CELL / 2, rowOf(landing) * CELL + CELL / 2, 0.34 * CELL, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
    return false;
  }

  /** "The road ends here" — a stub capped by a bar. No red, no new token; the rocket also refuses to rotate. */
  private drawRefusalStub(cell: number, direction: number): void {
    const { ctx } = this;
    const cx = colOf(cell) * CELL + CELL / 2;
    const cy = rowOf(cell) * CELL + CELL / 2;
    const dx = direction === 1 ? 1 : direction === 3 ? -1 : 0;
    const dy = direction === 2 ? 1 : direction === 0 ? -1 : 0;

    ctx.save();
    ctx.strokeStyle = this.palette.hair;
    ctx.setLineDash([5, 7]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx + dx * 0.30 * CELL, cy + dy * 0.30 * CELL);
    ctx.lineTo(cx + dx * 0.46 * CELL, cy + dy * 0.46 * CELL);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.lineWidth = 0.03 * CELL;
    ctx.beginPath();
    const bx = cx + dx * 0.46 * CELL;
    const by = cy + dy * 0.46 * CELL;
    ctx.moveTo(bx - dy * 0.08 * CELL, by - dx * 0.08 * CELL);
    ctx.lineTo(bx + dy * 0.08 * CELL, by + dx * 0.08 * CELL);
    ctx.stroke();
    ctx.restore();
  }

  /**
   * A flat dart with a notched tail, authored pointing UP in local space. Fill only — the silhouette does the
   * work, so it survives being 14px wide on a phone.
   */
  private drawRocket(cx: number, cy: number, angle: number, isTarget: boolean, engaged: boolean): void {
    const { ctx } = this;

    if (engaged) {
      // Drawn in WORLD space so the ring does not spin with the hull.
      ctx.strokeStyle = this.palette.text;
      ctx.lineWidth = 0.04 * CELL;
      ctx.beginPath();
      ctx.arc(cx, cy, 0.46 * CELL, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    ctx.fillStyle = isTarget ? this.palette.target : this.palette.robot;
    ctx.beginPath();
    ctx.moveTo(0, -0.40 * CELL);            // nose
    ctx.lineTo(0.185 * CELL, 0.10 * CELL);  // shoulder
    ctx.lineTo(0.145 * CELL, 0.31 * CELL);  // tail
    ctx.lineTo(0, 0.17 * CELL);             // notch — makes the rear unmistakable at small sizes
    ctx.lineTo(-0.145 * CELL, 0.31 * CELL);
    ctx.lineTo(-0.185 * CELL, 0.10 * CELL);
    ctx.closePath();
    ctx.fill();

    // The target keeps TWO shape cues, not just hue: with a rotating glyph a hue-only distinction gets worse
    // under deuteranopia, not better.
    if (isTarget) {
      ctx.strokeStyle = this.palette.void_;
      ctx.lineWidth = 0.05 * CELL;
      ctx.beginPath();
      ctx.arc(0, -0.06 * CELL, 0.10 * CELL, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(-0.145 * CELL, 0.31 * CELL);
      ctx.lineTo(0.145 * CELL, 0.31 * CELL);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawSolvedVeil(snapshot: LunarSnapshot, now: number): boolean {
    const { ctx } = this;
    const elapsed = this.animated ? now - this.solvedAt : VEIL_MS;
    const progress = Math.min(1, elapsed / VEIL_MS);

    ctx.save();
    ctx.globalAlpha = 0.55 * progress;
    ctx.fillStyle = this.palette.void_;
    ctx.fillRect(0, 0, LOGICAL, LOGICAL);
    ctx.restore();

    if (progress >= 1) {
      ctx.fillStyle = this.palette.text;
      ctx.font = `bold ${0.06 * LOGICAL}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const plural = snapshot.moves === 1 ? 'move' : 'moves';
      ctx.fillText(`Solved · ${snapshot.moves} ${plural}`, LOGICAL / 2, LOGICAL / 2);
    }
    return progress < 1;
  }
}

const rowOf = (cell: number) => Math.floor(cell / LUNAR_SIZE);
const colOf = (cell: number) => cell % LUNAR_SIZE;
const cellDistance = (a: number, b: number) => Math.abs(rowOf(a) - rowOf(b)) + Math.abs(colOf(a) - colOf(b));

/** Shortest arc: rotating from rest (−45°) to "left" must travel −45°, not +315°. */
function normalise(radians: number): number {
  let value = radians;
  while (value <= -Math.PI) value += Math.PI * 2;
  while (value > Math.PI) value -= Math.PI * 2;
  return value;
}
