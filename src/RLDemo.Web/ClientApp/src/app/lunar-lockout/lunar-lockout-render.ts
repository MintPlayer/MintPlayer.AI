/**
 * Lunar Lockout board renderer — Canvas 2D, deliberately sober (PRD §10.4).
 *
 * Flat fills, no gradients, no shadows, no glow. Five palette tokens, every one of them carrying meaning:
 * nothing, hairline, helper robot, target robot, goal. Identity comes from POSITION, not hue — Rush Hour's
 * 16-colour vehicle ramp is explicitly not adopted here.
 *
 * The renderer owns no game state. It is handed a snapshot and a slide to animate, and parks itself when the
 * animation finishes so a resting board costs zero frames (the same structure as the snake tube renderer).
 */

/** Robot positions as cell indices (`row * 5 + col`); index 0 is the target robot. */
export interface LunarSnapshot {
  robots: number[];
  /** Index of the currently selected robot, or -1. */
  selected: number;
  /** Landing cells for the selected robot's legal slides, for the hint overlay. */
  hints: number[];
  solved: boolean;
  moves: number;
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

const LIGHT: LunarPalette = {
  void_: '#f4f6fa', hair: '#d3d9e4', robot: '#5b6378', target: '#2563eb', goal: '#2f8f66', text: '#14171f',
};

export const LUNAR_SIZE = 5;
const CELL = 96;
const LOGICAL = LUNAR_SIZE * CELL;

/** Linear, unfussy. A slide of n cells takes 55ms each, capped so a long slide never drags. */
const MS_PER_CELL = 55;
const MAX_SLIDE_MS = 220;
const VEIL_MS = 200;

export class LunarLockoutRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private palette: LunarPalette;
  private snapshot: LunarSnapshot | null = null;

  /** In-flight slide: the robot, where it came from, and when it started. */
  private moving = -1;
  private fromCell = -1;
  private startedAt = 0;
  private durationMs = 0;
  private frame = 0;
  private solvedAt = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D is unavailable.');
    this.ctx = ctx;
    this.palette = matchMedia('(prefers-color-scheme: light)').matches ? LIGHT : DARK;
  }

  /** Honours the viewer's reduced-motion preference by collapsing every duration to zero. */
  private get animated(): boolean {
    return !matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** Draws a new position. `slide` animates one robot from its previous cell. */
  push(snapshot: LunarSnapshot, slide?: { robot: number; from: number }): void {
    const wasSolved = this.snapshot?.solved ?? false;
    this.snapshot = snapshot;

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

    this.palette = matchMedia('(prefers-color-scheme: light)').matches ? LIGHT : DARK;
    const { ctx } = this;

    // Device-pixel-ratio backing store, logical coordinates on top.
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

    let busy = false;
    const now = performance.now();
    let progress = 1;
    if (this.moving >= 0) {
      progress = this.durationMs <= 0 ? 1 : Math.min(1, (now - this.startedAt) / this.durationMs);
      if (progress < 1) busy = true;
      else this.moving = -1;
    }

    if (snapshot.selected >= 0 && !busy) this.drawHints(snapshot);

    for (let i = 0; i < snapshot.robots.length; i++) {
      let cx = colOf(snapshot.robots[i]) * CELL + CELL / 2;
      let cy = rowOf(snapshot.robots[i]) * CELL + CELL / 2;
      if (i === this.moving) {
        const fx = colOf(this.fromCell) * CELL + CELL / 2;
        const fy = rowOf(this.fromCell) * CELL + CELL / 2;
        cx = fx + (cx - fx) * progress;      // linear: no easing theatrics, no overshoot
        cy = fy + (cy - fy) * progress;
      }
      this.drawRobot(cx, cy, i === 0, i === snapshot.selected && !busy);
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

  /** Unobtrusive: a dashed hairline to each landing cell, and a hollow ring where the robot would stop. */
  private drawHints(snapshot: LunarSnapshot): void {
    const { ctx } = this;
    const from = snapshot.robots[snapshot.selected];
    ctx.save();
    ctx.setLineDash([5, 7]);
    ctx.strokeStyle = this.palette.hair;
    ctx.lineWidth = 1;
    for (const landing of snapshot.hints) {
      ctx.beginPath();
      ctx.moveTo(colOf(from) * CELL + CELL / 2, rowOf(from) * CELL + CELL / 2);
      ctx.lineTo(colOf(landing) * CELL + CELL / 2, rowOf(landing) * CELL + CELL / 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.lineWidth = 0.03 * CELL;
    for (const landing of snapshot.hints) {
      ctx.beginPath();
      ctx.arc(colOf(landing) * CELL + CELL / 2, rowOf(landing) * CELL + CELL / 2, 0.34 * CELL, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawRobot(cx: number, cy: number, isTarget: boolean, isSelected: boolean): void {
    const { ctx } = this;
    ctx.fillStyle = isTarget ? this.palette.target : this.palette.robot;
    ctx.beginPath();
    ctx.arc(cx, cy, 0.34 * CELL, 0, Math.PI * 2);
    ctx.fill();

    // The target robot is SHAPE-distinct as well as hue-distinct: under deuteranopia its blue and the goal's
    // green can converge, so it carries an inner ring no helper has.
    if (isTarget) {
      ctx.strokeStyle = this.palette.void_;
      ctx.lineWidth = 0.05 * CELL;
      ctx.beginPath();
      ctx.arc(cx, cy, 0.2 * CELL, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (isSelected) {
      ctx.strokeStyle = this.palette.text;
      ctx.lineWidth = 0.04 * CELL;
      ctx.beginPath();
      ctx.arc(cx, cy, 0.44 * CELL, 0, Math.PI * 2);
      ctx.stroke();
    }
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
      ctx.fillText(`Solved · ${snapshot.moves} moves`, LOGICAL / 2, LOGICAL / 2);
    }
    return progress < 1;
  }
}

const rowOf = (cell: number) => Math.floor(cell / LUNAR_SIZE);
const colOf = (cell: number) => cell % LUNAR_SIZE;
const cellDistance = (a: number, b: number) => Math.abs(rowOf(a) - rowOf(b)) + Math.abs(colOf(a) - colOf(b));
