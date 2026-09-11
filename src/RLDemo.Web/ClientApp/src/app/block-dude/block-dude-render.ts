/**
 * Block Dude renderer — Canvas 2D, deliberately primitive (PRD §10.3).
 *
 * Six palette tokens, one meaning each: nothing, terrain, carryable, you, goal, hairline. Flat fills, no
 * gradients, no shadows, no glow, no texture. Terrain is drawn full-bleed so contiguous rock fuses into one
 * silhouette rather than three hundred outlined boxes; the carryable block is the only rounded, inset thing on
 * the board, so "this one is loose" is said three ways at once — by value, by gap, and by corner.
 *
 * The dude is a figure, not a character: five rectangles, no face. Which way he faces is the single most
 * important piece of state, so it is carried by a rectangular brow notch that still reads when a cell is 22px
 * wide — where an eye dot would be sub-pixel.
 *
 * Motion exists only where it communicates state. The loop parks as soon as nothing is moving.
 */

export interface BlockDudeSnapshot {
  width: number;
  height: number;
  /** Terrain per cell, row-major: 0 empty, 1 wall, 2 door. */
  tiles: number[];
  /** Cell indices holding a carryable block. */
  blocks: number[];
  px: number;
  py: number;
  facingRight: boolean;
  carrying: boolean;
  won: boolean;
  moves: number;
}

/** A block that has just been released and is falling to its resting cell. */
export interface FallingBlock {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
}

const PALETTE = {
  void_: '#14171f',
  terrain: '#2b3245',
  block: '#aab2c5',
  dude: '#6ea8fe',
  exit: '#4caf82',
  hair: '#3a4154',
  text: '#e6e8ee',
};

const CELL = 48;

const WALK_MS = 110;
const CLIMB_MS = 150;
const CLIMB_LIFT = 0.40;   // fraction of the climb spent going up before moving across
const VEIL_MS = 260;
const FALL_MS_PER_CELL = 70;
const FALL_MAX_MS = 260;

export class BlockDudeRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private snapshot: BlockDudeSnapshot | null = null;

  private fromX = 0;
  private fromY = 0;
  private moveAt = 0;
  private moveMs = 0;
  private climbing = false;

  private falling: FallingBlock | null = null;
  private fallAt = 0;
  private fallMs = 0;

  private frame = 0;
  private wonAt = 0;

  private resizeObserver: ResizeObserver | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D is unavailable.');
    this.ctx = ctx;

    // The backing store is sized from the element, so it must follow the element. These boards vary from 19x8 to
    // 29x19, so the stage's aspect changes per level too — without this, a resize leaves a stale buffer that the
    // browser scales up.
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.kick());
      this.resizeObserver.observe(canvas);
    }
  }

  private get animated(): boolean {
    return !matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /** The board aspect, so the page can size the stage without knowing the cell size. */
  static aspect(snapshot: BlockDudeSnapshot): string {
    return `${snapshot.width} / ${snapshot.height}`;
  }

  push(snapshot: BlockDudeSnapshot, move?: { fromX: number; fromY: number; climb: boolean }, fall?: FallingBlock): void {
    const wasWon = this.snapshot?.won ?? false;
    this.snapshot = snapshot;

    if (move && this.animated) {
      this.fromX = move.fromX;
      this.fromY = move.fromY;
      this.climbing = move.climb;
      this.moveAt = performance.now();
      // A fall is a gravity event, not a step: its duration grows with the drop, so a long fall reads as heavy.
      const dropped = Math.abs(snapshot.py - move.fromY);
      this.moveMs = move.climb
        ? CLIMB_MS
        : dropped > 1
          ? Math.min(FALL_MS_PER_CELL * Math.sqrt(dropped), FALL_MAX_MS)
          : WALK_MS;
    } else {
      this.moveMs = 0;
    }

    if (fall && this.animated) {
      this.falling = fall;
      this.fallAt = performance.now();
      const distance = Math.max(1, Math.abs(fall.toY - fall.fromY));
      this.fallMs = Math.min(FALL_MS_PER_CELL * Math.sqrt(distance), FALL_MAX_MS);
    } else {
      this.falling = null;
    }

    if (snapshot.won && !wasWon) this.wonAt = performance.now();
    this.kick();
  }

  /** Snaps to the given position with no animation — for a level change or a restart. */
  reset(snapshot: BlockDudeSnapshot): void {
    this.snapshot = snapshot;
    this.moveMs = 0;
    this.falling = null;
    this.wonAt = 0;
    this.kick();
  }

  private kick(): void {
    if (this.frame) return;
    const step = () => {
      this.frame = 0;
      if (this.draw()) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  dispose(): void {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
  }

  private draw(): boolean {
    const snapshot = this.snapshot;
    if (!snapshot) return false;

    const { ctx } = this;
    const now = performance.now();
    const logicalW = snapshot.width * CELL;
    const logicalH = snapshot.height * CELL;

    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const cssWidth = this.canvas.clientWidth || logicalW;
    const backingW = Math.round(cssWidth * dpr);
    const backingH = Math.round(cssWidth * (logicalH / logicalW) * dpr);
    if (this.canvas.width !== backingW || this.canvas.height !== backingH) {
      this.canvas.width = backingW;
      this.canvas.height = backingH;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(backingW / logicalW, backingW / logicalW);

    ctx.fillStyle = PALETTE.void_;
    ctx.fillRect(0, 0, logicalW, logicalH);

    this.drawLattice(snapshot, logicalW, logicalH);
    this.drawTerrain(snapshot);
    this.drawDoor(snapshot);

    let busy = false;

    // ── blocks, one of which may be in flight ──
    let fallingCell = -1;
    if (this.falling) {
      const t = this.fallMs <= 0 ? 1 : Math.min(1, (now - this.fallAt) / this.fallMs);
      fallingCell = this.falling.toY * snapshot.width + this.falling.toX;
      const eased = t * t;                       // constant acceleration: no bounce, no squash
      const x = this.falling.fromX + (this.falling.toX - this.falling.fromX) * t;
      const y = this.falling.fromY + (this.falling.toY - this.falling.fromY) * eased;
      this.drawBlock(x, y);
      if (t < 1) busy = true;
      else this.falling = null;
    }
    for (const cell of snapshot.blocks) {
      if (cell === fallingCell && this.falling) continue;
      this.drawBlock(cell % snapshot.width, Math.floor(cell / snapshot.width));
    }

    // ── the dude ──
    let x = snapshot.px;
    let y = snapshot.py;
    let walking = 0;
    if (this.moveMs > 0) {
      const t = Math.min(1, (now - this.moveAt) / this.moveMs);
      if (this.climbing) {
        // Up first, then across — never a diagonal, which would look like floating.
        const lift = Math.min(1, t / CLIMB_LIFT);
        const across = Math.max(0, (t - CLIMB_LIFT) / (1 - CLIMB_LIFT));
        y = this.fromY + (snapshot.py - this.fromY) * lift;
        x = this.fromX + (snapshot.px - this.fromX) * across;
      } else {
        x = this.fromX + (snapshot.px - this.fromX) * t;
        y = this.fromY + (snapshot.py - this.fromY) * (t * t);
        walking = t;
      }
      if (t < 1) busy = true;
      else this.moveMs = 0;
    }
    this.drawDude(x, y, snapshot.facingRight, snapshot.carrying, walking);
    if (snapshot.carrying) this.drawBlock(x, y - 1);

    if (snapshot.won) busy = this.drawWonVeil(snapshot, now, logicalW, logicalH) || busy;
    return busy;
  }

  private drawLattice(snapshot: BlockDudeSnapshot, logicalW: number, logicalH: number): void {
    // The grid is load-bearing here: you count cells to plan a climb, so it is drawn at full hairline strength
    // rather than the whisper other games use.
    const { ctx } = this;
    ctx.strokeStyle = PALETTE.hair;
    ctx.lineWidth = 1;
    for (let x = 0; x <= snapshot.width; x++) {
      const at = Math.round(x * CELL) + 0.5;
      ctx.beginPath();
      ctx.moveTo(at, 0);
      ctx.lineTo(at, logicalH);
      ctx.stroke();
    }
    for (let y = 0; y <= snapshot.height; y++) {
      const at = Math.round(y * CELL) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, at);
      ctx.lineTo(logicalW, at);
      ctx.stroke();
    }
  }

  /** Full-bleed fills, then one hairline along region boundaries — a landscape, not a wall of boxes. */
  private drawTerrain(snapshot: BlockDudeSnapshot): void {
    const { ctx } = this;
    ctx.fillStyle = PALETTE.terrain;
    for (let y = 0; y < snapshot.height; y++) {
      for (let x = 0; x < snapshot.width; x++) {
        if (snapshot.tiles[y * snapshot.width + x] === 1) ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
      }
    }

    ctx.strokeStyle = PALETTE.hair;
    ctx.lineWidth = Math.max(1, 0.03 * CELL);
    for (let y = 0; y < snapshot.height; y++) {
      for (let x = 0; x < snapshot.width; x++) {
        if (snapshot.tiles[y * snapshot.width + x] !== 1) continue;
        const solid = (cx: number, cy: number) =>
          cx >= 0 && cy >= 0 && cx < snapshot.width && cy < snapshot.height &&
          snapshot.tiles[cy * snapshot.width + cx] === 1;
        ctx.beginPath();
        if (!solid(x, y - 1)) { ctx.moveTo(x * CELL, y * CELL); ctx.lineTo((x + 1) * CELL, y * CELL); }
        if (!solid(x, y + 1)) { ctx.moveTo(x * CELL, (y + 1) * CELL); ctx.lineTo((x + 1) * CELL, (y + 1) * CELL); }
        if (!solid(x - 1, y)) { ctx.moveTo(x * CELL, y * CELL); ctx.lineTo(x * CELL, (y + 1) * CELL); }
        if (!solid(x + 1, y)) { ctx.moveTo((x + 1) * CELL, y * CELL); ctx.lineTo((x + 1) * CELL, (y + 1) * CELL); }
        ctx.stroke();
      }
    }
  }

  /** A hollow aperture, open at the bottom: a doorway, not a box. Shape-coded, so it never relies on hue. */
  private drawDoor(snapshot: BlockDudeSnapshot): void {
    const index = snapshot.tiles.indexOf(2);
    if (index < 0) return;
    const x = (index % snapshot.width) * CELL;
    const y = Math.floor(index / snapshot.width) * CELL;
    const { ctx } = this;

    ctx.strokeStyle = PALETTE.exit;
    ctx.lineWidth = 0.06 * CELL;
    ctx.beginPath();
    ctx.moveTo(x + 0.18 * CELL, y + CELL);
    ctx.lineTo(x + 0.18 * CELL, y + 0.14 * CELL);
    ctx.lineTo(x + 0.82 * CELL, y + 0.14 * CELL);
    ctx.lineTo(x + 0.82 * CELL, y + CELL);
    ctx.stroke();

    ctx.lineWidth = 0.05 * CELL;
    ctx.beginPath();
    ctx.moveTo(x + 0.36 * CELL, y + 0.62 * CELL);
    ctx.lineTo(x + 0.50 * CELL, y + 0.46 * CELL);
    ctx.lineTo(x + 0.64 * CELL, y + 0.62 * CELL);
    ctx.stroke();
  }

  /** The only rounded, inset element on the board. */
  private drawBlock(x: number, y: number): void {
    const { ctx } = this;
    ctx.fillStyle = PALETTE.block;
    ctx.beginPath();
    ctx.roundRect(x * CELL + 0.10 * CELL, y * CELL + 0.10 * CELL, 0.80 * CELL, 0.80 * CELL, 0.08 * CELL);
    ctx.fill();
  }

  private drawDude(x: number, y: number, facingRight: boolean, carrying: boolean, walking: number): void {
    const { ctx } = this;
    const f = facingRight ? 1 : -1;
    const cx = x * CELL + 0.5 * CELL;
    const base = y * CELL + 0.96 * CELL;

    ctx.fillStyle = PALETTE.dude;

    // Torso.
    ctx.fillRect(cx - 0.22 * CELL, base - 0.66 * CELL, 0.44 * CELL, 0.42 * CELL);
    // Head, shifted toward the facing side.
    ctx.fillRect(cx - 0.15 * CELL + f * 0.05 * CELL, base - 0.84 * CELL, 0.30 * CELL, 0.20 * CELL);
    // Brow notch — the facing cue, and the only one. A rectangular prow survives a 22px cell.
    ctx.fillRect(cx + f * 0.20 * CELL, base - 0.80 * CELL, f * 0.10 * CELL, 0.07 * CELL);

    // Legs: a two-frame swap mid-step, not a cycle.
    const stride = walking > 0.25 && walking < 0.75 ? f * 0.07 * CELL : 0;
    ctx.fillRect(cx - 0.21 * CELL + stride, base - 0.24 * CELL, 0.13 * CELL, 0.24 * CELL);
    ctx.fillRect(cx + 0.08 * CELL - stride, base - 0.24 * CELL, 0.13 * CELL, 0.24 * CELL);

    // Carrying: two straight arms, no elbow. The held block is drawn by the caller in the SAME token as the
    // ones on the floor, so "what I hold is the same stuff" is literal.
    if (carrying) {
      ctx.fillRect(cx - 0.25 * CELL, base - 0.80 * CELL, 0.09 * CELL, 0.16 * CELL);
      ctx.fillRect(cx + 0.16 * CELL, base - 0.80 * CELL, 0.09 * CELL, 0.16 * CELL);
    }
  }

  private drawWonVeil(snapshot: BlockDudeSnapshot, now: number, logicalW: number, logicalH: number): boolean {
    const { ctx } = this;
    const elapsed = this.animated ? now - this.wonAt : VEIL_MS;
    const progress = Math.min(1, elapsed / VEIL_MS);

    ctx.save();
    ctx.globalAlpha = 0.55 * progress;
    ctx.fillStyle = PALETTE.void_;
    ctx.fillRect(0, 0, logicalW, logicalH);
    ctx.restore();

    if (progress >= 1) {
      ctx.fillStyle = PALETTE.text;
      ctx.font = `bold ${Math.max(18, 0.035 * logicalW)}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const plural = snapshot.moves === 1 ? 'move' : 'moves';
      ctx.fillText(`Level complete · ${snapshot.moves} ${plural}`, logicalW / 2, logicalH / 2);
    }
    return progress < 1;
  }
}
