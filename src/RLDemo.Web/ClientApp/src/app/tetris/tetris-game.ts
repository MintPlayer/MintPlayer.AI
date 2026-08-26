// Animating host around the single-source engine (tetris_solver.pg → PgTetris). Two drive modes over the
// SAME micro path (TETRIS_PRD.md §3.10, owner amendment 2026-08-26):
//  • human: shift/rotate/soft/hard drop under the NES gravity timer;
//  • watch: the AI picks a macro placement, then a PILOT plays it like a human would — the piece spawns
//    centered, visibly rotates and taps sideways to the target column (one input per ~90 ms), gravity
//    runs at the real level speed, and it hard-drops once aligned. The result is whatever the real micro
//    moves achieve: at kill-screen gravity even the AI's "fingers" can be outrun, authentically.

import { PgTetris } from './tetris_solver';

export const W = 10;
export const H = 20;

const FRAME_MS = 1000 / 60;   // NES gravity is specified in frames per row at 60 fps
const SOFT_DROP_MS = 55;      // gravity while the down key is held
const FLASH_MS = 220;         // line-clear highlight
const PILOT_INPUT_MS = 90;    // watch-mode cadence between the AI's simulated key presses

/** Watch-mode pilot: the placement the AI chose, played through the micro path. */
interface Pilot {
  rot: number;
  x: number;
  stuck: number; // consecutive no-progress inputs (blocked rotate/shift) — bail to hard drop at 2
}

export class TetrisGame {
  readonly board = new PgTetris();

  sevenBag = true;       // web default (PRD §1); watch tiers may run uniform for benchmark honesty
  garbageEvery = 0;      // 0 = off; the rising-garbage mode inserts a gapped row every N placements

  /** Lines/score flash after a clearing lock (render reads these). */
  flashMs = 0;
  flashLines = 0;

  private pilot: Pilot | null = null;
  private pilotAcc = 0;
  private gravityAcc = 0;
  private softDrop = false;

  constructor() {
    this.newGame();
  }

  newGame(): void {
    const seed = (Date.now() % 2147483646) + 1;
    this.board.reset(seed, this.sevenBag, this.garbageEvery);
    this.board.microSpawn();
    this.pilot = null;
    this.flashMs = 0;
    this.gravityAcc = 0;
    this.pilotAcc = 0;
    this.softDrop = false;
  }

  get gameOver(): boolean {
    return this.board.gameOver;
  }

  /** True while the watch-mode pilot is playing a piece (the director waits). */
  get animating(): boolean {
    return this.pilot !== null || this.flashMs > 0;
  }

  // ── Human input (micro API) ────────────────────────────────────────────────────────────────────────────

  moveLeft(): void { if (!this.gameOver) this.board.microShift(-1); }
  moveRight(): void { if (!this.gameOver) this.board.microShift(1); }
  rotate(): void { if (!this.gameOver) this.board.microRotate(); }
  setSoftDrop(on: boolean): void {
    // Engaging the fast interval must not SPEND the slow-gravity time already accumulated — without
    // this reset, up to a full gravity period (800 ms at level 0) converts into ~11 instant rows.
    if (on && !this.softDrop) this.gravityAcc = 0;
    this.softDrop = on;
  }

  hardDrop(): void {
    if (this.gameOver) return;
    this.board.microHardDrop();
    this.afterLock();
  }

  /** Per-frame update: NES gravity (both modes — the piece falls at the level's real speed), the
   * watch-mode pilot's simulated key presses, and the flash decay. */
  update(dtMs: number, human: boolean): void {
    if (this.flashMs > 0) this.flashMs = Math.max(0, this.flashMs - dtMs);
    if (this.gameOver) {
      this.pilot = null;
      return;
    }

    // The pilot "presses keys" between gravity steps: rotate first, then tap toward the column, then
    // hard-drop. Two consecutive blocked inputs (a wall of stack in the way) bail to an immediate drop.
    if (!human && this.pilot && this.board.activeLive) {
      this.pilotAcc += dtMs;
      while (this.pilotAcc >= PILOT_INPUT_MS && this.pilot) {
        this.pilotAcc -= PILOT_INPUT_MS;
        this.pilotStep();
      }
    }

    if ((human || this.pilot) && this.board.activeLive) {
      this.gravityAcc += dtMs;
      const gravityMs = this.board.gravityFrames(this.board.level) * FRAME_MS;
      const interval = human && this.softDrop ? Math.min(SOFT_DROP_MS, gravityMs) : gravityMs;
      while (this.gravityAcc >= interval) {
        this.gravityAcc -= interval;
        if (this.board.microDropStep()) {
          this.pilot = null; // gravity locked the piece (possibly short of the target — authentic)
          this.afterLock();
          break;
        }
      }
    }
  }

  /** Watch mode: hand the AI's chosen placement to the pilot (the piece is already spawned centered). */
  pilotTo(action: number): void {
    if (!this.board.activeLive && !this.gameOver) this.board.microSpawn();
    if (this.gameOver) return;
    this.pilot = { rot: this.board.actionRot(action), x: this.board.actionCol(action), stuck: 0 };
    this.pilotAcc = 0;
  }

  private pilotStep(): void {
    const b = this.board;
    const p = this.pilot!;
    let acted: boolean;
    if (b.activeRot !== p.rot) acted = b.microRotate();
    else if (b.activeX < p.x) acted = b.microShift(1);
    else if (b.activeX > p.x) acted = b.microShift(-1);
    else {
      this.pilot = null;
      if (b.microHardDrop()) this.afterLock();
      return;
    }
    p.stuck = acted ? 0 : p.stuck + 1;
    if (p.stuck >= 2) {
      // The route is blocked (stack in the way) — drop where it stands, like a player giving up the slide.
      this.pilot = null;
      if (b.microHardDrop()) this.afterLock();
    }
  }

  private afterLock(): void {
    // NES behavior: a lock cancels the soft drop — the next piece falls at gravity until the key is
    // pressed AGAIN (the component ignores auto-repeat), so holding ↓ never slams the next piece.
    this.softDrop = false;
    this.gravityAcc = 0;
    if (this.board.lastLinesCleared > 0) {
      this.flashMs = FLASH_MS;
      this.flashLines = this.board.lastLinesCleared;
    }
  }

  /** Ghost landing row for the human piece (render draws the outline). Scans from the piece's CURRENT
   * row, not the top — a piece slid under an overhang still gets its true landing. */
  ghostY(): number {
    const b = this.board;
    if (!b.activeLive) return -1;
    let y = b.activeY;
    while (b.fitsAt(b.current, b.activeRot, b.activeX, y + 1)) y++;
    return y;
  }

  /** Placements until the next garbage row (render shows the countdown); -1 when garbage is off. */
  garbageIn(): number {
    return this.garbageEvery > 0 ? this.garbageEvery - this.board.garbageCounter : -1;
  }
}
