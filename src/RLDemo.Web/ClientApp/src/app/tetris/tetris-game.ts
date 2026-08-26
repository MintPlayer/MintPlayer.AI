// Animating host around the single-source engine (tetris_solver.pg → PgTetris). Two drive modes over the
// SAME engine (TETRIS_PRD.md §3.10):
//  • human: the engine's micro API (shift/rotate/soft/hard drop) under a client-side gravity timer;
//  • watch: the AI picks a macro placement, the host animates a cosmetic fall, then applies it —
//    the animation replays what applyPlacement will do, it never invents rules.

import { PgTetris } from './tetris_solver';

export const W = 10;
export const H = 20;

const GRAVITY_MS = 650;       // human gravity step
const SOFT_DROP_MS = 45;      // gravity while the down key is held
const FLASH_MS = 220;         // line-clear highlight
const WATCH_FALL_MS = 160;    // watch-mode cosmetic drop time

/** Watch-mode drop animation: the piece glides from the top to its landing row, then locks. */
export interface DropAnim {
  action: number;
  piece: number;
  rot: number;
  x: number;
  yTo: number;
  t: number; // 0..1
}

export class TetrisGame {
  readonly board = new PgTetris();

  sevenBag = true;       // web default (PRD §1); watch tiers may run uniform for benchmark honesty
  garbageEvery = 0;      // 0 = off; the rising-garbage mode inserts a gapped row every N placements

  /** Lines/score flash after a clearing lock (render reads these). */
  flashMs = 0;
  flashLines = 0;

  /** Watch-mode animation in flight (board not yet mutated while set). */
  anim: DropAnim | null = null;

  private gravityAcc = 0;
  private softDrop = false;

  constructor() {
    this.newGame();
  }

  newGame(): void {
    const seed = (Date.now() % 2147483646) + 1;
    this.board.reset(seed, this.sevenBag, this.garbageEvery);
    this.board.microSpawn();
    this.anim = null;
    this.flashMs = 0;
    this.gravityAcc = 0;
    this.softDrop = false;
  }

  get gameOver(): boolean {
    return this.board.gameOver;
  }

  /** True while a watch-mode drop is animating (the director waits). */
  get animating(): boolean {
    return this.anim !== null || this.flashMs > 0;
  }

  // ── Human input (micro API) ────────────────────────────────────────────────────────────────────────────

  moveLeft(): void { if (!this.gameOver) this.board.microShift(-1); }
  moveRight(): void { if (!this.gameOver) this.board.microShift(1); }
  rotate(): void { if (!this.gameOver) this.board.microRotate(); }
  setSoftDrop(on: boolean): void { this.softDrop = on; }

  hardDrop(): void {
    if (this.gameOver) return;
    this.board.microHardDrop();
    this.afterLock();
  }

  /** Per-frame update: human gravity + flash decay + watch animation progress. */
  update(dtMs: number, human: boolean): void {
    if (this.flashMs > 0) this.flashMs = Math.max(0, this.flashMs - dtMs);
    if (human && !this.gameOver) {
      this.gravityAcc += dtMs;
      const interval = this.softDrop ? SOFT_DROP_MS : GRAVITY_MS;
      while (this.gravityAcc >= interval) {
        this.gravityAcc -= interval;
        if (this.board.microDropStep()) {
          this.afterLock();
          break;
        }
      }
    }
    if (this.anim) {
      this.anim.t += dtMs / WATCH_FALL_MS;
      if (this.anim.t >= 1) {
        const action = this.anim.action;
        this.anim = null;
        this.board.applyPlacement(action);
        this.afterLock();
      }
    }
  }

  /** Watch mode: start the cosmetic fall for a chosen placement (the director's move). */
  startDrop(action: number): void {
    const rot = this.board.actionRot(action);
    const x = this.board.actionCol(action);
    const yTo = this.board.dropY(this.board.current, rot, x);
    if (yTo < 0) {
      // Shouldn't happen (tiers return legal actions); apply directly as insurance.
      this.board.applyPlacement(action);
      return;
    }
    this.anim = { action, piece: this.board.current, rot, x, yTo, t: 0 };
  }

  private afterLock(): void {
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
