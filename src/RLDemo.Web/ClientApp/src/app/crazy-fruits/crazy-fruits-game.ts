// The animating host around the single-source engine (crazyfruits_solver.pg → PgCrazyFruits). The RULES all
// live in the engine; this class only drives the engine's stepwise move API (swapCells → clearStep →
// collapseColumns → finishMove) and records a timeline of animation steps for the renderer to play. The
// one-call applySwap path drains the same engine loop, so both are the same rules by construction.

import { PgCrazyFruits } from './crazyfruits_solver';

export const SIZE = PgCrazyFruits.Size;
export const CELLS = PgCrazyFruits.Cells;

/** A fruit sliding down a column: fromRow < 0 means it spawns above the board (refill). */
export interface FallMove {
  col: number;
  fromRow: number;
  toRow: number;
  fruit: number;
}

export type AnimStep =
  /** Two fruits glide into each other's cells (also used for the two legs of a revert). */
  | { kind: 'swap'; a: number; b: number; fruitA: number; fruitB: number; grid: number[]; duration: number }
  /** Matched/blasted cells scale/fade out with a floating score popup. `grid` is the board BEFORE the
   *  clear; `clearedBy` styles the effect per cell (1 match · 2 striped beam · 3 blast · 4 bomb zap);
   *  `created` sparkles new specials in as (cell, packedValue) pairs. */
  | { kind: 'pop'; cells: boolean[]; points: number; grid: number[]; clearedBy: number[]; created: number[]; duration: number }
  /** Survivors + refills slide down. `grid` is the board AFTER the collapse (the landing state). */
  | { kind: 'fall'; moves: FallMove[]; grid: number[]; duration: number }
  /** Deadlock: the re-dealt board fades in. */
  | { kind: 'reshuffle'; grid: number[]; duration: number };

const SWAP_MS = 150;
const POP_MS = 230;
const FALL_MS_PER_CELL = 55;
const FALL_MS_MAX = 320;
const RESHUFFLE_MS = 450;

export class CrazyFruitsGame {
  readonly board = new PgCrazyFruits();

  /** Pending animation steps for the current move; empty = idle (input accepted). */
  private queue: AnimStep[] = [];
  private elapsed = 0;

  /** Cell index of the tap-tap selection, or -1. */
  selected = -1;
  best = 0;
  /** Human play is 30-move rounds (SPECIALS PRD §3.8); true once the budget is spent and animations drained. */
  roundOver = false;

  constructor() {
    this.best = Number(localStorage.getItem('crazyfruits.best') ?? 0) || 0;
    this.newGame();
  }

  newGame(): void {
    this.board.reset((Date.now() % 2147483646) + 1);
    this.queue = [];
    this.elapsed = 0;
    this.selected = -1;
    this.roundOver = false;
  }

  get animating(): boolean {
    return this.queue.length > 0;
  }

  /** The step being played right now (null when idle) and its 0..1 progress. */
  get currentStep(): AnimStep | null {
    return this.queue[0] ?? null;
  }

  get progress(): number {
    const step = this.queue[0];
    return step ? Math.min(1, this.elapsed / step.duration) : 0;
  }

  update(dtMs: number): void {
    if (!this.queue.length) return;
    this.elapsed += dtMs;
    while (this.queue.length && this.elapsed >= this.queue[0].duration) {
      this.elapsed -= this.queue[0].duration;
      this.queue.shift();
    }
    if (!this.queue.length) this.elapsed = 0;
  }

  /** The swap action index for two orthogonally adjacent cells, or -1. */
  static actionFor(a: number, b: number): number {
    const [lo, hi] = a < b ? [a, b] : [b, a];
    const r = Math.floor(lo / SIZE);
    const c = lo % SIZE;
    if (hi === lo + 1 && c < SIZE - 1) return r * (SIZE - 1) + c;         // horizontal
    if (hi === lo + SIZE && r < SIZE - 1) return 56 + r * SIZE + c;       // vertical
    return -1;
  }

  /** Attempt a swap by its engine action index (the watch-AI path). */
  tryAction(action: number): boolean {
    return this.trySwap(this.board.cellA(action), this.board.cellB(action));
  }

  /**
   * Attempt the swap of two adjacent cells. A legal swap (match, bomb swap, or special+special combo) plays
   * swap → (pop → fall)* [→ reshuffle] and returns true; an illegal swap plays the glide-out-and-back
   * revert, costs nothing, and returns false. The engine's stepwise API does all the rules — including
   * specials creation, chained blasts and the wrapped double explosion — this class only records timelines.
   */
  trySwap(cellA: number, cellB: number): boolean {
    if (this.animating || this.roundOver) return false;
    const action = CrazyFruitsGame.actionFor(cellA, cellB);
    if (action < 0) return false;

    const board = this.board;
    const pre = [...board.grid];
    // Any attempted swap — legal or reverted — consumes the selection (standard match-3 feel).
    this.selected = -1;
    // cellB is the gesture's LAST-SELECTED cell (tap-tap: the second tap; drag: the dragged-to cell) —
    // combo blasts centre on it.
    if (!board.stageSwap(action, cellB)) {
      this.queue.push(
        { kind: 'swap', a: cellA, b: cellB, fruitA: pre[cellA], fruitB: pre[cellB], grid: pre, duration: SWAP_MS },
        { kind: 'swap', a: cellB, b: cellA, fruitA: pre[cellA], fruitB: pre[cellB], grid: pre, duration: SWAP_MS },
      );
      return false;
    }

    // stageSwap performed the swap and staged any combo; clearStep(0) will execute it.
    this.queue.push({ kind: 'swap', a: cellA, b: cellB, fruitA: pre[cellA], fruitB: pre[cellB], grid: pre, duration: SWAP_MS });

    let points = 0;
    for (let k = 0; ; k++) {
      const marked = new Array<boolean>(CELLS).fill(false);
      const preClear = [...board.grid];
      const stepPoints = board.clearStep(k, marked);
      if (stepPoints === 0) break;
      points += stepPoints;
      this.queue.push({
        kind: 'pop', cells: marked, points: stepPoints, grid: preClear,
        clearedBy: [...board.lastClearedBy], created: [...board.lastCreated],
        duration: POP_MS,
      });

      const preCollapse = [...board.grid]; // cleared cells are 0
      board.collapseColumns(true);
      const post = [...board.grid];
      const moves = CrazyFruitsGame.computeFalls(preCollapse, post);
      const maxDrop = moves.reduce((m, f) => Math.max(m, f.toRow - f.fromRow), 1);
      this.queue.push({
        kind: 'fall', moves, grid: post,
        duration: Math.min(FALL_MS_MAX, FALL_MS_PER_CELL * maxDrop + 80),
      });
    }

    const reshufflesBefore = board.reshuffles;
    board.finishMove(points);
    if (board.reshuffles !== reshufflesBefore)
      this.queue.push({ kind: 'reshuffle', grid: [...board.grid], duration: RESHUFFLE_MS });

    if (board.score > this.best) {
      this.best = board.score;
      localStorage.setItem('crazyfruits.best', String(this.best));
    }
    return true;
  }

  /** Called by the component once animations drain: closes the round at the 30-move budget (human mode). */
  checkRoundOver(movesPerRound: number): void {
    if (!this.animating && this.board.movesMade >= movesPerRound) this.roundOver = true;
  }

  /** Per-column fall moves between the post-clear (holes) and post-collapse boards — view geometry only. */
  private static computeFalls(preCollapse: number[], post: number[]): FallMove[] {
    const moves: FallMove[] = [];
    for (let c = 0; c < SIZE; c++) {
      let write = SIZE - 1;
      for (let r = SIZE - 1; r >= 0; r--) {
        const fruit = preCollapse[r * SIZE + c];
        if (fruit === 0) continue;
        if (write !== r) moves.push({ col: c, fromRow: r, toRow: write, fruit });
        write--;
      }
      // Refills land on rows 0..write, falling in from above the board in formation.
      const spawned = write + 1;
      for (let r = write; r >= 0; r--)
        moves.push({ col: c, fromRow: r - spawned, toRow: r, fruit: post[r * SIZE + c] });
    }
    return moves;
  }
}
