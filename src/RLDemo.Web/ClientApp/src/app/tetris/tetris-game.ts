// Animating host around the single-source engine (tetris_solver.pg → PgTetris). Two drive modes over the
// SAME micro path (TETRIS_PRD.md §3.10, owner amendment 2026-08-26):
//  • human: an NES-authentic fixed-timestep input machine (PLAN M55) — one logic tick per NES frame
//    (60.0988 Hz accumulator inside the rAF loop, NEVER the OS/browser key auto-repeat): frame-exact
//    DAS 16/10/6 with wall charge, hypertap latching, Down-blocks-horizontal, 3-then-2 soft drop, and
//    gravity folded into the same tick (max 1 row/frame). The machine itself lives in tetris-das.ts.
//  • watch: the AI picks a macro placement, then a PILOT plays it like a human would — the piece spawns
//    centered, visibly rotates and taps sideways to the target column (one input per ~90 ms), gravity
//    runs at the real level speed, and it hard-drops once aligned. The result is whatever the real micro
//    moves achieve: at kill-screen gravity even the AI's "fingers" can be outrun, authentically.

import { PgTetris } from './tetris_solver';
import { NES_FRAME_MS, NesInput, TECHNIQUE_FRAMES, type Technique } from './tetris-das';

export const W = 10;
export const H = 20;

const FRAME_MS = NES_FRAME_MS; // one NES frame (60.0988 Hz)
const FLASH_MS = 220;          // line-clear highlight
// M62.3 — the watch-mode pilot's cadence now comes from the technique dial rather than this constant.
// It used to be a flat 90 ms (≈11.1 Hz ≈ 5.4 frames), which was fiction: it sat almost exactly at
// hypertapping speed, so the "normal" AI was quietly hypertapping while being presented as ordinary play.

/** Watch-mode pilot: the placement the AI chose, played through the micro path. */
interface Pilot {
  /** M62.3: DAS only — whether this piece has already paid its 16-frame auto-shift charge. */
  charged: boolean;
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

  /** The NES input machine (human mode). The component reports key EDGES to it (press/release —
   * OS auto-repeat filtered out); all repeat timing is the machine's own, one tick per NES frame. */
  readonly input = new NesInput();

  private pilot: Pilot | null = null;
  private pilotAcc = 0;
  private gravityAcc = 0;
  private frameAcc = 0;

  private readonly dasHost = {
    shift: (dir: -1 | 1) => this.board.microShift(dir),
    dropStep: () => this.board.microDropStep(),
    gravityFrames: () => this.board.gravityFrames(this.board.level),
    gravityRowsPerStep: () => this.board.gravityRowsPerStep(this.board.level),
  };

  constructor() {
    this.newGame();
  }

  /**
   * M62.3 — the input technique. It governs BOTH the watch-mode pilot's tap cadence AND, through
   * `setTapRate`, the engine's own reachability budget, which the evaluator already consults
   * (`maxTapHeight` gates the tetris-ready reward and prices unreachable stacks). So the dial does not
   * merely slow the AI's hands down — it changes which placements the AI considers worth wanting.
   */
  technique: Technique = 'das';

  /** Push the current technique into the engine's reachability budget. Safe to call mid-game. */
  applyTechnique(): void {
    this.board.setTapRate(TECHNIQUE_FRAMES[this.technique]);
  }

  /** ms between the pilot's simulated presses, derived from the technique. */
  private get pilotInputMs(): number {
    return TECHNIQUE_FRAMES[this.technique] * FRAME_MS;
  }

  /** M62.2: NES start level (CTWC picks one per match). `reset` clears it, so newGame reapplies it. */
  startLevel = 0;
  /** M62.2: post-29 variant — 0 authentic, 1 CTM "39 halt", 2 CTWC 2xks. Survives reset by design. */
  killscreenMode = 0;

  newGame(): void {
    const seed = (Date.now() % 2147483646) + 1;
    this.board.reset(seed, this.sevenBag, this.garbageEvery);
    // Order matters: reset() zeroes startLevel and level, so both settings are reapplied afterwards.
    this.board.setKillscreenMode(this.killscreenMode);
    this.board.setTapRate(TECHNIQUE_FRAMES[this.technique]);
    if (this.startLevel > 0) this.board.setStartLevel(this.startLevel);
    this.board.microSpawn();
    this.pilot = null;
    this.flashMs = 0;
    this.gravityAcc = 0;
    this.pilotAcc = 0;
    this.frameAcc = 0;
    this.input.clear();
    this.input.onSpawn();
  }

  get gameOver(): boolean {
    return this.board.gameOver;
  }

  /** True while the watch-mode pilot is playing a piece (the director waits). */
  get animating(): boolean {
    return this.pilot !== null || this.flashMs > 0;
  }

  // ── Human input (micro API) ────────────────────────────────────────────────────────────────────────────

  // Immediate one-shot moves: the POINTER path (absolute-position drag/tap) and nothing else — the
  // keyboard goes through the NES input machine instead.
  moveLeft(): void { if (!this.gameOver) this.board.microShift(-1); }
  moveRight(): void { if (!this.gameOver) this.board.microShift(1); }
  rotate(): void { if (!this.gameOver) this.board.microRotate(); }
  // M62.1: the B button. On NES, A rotates clockwise and B counter-clockwise.
  rotateCcw(): void { if (!this.gameOver) this.board.microRotateCcw(); }

  hardDrop(): void {
    if (this.gameOver) return;
    this.board.microHardDrop();
    this.afterLock();
  }

  /** Per-frame update. Human mode runs the NES machine on a fixed 60.0988 Hz accumulator (input +
   * gravity in one frame-exact tick); watch mode runs the pilot's simulated key presses over the
   * ms-based gravity it always had. */
  update(dtMs: number, human: boolean): void {
    if (this.flashMs > 0) this.flashMs = Math.max(0, this.flashMs - dtMs);
    if (this.gameOver) {
      this.pilot = null;
      return;
    }

    if (human) {
      this.frameAcc += dtMs;
      while (this.frameAcc >= FRAME_MS) {
        this.frameAcc -= FRAME_MS;
        if (this.gameOver || !this.board.activeLive) break;
        if (this.input.tick(this.dasHost)) this.afterLock(); // the piece locked this frame
      }
      return;
    }

    // The pilot "presses keys" between gravity steps: rotate first, then tap toward the column, then
    // hard-drop. Two consecutive blocked inputs (a wall of stack in the way) bail to an immediate drop.
    if (this.pilot && this.board.activeLive) {
      this.pilotAcc += dtMs;
      const pilotMs = this.pilotInputMs;
      while (this.pilotAcc >= pilotMs && this.pilot) {
        this.pilotAcc -= pilotMs;
        this.pilotStep();
      }
    }

    if (this.pilot && this.board.activeLive) {
      this.gravityAcc += dtMs;
      const gravityMs = this.board.gravityFrames(this.board.level) * FRAME_MS;
      const rows = this.board.gravityRowsPerStep(this.board.level); // M62.2: 2 under the 2xks variant
      let locked = false;
      while (this.gravityAcc >= gravityMs && !locked) {
        this.gravityAcc -= gravityMs;
        for (let i = 0; i < rows && !locked; i++) {
          if (this.board.microDropStep()) {
            this.pilot = null; // gravity locked the piece (possibly short of the target — authentic)
            this.afterLock();
            locked = true;
          }
        }
      }
    }
  }

  /** Watch mode: hand the AI's chosen placement to the pilot (the piece is already spawned centered). */
  pilotTo(action: number): void {
    if (!this.board.activeLive && !this.gameOver) this.board.microSpawn();
    if (this.gameOver) return;
    this.pilot = { rot: this.board.actionRot(action), x: this.board.actionCol(action), stuck: 0, charged: false };
    this.pilotAcc = 0;
  }

  private pilotStep(): void {
    const b = this.board;
    const p = this.pilot!;
    // M62.3 — DAS pays the 16-frame charge before its FIRST lateral shift of a piece (a fresh direction,
    // uncharged). Hypertapping and rolling pay nothing: every tap shifts immediately. Modelled by burning
    // one pilot slot and pushing the accumulator back by the remaining charge, so the cost is in frames
    // rather than in skipped inputs. Rotation is one-per-press in every technique and is not charged.
    if (this.technique === 'das' && !p.charged && b.activeRot === p.rot && b.activeX !== p.x) {
      p.charged = true;
      this.pilotAcc -= (16 - TECHNIQUE_FRAMES.das) * FRAME_MS;
      return;
    }
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
    // NES spawn bookkeeping: soft-drop disengages and the drop/gravity counters reset — but the DAS
    // charge is PRESERVED (holding a direction through the spawn auto-shifts the new piece at once).
    this.input.onSpawn();
    this.gravityAcc = 0;
    if (this.board.lastLinesCleared > 0) {
      this.flashMs = FLASH_MS;
      this.flashLines = this.board.lastLinesCleared;
    }
    // M62.2: the CTM "level 39 halt" variant ends the run the moment 39 is reached — the game does not
    // top out, it simply stops, which is how the lower divisions bound a match.
    if (this.board.killscreenHalted()) this.board.forceGameOver();
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
