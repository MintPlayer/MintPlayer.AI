// Animating host around the single-source engine (tetris_solver.pg → PgTetris). Two drive modes over the
// SAME micro path (TETRIS_PRD.md §3.10, owner amendment 2026-08-26):
//  • human: an NES-authentic fixed-timestep input machine (PLAN M55) — one logic tick per NES frame
//    (60.0988 Hz accumulator inside the rAF loop, NEVER the OS/browser key auto-repeat): frame-exact
//    DAS 16/10/6 with wall charge, hypertap latching, Down-blocks-horizontal, 3-then-2 soft drop, and
//    gravity folded into the same tick (max 1 row/frame). The machine itself lives in tetris-das.ts.
//  • watch: the AI picks a macro placement AND the engine hands back the exact input timeline that
//    reaches it (M62.3b / PRD D9). The pilot REPLAYS that timeline on the same NES frame clock — it does
//    not re-derive a route, so the placement you watch is the placement the AI chose, by construction.
//    Tap cadence comes from the technique dial (DAS 10 Hz with its 16-frame charge, hypertapping 12 Hz,
//    rolling 20 Hz), so at kill-screen gravity the AI's "fingers" are outrun authentically rather than
//    by a fudge factor. Before M62.3b this was a flat 90 ms re-planner that silently substituted a
//    different placement when blocked — see the PRD's §7 corrections.

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
  /** M62.3b: the engine's input timeline as flat (frame, code) pairs — replayed, never re-derived. */
  timeline: number[];
  /** Read cursor into `timeline`, and the NES frame this piece is on. */
  ev: number;
  frame: number;
  rot: number;
  x: number;
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
  private frameAcc = 0;

  /**
   * M62.3b / G10. With reachability enforced the engine only offers placements it has simulated as
   * attainable, so a pilot that locks short of its target means the simulation and the micro path
   * disagree — a real bug. Logged rather than absorbed, which is precisely what the old `stuck >= 2`
   * hard-drop fallback never did.
   */
  reportDivergence = true;

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

  /** Push the current technique into the engine's input model. Safe to call mid-game. */
  applyTechnique(): void {
    // All three use the CHARGED model (PRD §4.1): DAS is held through ARE so it auto-shifts from frame 0
    // at its 6-frame repeat. The techniques differ by rate — 10 / 12 / 20 Hz — not by a startup penalty.
    const frames = TECHNIQUE_FRAMES[this.technique];
    this.board.setTapModel(frames, frames);
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
    // M62.4a: reachability is enforced in the browser, so the technique dial constrains what the AI can
    // actually DO rather than only how fast it looks. Safe for human play: enforcement is consulted by
    // placementReachable/hasLegalPlacement, which are reached only from the MACRO path (applyPlacement)
    // that the AI uses — a human plays through the micro API and is untouched.
    this.board.setReachEnforced(true);
    this.applyTechnique();
    if (this.startLevel > 0) this.board.setStartLevel(this.startLevel);
    this.board.microSpawn();
    this.pilot = null;
    this.flashMs = 0;
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

    // M62.3b — the pilot REPLAYS the engine's input timeline on the NES frame clock (PRD D9). It no
    // longer re-derives a route, so what you watch is exactly the placement the AI chose; if the piece
    // ever fails to arrive, that is a bug worth surfacing, not a fallback worth taking silently.
    if (this.pilot && this.board.activeLive) {
      this.frameAcc += dtMs;
      while (this.frameAcc >= FRAME_MS && this.pilot && this.board.activeLive) {
        this.frameAcc -= FRAME_MS;
        this.pilotFrameTick();
      }
      return;
    }

  }

  /**
   * One NES frame of watch mode: apply whatever inputs the engine's timeline scheduled for this frame,
   * then gravity. Both the timeline and the gravity clock come from the same engine that chose the
   * placement, so the piece arrives exactly where the AI said it would.
   */
  private pilotFrameTick(): void {
    const p = this.pilot!;
    const b = this.board;

    while (p.ev + 1 < p.timeline.length && p.timeline[p.ev] === p.frame) {
      switch (p.timeline[p.ev + 1]) {
        case 1: b.microShift(-1); break;
        case 2: b.microShift(1); break;
        case 3: b.microRotate(); break;
        case 4: b.microRotateCcw(); break;
      }
      p.ev += 2;
    }

    // Arrived: the remaining fall is pure gravity, so drop it and lock.
    if (b.activeRot === p.rot && b.activeX === p.x) {
      this.pilot = null;
      if (b.microHardDrop()) this.afterLock();
      return;
    }

    p.frame++;
    const g = b.gravityFrames(b.level);
    if (p.frame % g === 0) {
      const rows = b.gravityRowsPerStep(b.level); // M62.2: 2 under the 2xks variant
      for (let i = 0; i < rows; i++) {
        if (b.microDropStep()) {
          // Locked before arriving. Under an enforced reachability mask this should be unreachable —
          // the engine only offers placements it simulated as attainable — so it is reported rather
          // than absorbed. Without enforcement it is expected and harmless.
          if (p.timeline.length > 0 && this.reportDivergence) {
            console.error(`[tetris] pilot diverged: wanted rot ${p.rot} col ${p.x}, locked at rot ${b.activeRot} col ${b.activeX}`);
          }
          this.pilot = null;
          this.afterLock();
          return;
        }
      }
    }
  }

  /** Watch mode: hand the AI's chosen placement to the pilot (the piece is already spawned centered). */
  pilotTo(action: number): void {
    if (!this.board.activeLive && !this.gameOver) this.board.microSpawn();
    if (this.gameOver) return;
    // Ask the engine HOW to play this placement, not just where it ends up (PRD D9).
    this.pilot = {
      rot: this.board.actionRot(action),
      x: this.board.actionCol(action),
      timeline: this.board.reachTimelineFor(action),
      ev: 0,
      frame: 0,
    };
    this.frameAcc = 0;
  }


  private afterLock(): void {
    // NES spawn bookkeeping: soft-drop disengages and the drop/gravity counters reset — but the DAS
    // charge is PRESERVED (holding a direction through the spawn auto-shifts the new piece at once).
    this.input.onSpawn();
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
