// NES-authentic input state machine (PLAN M55): frame-exact DAS (delayed auto-shift), hypertapping,
// soft-drop cadence and gravity, per the NES Tetris disassembly (meatfighter.com/nintendotetrisai,
// tetris.wiki/DAS). Pure logic, no imports — unit-tested directly in node (tools/tetris_das_check.mjs).
//
// The machine ticks once per NES frame (60.0988 Hz) and consumes latched key edges, exactly like the
// NES NMI sampling its newly-pressed bitmask:
//  • fresh press: immediate 1-column shift, DAS := 0; a BLOCKED shift saturates DAS := 16 (wall charge);
//  • held: DAS++ to 16, shift, then DAS := 10 (⇒ 6-frame repeat ≈ 10 Hz); blocked ⇒ stay at 16;
//  • release / piece lock / spawn: DAS keeps its value — charge is only spent by a fresh press;
//  • hypertapping: each fresh press shifts immediately, capped at 1 shift per frame (~30 Hz ceiling);
//  • Down held: the horizontal routine is skipped entirely (DAS untouched) — you cannot DAS while
//    soft-dropping; a fresh Down press cannot engage while left/right is held;
//  • soft drop: first row 3 frames after engaging, then 1 row per 2 frames (1/2G); gravity and soft
//    drop never move the piece more than 1 row per frame (non-cumulative).
// Skipped NES quirks (deliberate): the −96-frame game-start Down lockout, pushdown scoring,
// left+right simultaneous (a D-pad impossibility — both-held resolves to no movement).

export const NES_FRAME_MS = 1000 / 60.0988;

const DAS_FULL = 16;   // frames to the first auto-shift (and the wall-charge saturation value)
const DAS_RESET = 10;  // counter value after an auto-shift ⇒ 16 − 10 = 6-frame repeat
const SOFT_FIRST = 3;  // frames from engaging Down to the first soft-dropped row
const SOFT_REPEAT = 2; // frames per row afterwards (1/2G)

/** What the machine drives each frame: shift/drop return false when blocked / true when LOCKED. */
export interface DasHost {
  shift(dir: -1 | 1): boolean;
  dropStep(): boolean; // one row down; true = the piece locked
  gravityFrames(): number;
}

export class NesInput {
  private das = 0;
  private heldL = false;
  private heldR = false;
  private heldDown = false;
  private latchL = false;
  private latchR = false;
  private latchDown = false;
  private downEngaged = false;
  private downCounter = 0;
  private downFirst = true;
  private gravCounter = 0;

  /** Key edges (component keydown with event.repeat filtered out / keyup). Latched between frames so a
   * tap shorter than one logic frame on a high-Hz display still registers as one press. */
  press(dir: -1 | 1): void {
    if (dir < 0) { this.heldL = true; this.latchL = true; }
    else { this.heldR = true; this.latchR = true; }
  }

  release(dir: -1 | 1): void {
    if (dir < 0) this.heldL = false;
    else this.heldR = false;
  }

  pressDown(): void { this.heldDown = true; this.latchDown = true; }
  releaseDown(): void { this.heldDown = false; }

  /** Drop all held/latched keys (pause, blur, pointer takeover). The DAS charge itself survives — on
   * the NES nothing but a fresh press rewrites it. */
  clear(): void {
    this.heldL = this.heldR = this.heldDown = false;
    this.latchL = this.latchR = this.latchDown = false;
    this.downEngaged = false;
  }

  /** Piece spawned (after a lock): soft-drop and gravity counters reset; the DAS charge is PRESERVED —
   * that carry-over is what makes wall charging a real technique. */
  onSpawn(): void {
    this.downEngaged = false;
    this.downCounter = 0;
    this.downFirst = true;
    this.gravCounter = 0;
  }

  /** One NES frame. Returns true when the active piece LOCKED this frame. */
  tick(host: DasHost): boolean {
    const newL = this.latchL, newR = this.latchR, newDown = this.latchDown;
    this.latchL = this.latchR = this.latchDown = false;

    // Down engages only on a fresh press with no horizontal held (NES rule); disengages on release.
    if (newDown && !this.heldL && !this.heldR && !this.downEngaged) {
      this.downEngaged = true;
      this.downCounter = 0;
      this.downFirst = true;
    }
    if (!this.heldDown) this.downEngaged = false;

    // Horizontal — skipped entirely while Down is held (the NES routine early-exits; DAS untouched).
    if (!this.heldDown) {
      let dir: -1 | 1 | 0 = 0;
      let fresh = false;
      if (newL !== newR) { dir = newL ? -1 : 1; fresh = true; } // a fresh press wins the frame
      else if (this.heldL !== this.heldR) dir = this.heldL ? -1 : 1; // both-held = no movement
      if (dir !== 0) {
        if (fresh) {
          this.das = 0;
          if (!host.shift(dir)) this.das = DAS_FULL; // wall charge
        } else {
          this.das++;
          if (this.das >= DAS_FULL) this.das = host.shift(dir) ? DAS_RESET : DAS_FULL;
        }
      }
    }

    // Soft drop cadence; at most one row per frame moves the piece (soft drop OR gravity, never both).
    let rowMoved = false;
    if (this.downEngaged) {
      this.downCounter++;
      if (this.downCounter >= (this.downFirst ? SOFT_FIRST : SOFT_REPEAT)) {
        this.downCounter = 0;
        this.downFirst = false;
        rowMoved = true;
        if (host.dropStep()) return true;
      }
    }

    this.gravCounter++;
    if (this.gravCounter >= host.gravityFrames()) {
      this.gravCounter = 0;
      if (!rowMoved && host.dropStep()) return true;
    }
    return false;
  }
}
