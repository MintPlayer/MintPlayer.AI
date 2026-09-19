import { describe, expect, it } from 'vitest';

import {
  DAS_CHARGE_FRAMES,
  NES_FRAME_MS,
  NesInput,
  TECHNIQUE_FRAMES,
  techniqueHz,
  type DasHost,
  type Technique,
} from './tetris-das';

// M63.x — the NES input state machine (PLAN M55).
//
// This is frame arithmetic, and frame arithmetic is exactly what nobody notices going wrong: a DAS
// charge that is one frame off still *feels* like Tetris, but it silently changes what the AI's
// reachability budget can promise and what a human can feed into column 9 at kill-screen gravity.
//
// The machine is driven here by an explicit frame loop with a recording host — never a real timer,
// never `setTimeout`. Every number below is a frame count, and the ones that matter are asserted
// against the module's own exported budget constants so the two cannot drift apart.

class RecordingHost implements DasHost {
  /** Directions of the shifts that actually happened (a blocked shift is not recorded here). */
  readonly shifts: (-1 | 1)[] = [];
  /** Every shift ATTEMPT, including blocked ones. */
  attempts = 0;
  drops = 0;
  blocked = false;
  lockAfterDrops = Number.POSITIVE_INFINITY;

  constructor(
    private readonly frames = 1_000_000, // effectively "gravity never fires" unless a test asks for it
    private readonly rowsPerStep = 1,
  ) {}

  shift(dir: -1 | 1): boolean {
    this.attempts++;
    if (this.blocked) return false;
    this.shifts.push(dir);
    return true;
  }

  dropStep(): boolean {
    this.drops++;
    return this.drops >= this.lockAfterDrops;
  }

  gravityFrames(): number { return this.frames; }
  gravityRowsPerStep(): number { return this.rowsPerStep; }
}

/** Ticks `frames` frames, returning the 1-based frame numbers on which a shift landed. */
function shiftFrames(machine: NesInput, host: RecordingHost, frames: number): number[] {
  const landed: number[] = [];
  for (let frame = 1; frame <= frames; frame++) {
    const before = host.shifts.length;
    machine.tick(host);
    if (host.shifts.length > before) landed.push(frame);
  }
  return landed;
}

/** Ticks `frames` frames, returning the 1-based frame numbers on which a row was dropped. */
function dropFrames(machine: NesInput, host: RecordingHost, frames: number): number[] {
  const landed: number[] = [];
  for (let frame = 1; frame <= frames; frame++) {
    const before = host.drops;
    machine.tick(host);
    if (host.drops > before) landed.push(frame);
  }
  return landed;
}

const gaps = (frames: number[]): number[] => frames.slice(1).map((f, i) => f - frames[i]);

describe('auto-shift timing', () => {
  it('shifts immediately on the press, then charges before auto-repeating at a fixed cadence', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);

    const landed = shiftFrames(machine, host, 30);

    // The press itself shifts on the very first frame it is sampled...
    expect(landed[0]).toBe(1);
    // ...then the ROM makes you wait out the full charge before the second shift...
    expect(landed[1] - landed[0]).toBe(DAS_CHARGE_FRAMES);
    // ...and from there it repeats at the steady-state rate the AI budgets with.
    expect(gaps(landed.slice(1))).toEqual(gaps(landed.slice(1)).map(() => TECHNIQUE_FRAMES.das));
    expect(landed.length).toBeGreaterThanOrEqual(4);
    expect(host.shifts.every(dir => dir === 1)).toBe(true);
  });

  it('never auto-repeats while the key is only tapped', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    machine.tick(host);
    machine.release(1);

    const landed = shiftFrames(machine, host, 60);

    expect(landed).toEqual([]);
    expect(host.shifts).toEqual([1]);
  });

  it('caps a burst of taps at one shift per frame', () => {
    const host = new RecordingHost();
    const machine = new NesInput();

    // Two presses inside the same frame are one latch, so they can only buy one shift.
    machine.press(1);
    machine.press(1);
    machine.tick(host);

    expect(host.shifts).toEqual([1]);
  });

  it('saturates the charge when the shift is blocked, so the wall charge pays off on the next frame', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    host.blocked = true;

    machine.press(1);
    machine.tick(host);           // the press is refused by the host — a wall
    expect(host.attempts).toBe(1);
    expect(host.shifts).toEqual([]);

    host.blocked = false;         // the obstruction is gone (in the game: a new piece, or a rotation)
    const landed = shiftFrames(machine, host, 10);

    // A fully charged counter means the very next frame auto-shifts, instead of paying 16 frames again.
    expect(landed[0]).toBe(1);
    // ...and it then falls back into the ordinary repeat cadence.
    expect(landed[1] - landed[0]).toBe(TECHNIQUE_FRAMES.das);
  });

  it('keeps the counter saturated while the auto-shift stays blocked', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    host.blocked = true;

    machine.press(1);
    shiftFrames(machine, host, 20);
    expect(host.shifts).toEqual([]);
    expect(host.attempts).toBe(20); // one attempt per frame once saturated — the counter never falls back

    host.blocked = false;
    expect(shiftFrames(machine, host, 1)).toEqual([1]);
  });
});

describe('reversing direction mid-charge', () => {
  it('lets a fresh press win the frame even while the opposite key is still held', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    shiftFrames(machine, host, 8);

    machine.press(-1); // right is still physically held
    machine.tick(host);

    expect(host.shifts[host.shifts.length - 1]).toBe(-1);
  });

  it('holds still while both directions are held', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(-1);
    machine.press(1);

    expect(shiftFrames(machine, host, 40)).toEqual([]);
    expect(host.attempts).toBe(0);
  });

  it('discards the accumulated charge, so the new direction pays the full charge again', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    shiftFrames(machine, host, 11); // most of the way to the first auto-shift, but not there yet
    expect(host.shifts).toEqual([1]);

    machine.release(1);
    machine.press(-1);
    const landed = shiftFrames(machine, host, 30);

    expect(landed[0]).toBe(1);                               // the reversal shifts immediately...
    expect(landed[1] - landed[0]).toBe(DAS_CHARGE_FRAMES);   // ...from a counter reset to zero
    expect(host.shifts.slice(1).every(dir => dir === -1)).toBe(true);
  });
});

describe('soft drop and gravity', () => {
  it('waits three frames for the first soft-dropped row, then drops every other frame', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.pressDown();

    const landed = dropFrames(machine, host, 12);

    expect(landed[0]).toBe(3);
    expect(gaps(landed)).toEqual(gaps(landed).map(() => 2)); // 1/2G afterwards
    expect(landed.length).toBeGreaterThanOrEqual(5);
  });

  it('does not engage the soft drop while a direction is held', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    machine.tick(host);
    machine.pressDown();

    expect(dropFrames(machine, host, 40)).toEqual([]);
  });

  it('suspends the horizontal routine entirely while Down is held, without spending the charge', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    machine.tick(host);            // the press shifts and zeroes the counter
    machine.pressDown();

    expect(shiftFrames(machine, host, 40)).toEqual([]);
    expect(host.attempts).toBe(1); // the routine is skipped, not merely blocked

    machine.releaseDown();
    const landed = shiftFrames(machine, host, 30);

    // The counter resumes from where it was left — still zero — so the first auto-shift is a full
    // charge away, exactly as if the Down interlude had never happened.
    expect(landed[0]).toBe(DAS_CHARGE_FRAMES);
  });

  it('drops one row per gravity interval', () => {
    const host = new RecordingHost(5);
    const machine = new NesInput();

    expect(dropFrames(machine, host, 16)).toEqual([5, 10, 15]);
  });

  it('drops two rows per interval under the 2xks variant', () => {
    const host = new RecordingHost(5, 2);
    const machine = new NesInput();

    for (let frame = 0; frame < 10; frame++) machine.tick(host);

    expect(host.drops).toBe(4); // two gravity steps, two rows each
  });

  it('never lets a soft drop plus gravity move the piece more than one row in a frame', () => {
    const host = new RecordingHost(3); // gravity every 3 frames, i.e. colliding with the soft drop
    const machine = new NesInput();
    machine.pressDown();

    const landed = dropFrames(machine, host, 30);

    // One row per drop-frame: the gravity step skips its first row when the soft drop already
    // took it, so the two sources can never stack inside a single frame.
    expect(host.drops).toBe(landed.length);
    expect(host.drops).toBeGreaterThan(10);
  });

  it('reports the lock as soon as the host says the piece landed', () => {
    const host = new RecordingHost(3);
    host.lockAfterDrops = 1;
    const machine = new NesInput();

    expect(machine.tick(host)).toBe(false);
    expect(machine.tick(host)).toBe(false);
    expect(machine.tick(host)).toBe(true);
  });
});

describe('charge persistence', () => {
  it('carries the charge across a piece spawn — that carry-over is what makes wall charging real', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    shiftFrames(machine, host, 11); // press frame + 10 frames of charge

    machine.onSpawn();
    const landed = shiftFrames(machine, host, 20);

    // 10 of the 16 charge frames were already banked, so only the remainder is still owed.
    expect(landed[0]).toBe(DAS_CHARGE_FRAMES - 10);
  });

  it('resets the soft-drop cadence on a spawn', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.pressDown();
    dropFrames(machine, host, 12); // well past the first row, so the cadence is in repeat mode

    machine.onSpawn();
    machine.pressDown();           // a fresh press is needed: onSpawn disengaged the soft drop
    const landed = dropFrames(machine, host, 6);

    expect(landed[0]).toBe(3);     // the new piece pays the initial delay again
  });

  it('stops all movement when the inputs are dropped', () => {
    const host = new RecordingHost();
    const machine = new NesInput();
    machine.press(1);
    machine.pressDown();
    machine.tick(host);

    machine.clear();

    expect(shiftFrames(machine, host, 40)).toEqual([]);
    expect(dropFrames(machine, host, 40)).toEqual([]);
  });
});

describe('technique budget', () => {
  it('orders the three techniques from slowest to fastest', () => {
    expect(TECHNIQUE_FRAMES.das).toBeGreaterThan(TECHNIQUE_FRAMES.hypertap);
    expect(TECHNIQUE_FRAMES.hypertap).toBeGreaterThan(TECHNIQUE_FRAMES.roll);
    expect(TECHNIQUE_FRAMES.roll).toBeGreaterThanOrEqual(1); // 1 shift/frame is the NMI sampling ceiling
  });

  it('reports a rate consistent with the frame budget and the NES frame length', () => {
    for (const technique of ['das', 'hypertap', 'roll'] as Technique[]) {
      const expected = 1000 / NES_FRAME_MS / TECHNIQUE_FRAMES[technique];
      expect(techniqueHz(technique)).toBeCloseTo(expected, 1);
    }

    // The published figures these constants exist to reproduce (tetris.wiki / Cheez's rolling).
    expect(techniqueHz('das')).toBe(10);
    expect(techniqueHz('hypertap')).toBe(12);
    expect(techniqueHz('roll')).toBe(20);
  });

  it('charges more for the first auto-shift than for every repeat after it', () => {
    expect(DAS_CHARGE_FRAMES).toBeGreaterThan(TECHNIQUE_FRAMES.das);
  });
});
