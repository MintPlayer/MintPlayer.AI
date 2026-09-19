import { describe, expect, it } from 'vitest';

import { PgTetris } from './tetris_solver';

// The browser-side half of the `.pg` coverage union for Tetris.
//
// `tetris_solver.ts` is transpiled from `tetris_solver.pg`, the same source that produces the C# the
// training backend runs, and coverage from both targets is overlaid onto the `.pg` line numbers.
// Training only ever uses the MACRO move: `applyPlacement(action)` computes a landing row and places
// the piece in one go. The browser plays the game the way a person does — one frame at a time through
// the micro-move API (`tetris-game.ts` drives `microShift` / `microDropStep` / `microRotate` /
// `microHardDrop`), so the micro path is browser-only by construction.
//
// The lines this file targets are `microDropStep`'s LOCKING branch (gravity ran out of room, hand
// over to `microLock`) and `microLock`'s TOP-OUT branch (`activeY < 0`, the piece locks above the
// ceiling and the game ends). Top-out is only reachable because rotation can move the active piece
// UP: `fitsAt` permits y down to -2, and the I piece's rotation offsets carry y from 0 to -2. So the
// setup below rotates an I piece upright above a blocker, leaving it with nowhere to fall.

/** A board with a known current piece. `reset` draws randomly; the micro path does not care which. */
function board(piece: number): PgTetris {
  const t = new PgTetris();
  t.reset(20260918, false, 0);
  t.current = piece;
  return t;
}

const bitsSet = (row: number): number => {
  let n = 0;
  for (let c = 0; c < PgTetris.W; c++) n += row >> c & 1;
  return n;
};

const I_PIECE = 0;

describe('PgTetris micro-moves — the human-play path', () => {
  it('spawns the active piece live at the top of an empty board', () => {
    const t = board(I_PIECE);

    expect(t.microSpawn()).toBe(true);
    expect(t.activeLive).toBe(true);
    expect(t.gameOver).toBe(false);
    expect(t.activeY).toBe(0);
  });

  it('falls one row per drop step and reports "not locked" the whole way down', () => {
    const t = board(I_PIECE);
    t.microSpawn();

    let y = t.activeY;
    let locked = false;
    for (let i = 0; i < PgTetris.H + 2 && !locked; i++) {
      const before = t.activeY;
      locked = t.microDropStep();
      if (!locked) {
        expect(t.activeY).toBe(before + 1);
        y = t.activeY;
      }
    }

    expect(locked).toBe(true);
    // A flat I piece is one row tall, so it comes to rest on the last row.
    expect(y).toBe(PgTetris.H - 1);
  });

  it('locks into the board and hands over to the next piece', () => {
    const t = board(I_PIECE);
    t.microSpawn();
    const x = t.activeX;

    let guard = 0;
    while (!t.microDropStep() && guard++ < PgTetris.H + 2) { /* fall */ }

    // Four contiguous cells, at the column the piece was actually standing in.
    expect(t.rows[PgTetris.H - 1]).toBe(0b1111 << x);
    for (let r = 0; r < PgTetris.H - 1; r++) expect(t.rows[r]).toBe(0);

    expect(t.piecesPlaced).toBe(1);
    expect(t.lines).toBe(0);            // four cells is not a full row
    expect(t.gameOver).toBe(false);
    expect(t.activeLive).toBe(true);    // microLock respawns
    expect(t.activeY).toBe(0);
  });

  it('a hard drop lands in the same place a run of drop steps does', () => {
    const stepped = board(I_PIECE);
    stepped.microSpawn();
    let guard = 0;
    while (!stepped.microDropStep() && guard++ < PgTetris.H + 2) { /* fall */ }

    const dropped = board(I_PIECE);
    dropped.microSpawn();
    expect(dropped.microHardDrop()).toBe(true);

    expect(dropped.rows).toEqual(stepped.rows);
  });

  it('tops out when the piece locks above the ceiling', () => {
    // The I piece spawns flat across columns 3..6. Rotating it upright moves it to column 5 and
    // lifts it to y = -2 (its rotation offsets do that). One blocker three rows down means it
    // cannot fall even a single row, so the very next gravity tick locks it with y < 0 — the
    // top-out the browser has to end the game on, and the one case `applyPlacement` never produces.
    const t = board(I_PIECE);
    t.rows[2] = 1 << 5;

    expect(t.microSpawn()).toBe(true);
    expect(t.microRotate()).toBe(true);
    expect(t.activeY).toBeLessThan(0);

    const rowsBefore = t.rows.slice();
    expect(t.microDropStep()).toBe(true);

    expect(t.gameOver).toBe(true);
    expect(t.activeLive).toBe(false);
    expect(t.piecesPlaced).toBe(0);     // a topped-out piece is never merged into the board
    expect(t.rows).toEqual(rowsBefore);
  });

  it('refuses every micro-move once the game is over', () => {
    const t = board(I_PIECE);
    t.rows[2] = 1 << 5;
    t.microSpawn();
    t.microRotate();
    t.microDropStep();
    expect(t.gameOver).toBe(true);

    expect(t.microDropStep()).toBe(false);
    expect(t.microHardDrop()).toBe(false);
    expect(t.microShift(-1)).toBe(false);
    expect(t.microShift(1)).toBe(false);
    expect(t.microRotate()).toBe(false);
    expect(t.microRotateCcw()).toBe(false);
  });

  it('shifts only into columns that are actually on the board', () => {
    const t = board(I_PIECE);
    t.microSpawn();

    let moved = 0;
    while (t.microShift(-1)) moved++;

    expect(moved).toBeGreaterThan(0);
    expect(t.activeX).toBe(0);
    expect(t.microShift(-1)).toBe(false); // the wall holds
  });

  it('round-trips a rotation: clockwise then counter-clockwise restores the spawn pose', () => {
    const t = board(I_PIECE);
    t.microSpawn();
    const rot = t.activeRot;
    const x = t.activeX;
    const y = t.activeY;

    expect(t.microRotate()).toBe(true);
    expect(t.activeRot).not.toBe(rot);
    expect(t.microRotateCcw()).toBe(true);

    expect(t.activeRot).toBe(rot);
    expect(t.activeX).toBe(x);
    expect(t.activeY).toBe(y);
  });

  it('clears a completed row through the micro path', () => {
    const t = board(I_PIECE);
    // Leave columns 3..6 of the bottom row open, so one flat I fills it exactly.
    t.rows[PgTetris.H - 1] = PgTetris.FullRow & ~(0b1111 << 3);
    expect(bitsSet(t.rows[PgTetris.H - 1])).toBe(PgTetris.W - 4);

    t.microSpawn();
    expect(t.activeX).toBe(3);
    let guard = 0;
    while (!t.microDropStep() && guard++ < PgTetris.H + 2) { /* fall */ }

    expect(t.lines).toBe(1);
    expect(t.rows[PgTetris.H - 1]).toBe(0);
    expect(t.score).toBeGreaterThan(0);
  });
});
