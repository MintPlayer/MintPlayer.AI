import { describe, expect, it } from 'vitest';

import {
  ACTION_DOWN,
  ACTION_LEFT,
  ACTION_RIGHT,
  ACTION_UP,
  anyMoveAvailable,
  applyMove,
  exponentOf,
  maxTile,
  spawn,
  type Board,
} from './game-2048-logic';

// M63.x — the wire-format 2048 rules. This module is the client's copy of the server's `Board2048`,
// so every subtle merge rule here is one the two implementations have to agree on; a silent drift
// would show up as a playback reconstruction that diverges from the recorded episode.
//
// Boards are 16 EXPONENTS row-major (0 = empty, 1 = the tile "2"), which is why the fixtures below
// read as small integers rather than tile faces. Pure array maths: no DOM, no Angular, no fixtures.

/** Row-major board from four rows of four exponents — keeps the fixtures readable as a grid. */
const board = (...rows: number[][]): Board => rows.flat();

const EMPTY = [0, 0, 0, 0];

describe('applyMove', () => {
  it('merges each pair once, and never merges the tile it just produced', () => {
    // Four equal tiles collapse into TWO pairs, not into one doubled-twice tile: the classic
    // "1,1,1,1 -> 2,2" rule. If the merged result were eligible again in the same move the row
    // would read [3,0,0,0] and the score would be 12 instead of 8.
    const b = board([1, 1, 1, 1], EMPTY, EMPTY, EMPTY);

    const { moved, gained } = applyMove(b, ACTION_LEFT);

    expect(b.slice(0, 4)).toEqual([2, 2, 0, 0]);
    expect(moved).toBe(true);
    expect(gained).toBe(8); // two "4" tiles created
  });

  it('merges the two pairs of a mixed row independently', () => {
    const b = board([1, 1, 2, 2], EMPTY, EMPTY, EMPTY);

    const { gained } = applyMove(b, ACTION_LEFT);

    expect(b.slice(0, 4)).toEqual([2, 3, 0, 0]);
    // The score delta is the sum of the FACE VALUES created, not the count of merges.
    expect(gained).toBe(4 + 8);
  });

  it('compacts across gaps, merging tiles that are not adjacent', () => {
    const b = board([0, 1, 0, 1], EMPTY, EMPTY, EMPTY);

    const { moved, gained } = applyMove(b, ACTION_LEFT);

    expect(b.slice(0, 4)).toEqual([2, 0, 0, 0]);
    expect(moved).toBe(true);
    expect(gained).toBe(4);
  });

  it('slides unequal tiles together without scoring', () => {
    const b = board([0, 0, 1, 2], EMPTY, EMPTY, EMPTY);

    const { moved, gained } = applyMove(b, ACTION_LEFT);

    expect(b.slice(0, 4)).toEqual([1, 2, 0, 0]);
    expect(moved).toBe(true);
    expect(gained).toBe(0);
  });

  it('reports a move that changes nothing as a no-op', () => {
    // Already packed left with no equal neighbours — the board must come back untouched AND be
    // reported as not moved, because the caller uses `moved` to decide whether to spawn a tile.
    const before = board([1, 2, 3, 4], EMPTY, EMPTY, EMPTY);
    const b = [...before];

    const { moved, gained } = applyMove(b, ACTION_LEFT);

    expect(b).toEqual(before);
    expect(moved).toBe(false);
    expect(gained).toBe(0);
  });

  it('treats left and right as mirror images of one another', () => {
    const left = board([1, 1, 2, 2], EMPTY, EMPTY, EMPTY);
    const right = board([2, 2, 1, 1], EMPTY, EMPTY, EMPTY);

    const leftResult = applyMove(left, ACTION_LEFT);
    const rightResult = applyMove(right, ACTION_RIGHT);

    expect(left.slice(0, 4)).toEqual([2, 3, 0, 0]);
    expect(right.slice(0, 4)).toEqual([...left.slice(0, 4)].reverse());
    expect(rightResult.gained).toBe(leftResult.gained);
  });

  it('treats up and down as mirror images of one another', () => {
    // Column 0 only: indices 0, 4, 8, 12.
    const up = board([1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]);
    const down = board([2, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]);

    const upResult = applyMove(up, ACTION_UP);
    const downResult = applyMove(down, ACTION_DOWN);

    const upColumn = [up[0], up[4], up[8], up[12]];
    const downColumn = [down[0], down[4], down[8], down[12]];

    expect(upColumn).toEqual([2, 3, 0, 0]);
    expect(downColumn).toEqual([...upColumn].reverse());
    expect(downResult.gained).toBe(upResult.gained);
  });

  it('applies the same rule to every line of the board in one move', () => {
    const b = board([1, 1, 1, 1], [1, 1, 1, 1], EMPTY, [2, 2, 0, 0]);

    const { gained } = applyMove(b, ACTION_LEFT);

    expect(b).toEqual(board([2, 2, 0, 0], [2, 2, 0, 0], EMPTY, [3, 0, 0, 0]));
    expect(gained).toBe(8 + 8 + 8);
  });
});

describe('anyMoveAvailable', () => {
  it('is true whenever an empty cell exists', () => {
    expect(anyMoveAvailable(board([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]))).toBe(true);
  });

  it('is false on a full board whose neighbours all differ', () => {
    // Alternating parity in both axes, so no horizontal or vertical pair is equal.
    const locked = board([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]);

    expect(anyMoveAvailable(locked)).toBe(false);
    // ...and that agrees with the rules: no direction actually changes the board.
    for (const action of [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]) {
      expect(applyMove([...locked], action).moved).toBe(false);
    }
  });

  it('is true on a full board that still has one equal pair', () => {
    const b = board([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 1, 2]);

    expect(anyMoveAvailable(b)).toBe(true);
  });
});

describe('spawn', () => {
  it('fills the only empty cell with a 2 or a 4, leaving everything else alone', () => {
    const b = board([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 0]);

    spawn(b);

    expect(b[15]).toBeGreaterThanOrEqual(1);
    expect(b[15]).toBeLessThanOrEqual(2); // exponent 1 or 2, i.e. a 2 or a 4
    expect(b.slice(0, 15)).toEqual([1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2]);
  });

  it('leaves a full board untouched', () => {
    const before = board([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]);
    const b = [...before];

    spawn(b);

    expect(b).toEqual(before);
  });
});

describe('maxTile', () => {
  it('reports the face value of the largest tile, not its exponent', () => {
    expect(maxTile(board([1, 5, 0, 0], EMPTY, EMPTY, EMPTY))).toBe(32);
  });

  it('is 0 on an empty board', () => {
    expect(maxTile(board(EMPTY, EMPTY, EMPTY, EMPTY))).toBe(0);
  });
});

describe('exponentOf', () => {
  it('inverts the exponent encoding used by the board', () => {
    for (const exponent of [1, 2, 3, 10, 11]) {
      expect(exponentOf(1 << exponent)).toBe(exponent);
    }
  });

  it('maps the empty cell to 0', () => {
    expect(exponentOf(0)).toBe(0);
  });
});
