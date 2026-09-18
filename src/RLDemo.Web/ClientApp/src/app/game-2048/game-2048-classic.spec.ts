import { describe, expect, it } from 'vitest';

import { ClassicEngine, serverActionToClassicDirection } from './game-2048-classic';
import { ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, applyMove } from './game-2048-logic';

// M63.x — the classic (animated) 2048 engine.
//
// This is Cirulli's original traversal-order implementation, kept alongside the flat `applyMove`
// rules so the DOM renderer can play slide/pop animations. That duplication is exactly the risk:
// the file's own header promises the two produce IDENTICAL merge results, and nothing enforced it.
// The last describe block does, by running both over the same boards.
//
// Note the coordinate convention: the grid is indexed [x][y] (column, then row) internally, while
// the exponent wire format is 16 cells ROW-major. The fixtures below are written in wire format.

/** Row-major exponent board from four rows of four — keeps the fixtures readable as a grid. */
const cells = (...rows: number[][]): number[] => rows.flat();

const EMPTY = [0, 0, 0, 0];

describe('serverActionToClassicDirection', () => {
  it('maps each server action onto the classic direction with the same meaning', () => {
    // Server ids: 0=left 1=down 2=right 3=up. Classic: 0=up 1=right 2=down 3=left.
    expect(serverActionToClassicDirection(ACTION_LEFT)).toBe(3);
    expect(serverActionToClassicDirection(ACTION_DOWN)).toBe(2);
    expect(serverActionToClassicDirection(ACTION_RIGHT)).toBe(1);
    expect(serverActionToClassicDirection(ACTION_UP)).toBe(0);
  });
});

describe('ClassicEngine board round-trip', () => {
  it('reloads an exponent board unchanged', () => {
    const board = cells([1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 0, 0], [0, 0, 6, 0]);

    expect(ClassicEngine.fromExponents(board).toExponents()).toEqual(board);
  });

  it('reports the largest face value on the board', () => {
    expect(ClassicEngine.fromExponents(cells([1, 5, 0, 0], EMPTY, EMPTY, EMPTY)).maxTile()).toBe(32);
    expect(ClassicEngine.fromExponents(cells(EMPTY, EMPTY, EMPTY, EMPTY)).maxTile()).toBe(0);
  });
});

describe('ClassicEngine.move', () => {
  it('merges each pair once, and never merges the tile it just produced', () => {
    // The `mergedFrom` guard is the entire point of this implementation: four equal tiles must
    // collapse into two, not cascade into one. A broken guard yields [3,0,0,0] and 12 points.
    const engine = ClassicEngine.fromExponents(cells([1, 1, 1, 1], EMPTY, EMPTY, EMPTY));

    const { moved, gained } = engine.move(ACTION_LEFT);

    expect(engine.toExponents().slice(0, 4)).toEqual([2, 2, 0, 0]);
    expect(moved).toBe(true);
    expect(gained).toBe(8);
    expect(engine.score).toBe(8);
  });

  it('merges the two pairs of a mixed row independently and scores their face values', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 1, 2, 2], EMPTY, EMPTY, EMPTY));

    const { gained } = engine.move(ACTION_LEFT);

    expect(engine.toExponents().slice(0, 4)).toEqual([2, 3, 0, 0]);
    expect(gained).toBe(4 + 8);
  });

  it('compacts across gaps', () => {
    const engine = ClassicEngine.fromExponents(cells([0, 1, 0, 1], EMPTY, EMPTY, EMPTY));

    const { moved, gained } = engine.move(ACTION_LEFT);

    expect(engine.toExponents().slice(0, 4)).toEqual([2, 0, 0, 0]);
    expect(moved).toBe(true);
    expect(gained).toBe(4);
  });

  it('accumulates the score across moves', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 1, 0, 0], [1, 1, 0, 0], EMPTY, EMPTY));

    const first = engine.move(ACTION_LEFT);  // two 4s
    const second = engine.move(ACTION_UP);   // the two 4s become an 8

    expect(first.gained).toBe(8);
    expect(second.gained).toBe(8);
    expect(engine.score).toBe(first.gained + second.gained);
  });

  it('reports a move that changes nothing as a no-op', () => {
    const board = cells([1, 2, 3, 4], EMPTY, EMPTY, EMPTY);
    const engine = ClassicEngine.fromExponents(board);

    const { moved, gained } = engine.move(ACTION_LEFT);

    expect(moved).toBe(false);
    expect(gained).toBe(0);
    expect(engine.toExponents()).toEqual(board);
    expect(engine.score).toBe(0);
  });

  it('treats left and right as mirror images of one another', () => {
    const left = ClassicEngine.fromExponents(cells([1, 1, 2, 2], EMPTY, EMPTY, EMPTY));
    const right = ClassicEngine.fromExponents(cells([2, 2, 1, 1], EMPTY, EMPTY, EMPTY));

    const leftResult = left.move(ACTION_LEFT);
    const rightResult = right.move(ACTION_RIGHT);

    expect(right.toExponents().slice(0, 4)).toEqual([...left.toExponents().slice(0, 4)].reverse());
    expect(rightResult.gained).toBe(leftResult.gained);
  });

  it('raises `won` when a 2048 tile is created', () => {
    const engine = ClassicEngine.fromExponents(cells([10, 10, 0, 0], EMPTY, EMPTY, EMPTY));

    expect(engine.won).toBe(false);
    const { gained } = engine.move(ACTION_LEFT);

    expect(engine.won).toBe(true);
    expect(gained).toBe(2048);
    expect(engine.maxTile()).toBe(2048);
  });
});

describe('ClassicEngine.movesAvailable', () => {
  it('is true whenever an empty cell exists', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 0]));

    expect(engine.movesAvailable()).toBe(true);
  });

  it('is false on a full board whose neighbours all differ', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]));

    expect(engine.movesAvailable()).toBe(false);
    for (const action of [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]) {
      expect(ClassicEngine.fromExponents(engine.toExponents()).move(action).moved).toBe(false);
    }
  });
});

describe('ClassicEngine spawning', () => {
  it('places a scripted tile at the row-major index the server sent', () => {
    const engine = ClassicEngine.fromExponents(cells(EMPTY, EMPTY, EMPTY, EMPTY));

    engine.addSpecificTile(5, 4);

    const board = engine.toExponents();
    expect(board[5]).toBe(2); // exponent of the face value 4
    expect(board.filter(c => c !== 0)).toHaveLength(1);
  });

  it('fills the only empty cell with a 2 or a 4 when spawning at random', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 0]));

    engine.addRandomTile();

    const board = engine.toExponents();
    expect(board[15]).toBeGreaterThanOrEqual(1);
    expect(board[15]).toBeLessThanOrEqual(2);
    expect(board.filter(c => c === 0)).toHaveLength(0);
  });
});

describe('ClassicEngine.renderTiles', () => {
  it('keeps both merge sources on the board under the merged tile so the pop can animate', () => {
    const engine = ClassicEngine.fromExponents(cells([1, 1, 0, 0], EMPTY, EMPTY, EMPTY));

    engine.move(ACTION_LEFT);
    const tiles = engine.renderTiles();

    // Three render entries for two board tiles: the merged "4" plus the two "2"s that produced it,
    // all parked on the destination cell.
    expect(tiles).toHaveLength(3);
    expect(tiles.every(t => t.x === 0 && t.y === 0)).toBe(true);

    const sources = tiles.filter(t => t.state === 'none');
    expect(sources.map(t => t.value)).toEqual([2, 2]);

    const merged = tiles.filter(t => t.state === 'merged');
    expect(merged).toHaveLength(1);
    expect(merged[0].value).toBe(4);

    expect(tiles.map(t => t.id)).toEqual([...tiles.map(t => t.id)].sort((a, b) => a - b));
  });

  it('flags a freshly spawned tile as new, and clears the flag on the next move', () => {
    const engine = ClassicEngine.fromExponents(cells(EMPTY, EMPTY, EMPTY, EMPTY));

    engine.addSpecificTile(0, 2);
    expect(engine.renderTiles().map(t => t.state)).toEqual(['new']);

    engine.move(ACTION_RIGHT);
    expect(engine.renderTiles().map(t => t.state)).toEqual(['none']);
  });
});

describe('the two 2048 implementations agree', () => {
  // The classic engine's header promises its merge RESULTS match `applyMove` exactly. That is the
  // contract that lets the animated client and the server's recorded episodes stay in lockstep, so
  // assert it directly instead of trusting the comment.
  const boards = [
    cells([1, 1, 2, 2], EMPTY, [3, 0, 3, 0], [1, 0, 0, 1]),
    cells([1, 1, 1, 1], [1, 1, 1, 1], EMPTY, [2, 2, 0, 0]),
    cells([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2], [0, 2, 0, 2]),
    cells([5, 0, 0, 5], [0, 5, 5, 0], [0, 0, 0, 0], [6, 6, 6, 6]),
  ];

  for (const action of [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]) {
    it(`produces the same board and score for action ${action}`, () => {
      boards.forEach((board, index) => {
        const flat = [...board];
        const flatResult = applyMove(flat, action);

        const engine = ClassicEngine.fromExponents(board);
        const classicResult = engine.move(action);

        expect({ index, board: engine.toExponents(), gained: classicResult.gained })
          .toEqual({ index, board: flat, gained: flatResult.gained });
      });
    });
  }
});
