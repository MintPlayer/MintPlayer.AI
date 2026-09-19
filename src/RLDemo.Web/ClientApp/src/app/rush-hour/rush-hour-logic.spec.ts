import { describe, expect, it } from 'vitest';

import type { VehicleDto } from './rush-hour-api';
import {
  EXIT_ROW,
  MAX_VEHICLES,
  SIZE,
  canMove,
  canPlace,
  clampRange,
  initialPositions,
  isSolved,
  occupancy,
} from './rush-hour-logic';

// M63.x — the client's Rush Hour board rules, mirroring `RushHourBoard` on the server.
//
// The interesting piece is `clampRange`: it exists purely as an optimisation (one grid rebuild per
// drag instead of one per pointer event), and its doc comment states it "must agree with iterating
// canMove one step at a time". That equivalence is asserted below over every vehicle of a fixture,
// which is the only thing standing between the drag handler and a car that slides through another.
//
// Coordinates: `positions[i]` is vehicle i's VARIABLE coordinate — its column when horizontal, its
// row when vertical. The fixed coordinate stays on the DTO.

const car = (row: number, col: number, length: number, horizontal: boolean): VehicleDto =>
  ({ row, col, length, horizontal });

/**
 *      0 1 2 3 4 5
 *   0  2 2 . 1 . .
 *   1  . . . 1 . .
 *   2  0 0 . 1 . .      <- exit row, vehicle 0 is the red car
 *   3  . . . . . 3
 *   4  . . . . . 3
 *   5  . . . . . .
 */
const board = (): VehicleDto[] => [
  car(EXIT_ROW, 0, 2, true),  // 0: the red car, blocked by the truck in column 3
  car(0, 3, 3, false),        // 1: vertical truck
  car(0, 0, 2, true),         // 2
  car(3, 5, 2, false),        // 3
];

describe('initialPositions', () => {
  it('takes the column of a horizontal vehicle and the row of a vertical one', () => {
    const vehicles = [car(2, 4, 2, true), car(1, 5, 3, false)];

    expect(initialPositions(vehicles)).toEqual([4, 1]);
  });
});

describe('occupancy', () => {
  it('stamps every cell a vehicle covers with its index and leaves the rest empty', () => {
    const vehicles = board();
    const grid = occupancy(vehicles, initialPositions(vehicles));

    expect(grid).toHaveLength(SIZE * SIZE);
    expect(grid.filter(v => v >= 0)).toHaveLength(2 + 3 + 2 + 2); // no vehicle overlaps another

    for (let i = 0; i < vehicles.length; i++) {
      expect(grid.filter(v => v === i)).toHaveLength(vehicles[i].length);
    }

    expect(grid[EXIT_ROW * SIZE + 0]).toBe(0);
    expect(grid[EXIT_ROW * SIZE + 1]).toBe(0);
    expect(grid[EXIT_ROW * SIZE + 3]).toBe(1); // the truck, sitting across the exit row
    expect(grid[EXIT_ROW * SIZE + 2]).toBe(-1);
  });

  it('follows the positions it is given rather than the DTO own coordinates', () => {
    const vehicles = board();
    const moved = initialPositions(vehicles);
    moved[0] = 1; // slide the red car one cell right

    const grid = occupancy(vehicles, moved);

    expect(grid[EXIT_ROW * SIZE + 0]).toBe(-1);
    expect(grid[EXIT_ROW * SIZE + 2]).toBe(0);
  });
});

describe('canMove', () => {
  it('refuses to move off the near edge of the board', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    expect(positions[0]).toBe(0);
    expect(canMove(vehicles, positions, 0, 0)).toBe(false); // the red car is already against column 0
  });

  it('refuses to move off the far edge of the board', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);
    positions[3] = SIZE - vehicles[3].length; // park the vertical car against the bottom

    expect(canMove(vehicles, positions, 3, 1)).toBe(false);
  });

  it('refuses to move into an occupied cell', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);
    positions[0] = 1; // the red car now ends at column 2, one short of the truck

    expect(canMove(vehicles, positions, 0, 1)).toBe(false);
    expect(canMove(vehicles, positions, 0, 0)).toBe(true); // ...but it can still back up
  });

  it('allows a step into free space', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    expect(canMove(vehicles, positions, 0, 1)).toBe(true);
    expect(canMove(vehicles, positions, 1, 1)).toBe(true);
    expect(canMove(vehicles, positions, 3, 0)).toBe(true);
  });
});

describe('clampRange', () => {
  it('agrees with stepping canMove one cell at a time', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    for (let i = 0; i < vehicles.length; i++) {
      const { lo, hi } = clampRange(vehicles, positions, i);

      expect(lo).toBeLessThanOrEqual(positions[i]);
      expect(hi).toBeGreaterThanOrEqual(positions[i]);
      // A single step is available in a direction exactly when the range extends that way.
      expect(canMove(vehicles, positions, i, 0)).toBe(lo < positions[i]);
      expect(canMove(vehicles, positions, i, 1)).toBe(hi > positions[i]);
    }
  });

  it('stops at the vehicle in the way and at the board edge', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    // The red car: pinned against column 0, and it cannot reach the truck's column.
    expect(clampRange(vehicles, positions, 0)).toEqual({ lo: 0, hi: 1 });
    // The vertical car in column 5 has the whole column to itself.
    expect(clampRange(vehicles, positions, 3)).toEqual({ lo: 0, hi: SIZE - vehicles[3].length });
  });

  it('keeps the whole vehicle on the board at both ends of the range', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    for (let i = 0; i < vehicles.length; i++) {
      const { lo, hi } = clampRange(vehicles, positions, i);

      expect(lo).toBeGreaterThanOrEqual(0);
      expect(hi + vehicles[i].length - 1).toBeLessThanOrEqual(SIZE - 1);
    }
  });
});

describe('isSolved', () => {
  it('is true exactly when the red car nose reaches the last column', () => {
    const vehicles = board();
    const positions = initialPositions(vehicles);

    expect(isSolved(vehicles, positions)).toBe(false);

    positions[0] = SIZE - vehicles[0].length;
    expect(isSolved(vehicles, positions)).toBe(true);

    positions[0] -= 1;
    expect(isSolved(vehicles, positions)).toBe(false);
  });
});

describe('canPlace', () => {
  it('accepts a vehicle that fits in free space', () => {
    expect(canPlace(board(), car(5, 0, 2, true))).toBe(true);
  });

  it('rejects a vehicle that overlaps an existing one', () => {
    expect(canPlace(board(), car(EXIT_ROW, 1, 2, true))).toBe(false);
  });

  it('rejects a vehicle that hangs off the board', () => {
    expect(canPlace(board(), car(5, SIZE - 1, 2, true))).toBe(false);  // runs off the right edge
    expect(canPlace(board(), car(SIZE - 1, 1, 2, false))).toBe(false); // runs off the bottom edge
    expect(canPlace(board(), car(-1, 0, 2, true))).toBe(false);
    expect(canPlace(board(), car(0, -1, 2, false))).toBe(false);
  });

  it('accepts a vehicle that ends exactly on the last cell', () => {
    expect(canPlace(board(), car(5, SIZE - 2, 2, true))).toBe(true);
    expect(canPlace(board(), car(SIZE - 2, 0, 2, false))).toBe(true);
  });

  it('rejects any vehicle once the board is full of them', () => {
    // The cap is checked before the geometry is, so the contents of the list do not matter here.
    const full = Array.from({ length: MAX_VEHICLES }, () => car(0, 0, 2, true));

    expect(canPlace(full, car(5, 0, 2, true))).toBe(false);
  });
});
