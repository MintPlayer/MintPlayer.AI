import { VehicleDto } from './rush-hour-api';

// Mirrors RushHourBoard in RL.NET.Environments: 6×6 board, vehicle 0 is the red car
// (horizontal, exit row 2); positions[i] is vehicle i's variable coordinate
// (col when horizontal, row when vertical).
export const SIZE = 6;
export const EXIT_ROW = 2;
export const MAX_VEHICLES = 16;

export function initialPositions(vehicles: VehicleDto[]): number[] {
  return vehicles.map(v => (v.horizontal ? v.col : v.row));
}

/** 36-cell occupancy grid with the occupying vehicle index (−1 = empty). */
export function occupancy(vehicles: VehicleDto[], positions: number[]): number[] {
  const grid = new Array<number>(SIZE * SIZE).fill(-1);
  vehicles.forEach((v, i) => {
    for (let k = 0; k < v.length; k++) {
      const row = v.horizontal ? v.row : positions[i] + k;
      const col = v.horizontal ? positions[i] + k : v.col;
      grid[row * SIZE + col] = i;
    }
  });
  return grid;
}

/** Direction 0 = left/up (toward smaller coordinates), 1 = right/down. */
export function canMove(vehicles: VehicleDto[], positions: number[], vehicle: number, direction: number): boolean {
  const grid = occupancy(vehicles, positions);
  const v = vehicles[vehicle];
  const pos = positions[vehicle];

  if (direction === 0) {
    if (pos === 0) return false;
    const row = v.horizontal ? v.row : pos - 1;
    const col = v.horizontal ? pos - 1 : v.col;
    return grid[row * SIZE + col] < 0;
  }
  if (pos + v.length > SIZE - 1) return false;
  const row = v.horizontal ? v.row : pos + v.length;
  const col = v.horizontal ? pos + v.length : v.col;
  return grid[row * SIZE + col] < 0;
}

export function isSolved(vehicles: VehicleDto[], positions: number[]): boolean {
  return positions[0] + vehicles[0].length - 1 === SIZE - 1;
}

/** Can the vehicle be added to the drawing without leaving the board or overlapping? */
export function canPlace(vehicles: VehicleDto[], candidate: VehicleDto): boolean {
  if (vehicles.length >= MAX_VEHICLES) return false;
  const endRow = candidate.row + (candidate.horizontal ? 0 : candidate.length - 1);
  const endCol = candidate.col + (candidate.horizontal ? candidate.length - 1 : 0);
  if (candidate.row < 0 || candidate.col < 0 || endRow >= SIZE || endCol >= SIZE) return false;

  const grid = occupancy(vehicles, initialPositions(vehicles));
  for (let k = 0; k < candidate.length; k++) {
    const row = candidate.horizontal ? candidate.row : candidate.row + k;
    const col = candidate.horizontal ? candidate.col + k : candidate.col;
    if (grid[row * SIZE + col] >= 0) return false;
  }
  return true;
}
