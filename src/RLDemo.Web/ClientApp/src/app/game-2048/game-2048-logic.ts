// Client-side 2048 rules, mirroring Board2048 in MintPlayer.AI.ReinforcementLearning.Environments exactly
// (same action ids and merge semantics) so manual play and playback reconstruction
// behave identically to the server. Boards are 16 EXPONENTS row-major (0 = empty).

export type Board = number[];

export const ACTION_LEFT = 0;
export const ACTION_DOWN = 1;
export const ACTION_RIGHT = 2;
export const ACTION_UP = 3;

const LINES: number[][][] = buildLines();

function buildLines(): number[][][] {
  const range = [0, 1, 2, 3];
  return [
    range.map(r => range.map(c => r * 4 + c)),            // left
    range.map(c => range.map(r => (3 - r) * 4 + c)),      // down
    range.map(r => range.map(c => r * 4 + (3 - c))),      // right
    range.map(c => range.map(r => r * 4 + c)),            // up
  ];
}

/** Applies a move in place; returns whether it changed the board and the classic score gained. */
export function applyMove(board: Board, action: number): { moved: boolean; gained: number } {
  let moved = false;
  let gained = 0;

  for (const map of LINES[action]) {
    const line = map.map(i => board[i]);
    const result = [0, 0, 0, 0];
    let write = 0;
    let pending = 0;

    for (const tile of line) {
      if (tile === 0) continue;
      if (pending === 0) {
        pending = tile;
      } else if (pending === tile) {
        const merged = Math.min(pending + 1, 15);
        result[write++] = merged;
        gained += 1 << merged;
        pending = 0;
      } else {
        result[write++] = pending;
        pending = tile;
      }
    }
    if (pending !== 0) result[write++] = pending;

    for (let i = 0; i < 4; i++) {
      if (line[i] !== result[i]) moved = true;
      board[map[i]] = result[i];
    }
  }
  return { moved, gained };
}

export function anyMoveAvailable(board: Board): boolean {
  if (board.some(t => t === 0)) return true;
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      if (c < 3 && board[r * 4 + c] === board[r * 4 + c + 1]) return true;
      if (r < 3 && board[r * 4 + c] === board[(r + 1) * 4 + c]) return true;
    }
  return false;
}

/** Spawns a 2 (90%) or 4 (10%) in a random empty cell — manual play only. */
export function spawn(board: Board): void {
  const empty = board.map((t, i) => (t === 0 ? i : -1)).filter(i => i >= 0);
  if (empty.length === 0) return;
  const index = empty[Math.floor(Math.random() * empty.length)];
  board[index] = Math.random() < 0.9 ? 1 : 2;
}

export function maxTile(board: Board): number {
  const exponent = Math.max(...board);
  return exponent === 0 ? 0 : 1 << exponent;
}

export const exponentOf = (value: number): number => (value === 0 ? 0 : Math.round(Math.log2(value)));
