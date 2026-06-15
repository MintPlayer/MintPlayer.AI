// Pure client-side Snake engine for HUMAN play (PRD §7.1: human mode is client-driven, ticked by a JS
// timer — no backend in the loop). Mirrors SnakeEnv's rules (grid, reversal guard, eat/grow, collision)
// so the human and the AI obey the same game.

export const SIZE = 12;
export const CELLS = SIZE * SIZE;

export type Dir = 0 | 1 | 2 | 3; // Up, Down, Left, Right — same indexing as SnakeEnv
const DELTAS: ReadonlyArray<readonly [number, number]> = [[-1, 0], [1, 0], [0, -1], [0, 1]];

export class SnakeGame {
  /** Cell indices, head first. */
  body: number[] = [];
  occupied = new Set<number>();
  food = 0;
  foodEaten = 0;
  dead = false;
  private heading: Dir = 3; // facing Right at start

  constructor() { this.reset(); }

  reset(): void {
    this.body = [];
    this.occupied.clear();
    const row = Math.floor(SIZE / 2);
    const headCol = Math.floor(SIZE / 2);
    for (let c = headCol; c >= headCol - 2; c--) {
      const cell = row * SIZE + c;
      this.body.push(cell);
      this.occupied.add(cell);
    }
    this.heading = 3;
    this.foodEaten = 0;
    this.dead = false;
    this.spawnFood();
  }

  /** Queue a heading; ignores the 180° reversal (the move onto the neck), exactly as the env masks it. */
  setDirection(dir: Dir): void {
    if (this.body.length >= 2) {
      const head = this.body[0];
      const [dr, dc] = DELTAS[dir];
      const r = Math.floor(head / SIZE) + dr;
      const c = (head % SIZE) + dc;
      if (r * SIZE + c === this.body[1]) return; // reversal — ignored
    }
    this.heading = dir;
  }

  /** Advance one step in the current heading. Sets `dead` on wall/self collision. */
  tick(): void {
    if (this.dead) return;
    const head = this.body[0];
    const [dr, dc] = DELTAS[this.heading];
    const r = Math.floor(head / SIZE) + dr;
    const c = (head % SIZE) + dc;
    if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) { this.dead = true; return; }

    const newHead = r * SIZE + c;
    const eating = newHead === this.food;
    const tail = this.body[this.body.length - 1];
    if (this.occupied.has(newHead) && !(newHead === tail && !eating)) { this.dead = true; return; }

    this.body.unshift(newHead);
    this.occupied.add(newHead);
    if (eating) {
      this.foodEaten++;
      if (this.occupied.size < CELLS) this.spawnFood();
    } else {
      this.occupied.delete(tail);
      this.body.pop();
    }
  }

  private spawnFood(): void {
    const free: number[] = [];
    for (let i = 0; i < CELLS; i++) if (!this.occupied.has(i)) free.push(i);
    this.food = free[Math.floor(Math.random() * free.length)];
  }
}
