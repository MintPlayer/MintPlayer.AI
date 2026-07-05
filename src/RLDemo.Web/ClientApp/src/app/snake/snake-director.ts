import { PgSnakeEnv, PgSnakeNet } from './snake_solver';
import { loadSnakeNet } from './snake-net';

// Client-side "watch the AI" director — the whole Snake AI runs in the browser (M33). It drives the single-source
// PgSnakeEnv (dynamics + 177-dim observation + the anti-self-trap flood-fill action mask) and the masked-greedy
// chooseAction over the trained net loaded from the shipped checkpoint — no server, no WebSocket. Discrete-tick
// (one AI move per tick), so the component drives it on a plain interval, like human play.

const SIZE = 12;                 // the shipped net was trained on a 12×12 board
const SAFE_MASK = true;          // the deployed net was trained WITH the flood-fill shield — inference must match
const STEP_PENALTY = -0.01;      // training-only; irrelevant to greedy inference
const DEAD_HOLD_TICKS = 8;       // show the dead board briefly before auto-restarting

export interface SnakeAiFrame {
  body: number[]; // head first
  food: number;
  foodEaten: number;
  done: boolean;
  length: number;
}

export class SnakeDirector {
  private readonly core = new PgSnakeEnv(SIZE, STEP_PENALTY, SAFE_MASK);
  private net: PgSnakeNet | null = null;
  private ready = false;
  private deadHold = 0;

  constructor() {
    void loadSnakeNet().then(n => {
      this.net = n; // null (missing checkpoint) → the board just sits; the checkpoint is shipped, so this is a safety net
      this.newGame();
      this.ready = true;
    });
  }

  private randFree(): number {
    return Math.floor(Math.random() * this.core.freeCount()); // the browser owns the food RNG now
  }

  private newGame(): void {
    this.core.reset();
    this.core.spawnFood(this.randFree());
    this.deadHold = 0;
  }

  /** Advance one AI move. Returns the current frame, or null while the checkpoint is still loading. */
  step(): SnakeAiFrame | null {
    if (!this.ready) return null;
    if (this.core.done) {
      if (this.deadHold > 0) { this.deadHold--; return this.frame(); }
      this.newGame();
      return this.frame();
    }
    if (this.net === null) return this.frame();

    const action = this.core.chooseAction(this.net);
    if (action < 0) { this.deadHold = DEAD_HOLD_TICKS; return this.frame(); } // no legal move (shouldn't happen)
    this.core.step(action);
    if (this.core.needsFood) this.core.spawnFood(this.randFree());
    if (this.core.done) this.deadHold = DEAD_HOLD_TICKS;
    return this.frame();
  }

  private frame(): SnakeAiFrame {
    const n = this.core.body.length;
    const body = new Array<number>(n);
    for (let k = 0; k < n; k++) body[k] = this.core.body[n - 1 - k]; // core stores head-at-end → head first
    return { body, food: this.core.food, foodEaten: this.core.foodEaten, done: this.core.done, length: n };
  }
}
