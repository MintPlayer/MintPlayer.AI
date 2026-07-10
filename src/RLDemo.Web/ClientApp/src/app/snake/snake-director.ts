import { PgSnakeEnv, PgSnakeNet } from './snake_solver';
import { loadSnakeNet } from './snake-net';

// Client-side "watch the AI" director — the whole Snake AI runs in the browser (M33 + M34 search). It drives the
// single-source PgSnakeEnv (dynamics + 177-dim observation + flood-fill survivability) and the net-guided
// multi-ply look-ahead `chooseActionSearch` over the trained net loaded from the shipped checkpoint — no server,
// no WebSocket. The search is the lever that lifts play from the reactive ~50-food plateau to ~75+ (M34): it
// simulates every legal line and keeps the snake out of boxes it can't escape, with the net scoring the leaves.
// Discrete-tick (one AI move per tick), so the component drives it on a plain interval, like human play.

const SIZE = 12;                 // the shipped net was trained on a 12×12 board
const SAFE_MASK = false;         // the planner's survival scoring supersedes the reactive 1-ply shield (it plans deeper)
const STEP_PENALTY = -0.01;      // training-only; irrelevant to greedy/search inference

// Net-tiebroken look-ahead tuning (M34). The flood-fill survival search does the heavy lifting; the net breaks ties
// between equally-safe moves (one forward per move). Depth 12 / beam 16 is the measured sweet spot — deeper/wider
// scored WORSE (beam pruning misranks deep lines, per PR #11's sweep) and slower. Mirrors SnakeSearchConfig in C#.
const SEARCH_DEPTH = 12;
const SEARCH_BEAM = 16;
const W_FOOD = 10_000;
const W_TRAP = 50_000;
const W_NET = 50;   // small: the net only breaks ties between equally-safe root moves (a big weight slightly hurts — measured)
const W_SPACE = 50;
const W_DIST = 1;
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

    const action = this.core.chooseActionSearch(this.net, SEARCH_DEPTH, SEARCH_BEAM, W_FOOD, W_TRAP, W_NET, W_SPACE, W_DIST);
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
