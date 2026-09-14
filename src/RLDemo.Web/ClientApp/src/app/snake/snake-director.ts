import { PgSnakeEnv, PgSnakeNet } from './snake_solver';
import { loadSnakeNet } from './snake-net';

// Client-side "watch the AI" director — the whole Snake AI runs in the browser (M33 + M34 search). It drives the
// single-source PgSnakeEnv (dynamics + 177-dim observation + flood-fill survivability) and the net-guided
// multi-ply look-ahead `chooseActionSearch` over the trained net loaded from the shipped checkpoint — no server,
// no WebSocket. The search is the lever that lifts play from the reactive ~50-food plateau to ~75+ (M34): it
// simulates every legal line and keeps the snake out of boxes it can't escape, with the net scoring the leaves.
// Discrete-tick (one AI move per tick), so the component drives it on a plain interval, like human play.
//
// The 'cycle' strategy (M48) instead drives `chooseActionCycle`: the snake always holds a Hamiltonian cycle it
// provably cannot die on, rebuilds it per food to route straight at the food, and the net ranks the safe
// shortcuts — games end board-full (a perfect game), never in a death.

// The board edge is a visitor setting (M61). The shipped net was trained on 12×12, but its 177-dim observation
// is size-independent — a 9×9 local obstacle patch plus scalars normalised by the board edge — so the same
// checkpoint drives any size. Only the Hamiltonian mode constrains it: a square grid graph has a Hamiltonian
// cycle only when an edge is EVEN (odd×odd is bipartite with unequal colour classes), so the picker steps by 2.
const SAFE_MASK = false;         // the planner's survival scoring supersedes the reactive 1-ply shield (it plans deeper)
const STEP_PENALTY = -0.01;      // training-only; irrelevant to greedy/search inference

// Net-tiebroken look-ahead tuning (M34). The flood-fill survival search does the heavy lifting; the net breaks ties
// between equally-safe moves (one forward per move). Depth 12 / beam 16 is the measured sweet spot — deeper/wider
// scored WORSE (beam pruning misranks deep lines, per PR #11's sweep) and slower. Mirrors SnakeSearchConfig in C#.
const SEARCH_DEPTH = 12;
const SEARCH_BEAM = 16;
const SEARCH_TUNED_CELLS = 12 * 12;   // the board those two numbers were tuned on
const MIN_SEARCH_DEPTH = 4;
const MIN_SEARCH_BEAM = 2;
const W_FOOD = 10_000;
const W_TRAP = 50_000;
const W_NET = 50;   // small: the net only breaks ties between equally-safe root moves (a big weight slightly hurts — measured)
const W_SPACE = 50;
const W_DIST = 1;
const W_RATIO = 100_000;   // anti-fragmentation: fraction of free cells still reachable. Biggest lever — ~71 → ~81 food@12 (M34)

// Cycle-mode tuning (M48). Mirrors SnakeCycleConfig in C#.
const W_CYCLE_NET = 50;          // net Q nudge between equally-safe shortcut options
const W_CYCLE_PROGRESS = 1_000;  // pull per cycle position a shortcut skips
const CYCLE_MARGIN = 4;          // positions a shortcut must leave before the tail (growth slack)
const DEAD_HOLD_TICKS = 8;       // show the finished board briefly before auto-restarting

export interface SnakeAiFrame {
  body: number[]; // head first
  food: number;
  foodEaten: number;
  done: boolean;
  length: number;
  /** The safety cycle in travel order (M60), or null in 'search' mode — there is no cycle there. */
  cycle: number[] | null;
  /** Bumped whenever the cycle changes; the view keys its cached path and its rebuild flash off this. */
  cycleEpoch: number;
}

export class SnakeDirector {
  private readonly core: PgSnakeEnv;
  /** Shortcuts are allowed while more than half the board is still free — a fraction, so it scales with size. */
  private readonly cycleMinFree: number;
  private readonly depth: number;
  private readonly beam: number;
  private net: PgSnakeNet | null = null;
  private ready = false;
  private deadHold = 0;

  // Cycle change-detection (M60). Reference identity alone is not enough in EITHER direction: initCycle()
  // mutates the array in place (`length = 0` + push), so the very first build keeps the old reference, and
  // reset() empties that same array on a new game. tryRebuildCycle() does assign a fresh array, which the
  // reference check catches — including on a non-food tick, since a failed rebuild is retried every tick.
  private lastCycleRef: number[] | null = null;
  private lastCycleLen = -1;
  private cycleEpoch = 0;
  private cycleCopy: number[] | null = null;

  constructor(private readonly strategy: 'search' | 'cycle' = 'search', size = 12) {
    this.core = new PgSnakeEnv(size, STEP_PENALTY, SAFE_MASK);
    this.cycleMinFree = size * size / 2;
    // Search budget, scaled to the board (M61). Every node of the look-ahead flood-fills the whole grid, so the
    // cost of the tuned depth-12 / beam-16 search grows super-linearly in cells: MEASURED in-browser at speed 30,
    // a step costs <50 ms at 12×12 but ~233 ms at 24×24, ~1.1 s at 40×40 and ~2.2 s at 50×50 — the main thread
    // is pegged and the board crawls. Shrinking the beam (∝ cells) and the depth (∝ √cells) holds a step at
    // roughly its 12×12 cost at every size. At and below 12×12 this is exactly the tuned pair, unchanged; the
    // floors keep a shallow but real search on the biggest boards, where the anti-fragmentation ratio term —
    // the measured big lever (M34) — is doing most of the work anyway.
    const scale = SEARCH_TUNED_CELLS / (size * size);
    this.depth = Math.max(MIN_SEARCH_DEPTH, Math.min(SEARCH_DEPTH, Math.round(SEARCH_DEPTH * Math.sqrt(scale))));
    this.beam = Math.max(MIN_SEARCH_BEAM, Math.min(SEARCH_BEAM, Math.round(SEARCH_BEAM * scale)));
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

    const action = this.strategy === 'cycle'
      ? this.core.chooseActionCycle(this.net, W_CYCLE_NET, W_CYCLE_PROGRESS, CYCLE_MARGIN, true, this.cycleMinFree)
      : this.core.chooseActionSearch(this.net, this.depth, this.beam, W_FOOD, W_TRAP, W_NET, W_SPACE, W_DIST, W_RATIO);
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
    return {
      body, food: this.core.food, foodEaten: this.core.foodEaten, done: this.core.done, length: n,
      cycle: this.syncCycle(), cycleEpoch: this.cycleEpoch,
    };
  }

  /**
   * Hand the view a SNAPSHOT of the cycle, never the engine's live array — `core.cycle` is engine-owned and
   * mutated in place, so a live reference would let a rebuild tear a frame the rAF loop is mid-draw on. The
   * copy is taken once per epoch (≈ once per food), not once per tick.
   */
  private syncCycle(): number[] | null {
    if (this.strategy !== 'cycle') return null;
    const live = this.core.cycle;
    if (live !== this.lastCycleRef || live.length !== this.lastCycleLen) {
      this.lastCycleRef = live;
      this.lastCycleLen = live.length;
      this.cycleEpoch++;
      this.cycleCopy = live.length > 0 ? live.slice() : null;
    }
    return this.cycleCopy;
  }
}
