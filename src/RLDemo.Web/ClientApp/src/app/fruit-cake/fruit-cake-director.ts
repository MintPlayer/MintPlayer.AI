import { PgFruitCakeWorld } from './fruitcake_solver';
import { FruitCakeFrame } from './fruit-cake-frame';
import type { AiDrop, AiResponse } from './fruit-cake-ai-protocol';

// Client-side "watch the AI" director — the whole AI runs in the browser (PRD FRUITCAKE_CLIENT_SIDE_AI, M32).
//
// M53: this class no longer decides anything. The worker (fruit-cake-ai.worker.ts) owns the game and plays
// it ahead with no animation pacing; the director REPLAYS the drops it streams back. That inversion is what
// removes the freeze: the depth-3 search costs 0.97–5.7 s per drop, and it used to run inline in the rAF
// callback, blocking the whole tab once per drop. Now it happens while the previous fruit is still falling.
//
// The replay is exact rather than approximate. Both sides run the same generated deterministic physics
// (`fruitcake_solver`, from the single-source `.pg`), so spawning the same fruit in the same column and
// stepping the same number of sub-steps reproduces the authority's board bit for bit. `drop.substeps` is
// the contract; the per-drop snapshot is only insurance, and the drift counter proves it stays at zero.
//
// State machine: LOADING/WAITING (no drop available yet) → SETTLE (animate exactly drop.substeps in real
// time) → BETWEEN (brief pause) → next queued drop, with GAMEOVER on a lost board.

type Phase = 'waiting' | 'settle' | 'between' | 'gameover';

const STEP = 1 / 60;                     // physics sub-step (matches the worker's settle loop)
const BETWEEN_S = 0.25;                  // pause between drops
const GAMEOVER_S = 1.8;                  // pause on the game-over board before restarting

export class FruitCakeDirector {
  private world = new PgFruitCakeWorld(true); // rotation on so fruit visibly roll (search clones rotation-off)
  private readonly worker: Worker;
  private phase: Phase = 'waiting';

  /** Drops decided by the worker but not yet animated. Its depth is the buffer that hides the search. */
  private readonly queue: AiDrop[] = [];
  private active: AiDrop | null = null;
  private remaining = 0;                  // sub-steps left in the current drop's animation
  private acc = 0;
  private timer = 0;
  private score = 0;
  private gen = 0;

  /** Instrumentation (M53.2 gates): how often the viewer caught up with the AI, and whether the replay
   *  ever diverged from the authority. Both are expected to stay at zero on a desktop. */
  private starvedDrops = 0;
  private driftedDrops = 0;

  constructor() {
    this.worker = new Worker(new URL('./fruit-cake-ai.worker', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (event: MessageEvent<AiResponse>) => {
      const { drop } = event.data;
      if (drop.gen === this.gen) this.queue.push(drop);
    });
  }

  /** Restart a fresh game (the worker keeps its already-loaded net). */
  reset(): void {
    this.clearBoard();
    this.requestNewGame();
    this.phase = 'waiting';
  }

  /** Tell the worker to begin a new game. Bumping the generation makes every in-flight drop from the
   *  abandoned game get discarded on arrival. */
  private requestNewGame(): void {
    this.gen++;
    this.queue.length = 0;
    this.worker.postMessage({ type: 'reset' });
  }

  private clearBoard(): void {
    this.world.clear();
    this.active = null;
    this.score = 0;
    this.acc = 0;
    this.timer = 0;
    this.remaining = 0;
  }

  /** Release the worker. The component calls this on destroy. */
  dispose(): void {
    this.worker.terminate();
  }

  /** How many decided drops are buffered ahead of the viewer (0 ⇒ the AI is the bottleneck). */
  get lookAhead(): number {
    return this.queue.length;
  }

  /** Times the viewer had to wait for the AI, and times the replay diverged from the authority. */
  get health(): { starved: number; drifted: number } {
    return { starved: this.starvedDrops, drifted: this.driftedDrops };
  }

  /** Advance the AI game by real elapsed time (seconds). Called from the RAF loop while in watch mode. */
  update(dt: number): void {
    switch (this.phase) {
      case 'waiting':
        this.beginNextDrop();
        return;

      case 'settle': {
        this.acc += dt;
        while (this.acc >= STEP && this.remaining > 0) {
          this.world.step(STEP);
          this.remaining--;
          this.acc -= STEP;
        }
        if (this.remaining === 0) this.finishDrop();
        return;
      }

      case 'between':
        this.timer += dt;
        if (this.timer >= BETWEEN_S) this.beginNextDrop();
        return;

      case 'gameover':
        this.timer += dt;
        // The next game was already requested when the board was lost, so the worker has been searching
        // throughout this pause — the fresh board starts animating immediately instead of after a search.
        if (this.timer >= GAMEOVER_S) {
          this.clearBoard();
          this.phase = 'waiting';
        }
        return;
    }
  }

  private beginNextDrop(): void {
    const drop = this.queue.shift();
    if (!drop) {
      // The AI has not decided the next drop yet. On a desktop this never happens; on a slow device it is
      // the signal that the look-ahead buffer has drained (PRD M53.3).
      if (this.phase !== 'waiting') this.starvedDrops++;
      this.phase = 'waiting';
      return;
    }
    this.active = drop;
    this.world.spawnFruit(
      drop.tier, PgFruitCakeWorld.columnX(drop.column, drop.tier), PgFruitCakeWorld.heldY(drop.tier));
    this.remaining = drop.substeps;
    this.acc = 0;
    this.phase = 'settle';
  }

  private finishDrop(): void {
    const drop = this.active;
    if (!drop) return;
    if (this.drifted(drop)) {
      this.driftedDrops++;
      this.snapTo(drop); // the worker is the authority — never let the replay accumulate error
    }
    this.score = drop.scoreAfter;
    this.worker.postMessage({ type: 'ack', index: drop.index });
    this.timer = 0;
    if (drop.lost) {
      this.phase = 'gameover';
      this.requestNewGame(); // let the AI search the opening while the game-over board is on screen
    } else {
      this.phase = 'between';
    }
  }

  /** Sub-pixel comparison against the authority's settled board. Deterministic replay should make this
   *  exactly false; a true here means the two worlds stopped agreeing and the drift counter should show it. */
  private drifted(drop: AiDrop): boolean {
    const bodies = this.world.bodies;
    if (bodies.length !== drop.snapshot.length) return true;
    for (let i = 0; i < bodies.length; i++) {
      const a = bodies[i], b = drop.snapshot[i];
      if (a.tier !== b.tier || Math.abs(a.x - b.x) > 0.5 || Math.abs(a.y - b.y) > 0.5) return true;
    }
    return false;
  }

  private snapTo(drop: AiDrop): void {
    this.world.clear();
    for (const b of drop.snapshot) {
      const body = this.world.spawnFruit(b.tier, b.x, b.y);
      body.angle = b.angle;
      body.vx = b.vx;
      body.vy = b.vy;
    }
  }

  /** A render frame in the shape renderFrame() consumes. */
  toFrame(): FruitCakeFrame {
    return {
      fruit: this.world.bodies.map(b => ({ x: b.x, y: b.y, angle: b.angle, tier: b.tier })),
      heldTier: this.active?.tier ?? 0,
      nextTier: this.active?.nextTier ?? this.queue[0]?.tier ?? 0,
      score: this.score,
      danger: this.world.bodies.some(b => b.y < PgFruitCakeWorld.DangerLineY),
      done: this.phase === 'gameover',
    };
  }
}
