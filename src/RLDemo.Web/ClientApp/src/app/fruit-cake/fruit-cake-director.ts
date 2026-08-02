import { PgFruitCakeWorld } from './fruitcake_solver';
import { FruitCakeFrame } from './fruit-cake-frame';
import type { AiRequest, AiResponse } from './fruit-cake-ai-protocol';

// Client-side "watch the AI" director — the whole AI runs in the browser (PRD FRUITCAKE_CLIENT_SIDE_AI, M32).
// It drives the single-source physics (PgFruitCakeWorld) and renders it; the depth-3 search over the trained
// net runs in a Web Worker (M53, FRUITCAKE_WATCH_AI_STALL_PRD.md) — no server, no WebSocket.
//
// M53 — why the search is not inline any more: it costs 784 clone+settle rollouts and ~3920 net forward
// passes per drop, which measured 0.97–5.7 s of *blocked main thread* when it ran inside the rAF callback.
// rAF stopped firing entirely and the tab froze once per drop, growing +240 ms per fruit on the board. The
// search config (depth 3 / topK 5 / topK2 2) is deliberately unchanged: the worker exists so we can keep it.
//
// A small real-time state machine mirrors the old server loop: THINK (ask the worker, spawn on its answer) →
// SETTLE (step the physics to rest, animating the fall/merges in real time) → BETWEEN (brief pause) → repeat,
// with GAMEOVER on a lost board. The net loads inside the worker; until it is ready the board stays empty.

type Phase = 'loading' | 'think' | 'thinking' | 'settle' | 'between' | 'gameover';

const STEP = 1 / 60;                     // physics sub-step (matches the .pg settle loop)
const BETWEEN_S = 0.25;                  // pause between drops
const GAMEOVER_S = 1.8;                  // pause on the game-over board before restarting

export class FruitCakeDirector {
  private world = new PgFruitCakeWorld(true); // rotation on so fruit visibly roll (search clones rotation-off)
  private readonly worker: Worker;
  private ready = false;
  private phase: Phase = 'loading';
  private current = this.randTier();
  private next = this.randTier();
  private score = 0;
  private substeps = 0;
  private acc = 0;
  private timer = 0;
  /** Correlates a worker answer with the request that asked for it, so a reply for a game we have already
   *  restarted is dropped instead of spawning a fruit into the new board. */
  private requestId = 0;

  constructor() {
    this.worker = new Worker(new URL('./fruit-cake-ai.worker', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (event: MessageEvent<AiResponse>) => this.onMessage(event.data));
  }

  /** Restart a fresh game (the worker keeps its already-loaded net). */
  reset(): void {
    this.world.clear();
    this.score = 0;
    this.current = this.randTier();
    this.next = this.randTier();
    this.acc = 0;
    this.timer = 0;
    this.substeps = 0;
    this.requestId++; // invalidate any answer still in flight for the old board
    this.phase = this.ready ? 'think' : 'loading';
  }

  /** Release the worker. The component calls this on destroy. */
  dispose(): void {
    this.worker.terminate();
  }

  /** Advance the AI game by real elapsed time (seconds). Called from the RAF loop while in watch mode. */
  update(dt: number): void {
    switch (this.phase) {
      case 'loading':
      case 'thinking':
        return; // waiting on the worker; the board is at rest and nothing may change under it

      case 'think': {
        const request: AiRequest = {
          type: 'search',
          id: this.requestId,
          bodies: this.world.bodies.map(b => ({ tier: b.tier, x: b.x, y: b.y, vx: b.vx, vy: b.vy })),
          current: this.current,
          next: this.next,
        };
        this.worker.postMessage(request);
        this.phase = 'thinking'; // guard: post once, not once per frame
        return;
      }

      case 'settle': {
        this.acc += dt;
        let settled = false;
        while (this.acc >= STEP && !settled) {
          this.score += this.world.step(STEP);
          this.substeps++;
          this.acc -= STEP;
          const quiet = this.substeps >= PgFruitCakeWorld.MinSettleSubsteps &&
            this.world.maxSpeed() < PgFruitCakeWorld.SettleSpeedPx;
          if (quiet || this.substeps >= PgFruitCakeWorld.MaxSubsteps) settled = true;
        }
        if (settled) {
          const lost = this.world.anyEjected() || this.world.anyRestingAboveDangerLine(PgFruitCakeWorld.RestSpeedPx);
          this.timer = 0;
          this.phase = lost ? 'gameover' : 'between';
        }
        return;
      }

      case 'between':
        this.timer += dt;
        if (this.timer >= BETWEEN_S) {
          this.current = this.next;
          this.next = this.randTier();
          this.phase = 'think';
        }
        return;

      case 'gameover':
        this.timer += dt;
        if (this.timer >= GAMEOVER_S) this.reset();
        return;
    }
  }

  private onMessage(msg: AiResponse): void {
    if (msg.type === 'ready') {
      this.ready = true;
      if (this.phase === 'loading') this.phase = 'think';
      return;
    }
    // A stale answer (the game restarted while the worker was searching) must not spawn into the new board.
    if (msg.id !== this.requestId || this.phase !== 'thinking') return;
    this.world.spawnFruit(
      this.current, PgFruitCakeWorld.columnX(msg.column, this.current), PgFruitCakeWorld.heldY(this.current));
    this.substeps = 0;
    this.acc = 0;
    this.phase = 'settle';
  }

  /** A render frame in the shape renderFrame() consumes. */
  toFrame(): FruitCakeFrame {
    return {
      fruit: this.world.bodies.map(b => ({ x: b.x, y: b.y, angle: b.angle, tier: b.tier })),
      heldTier: this.current,
      nextTier: this.next,
      score: this.score,
      danger: this.world.bodies.some(b => b.y < PgFruitCakeWorld.DangerLineY),
      done: this.phase === 'gameover',
    };
  }

  private randTier(): number {
    return 1 + Math.floor(Math.random() * PgFruitCakeWorld.MaxDroppableTier); // droppable tiers 1..5
  }
}
