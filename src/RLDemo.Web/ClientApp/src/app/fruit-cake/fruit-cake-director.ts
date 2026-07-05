import { PgDuelingNet, PgFruitCakeWorld } from './fruitcake_solver';
import { FruitCakeFrame } from './fruit-cake-api';
import { loadFruitCakeNet } from './fruitcake-net';

// Client-side "watch the AI" director — the whole AI now runs in the browser (PRD FRUITCAKE_CLIENT_SIDE_AI, M32).
// It drives the single-source physics (PgFruitCakeWorld) with the single-source depth-3 search (chooseColumn) over
// the trained net loaded from the shipped checkpoint — no server, no WebSocket. This replaces the old
// server-authoritative stream: the browser owns the physics AND the decision, so per-viewer server cost is zero.
//
// A small real-time state machine mirrors the old server loop: THINK (run the search, spawn) → SETTLE (step the
// physics to rest, animating the fall/merges in real time) → BETWEEN (brief pause) → repeat, with GAMEOVER on a
// lost board. The net loads asynchronously; until it's ready the board stays empty (LOADING).

type Phase = 'loading' | 'think' | 'settle' | 'between' | 'gameover';

const DEPTH = 3, TOPK = 5, TOPK2 = 2;   // serving search config (matches the retired C# controller)
const STEP = 1 / 60;                     // physics sub-step (matches the .pg settle loop)
const BETWEEN_S = 0.25;                  // pause between drops
const GAMEOVER_S = 1.8;                  // pause on the game-over board before restarting

export class FruitCakeDirector {
  private world = new PgFruitCakeWorld(true); // rotation on so fruit visibly roll (search clones rotation-off)
  private net: PgDuelingNet | null = null;
  private ready = false;
  private phase: Phase = 'loading';
  private current = this.randTier();
  private next = this.randTier();
  private score = 0;
  private substeps = 0;
  private acc = 0;
  private timer = 0;

  constructor() {
    void loadFruitCakeNet().then(net => {
      this.net = net; // null => the greedy fallback keeps it playing (as the server did with no net)
      this.ready = true;
      if (this.phase === 'loading') this.phase = 'think';
    });
  }

  /** Restart a fresh game (keeps the already-loaded net). */
  reset(): void {
    this.world.clear();
    this.score = 0;
    this.current = this.randTier();
    this.next = this.randTier();
    this.acc = 0;
    this.timer = 0;
    this.substeps = 0;
    this.phase = this.ready ? 'think' : 'loading';
  }

  /** Advance the AI game by real elapsed time (seconds). Called from the RAF loop while in watch mode. */
  update(dt: number): void {
    switch (this.phase) {
      case 'loading':
        if (this.ready) this.phase = 'think';
        return;

      case 'think': {
        // One synchronous search per drop (a brief "thinking" hitch — acceptable for a watch view).
        const col = this.net
          ? this.world.chooseColumn(this.net, this.current, this.next, DEPTH, TOPK, TOPK2)
          : this.fallbackColumn();
        this.world.spawnFruit(this.current, PgFruitCakeWorld.columnX(col, this.current), PgFruitCakeWorld.heldY(this.current));
        this.substeps = 0;
        this.acc = 0;
        this.phase = 'settle';
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

  /** A render frame in the same shape the old server stream used, so renderFrame() is unchanged. */
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

  // Greedy one-drop fallback used only if the checkpoint is missing (net === null): pick the non-losing column
  // with the most immediate merge points, tie-broken by the lower resulting pile. Mirrors the server's -pileHeight
  // heuristic intent so the game still plays sensibly without a net.
  private fallbackColumn(): number {
    let best = Math.floor(PgFruitCakeWorld.Width / 2 / (PgFruitCakeWorld.Width / PgFruitCakeWorld.ColumnCount));
    let bestPts = -1;
    let bestPile = Number.POSITIVE_INFINITY;
    for (let c = 0; c < PgFruitCakeWorld.ColumnCount; c++) {
      const sim = this.world.clone(false);
      sim.spawnFruit(this.current, PgFruitCakeWorld.columnX(c, this.current), PgFruitCakeWorld.heldY(this.current));
      const pts = sim.settleAfterDrop(PgFruitCakeWorld.SettleSpeedPx, PgFruitCakeWorld.MinSettleSubsteps, PgFruitCakeWorld.MaxSubsteps, STEP);
      if (sim.anyEjected() || sim.anyRestingAboveDangerLine(PgFruitCakeWorld.RestSpeedPx)) continue;
      const pile = sim.pileHeight();
      if (pts > bestPts || (pts === bestPts && pile < bestPile)) { best = c; bestPts = pts; bestPile = pile; }
    }
    return best;
  }
}
