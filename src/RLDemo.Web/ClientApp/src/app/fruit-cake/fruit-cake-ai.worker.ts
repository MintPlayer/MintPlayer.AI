/// <reference lib="webworker" />
import { PgDuelingNet, PgFruitCakeWorld } from './fruitcake_solver';
import { loadFruitCakeNet } from './fruitcake-net';
import type { AiBody, AiRequest, AiResponse } from './fruit-cake-ai-protocol';

// The FruitCake AI, off the main thread (M53 — see FRUITCAKE_WATCH_AI_STALL_PRD.md).
//
// This worker owns the net and answers "which column?". It runs the SAME generated search
// (`fruitcake_solver.ts`, transpiled from the single-source `.pg`) the director used to run inline, so the
// column it returns is identical — the only thing that changed is which thread pays for it.
//
// Reconstruction note: a search only ever reads `tier/x/y/vx/vy`. `chooseColumn` evaluates every candidate
// through `clone(false)`, which zeroes angle and angularVel and rebuilds each body's inverse inertia from
// the *clone's* rotation flag — so the parent world's rotation setting cannot reach the result. That is why
// the wire format omits orientation and why this world is built rotation-off without changing any outcome.

const DEPTH = 3, TOPK = 5, TOPK2 = 2; // unchanged from the inline path — the worker exists so we can KEEP these

/** Minimal local shape for the worker global. This workspace's `lib` is DOM-only (adding "webworker" would
 *  collide with DOM in the same program), so we describe just what we use — the same idiom as
 *  `screen-wake-lock.ts`. esbuild strips it either way; this keeps the editor honest. */
const ctx = globalThis as unknown as {
  postMessage(message: AiResponse): void;
  addEventListener(type: 'message', listener: (event: { data: AiRequest }) => void): void;
};

let net: PgDuelingNet | null = null;

void loadFruitCakeNet().then(loaded => {
  net = loaded; // null ⇒ the greedy fallback keeps it playing, exactly as the inline path did
  ctx.postMessage({ type: 'ready', hasNet: net !== null });
});

ctx.addEventListener('message', event => {
  const msg = event.data;
  if (msg.type !== 'search') return;
  const world = rebuild(msg.bodies);
  const column = net
    ? world.chooseColumn(net, msg.current, msg.next, DEPTH, TOPK, TOPK2)
    : fallbackColumn(world, msg.current);
  ctx.postMessage({ type: 'result', id: msg.id, column });
});

function rebuild(bodies: AiBody[]): PgFruitCakeWorld {
  const world = new PgFruitCakeWorld(false);
  for (const b of bodies) {
    const body = world.spawnFruit(b.tier, b.x, b.y);
    body.vx = b.vx;
    body.vy = b.vy;
  }
  return world;
}

/** Greedy one-drop fallback for a missing/unreadable checkpoint: the non-losing column with the most
 *  immediate merge points, tie-broken by the lower resulting pile. Moved here from the director so the
 *  net-missing path is off the main thread too — it is 14 full rollouts, which blocked just as visibly. */
function fallbackColumn(world: PgFruitCakeWorld, current: number): number {
  let best = Math.floor(PgFruitCakeWorld.Width / 2 / (PgFruitCakeWorld.Width / PgFruitCakeWorld.ColumnCount));
  let bestPts = -1;
  let bestPile = Number.POSITIVE_INFINITY;
  for (let c = 0; c < PgFruitCakeWorld.ColumnCount; c++) {
    const sim = world.clone(false);
    sim.spawnFruit(current, PgFruitCakeWorld.columnX(c, current), PgFruitCakeWorld.heldY(current));
    const pts = sim.settleAfterDrop(
      PgFruitCakeWorld.SettleSpeedPx, PgFruitCakeWorld.MinSettleSubsteps, PgFruitCakeWorld.MaxSubsteps, 1 / 60);
    if (sim.anyEjected() || sim.anyRestingAboveDangerLine(PgFruitCakeWorld.RestSpeedPx)) continue;
    const pile = sim.pileHeight();
    if (pts > bestPts || (pts === bestPts && pile < bestPile)) { best = c; bestPts = pts; bestPile = pile; }
  }
  return best;
}
