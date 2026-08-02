/// <reference lib="webworker" />
import { PgDuelingNet, PgFruitCakeWorld } from './fruitcake_solver';
import { loadFruitCakeNet } from './fruitcake-net';
import type { AiDrop, AiRequest, AiResponse } from './fruit-cake-ai-protocol';

// The FruitCake AI plays the game here, off the main thread and ahead of the animation
// (M53 — see FRUITCAKE_WATCH_AI_STALL_PRD.md).
//
// This worker is the AUTHORITY: it owns the board, the net, and the fruit sequence, and it runs the same
// think → settle loop the director used to run — but with no rAF pacing, so a settle costs microseconds
// instead of the seconds it takes to watch. It stays LOOK_AHEAD drops in front of the viewer; the main
// thread merely replays what already happened. Since the next drop is decided while the previous one is
// still falling, a 1–5.7 s search never reaches the screen.
//
// Everything below runs the generated single-source solver unchanged. `fruitcake_solver.pg` is NOT touched
// by M53: it is shared with the C# training path under a bitwise-parity guarantee.

const DEPTH = 3, TOPK = 5, TOPK2 = 2; // unchanged from the inline path — the worker exists so we can KEEP these
const STEP = 1 / 60;                  // physics sub-step, identical to the replay's
const LOOK_AHEAD = 4;                 // decided drops held in front of the viewer

/** Minimal local shape for the worker global. This workspace's `lib` is DOM-only (adding "webworker" would
 *  collide with DOM in the same program), so we describe just what we use — the same idiom as
 *  `screen-wake-lock.ts`. esbuild strips it either way; this keeps the editor honest. */
const ctx = globalThis as unknown as {
  postMessage(message: AiResponse): void;
  addEventListener(type: 'message', listener: (event: { data: AiRequest }) => void): void;
};

const world = new PgFruitCakeWorld(true); // rotation ON so the authority matches the world being replayed
let net: PgDuelingNet | null = null;
let netReady = false;

let gen = 0;
let produced = 0;
let acked = 0;
let score = 0;
let over = true;      // no game until the first reset
let current = randTier();
let next = randTier();
let scheduled = false;

void loadFruitCakeNet().then(loaded => {
  net = loaded; // null ⇒ the greedy fallback keeps it playing, exactly as the inline path did
  netReady = true;
  schedule();
});

ctx.addEventListener('message', event => {
  const msg = event.data;
  if (msg.type === 'reset') {
    gen++;
    world.clear();
    produced = 0;
    acked = 0;
    score = 0;
    over = false;
    current = randTier();
    next = randTier();
    schedule();
  } else if (msg.type === 'ack') {
    acked = Math.max(acked, msg.index + 1);
    schedule();
  }
});

/** Produce drops one macrotask at a time, so `ack`/`reset` are still delivered between them. A drop costs
 *  1–5.7 s of solid compute; without the yield the worker would never read its own inbox. */
function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  setTimeout(() => { scheduled = false; pump(); }, 0);
}

function pump(): void {
  if (over || !netReady) return;
  if (produced - acked >= LOOK_AHEAD) return; // full — an `ack` will wake us
  produceDrop();
  schedule();
}

function produceDrop(): void {
  const column = net
    ? world.chooseColumn(net, current, next, DEPTH, TOPK, TOPK2)
    : fallbackColumn(current);

  world.spawnFruit(current, PgFruitCakeWorld.columnX(column, current), PgFruitCakeWorld.heldY(current));

  // The settle rule is the director's, verbatim — the replay reproduces this loop step for step, so the
  // two must agree on when to stop. `substeps` is the contract that keeps them in lockstep.
  let substeps = 0;
  for (;;) {
    score += world.step(STEP);
    substeps++;
    const quiet = substeps >= PgFruitCakeWorld.MinSettleSubsteps &&
      world.maxSpeed() < PgFruitCakeWorld.SettleSpeedPx;
    if (quiet || substeps >= PgFruitCakeWorld.MaxSubsteps) break;
  }

  const lost = world.anyEjected() || world.anyRestingAboveDangerLine(PgFruitCakeWorld.RestSpeedPx);
  const drop: AiDrop = {
    gen,
    index: produced,
    tier: current,
    nextTier: next,
    column,
    substeps,
    scoreAfter: score,
    lost,
    snapshot: world.bodies.map(b => ({ tier: b.tier, x: b.x, y: b.y, angle: b.angle, vx: b.vx, vy: b.vy })),
  };
  produced++;
  if (lost) over = true; // stop here; the main thread will `reset` after showing the game-over board
  else { current = next; next = randTier(); }
  ctx.postMessage({ type: 'drop', drop });
}

function randTier(): number {
  return 1 + Math.floor(Math.random() * PgFruitCakeWorld.MaxDroppableTier); // droppable tiers 1..5
}

/** Greedy one-drop fallback for a missing/unreadable checkpoint: the non-losing column with the most
 *  immediate merge points, tie-broken by the lower resulting pile. */
function fallbackColumn(tier: number): number {
  let best = Math.floor(PgFruitCakeWorld.Width / 2 / (PgFruitCakeWorld.Width / PgFruitCakeWorld.ColumnCount));
  let bestPts = -1;
  let bestPile = Number.POSITIVE_INFINITY;
  for (let c = 0; c < PgFruitCakeWorld.ColumnCount; c++) {
    const sim = world.clone(false);
    sim.spawnFruit(tier, PgFruitCakeWorld.columnX(c, tier), PgFruitCakeWorld.heldY(tier));
    const pts = sim.settleAfterDrop(
      PgFruitCakeWorld.SettleSpeedPx, PgFruitCakeWorld.MinSettleSubsteps, PgFruitCakeWorld.MaxSubsteps, STEP);
    if (sim.anyEjected() || sim.anyRestingAboveDangerLine(PgFruitCakeWorld.RestSpeedPx)) continue;
    const pile = sim.pileHeight();
    if (pts > bestPts || (pts === bestPts && pile < bestPile)) { best = c; bestPts = pts; bestPile = pile; }
  }
  return best;
}
