import { describe, expect, it } from 'vitest';

import { PgSnakeBeamNode, PgSnakeEnv, PgSnakeNet } from './snake_solver';

// The browser-side half of the `.pg` coverage union for Snake.
//
// `snake_solver.ts` is transpiled from `snake_solver.pg`, the same source that produces the C# the
// training backend runs, and coverage from both targets is overlaid onto the `.pg` line numbers. This
// file exists for the lines the C# side structurally cannot reach:
//
//  * The M34 receding-horizon BEAM PLANNER (`chooseActionSearch` and everything it pulls in —
//    `clone`, `simSpawnFood`, `freeSpaceAhead`, `firstLegalAction`, `leafScoreSearch`, `pruneBeam`).
//    It is serving-only: training plays the reactive masked-greedy policy, the browser plans. The
//    real call site is `snake-director.ts` (`chooseActionSearch(net, depth, beam, wFood, wTrap,
//    wNet, wSpace, wDist, wRatio)`), and the argument shapes here mirror it.
//  * `PgSnakeNet.forward`'s hidden-layer loop and `relu`. The C# fixture net has ZERO hidden layers,
//    so the loop body never executes there; the shipped checkpoint the browser loads has several.
//  * The safe-mask branch of `currentActionMask`. The director hard-codes the mask OFF (the planner
//    supersedes the 1-ply shield), so only a test constructs an env with it on.
//
// Everything is asserted as an invariant — legality, deep-copy independence, the ordering of a
// terminal leaf against a survivable one — rather than against constants copied out of the twin.
// Boards are 6x6 and depths are shallow: the flood fill is O(cells^2) per node.

const SIZE = 6;
const STEP_PENALTY = -0.01;

// Director weights (snake-director.ts). Their exact values are tuning, not contract, so nothing below
// asserts on them; they are here so the planner runs on the same scale the browser actually uses.
const W_FOOD = 10_000;
const W_TRAP = 50_000;
const W_NET = 50;
const W_SPACE = 50;
const W_DIST = 1;
const W_RATIO = 100_000;

const INPUT = PgSnakeEnv.ObservationSize;
const ACTIONS = PgSnakeEnv.ActionCount;
const HIDDEN = [3];

/** Deterministic weights — an LCG, so there is no `Math.random` anywhere in this file. */
function weights(count: number, seed: number): number[] {
  let s = seed >>> 0;
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out.push(s / 4294967296 - 0.5);
  }
  return out;
}

/**
 * A net with a real hidden layer. Shapes follow `PgSnakeNet.linear`'s indexing
 * (`w[wOff + i * outDim + o]`), i.e. row-major [in][out].
 */
function makeNet(seed = 12345): PgSnakeNet {
  return new PgSnakeNet(
    INPUT,
    ACTIONS,
    HIDDEN,
    weights(INPUT * HIDDEN[0], seed),
    weights(HIDDEN[0], seed + 1),
    weights(HIDDEN[0] * 1, seed + 2),
    weights(1, seed + 3),
    weights(HIDDEN[0] * ACTIONS, seed + 4),
    weights(ACTIONS, seed + 5),
  );
}

/** Reset, then put the food on the cell immediately right of the head. */
function newEnv(safeMask = false): PgSnakeEnv {
  const env = new PgSnakeEnv(SIZE, STEP_PENALTY, safeMask);
  env.reset();
  // `spawnFood` counts free cells, not raw cells. The body occupies three cells in the middle row,
  // so the 20th free cell (index 19) is the one just past the head.
  env.spawnFood(19);
  return env;
}

/**
 * A 4x4 board holding a 9-cell boustrophedon snake (tail 0 -> head 8). Free cells: 7, so no
 * reachable region can ever be as large as the body.
 */
function boxedIn(safeMask: boolean): PgSnakeEnv {
  const env = new PgSnakeEnv(4, STEP_PENALTY, safeMask);
  env.reset();
  env.body.length = 0;
  for (let i = 0; i < env.cells; i++) env.occupied[i] = false;
  for (const cell of [0, 1, 2, 3, 7, 6, 5, 4, 8]) {
    env.body.push(cell);
    env.occupied[cell] = true;
  }
  env.food = 15;
  return env;
}

describe('PgSnakeNet.forward (hidden layers + relu)', () => {
  it('produces one Q value per action through a hidden layer', () => {
    const q = makeNet().forward(newEnv().buildObservation());

    expect(q).toHaveLength(ACTIONS);
    for (const v of q) expect(Number.isFinite(v)).toBe(true);
  });

  it('is deterministic for the same observation', () => {
    const net = makeNet();
    const obs = newEnv().buildObservation();

    expect(net.forward(obs)).toEqual(net.forward(obs));
  });

  it('floors negative hidden pre-activations at zero, so a saturated-negative trunk matches a zero trunk', () => {
    // relu is only observable through what it erases. With the trunk weights zeroed, every hidden
    // unit's pre-activation is exactly its bias — so a strongly NEGATIVE bias must give the same
    // output as a zero bias (both relu to 0), while a positive one must not.
    const obs = newEnv().buildObservation();
    const trunkW = new Array<number>(INPUT * HIDDEN[0]).fill(0);
    const valueW = weights(HIDDEN[0], 7);
    const valueB = [0.25];
    const advW = weights(HIDDEN[0] * ACTIONS, 8);
    const advB = [0.5, -1.5, 0.75, 2.0];
    const withBias = (b: number) =>
      new PgSnakeNet(INPUT, ACTIONS, HIDDEN, trunkW, new Array<number>(HIDDEN[0]).fill(b),
        valueW, valueB, advW, advB).forward(obs);

    expect(withBias(-5)).toEqual(withBias(0));
    expect(withBias(3)).not.toEqual(withBias(0));
  });

  it('aggregates duelling heads as advantage + (value - mean advantage)', () => {
    // Silence the hidden layer (relu(-5) = 0), so the value head collapses to its bias and the
    // advantage head to its own — which makes the aggregation itself, and only it, observable.
    const advB = [0.5, -1.5, 0.75, 2.0];
    const valueB = [0.25];
    const q = new PgSnakeNet(
      INPUT, ACTIONS, HIDDEN,
      new Array<number>(INPUT * HIDDEN[0]).fill(0),
      new Array<number>(HIDDEN[0]).fill(-5),
      weights(HIDDEN[0], 7), valueB,
      weights(HIDDEN[0] * ACTIONS, 8), advB,
    ).forward(newEnv().buildObservation());

    const meanAdv = advB.reduce((a, b) => a + b, 0) / ACTIONS;
    for (let a = 0; a < ACTIONS; a++) expect(q[a]).toBeCloseTo(advB[a] + (valueB[0] - meanAdv), 12);
  });
});

describe('PgSnakeEnv.spawnFood / simSpawnFood', () => {
  it('spawns on the pick-th FREE cell, skipping the body', () => {
    const env = new PgSnakeEnv(SIZE, STEP_PENALTY, false);
    env.reset();
    env.spawnFood(0);
    expect(env.occupied[env.food]).toBe(false);
    expect(env.food).toBe(0);

    // The body sits on three consecutive middle-row cells, so free cell #19 is the first one past it.
    env.spawnFood(19);
    expect(env.occupied[env.food]).toBe(false);
    expect(env.food).toBe(env.headCell() + 1);
  });

  it('places search-time food on the first free cell (deterministic, no RNG inside the planner)', () => {
    const env = newEnv();
    env.simSpawnFood();

    expect(env.food).toBe(0);
    expect(env.occupied[env.food]).toBe(false);
  });
});

describe('PgSnakeEnv.clone', () => {
  it('is a deep copy — mutating the clone leaves the original untouched', () => {
    const env = newEnv();
    const before = env.body.slice();
    const copy = env.clone();

    copy.step(3);
    copy.step(0);

    expect(env.body).toEqual(before);
    expect(env.foodEaten).toBe(0);
    expect(env.done).toBe(false);
    expect(copy.body).not.toEqual(before);
    expect(copy.occupied).not.toBe(env.occupied);
    expect(copy.body).not.toBe(env.body);
  });

  it('carries the state the planner scores on', () => {
    const env = newEnv();
    env.step(3); // eat the adjacent food
    const copy = env.clone();

    expect(copy.size).toBe(env.size);
    expect(copy.food).toBe(env.food);
    expect(copy.foodEaten).toBe(env.foodEaten);
    expect(copy.heading).toBe(env.heading);
    expect(copy.body).toEqual(env.body);
    expect(copy.occupied).toEqual(env.occupied);
  });
});

describe('PgSnakeEnv death', () => {
  it('dies on self-collision (stepping back into the neck)', () => {
    const env = newEnv();
    const head = env.headCell();
    const neck = env.neckCell();
    expect(neck).toBe(head - 1); // reset lays the body out left-to-right, heading right

    env.step(2); // left, straight into the neck — `step` does not consult the mask

    expect(env.done).toBe(true);
    expect(env.lastTerminated).toBe(true);
    expect(env.lastTruncated).toBe(false);
    expect(env.lastReward).toBe(PgSnakeEnv.DeathReward);
  });

  it('dies on the wall', () => {
    const env = newEnv();
    // Straight up from the middle row: three moves stay on the board, the fourth leaves it.
    for (let i = 0; i < 3; i++) {
      env.step(0);
      expect(env.done).toBe(false);
    }
    env.step(0);

    expect(env.done).toBe(true);
    expect(env.lastTerminated).toBe(true);
    expect(env.lastReward).toBe(PgSnakeEnv.DeathReward);
  });
});

describe('PgSnakeEnv.leafScoreSearch', () => {
  it('scores a dead leaf far below any survivable one', () => {
    const alive = newEnv();
    alive.step(3);
    const dead = newEnv();
    dead.step(2); // self-collision

    const score = (e: PgSnakeEnv, depth: number) =>
      e.leafScoreSearch(0, depth, W_FOOD, W_TRAP, W_SPACE, W_DIST, W_RATIO);

    expect(score(dead, 1)).toBeLessThan(score(alive, 1));
  });

  it('prefers dying LATER — a death deeper in the line is less bad', () => {
    const dead = newEnv();
    dead.step(2);

    const score = (depth: number) =>
      dead.leafScoreSearch(0, depth, W_FOOD, W_TRAP, W_SPACE, W_DIST, W_RATIO);

    expect(score(5)).toBeGreaterThan(score(1));
  });

  it('rewards eating: a leaf that gained food outscores the same leaf counted as no gain', () => {
    const eaten = newEnv();
    eaten.step(3);
    expect(eaten.foodEaten).toBe(1);

    const gained = eaten.leafScoreSearch(0, 1, W_FOOD, W_TRAP, W_SPACE, W_DIST, W_RATIO);
    const flat = eaten.leafScoreSearch(1, 1, W_FOOD, W_TRAP, W_SPACE, W_DIST, W_RATIO);

    expect(gained - flat).toBeCloseTo(W_FOOD, 6);
  });
});

describe('PgSnakeEnv.freeSpaceAhead / firstLegalAction', () => {
  it('reports the best reachable region over the four neighbours, bounded by the free cells', () => {
    const env = newEnv();
    const free = env.freeSpaceAhead();

    expect(free).toBeGreaterThan(0);
    // The flood fill counts the neighbour itself plus what it reaches, and the tail cell is
    // treated as free, so it can never exceed the free-cell count plus that one tail cell.
    expect(free).toBeLessThanOrEqual(env.freeCount() + 1);
  });

  it('returns an action the mask actually allows', () => {
    const env = newEnv();
    const a = env.firstLegalAction();

    expect(env.currentActionMask()[a]).toBe(true);
  });
});

describe('PgSnakeEnv.pruneBeam', () => {
  const node = (score: number) => new PgSnakeBeamNode(newEnv(), 0, score);

  it('returns the input untouched when it already fits in the beam', () => {
    const nodes = [node(1), node(2)];

    expect(PgSnakeEnv.pruneBeam(nodes, 4)).toBe(nodes);
    expect(PgSnakeEnv.pruneBeam(nodes, 2)).toBe(nodes);
  });

  it('keeps exactly the top-k by score, best first', () => {
    const nodes = [node(-3), node(10), node(0), node(7), node(-100)];

    const kept = PgSnakeEnv.pruneBeam(nodes, 3);

    expect(kept).toHaveLength(3);
    expect(kept.map(n => n.score)).toEqual([10, 7, 0]);
  });

  it('survives negative-only scores (the selector must not treat 0 as a floor)', () => {
    const kept = PgSnakeEnv.pruneBeam([node(-5), node(-1), node(-9)], 2);

    expect(kept.map(n => n.score)).toEqual([-1, -5]);
  });
});

describe('PgSnakeEnv.currentActionMask', () => {
  it('forbids reversing into the neck', () => {
    const env = newEnv();
    const mask = env.currentActionMask();

    // Heading right after reset, so "left" is the reversal.
    expect(mask[2]).toBe(false);
    expect(mask.filter(Boolean)).toHaveLength(3);
  });

  it('with the safe mask on, only keeps moves into a region big enough to hold the snake', () => {
    // The director disables this (the planner supersedes it), so this branch is browser-reachable
    // only through the constructor flag.
    const env = newEnv(true);
    const mask = env.currentActionMask();
    const head = env.headCell();
    const hr = Math.floor(head / SIZE);
    const hc = head % SIZE;
    const tail = env.tailCell();

    expect(mask.some(Boolean)).toBe(true);
    for (let a = 0; a < ACTIONS; a++) {
      if (!mask[a]) continue;
      const room = env.reachableFreeSpace(
        hr + PgSnakeEnv.drOf(a), hc + PgSnakeEnv.dcOf(a), tail);
      expect(room).toBeGreaterThanOrEqual(env.length);
    }
  });

  it('falls back to the plain mask when NO move leaves enough room', () => {
    // A 4x4 board carrying a 9-cell snake. Only 7 cells are free, so EVERY neighbour region is
    // strictly smaller than the body no matter how it is shaped — which forces the `any == false`
    // fallback. Without it the caller would be handed an all-false mask and no move at all.
    const mask = boxedIn(true).currentActionMask();
    const plain = boxedIn(false).currentActionMask();

    expect(mask.some(Boolean)).toBe(true);
    expect(mask).toEqual(plain);
  });
});

describe('PgSnakeEnv.chooseAction (masked greedy)', () => {
  it('returns an action the mask allows, and never the reversal', () => {
    const env = newEnv();
    const a = env.chooseAction(makeNet());

    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(ACTIONS);
    expect(env.currentActionMask()[a]).toBe(true);
  });

  it('picks the highest-Q legal action', () => {
    // A net whose advantages are a fixed ranking: action 2 (the reversal) is best but illegal, so
    // the greedy pick must be the best of the rest.
    const env = newEnv();
    const advB = [1, 2, 99, 3];
    const net = new PgSnakeNet(
      INPUT, ACTIONS, HIDDEN,
      new Array<number>(INPUT * HIDDEN[0]).fill(0),
      new Array<number>(HIDDEN[0]).fill(-5),
      new Array<number>(HIDDEN[0]).fill(0), [0],
      new Array<number>(HIDDEN[0] * ACTIONS).fill(0), advB,
    );

    expect(env.currentActionMask()[2]).toBe(false);
    expect(env.chooseAction(net)).toBe(3);
  });

  it('runs with the safe mask enabled too', () => {
    const env = newEnv(true);
    const a = env.chooseAction(makeNet());

    expect(env.currentActionMask()[a]).toBe(true);
  });
});

describe('PgSnakeEnv.chooseActionSearch (M34 beam planner)', () => {
  // Beam 2 is deliberately SMALLER than the branching factor (three legal moves per node), so
  // `pruneBeam` takes its top-k path on every ply instead of early-returning.
  const plan = (env: PgSnakeEnv, net: PgSnakeNet, depth = 4, beam = 2) =>
    env.chooseActionSearch(net, depth, beam, W_FOOD, W_TRAP, W_NET, W_SPACE, W_DIST, W_RATIO);

  it('returns a legal action', () => {
    const env = newEnv();
    const a = plan(env, makeNet());

    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(ACTIONS);
    expect(env.currentActionMask()[a]).toBe(true);
  });

  it('leaves the env it planned from completely unchanged', () => {
    // The planner clones the root; if it ever planned in place, the director's board would jump.
    const env = newEnv();
    const body = env.body.slice();
    const occupied = env.occupied.slice();

    plan(env, makeNet());

    expect(env.body).toEqual(body);
    expect(env.occupied).toEqual(occupied);
    expect(env.food).toBe(body[body.length - 1] + 1);
    expect(env.foodEaten).toBe(0);
    expect(env.done).toBe(false);
    expect(env.elapsedSteps).toBe(0);
  });

  it('is deterministic — same state, same plan', () => {
    const net = makeNet();

    expect(plan(newEnv(), net)).toBe(plan(newEnv(), net));
  });

  it('stays legal at every beam width, including ones wider than the branching factor', () => {
    const net = makeNet();
    for (const beam of [1, 2, 3, 8]) {
      const env = newEnv();
      const a = plan(env, net, 4, beam);
      expect(env.currentActionMask()[a]).toBe(true);
    }
  });

  it('stays legal at depth 1, where the beam is pruned before it is ever expanded', () => {
    const env = newEnv();
    const a = plan(env, makeNet(), 1, 2);

    expect(env.currentActionMask()[a]).toBe(true);
  });

  it('drives a whole game without ever choosing an illegal move', () => {
    // The end-to-end shape the director runs: plan, step, respawn food. Short and shallow, but it
    // is the only thing that exercises eating mid-search (`needsFood` -> `simSpawnFood`).
    const env = newEnv();
    const net = makeNet();
    let steps = 0;

    while (!env.done && steps < 25) {
      const a = plan(env, net, 3, 2);
      expect(env.currentActionMask()[a]).toBe(true);
      env.step(a);
      if (env.needsFood) env.simSpawnFood();
      steps++;
    }

    expect(steps).toBeGreaterThan(0);
    expect(env.body.length).toBeGreaterThanOrEqual(3);
  });

  it('ignores the net entirely when its weight is zero', () => {
    // W_NET only breaks ties between equally-safe roots; with it zeroed the plan must not depend
    // on the net at all, which pins the search score as the primary term.
    const envA = newEnv();
    const envB = newEnv();

    const a = envA.chooseActionSearch(makeNet(1), 4, 2, W_FOOD, W_TRAP, 0, W_SPACE, W_DIST, W_RATIO);
    const b = envB.chooseActionSearch(makeNet(999), 4, 2, W_FOOD, W_TRAP, 0, W_SPACE, W_DIST, W_RATIO);

    expect(a).toBe(b);
  });
});
