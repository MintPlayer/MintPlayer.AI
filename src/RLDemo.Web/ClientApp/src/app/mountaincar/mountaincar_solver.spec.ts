import { describe, expect, it } from 'vitest';

import { PgMlpNet, PgMountainCarEnv, clampF } from './mountaincar_solver';

// M63.6 — the first frontend test in this repo, and deliberately aimed at a GENERATED file.
//
// `mountaincar_solver.ts` is transpiled from `mountaincar_solver.pg`, the same source that
// produces the C# the backend runs. Covering it here is the TypeScript half of the `.pg`
// coverage union (PRD §4): the C# side already attributes through `#line` pragmas, and this
// side attributes through the v3 source map Polyglot emits beside the twin.
//
// These assertions are on the transpiled contract, not on Angular. They need no TestBed, no
// DOM and no fixtures — which is why the solver twins are the right place to start a suite
// that has to exist from zero.

describe('clampF', () => {
  it('returns the value when it is already inside the range', () => {
    expect(clampF(0.5, 0, 1)).toBe(0.5);
  });

  it('clamps to the bounds', () => {
    expect(clampF(-3, -1.2, 0.6)).toBe(-1.2);
    expect(clampF(99, -1.2, 0.6)).toBe(0.6);
  });

  it('returns the bound itself when the value sits exactly on it', () => {
    expect(clampF(-1.2, -1.2, 0.6)).toBe(-1.2);
    expect(clampF(0.6, -1.2, 0.6)).toBe(0.6);
  });
});

describe('PgMountainCarEnv', () => {
  it('resets to the given start position, at rest', () => {
    const env = new PgMountainCarEnv(200, false);
    env.reset(-0.5);

    expect(env.position).toBe(-0.5);
    expect(env.velocity).toBe(0);
    expect(env.elapsedSteps).toBe(0);
    expect(env.done).toBe(false);
  });

  it('keeps position within the track bounds under sustained left thrust', () => {
    const env = new PgMountainCarEnv(500, false);
    env.reset(-0.5);

    // Action 0 is "push left". Far more steps than it takes to pin the car against the wall,
    // so this asserts the clamp holds rather than that one step happens to stay in range.
    for (let i = 0; i < 300; i++) env.step(0);

    expect(env.position).toBeGreaterThanOrEqual(PgMountainCarEnv.MinPosition);
    expect(env.position).toBeLessThanOrEqual(PgMountainCarEnv.MaxPosition);
    expect(Math.abs(env.velocity)).toBeLessThanOrEqual(PgMountainCarEnv.MaxSpeed);
  });

  it('truncates at maxEpisodeSteps without terminating', () => {
    const env = new PgMountainCarEnv(10, false);
    env.reset(-0.5);

    // Idling (action 1) cannot reach the goal from the valley, so the episode can only end by
    // hitting the step cap — which separates truncation from termination.
    for (let i = 0; i < 10; i++) env.step(1);

    expect(env.elapsedSteps).toBe(10);
    expect(env.done).toBe(true);
    expect(env.lastTruncated).toBe(true);
    expect(env.lastTerminated).toBe(false);
    expect(env.position).toBeLessThan(PgMountainCarEnv.GoalPosition);
  });

  it('normalises the observation rather than emitting raw state', () => {
    // The net is fed `(position + 0.3) / 0.9` and `velocity / MaxSpeed`, NOT the raw values —
    // so the two channels are centred near 0 and scaled to roughly unit range. Asserted through
    // the defining points rather than by pasting in computed constants: position -0.3 is the
    // centre of the track encoding, and MaxSpeed is by definition 1.0 in velocity units.
    const env = new PgMountainCarEnv(200, false);

    env.setState(-0.3, 0);
    expect(env.buildObservation()).toHaveLength(2);
    expect(env.buildObservation()[0]).toBeCloseTo(0, 12);
    expect(env.buildObservation()[1]).toBeCloseTo(0, 12);

    env.setState(0.6, PgMountainCarEnv.MaxSpeed);
    expect(env.buildObservation()[0]).toBeCloseTo(1, 12);
    expect(env.buildObservation()[1]).toBeCloseTo(1, 12);

    env.setState(-1.2, -PgMountainCarEnv.MaxSpeed);
    expect(env.buildObservation()[0]).toBeCloseTo(-1, 12);
    expect(env.buildObservation()[1]).toBeCloseTo(-1, 12);
  });
});

// ---------------------------------------------------------------------------------------------
// The policy net, and why it is here.
//
// `PgMlpNet` exists SOLELY so the browser can run the trained PPO policy client-side: the C#
// training path uses the real tensor library and never touches this hand-rolled matmul. So every
// line of it is dark in the C# coverage report, and no amount of backend testing will ever light
// it — this file is the only thing that can. That is the whole argument for the `.pg` coverage
// union in one class: two targets, two callers, two different halves of one source.
//
// Weights are chosen so every expected value is exact in binary floating point (halves and
// quarters), letting the assertions be equalities rather than tolerances wherever tanh is not
// involved.
// ---------------------------------------------------------------------------------------------

describe('PgMlpNet', () => {
  // 2 -> 2 identity-ish single layer: w = [[1,0],[0,1]] row-major, b = [0,0].
  const identity = () => new PgMlpNet([2, 2], [1, 0, 0, 1], [0, 0]);

  it('applies weights and biases as a row-major [in, out] matrix', () => {
    // w is laid out row-major [in, out], so w[i * outDim + o] is input i's contribution to output o.
    // Transposing it is the classic silent bug here: with a square matrix the shapes still line up
    // and the net simply computes the wrong thing, which is why the fixture is deliberately
    // asymmetric.
    const net = new PgMlpNet([2, 2], [1, 2, 4, 8], [0.5, -0.5]);

    // out[0] = 1*x0 + 4*x1 + 0.5 ; out[1] = 2*x0 + 8*x1 - 0.5
    expect(net.forward([1, 0])).toEqual([1.5, 1.5]);
    expect(net.forward([0, 1])).toEqual([4.5, 7.5]);
    expect(net.forward([1, 1])).toEqual([5.5, 9.5]);
  });

  it('returns one logit per output unit', () => {
    expect(new PgMlpNet([2, 3], [0, 0, 0, 0, 0, 0], [1, 2, 3]).forward([0, 0])).toEqual([1, 2, 3]);
  });

  it('leaves the output layer linear', () => {
    // A single-layer net is all output and no hidden, so tanh must NOT be applied — otherwise every
    // logit would be squashed into (-1, 1) and a bias of 9 could never outrank one of 3.
    const net = new PgMlpNet([1, 2], [0, 0], [9, 3]);

    expect(net.forward([0])).toEqual([9, 3]);
  });

  it('applies tanh to hidden layers but not to the output', () => {
    // [1, 1, 1]: one hidden unit, one output unit. Hidden pre-activation is 2, so the hidden
    // output is tanh(2); the output layer then scales it by 1 and adds 0, un-squashed.
    const net = new PgMlpNet([1, 1, 1], [2, 1], [0, 0]);

    const [out] = net.forward([1]);
    expect(out).toBeCloseTo(Math.tanh(2), 12);
    // The defining property, asserted rather than the constant: a hidden tanh bounds its own
    // output, so a large hidden pre-activation saturates instead of growing.
    const saturated = new PgMlpNet([1, 1, 1], [1000, 1], [0, 0]);
    expect(saturated.forward([1])[0]).toBeCloseTo(1, 9);
  });

  it('walks the weight and bias offsets forward across layers', () => {
    // The offset arithmetic (wOff += inDim * outDim, bOff += outDim) is the part most likely to
    // drift: a wrong stride reads another layer's weights and still produces plausible numbers.
    // 2 -> 2 -> 2, both layers identity, biases 0 then [1, 2]. If the second layer read the first
    // layer's biases the answer would be [tanh(1), tanh(1)] rather than the shifted pair.
    const net = new PgMlpNet([2, 2, 2], [1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 2]);

    const out = net.forward([1, 0]);
    expect(out[0]).toBeCloseTo(Math.tanh(1) + 1, 12);
    expect(out[1]).toBeCloseTo(Math.tanh(0) + 2, 12);
  });

  it('handles a net whose layers change width', () => {
    // 3 -> 2: guards the non-square case, where a transposed index would read out of bounds and
    // surface as NaN rather than as a wrong-but-finite answer.
    const net = new PgMlpNet([3, 2], [1, 0, 0, 1, 1, 1], [0, 0]);

    expect(net.forward([1, 2, 4])).toEqual([5, 6]);
    expect(net.forward([0, 0, 0])).toEqual([0, 0]);
  });

  it('does not mutate the input observation', () => {
    // forward() starts with `x = input` and reassigns rather than writing in place. If that ever
    // became an in-place write, the caller's observation would be corrupted between frames.
    const obs = [1, 0];
    identity().forward(obs);

    expect(obs).toEqual([1, 0]);
  });
});

describe('PgMountainCarEnv.chooseAction', () => {
  /** A 2 -> n net that ignores its input and returns `logits` verbatim, via the bias vector. */
  const constantLogits = (logits: number[]) =>
    new PgMlpNet([2, logits.length], new Array(2 * logits.length).fill(0), logits);

  it('picks the argmax of the policy logits', () => {
    const env = new PgMountainCarEnv(200, false);
    env.setState(-0.5, 0);

    expect(env.chooseAction(constantLogits([0, 0, 1]))).toBe(2);
    expect(env.chooseAction(constantLogits([0, 1, 0]))).toBe(1);
    expect(env.chooseAction(constantLogits([1, 0, 0]))).toBe(0);
  });

  it('keeps the first index on a tie', () => {
    // The scan is strictly-greater, so equal logits keep the earliest action. This mirrors the C#
    // PolicyAgent's greedy argmax; if the two disagreed on ties the browser and the backend would
    // play measurably different policies from identical weights.
    const env = new PgMountainCarEnv(200, false);
    env.setState(-0.5, 0);

    expect(env.chooseAction(constantLogits([1, 1, 1]))).toBe(0);
    expect(env.chooseAction(constantLogits([0, 5, 5]))).toBe(1);
  });

  it('handles negative logits', () => {
    // `best` starts at index 0 rather than at -Infinity, so an all-negative logit vector is the
    // case that would break a naive "track the max seen" implementation.
    const env = new PgMountainCarEnv(200, false);
    env.setState(-0.5, 0);

    expect(env.chooseAction(constantLogits([-5, -1, -3]))).toBe(1);
  });

  it('always returns an index within the action space', () => {
    // Driven from real observations across the whole track, through a real (if tiny) two-layer net
    // rather than a constant one -- this is the path the director actually takes each tick.
    const env = new PgMountainCarEnv(200, false);
    const net = new PgMlpNet([2, 3, 3], [1, -1, 0, 0, 1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]);

    for (const position of [-1.2, -0.6, -0.3, 0, 0.5]) {
      for (const velocity of [-PgMountainCarEnv.MaxSpeed, 0, PgMountainCarEnv.MaxSpeed]) {
        env.setState(position, velocity);
        const action = env.chooseAction(net);
        expect(Number.isInteger(action)).toBe(true);
        expect(action).toBeGreaterThanOrEqual(0);
        expect(action).toBeLessThan(3);
      }
    }
  });

  it('is a pure read of the environment state', () => {
    // Choosing must not step the world: the director calls chooseAction and step separately
    // (`this.core.step(this.core.chooseAction(this.net))`), so a side effect here would advance the
    // simulation twice per tick.
    const env = new PgMountainCarEnv(200, false);
    env.setState(-0.4, 0.01);

    env.chooseAction(constantLogits([0, 0, 1]));

    expect(env.position).toBe(-0.4);
    expect(env.velocity).toBe(0.01);
    expect(env.elapsedSteps).toBe(0);
  });
});
