import { describe, expect, it } from 'vitest';

import { PgMountainCarEnv, clampF } from './mountaincar_solver';

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
