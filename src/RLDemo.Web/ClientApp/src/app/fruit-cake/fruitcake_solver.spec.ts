import { describe, expect, it } from 'vitest';

import { PgFruitCakeWorld, byTier } from './fruitcake_solver';

// The browser-side half of the `.pg` coverage union for FruitCake.
//
// `fruitcake_solver.ts` is transpiled from `fruitcake_solver.pg`, the same source that produces the
// C# the training backend runs, and coverage from both targets is overlaid onto the `.pg` line
// numbers. The world is constructed with ROTATION ON in every browser path — `fruit-cake-physics.ts`
// (`new PgFruitCakeWorld(true)`, "fruit roll during human play"), the director and the AI worker —
// and with rotation OFF in training, where spin is dead weight the planner would have to simulate.
//
// So the angular half of the integrator is browser-only: the `angle += angularVel * dt` /
// `angularVel *= AngularDamping` block inside `step`, and the `1 / inertia` branch of `spawnFruit`
// that gives a body a non-zero inverse inertia in the first place. This file drives exactly that,
// and asserts the two properties that make rolling look right: spin INTEGRATES into the angle, and
// it DAMPS towards rest rather than persisting forever.

const DT = 1 / 60;
const TIER = 1;
const RADIUS = byTier(TIER).radiusPx;

/** Well clear of every wall and of the floor, so nothing but gravity and spin acts on the body. */
const FREE_X = PgFruitCakeWorld.Width / 2;
const FREE_Y = 100;

describe('PgFruitCakeWorld rotation (browser-only physics)', () => {
  it('only gives a body rotational inertia when the world was built with rotation on', () => {
    const spinning = new PgFruitCakeWorld(true).spawnFruit(TIER, FREE_X, FREE_Y);
    const flat = new PgFruitCakeWorld(false).spawnFruit(TIER, FREE_X, FREE_Y);

    expect(spinning.invI).toBeGreaterThan(0);
    expect(flat.invI).toBe(0);
    // Training's default: no rotation unless asked for.
    expect(new PgFruitCakeWorld().rotation).toBe(false);
  });

  it('integrates angular velocity into the angle, and damps it, on every step', () => {
    const world = new PgFruitCakeWorld(true);
    const body = world.spawnFruit(TIER, FREE_X, FREE_Y);
    body.angularVel = 1.2;

    expect(world.step(DT)).toBe(0); // a lone fruit merges with nothing

    expect(body.angle).toBeCloseTo(1.2 * DT, 12);
    expect(body.angularVel).toBeCloseTo(1.2 * PgFruitCakeWorld.AngularDamping, 12);
  });

  it('leaves angle and spin alone when rotation is off — the training world is unaffected', () => {
    const world = new PgFruitCakeWorld(false);
    const body = world.spawnFruit(TIER, FREE_X, FREE_Y);
    body.angularVel = 1.2; // forced in by hand; nothing in a rotation-off world can produce it

    world.step(DT);

    expect(body.angle).toBe(0);
    expect(body.angularVel).toBe(1.2);
  });

  it('advances the angle monotonically while the spin decays towards zero', () => {
    const world = new PgFruitCakeWorld(true);
    const body = world.spawnFruit(TIER, FREE_X, FREE_Y);
    const w0 = 2.5;
    body.angularVel = w0;

    const steps = 30;
    for (let i = 0; i < steps; i++) {
      const angle = body.angle;
      const spin = body.angularVel;
      world.step(DT);
      expect(body.angle).toBeGreaterThan(angle);        // positive spin -> the angle advances
      expect(body.angularVel).toBeLessThan(spin);       // ...and the spin is strictly bled off
      expect(body.angularVel).toBeGreaterThan(0);       // damping decays, it never flips the sign
    }

    // Geometric decay: one damping factor applied per step.
    expect(body.angularVel).toBeCloseTo(w0 * PgFruitCakeWorld.AngularDamping ** steps, 10);
    // Still in free fall, so nothing but the integrator touched the angle.
    expect(body.y).toBeLessThan(PgFruitCakeWorld.Height - RADIUS);
    expect(body.y).toBeGreaterThan(FREE_Y);
  });

  it('turns floor friction into spin — a fruit skidding along the bottom starts rolling', () => {
    // Overlapping the floor by a pixel so the contact exists on the very first step, with a
    // sideways velocity so the tangential (friction) impulse has something to bite on.
    const drop = (rotation: boolean) => {
      const world = new PgFruitCakeWorld(rotation);
      const body = world.spawnFruit(TIER, FREE_X, PgFruitCakeWorld.Height - RADIUS + 1);
      body.vx = 300;
      world.step(DT);
      return body;
    };

    const rolling = drop(true);
    const sliding = drop(false);

    expect(rolling.angularVel).not.toBe(0);
    expect(rolling.angle).not.toBe(0);
    // Rolling the way it is travelling: moving right (+x) spins clockwise, i.e. a growing angle.
    expect(Math.sign(rolling.angularVel)).toBe(1);
    // Without rotational inertia the same impulse produces no spin at all.
    expect(sliding.angularVel).toBe(0);
    expect(sliding.angle).toBe(0);
  });

  it('clones a rotating world with its spin, and a non-rotating one without', () => {
    const world = new PgFruitCakeWorld(true);
    const body = world.spawnFruit(TIER, FREE_X, FREE_Y);
    body.angle = 0.7;
    body.angularVel = -1.3;
    body.vx = 12;
    body.vy = -4;

    const withSpin = world.clone(true);
    const withoutSpin = world.clone(false);

    expect(withSpin.count).toBe(1);
    expect(withSpin.bodies[0].angle).toBe(0.7);
    expect(withSpin.bodies[0].angularVel).toBe(-1.3);
    expect(withSpin.bodies[0].invI).toBeGreaterThan(0);
    expect(withSpin.bodies[0].vx).toBe(12);
    expect(withSpin.bodies[0].vy).toBe(-4);

    // The planner's world drops the spin deliberately: it is state it will never integrate.
    expect(withoutSpin.bodies[0].angle).toBe(0);
    expect(withoutSpin.bodies[0].angularVel).toBe(0);
    expect(withoutSpin.bodies[0].invI).toBe(0);

    // ...and neither clone aliases the original.
    expect(withSpin.bodies[0]).not.toBe(body);
    expect(body.angle).toBe(0.7);
  });
});
