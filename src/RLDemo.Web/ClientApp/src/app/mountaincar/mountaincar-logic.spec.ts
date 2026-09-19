import { describe, expect, it } from 'vitest';

import { GOAL, MAX_POS, MAX_SPEED, MAX_STEPS, MIN_POS, MountainCarGame } from './mountaincar-logic';

// M63.x — the client-side MountainCar physics that HUMAN play runs on (PRD §7.1).
//
// The AI drives the `.pg` twin (`mountaincar_solver.ts`, covered separately); this is the hand-written
// mirror of it, and the whole point of the file is that the two obey the same dynamics. So the
// assertions here are on the invariants the env promises — bounded state, the left-wall rule, the
// truncation cap — rather than on hand-computed trajectories.
//
// `reset()` randomises the start position, so nothing below depends on where the car begins.

describe('reset', () => {
  it('starts at rest somewhere in the valley, with a clean episode', () => {
    for (let trial = 0; trial < 50; trial++) {
      const game = new MountainCarGame();

      expect(game.position).toBeGreaterThanOrEqual(-0.6);
      expect(game.position).toBeLessThanOrEqual(-0.4);
      expect(game.position).toBeLessThan(GOAL);
      expect(game.velocity).toBe(0);
      expect(game.steps).toBe(0);
      expect(game.done).toBe(false);
      expect(game.reachedGoal).toBe(false);
    }
  });
});

describe('step', () => {
  it('keeps position and velocity inside the track bounds under sustained left thrust', () => {
    const game = new MountainCarGame();

    // Action 0 is "push left" — far more steps than it takes to reach the left wall, so this
    // asserts the clamps hold rather than that one step happens to stay in range.
    for (let i = 0; i < MAX_STEPS; i++) {
      game.step(0);

      expect(game.position).toBeGreaterThanOrEqual(MIN_POS);
      expect(game.position).toBeLessThanOrEqual(MAX_POS);
      expect(Math.abs(game.velocity)).toBeLessThanOrEqual(MAX_SPEED);
      // The left wall is inelastic: the car cannot keep pressing into it.
      if (game.position === MIN_POS) expect(game.velocity).toBeGreaterThanOrEqual(0);
    }
  });

  it('truncates at the step cap without reaching the goal', () => {
    const game = new MountainCarGame();

    // Idling (action 1) cannot climb out of the valley, so the episode can only end by hitting the
    // cap — which separates truncation from termination.
    for (let i = 0; i < MAX_STEPS; i++) game.step(1);

    expect(game.steps).toBe(MAX_STEPS);
    expect(game.done).toBe(true);
    expect(game.reachedGoal).toBe(false);
    expect(game.position).toBeLessThan(GOAL);
  });

  it('is inert once the episode is over', () => {
    const game = new MountainCarGame();
    for (let i = 0; i < MAX_STEPS; i++) game.step(1);
    const { position, velocity, steps } = game;

    game.step(2);

    expect(game.position).toBe(position);
    expect(game.velocity).toBe(velocity);
    expect(game.steps).toBe(steps);
  });

  it('terminates at the goal, and flags it as a goal rather than a timeout', () => {
    const game = new MountainCarGame();
    // Placed just short of the flag at full speed: one step carries it across.
    game.position = GOAL - MAX_SPEED / 2;
    game.velocity = MAX_SPEED;

    game.step(2);

    expect(game.position).toBeGreaterThanOrEqual(GOAL);
    expect(game.done).toBe(true);
    expect(game.reachedGoal).toBe(true);
    expect(game.steps).toBeLessThan(MAX_STEPS);
  });

  it('accelerates right and decelerates left from the same state', () => {
    // The three actions differ only by the applied force, so from an identical state the ordering
    // of the resulting velocities is the whole contract of the action space.
    const velocityAfter = (action: number): number => {
      const game = new MountainCarGame();
      game.position = -0.5;
      game.velocity = 0;
      game.step(action);
      return game.velocity;
    };

    expect(velocityAfter(2)).toBeGreaterThan(velocityAfter(1));
    expect(velocityAfter(1)).toBeGreaterThan(velocityAfter(0));
  });

  it('counts every step it takes', () => {
    const game = new MountainCarGame();

    for (let i = 1; i <= 10; i++) {
      game.step(1);
      expect(game.steps).toBe(i);
    }
    expect(game.done).toBe(false);
  });
});
