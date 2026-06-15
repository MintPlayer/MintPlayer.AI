// Pure client-side MountainCar physics for HUMAN play (PRD §7.1: human mode is client-driven on a JS timer).
// Mirrors MountainCarEnv's dynamics exactly so the human drives the same car the AI does.

export const MIN_POS = -1.2;
export const MAX_POS = 0.6;
export const MAX_SPEED = 0.07;
export const GOAL = 0.5;
const FORCE = 0.001;
const GRAVITY = 0.0025;
export const MAX_STEPS = 200;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export class MountainCarGame {
  position = -0.5;
  velocity = 0;
  steps = 0;
  done = false;
  reachedGoal = false;

  constructor() { this.reset(); }

  reset(): void {
    this.position = -0.6 + Math.random() * 0.2; // U[-0.6, -0.4]
    this.velocity = 0;
    this.steps = 0;
    this.done = false;
    this.reachedGoal = false;
  }

  /** action: 0 = push left, 1 = none, 2 = push right. */
  step(action: number): void {
    if (this.done) return;
    let v = this.velocity + (action - 1) * FORCE + Math.cos(3 * this.position) * -GRAVITY;
    v = clamp(v, -MAX_SPEED, MAX_SPEED);
    let p = clamp(this.position + v, MIN_POS, MAX_POS);
    if (p <= MIN_POS && v < 0) v = 0;
    this.position = p;
    this.velocity = v;
    this.steps++;
    if (p >= GOAL) { this.done = true; this.reachedGoal = true; }
    else if (this.steps >= MAX_STEPS) { this.done = true; }
  }
}
