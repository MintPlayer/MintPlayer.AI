// Pure client-side Pendulum physics for HUMAN play (PRD §7.1: human mode is client-driven on a JS timer).
// Mirrors PendulumEnv's dynamics exactly so the human swings the same rod the AI does.

const G = 10;
const M = 1;
const L = 1;
const DT = 0.05;
export const MAX_TORQUE = 2;
export const MAX_SPEED = 8;
export const MAX_STEPS = 200;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export class PendulumGame {
  theta = Math.PI;   // start hanging down
  thetaDot = 0;
  steps = 0;
  done = false;

  constructor() { this.reset(); }

  reset(): void {
    this.theta = (Math.random() * 2 - 1) * Math.PI; // U[-π, π]
    this.thetaDot = Math.random() * 2 - 1;          // U[-1, 1]
    this.steps = 0;
    this.done = false;
  }

  /** torque: continuous, clamped to ±MAX_TORQUE. Semi-implicit Euler, matching the env. */
  step(torque: number): void {
    if (this.done) return;
    const u = clamp(torque, -MAX_TORQUE, MAX_TORQUE);
    let newThetaDot = this.thetaDot + (3 * G / (2 * L) * Math.sin(this.theta) + 3 / (M * L * L) * u) * DT;
    newThetaDot = clamp(newThetaDot, -MAX_SPEED, MAX_SPEED);
    this.theta += newThetaDot * DT;
    this.thetaDot = newThetaDot;
    this.steps++;
    if (this.steps >= MAX_STEPS) this.done = true;
  }

  /** Upright-ness in [0,1] for a simple human score (1 = perfectly balanced). */
  get upright(): number {
    return (Math.cos(this.theta) + 1) / 2;
  }
}
