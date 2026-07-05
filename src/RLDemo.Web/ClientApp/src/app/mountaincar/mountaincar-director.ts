import { PgMlpNet, PgMountainCarEnv } from './mountaincar_solver';
import { loadMountainCarNet } from './mountaincar-net';

// Client-side "watch the AI" director — the whole MountainCar AI runs in the browser (M33): the single-source
// PgMountainCarEnv dynamics + the PPO policy (PgMlpNet, argmax over logits) loaded from the shipped checkpoint.
// No server, no WebSocket. Discrete-tick (one policy move per tick), so the component drives it on an interval.

const MAX_STEPS = 200;      // v0 episode cap
const DEAD_HOLD_TICKS = 12; // hold the terminal frame briefly before restarting

export interface McAiFrame {
  position: number;
  done: boolean;
  reachedGoal: boolean;
}

export class MountainCarDirector {
  private readonly core = new PgMountainCarEnv(MAX_STEPS, false);
  private net: PgMlpNet | null = null;
  private ready = false;
  private deadHold = 0;

  constructor() {
    void loadMountainCarNet().then(n => {
      this.net = n; // null (missing checkpoint) → the car sits at the start; the checkpoint is shipped
      this.newGame();
      this.ready = true;
    });
  }

  private newGame(): void {
    this.core.reset(-0.6 + Math.random() * 0.2); // start ~U[-0.6,-0.4]; the browser owns the RNG
    this.deadHold = 0;
  }

  /** Advance one policy step. Returns the current frame, or null while the checkpoint is still loading. */
  step(): McAiFrame | null {
    if (!this.ready) return null;
    if (this.core.done) {
      if (this.deadHold > 0) { this.deadHold--; return this.frame(); }
      this.newGame();
      return this.frame();
    }
    if (this.net === null) return this.frame();

    this.core.step(this.core.chooseAction(this.net));
    if (this.core.done) this.deadHold = DEAD_HOLD_TICKS;
    return this.frame();
  }

  private frame(): McAiFrame {
    return { position: this.core.position, done: this.core.done, reachedGoal: this.core.position >= 0.5 };
  }
}
