// Message contract between the FruitCake watch-mode director (main thread) and the AI worker
// (fruit-cake-ai.worker.ts). Types only — importing this from either side costs nothing at runtime, so the
// worker's code never leaks into the main bundle.
//
// Why a worker at all: the depth-3 search costs 784 clone+settle rollouts and ~3920 net forward passes per
// drop, which measured 0.97–5.7 s of *blocked main thread* when it ran inside the rAF callback — the whole
// tab froze once per drop (M53, `FRUITCAKE_WATCH_AI_STALL_PRD.md`).

/** One fruit as sent across the wire. Matches what `PgFruitCakeWorld.clone()` actually preserves: the
 *  search discards angle/angularVel (it clones rotation-off), so sending them would be dead weight. */
export interface AiBody {
  tier: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** Main thread → worker. */
export type AiRequest =
  | { type: 'search'; id: number; bodies: AiBody[]; current: number; next: number };

/** Worker → main thread. */
export type AiResponse =
  /** The net finished loading (or failed, in which case the worker uses the greedy fallback). */
  | { type: 'ready'; hasNet: boolean }
  /** The column to drop in, answering the `search` with the same `id`. */
  | { type: 'result'; id: number; column: number };
