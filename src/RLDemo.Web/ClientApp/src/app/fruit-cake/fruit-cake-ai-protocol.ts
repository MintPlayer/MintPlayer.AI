// Message contract between the FruitCake watch-mode director (main thread) and the AI worker
// (fruit-cake-ai.worker.ts). Types only — importing this from either side costs nothing at runtime, so the
// worker's code never leaks into the main bundle.
//
// The worker owns the GAME, not just the search (M53.2, `FRUITCAKE_WATCH_AI_STALL_PRD.md`): it plays ahead
// with no animation pacing and streams decided drops, while the main thread animates them a few drops
// behind. The animation delay was only ever a presentation constraint — the AI never needed it. That is why
// the viewer never waits for a search: by the time a fruit finishes falling, the next one is already decided.
//
// The wire format is deliberately tiny. The physics is deterministic single-source code (`fruitcake_solver`,
// transpiled from the `.pg`), so the main thread reproduces the worker's world by REPLAYING the same drop
// for the same number of sub-steps — no trajectory is transmitted. `substeps` is what makes the replay
// exact: the two sides must stop stepping at the same instant, or their worlds diverge from that drop on.

/** A settled body, as sent for the per-drop drift check. */
export interface AiSnapshotBody {
  tier: number;
  x: number;
  y: number;
  angle: number;
  vx: number;
  vy: number;
}

/** One decided drop: what to spawn, where, and exactly how long to animate it. */
export interface AiDrop {
  /** Bumped on every reset; the main thread discards drops from a game it has already abandoned. */
  gen: number;
  index: number;
  /** The fruit being dropped, and the one after it (HUD preview). */
  tier: number;
  nextTier: number;
  column: number;
  /** Sub-steps the authoritative world took to settle. The replay MUST run exactly this many. */
  substeps: number;
  /** Authoritative score once this drop has settled — the HUD follows the AI's reality, not the replay's. */
  scoreAfter: number;
  /** This drop ended the game. */
  lost: boolean;
  /** The settled board, for the drift check (insurance — the replay should already match bit for bit). */
  snapshot: AiSnapshotBody[];
}

/** Main thread → worker. */
export type AiRequest =
  /** Start (or restart) a game. Also the signal that begins production. */
  | { type: 'reset' }
  /** The main thread has finished animating this drop — release one slot of look-ahead. */
  | { type: 'ack'; index: number };

/** Worker → main thread. */
export type AiResponse = { type: 'drop'; drop: AiDrop };
