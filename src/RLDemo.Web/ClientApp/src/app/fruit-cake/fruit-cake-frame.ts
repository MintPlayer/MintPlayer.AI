// The render-frame shape for FruitCake's AI "watch" mode. The client-side FruitCakeDirector produces these and
// fruit-cake-render.renderFrame consumes them. (Formerly the server-stream DTO in fruit-cake-api.ts; the server
// path was retired in M32 — the AI runs entirely in the browser now, and since M53 the search runs in a Web
// Worker that plays ahead of the animation.)

/** One fruit in a render frame. */
export interface FruitFrameItem {
  x: number;
  y: number;
  angle: number;
  tier: number;
}

/** One render frame of the AI FruitCake game. */
export interface FruitCakeFrame {
  fruit: FruitFrameItem[];
  heldTier: number;
  nextTier: number;
  score: number;
  danger: boolean;
  done: boolean;
}
