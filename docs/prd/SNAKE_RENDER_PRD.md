# Snake — curved-tube rendering (client-side view layer) — PRD

**Status:** planned · 2026-07-10 · branch `m35-snake-tube` (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M35 · **Depends on:** M33 (client-side Snake) — purely cosmetic, **no AI/logic change**

## 1. Problem

The Snake board renders as **flat coloured squares on a CSS grid**. Each of the 144 cells is a fixed `<div>`; every
tick the component recolours the head / body / food cells (`snake.ts` `cellClass()` + `render()`, `snake.html`
`@for` grid, `snake.scss` `.cell/.body/.head/.food`). Movement is an **instant per-tick snap** (`setInterval` 120 ms
watch / 150 ms human — no `requestAnimationFrame`, no interpolation). The result is functional but reads as a 1980s
blocky snake: disconnected squares, hard 90° elbows, no head, no sense of a body gliding.

We want the snake rendered as a **smooth, connected, curved tube** — a rounded body that flows around corners, a
distinct head oriented to its direction of travel (eyes, optional tongue flick), a tapering tail, and light 3-D
shading — and to **glide** between grid ticks rather than teleport.

## 2. Goal & success criteria

Replace the Snake **view layer only** with a curved-tube renderer, leaving all game logic and the AI untouched.

- **Visual gate (human judgement, in-browser):** the snake reads as one continuous rounded tube that curves smoothly
  around turns (no visible square segments, no sharp elbows); it has a recognisable head facing where it moves and a
  tapered tail; motion between ticks is smooth (glides one cell per tick, no snapping); food reads as a distinct piece.
  Both **Watch AI** and **Play yourself** modes look identical and correct.
- **Zero logic/AI change:** `snake-logic.ts`, `snake-director.ts`, `snake_solver.ts`, `snake-net.ts` and the `.pg`
  source are **not touched**. Food@12 strength (M34, ~81) is unaffected — this is presentation only.
- **Performance:** a steady 60 fps for a snake up to the full board (~144 segments) on a mid-range laptop; no
  measurable regression to the AI cadence (the game loop stays on its `setInterval`; only *drawing* moves to rAF).
- **Crisp on hi-DPI:** sharp on retina / fractional `devicePixelRatio` displays (no blur).
- **Graceful teardown:** the rAF loop stops with the game (`stop()` / component destroy) — no leaked animation frames
  or wake-lock interaction changes.

**Non-goals.** No WebGL/3-D engine; no change to board size (12×12), tick rate, controls, or scoring; no new AI
behaviour; no sprite assets (everything drawn procedurally on Canvas 2D — keeps it dependency-free and themeable).

## 3. Key decision — Canvas 2D over the untouched logic (design settled by a 2-agent investigation, 2026-07-10)

Two agents mapped the current renderer and researched tube-drawing techniques. Findings:

**The current view is DOM/CSS-grid, not Canvas.** So this is *introducing* a `<canvas>`, not editing canvas code —
which is the clean case: the game logic already centralises all display state through one `render(body, food, eaten)`
call, and the body is a flat `number[]` of cell indices (head-first), so `col = i % 12`, `row = Math.floor(i / 12)`
maps directly to pixel centres. The view swap needs **no logic change**.

**Canvas 2D is the right tool — not SVG, not WebGL.** For one snake of ≤144 segments at 60 fps this is a trivial
draw load. SVG (rewriting a `<path>` `d` 60×/s) thrashes the DOM/layout pipeline and is slower for a
redrawn-every-frame loop, and its stroke can't taper any more than Canvas can. WebGL only wins at scales we don't have
(hundreds of snakes / true per-pixel lighting) and costs shaders + mesh triangulation for no visible gain here. Canvas
2D with a spline centreline + multi-pass stroke is the sweet spot of realism-per-effort.

**The four techniques that produce the look:**

1. **Catmull-Rom spline centreline → cubic Bézier.** Treat the (interpolated) cell centres as an interpolating
   spline's control points and emit each span as one `bezierCurveTo`. The load-bearing conversion for a span
   `p1→p2` with neighbours `p0,p3`: `cp1 = p1 + (p2−p0)/6`, `cp2 = p2 − (p3−p1)/6` (uniform Catmull-Rom; a
   `tension` knob `(1−tension)/6` tightens the curve toward a polyline). Clamp the endpoints (duplicate first/last)
   so the curve terminates exactly on head and tail. The snake never wraps (it dies at the wall) → always a simple
   open chain, no discontinuities.
2. **Cornering for free.** Because the spline passes through cell centres and blends toward neighbour directions, a
   turn becomes a natural quarter-curve — no elbow, no per-corner special-casing. `tension` (~0.2–0.4) is the single
   "how curvy" dial if tight S-turns bulge past the cell band.
3. **Taper + head.** A stroked path is uniform width, so the tail taper needs either **stamped filled circles** with
   a radius profile (simplest, ~90% as good) or a **variable-width ribbon polygon** — offset each sampled centreline
   point by `±radius` along the local normal into one closed `fill()` (crispest, also gives a clean outline). The
   **head** is drawn last, on top: `ctx.rotate(atan2(dir))` to face travel, an elongated blob, two eyes, an optional
   tongue flick — all in head-local coords so the diagonal angles that interpolation produces just work.
4. **Interpolation = the biggest single visual win.** At 120–150 ms/tick the logic snaps a whole cell. A
   `requestAnimationFrame` loop animates `p ∈ [0,1]` across each tick and builds the point list from **two**
   consecutive states: the head point lerps old→new head; the tail point lerps old→new tail **unless growing** (on
   the eat-tick the tail stays put and the snake visibly lengthens). rAF only *reads* the latest snapshot — the game
   loop stays on `setInterval`, so no logic timing changes. Clamp `p` at 1 so a throttled/background frame parks the
   snake exactly on-cell (dovetails with the existing `visibilitychange` handling).

Shading (cheap 3-D, all optional polish, ordered by payoff): a dark **outline**; a **multi-pass stroke** (same path
stroked 3× wide→narrow, dark→light, faking a cylindrical cross-section); a low-alpha **glossy highlight** line; a
cached head→tail **gradient**. Avoid `shadowBlur` (the one genuinely expensive Canvas 2D feature) — fake any glow with
a wide low-alpha pass.

## 4. Design — where the code goes

- **New `snake-renderer.ts`** — a pure view class `SnakeTubeRenderer`, no game logic. Owns the `<canvas>` context,
  `devicePixelRatio` backing-store scaling (`canvas.width = boardPx*dpr; ctx.scale(dpr,dpr)`), the `cell` size, the
  `center(i)` index→pixel map, `buildPoints(next, prev, p, growing)`, the spline/ribbon/head draw, and the rAF loop.
  Exposes something like `push(body, food, eaten)` (snapshot + kick the loop) and `stop()`.
- **`snake.html`** — replace the `@for` grid of `<div>`s with a single `<canvas #board width="480" height="480">`.
- **`snake.scss`** — drop `.cell/.body/.head/.food/.board` grid styles; the board is now the canvas element (keep the
  rounded dark frame). Cell-index → pixel note: the current 480 px box bakes in 4 px padding + 1 px gaps, so
  `cell = 480/12 = 40 px` is approximate — **drop the padding** on canvas (cleaner) so `cell` is exact.
- **`snake.ts`** — construct the renderer once the canvas exists (`afterNextRender`/`viewChild`), route the existing
  `render(body, food, eaten)` writes into `renderer.push(...)` (signals may stay for the score readout), and call
  `renderer.stop()` from `stop()` / `DestroyRef`. **No change** to the tick `setInterval`s, the director, or the
  keyboard handler.
- **Untouched:** `snake-logic.ts`, `snake-director.ts`, `snake_solver.ts`, `snake-net.ts`, `snake_solver.pg`.

**Colours (reuse today's palette so the page's identity holds):** body `#4caf82`, head bright `#7cffb2`, food
`#ff6b6b`, board frame `#1c2230`, empty background `#2a3142`. The multi-pass tube shading derives its rim/spine
shades from the body green.

## 5. Risks / watch-items

- **Per-frame allocations.** `buildPoints`/spline sampling allocate arrays every frame; at 60 fps × ~800 points this
  is GC pressure. Reuse scratch arrays; cache gradients/patterns (never `createLinearGradient` in the loop).
- **Growing-tick detection.** The tail-lerp must be suppressed exactly on the eat-tick (`eaten` increased) or the tail
  visibly stutters. Drive it off the `foodEaten` delta already available to `render()`.
- **Head/tail source-of-truth.** `snake-logic.ts` stores body head-first; the director already reverses
  `PgSnakeEnv`'s head-at-end body to head-first in `frame()`. The renderer consumes head-first uniformly — verify both
  modes feed the same orientation (they do, via the shared `render()` signature).
- **Death frame.** On death the loop should draw the final resting snake and stop cleanly (Watch-AI auto-restarts;
  human shows game-over) — no half-interpolated frame left on screen.
- **Verification is visual.** There is no numeric gate; confirm in-browser via the running dev server (the ASP.NET
  host proxies the Angular dev server — **do not** run `ng serve`/`ng build`; save files and let it live-reload).
  Capture a before/after screenshot (or short clip) of both modes for the PR.

## 6. Rollout

Single small PR (view-only, no logic diff, no model change, no server change). Because nothing in the AI or physics
moves, the M34 strength numbers and all tests stand as-is; the review is a visual before/after plus a quick 60 fps /
full-board sanity check. Ships to `master` behind the same page — no feature flag needed.
