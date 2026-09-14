# Snake — render the planned Hamiltonian cycle on the board — PRD

**Status:** planned · 2026-09-14 · branch `m60-snake-cycle-overlay` (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M60 · **Depends on:** M48 (`SNAKE_HAMILTONIAN_PRD.md`), M35 (renderer, `SNAKE_RENDER_PRD.md`)

## 1. Problem

M48 shipped **"Watch AI (Hamiltonian cycle)"**: the snake permanently holds a full-board cycle it provably
cannot die on, rebuilds it on every food spawn to route straight at the food, and lets the net rank the safe
shortcuts along it. Every game ends board-full.

**But the cycle is invisible.** On screen it looks like an ordinary snake that happens never to die — the one
thing that makes the mode interesting (there is a rail underneath, covering every cell, and the snake is only
ever allowed to move forward along it) is not shown. `SNAKE_HAMILTONIAN_PRD.md` §4.5 already named this as a
stretch — *"faint overlay drawing the current cycle — great for demoing why it never dies"* — and M48.3 shipped
without it (PLAN M48.3: *"cycle overlay stretch not built"*). This PRD is that stretch, promoted to a milestone.

A second, smaller problem surfaced while specifying the first and is fixed here rather than deferred (owner's
one-PR rule): **Snake is the only game page whose board does not shrink on a phone.** `snake.scss:20` is a hard
`width: 480px; height: 480px` with no `max-width` and no media query, while `crazy-fruits.scss:42`
(`width: min(620px, 100%)`) and `tetris.scss:34` (`width: calc(min(560px, 100%) * var(--zoom, 1))`) both scale.

## 2. Goal & success criteria

Draw the currently-held cycle on the board during the Hamiltonian mode, so a viewer can see the plan, see the
corridor the snake is allowed to move inside, and watch the plan get redrawn.

- **Visible plan:** the whole cycle is drawn as a thin line through every cell; the arc from the head forward to
  the food is drawn brighter; the moment the cycle changes is legible on screen.
- **Hard gate — zero AI change.** `snake_solver.pg`, the C# twin, the `snake-net.ckpt` checkpoint and both
  `chooseAction*` paths are **untouched**. `git diff` against `master` must show no change under
  `src/MintPlayer.AI.ReinforcementLearning.Environments/` (`snake_solver.ts` is gitignored build output,
  `.gitignore:52` — the `.pg` is the file that must not move). Same seed ⇒ same trajectory.
- **Hard gate — the other two modes keep identical drawing logic.** "Watch AI" (search) and "Play yourself" run
  exactly the same tube/eyes/food/grid code paths; the overlay is unreachable outside `watch-cycle`. Their
  canvas *dimensions* now follow the viewport (§5.2) — deliberate, and the same behaviour every other game page
  already has. This gate is about logic, not pixels; it was written as "pixel-identical" and is narrowed here
  because §5.2 makes that promise false on narrow screens.
- **Perf gate — no permanent animation loop.** M35's renderer parks its rAF loop once a tick has played out
  (`p ≥ 1`). The rebuild flash (§5.3) may keep it alive, but only in **bounded, self-terminating** runs of
  ~300 ms. Verified as 60 fps with no long tasks over ≥30 s of live watching.
- **Verified live** in the running host at 1280×800 **and** 390×844, by screenshot.

## 3. Decisions (owner, 2026-09-14 — design interview)

Settled before implementation, so there are no open taste calls. Each records what was rejected and why.

### 3.1 The bright arc is a **corridor**, not a route

Layer r1 is the cycle arc from the head forward to the food. It is tempting — and was the first draft's
wording — to call it "the stretch the snake is about to traverse". **The code says otherwise.**
`chooseActionCycle` (`snake_solver.ts:605–653`) scores candidates as `d * progressWeight + rootQ[a] * netWeight`
with `W_CYCLE_PROGRESS = 1000` against `W_CYCLE_NET = 50` (`snake-director.ts:33–34`), where `d` is how far
forward along the cycle a move jumps. So while `freeCount() > 72` the policy takes the **largest legal `d`** on
essentially every tick and the net only breaks ties — shortcutting is the default, not an occasional flourish.
The snake will visibly cut *across* r1 rather than follow it.

What rescues r1 is that shortcuts are bounded: `d > dFood` is rejected, so a jump can never pass the food.
**r1 is a genuine invariant — head and food are both on it, and the snake provably stays inside it.** The page
copy must say *corridor it is allowed to move inside*, never *path it will follow*.

**Accepted consequence:** `dFood` can be ~140 right after `initCycle` (or after a run of failed rebuilds), so r1
can cover almost the whole loop and r2 nearly vanishes. That is *true* — the food really is that far ahead in
cycle order — and it is accepted rather than capped. *(Rejected: capping r1 to a fixed ~24-cell lookahead. It
would read as a stable "what's next" at every moment, but it throws away the one thing r1 gets right — that
there is a guaranteed enclosure around the food — and the cap length would be an arbitrary number chosen
without seeing a board.)*

### 3.2 Rebuild: snap the geometry, flash r1 once

`tryRebuildCycle` runs inside `chooseActionCycle`, so a successful rebuild swaps the entire loop within one
`step()`. The geometry **snaps** (no cross-fade, no keeping the old loop alive) and r1 **flashes once** at full
brightness, easing back to its resting alpha over ~300 ms.

**Be honest about what the flash means:** *the plan changed*, not *it ate*. A failed rebuild leaves
`cycleFoodTarget` stale and is **retried every tick** until it succeeds, so in the early game the flash
coincides with eating, and in the late game — where rebuilds fail under fragmentation — it lands several ticks
later, or not at all until the next food. *(Rejected: a cross-fade of old→new loop — same unpredictable timing,
but needs the previous cycle retained and doubles the drawn geometry for 200 ms. Rejected: marking the event in
side-panel text only — zero canvas cost, but nobody watching a canvas is reading the sidebar.)*

### 3.3 Two flat strokes, both 5 CSS px, distinguished only by colour

| | colour | source |
|---|---|---|
| **r1** — head → food | `rgba(255, 212, 121, 0.45)` | `#ffd479`, already in `snake.scss` as `.banner.training` |
| **r2** — the rest of the loop | `#3a4154` | the `hair` token from `lunar-lockout-render.ts:65` |

Both 5 CSS px — the owner's call, and deliberately far below the tube's `cell * 0.72` (~29 px on desktop): the
overlay is a hairline plan, not a second snake. Width therefore cannot carry the r1/r2 distinction; colour does.

**No gradient, no dash, no glow.** `lunar-lockout-render.ts:4` states the house rule outright — *"Flat fills, no
gradients, no shadows, no glow"* — and `snake-renderer.ts` contains no gradient either. The first draft specced
a head→food alpha gradient to make direction readable without animation; that justification does not survive
contact with the board, because r1's endpoints are **already marked** — the head has eyes, the food is a red
dot — so direction is inferable from geometry alone. The flash (§3.2) reuses r1's own token at alpha 1.0, so
the overlay introduces exactly two colours, both pre-existing.

The `0.45` is deliberate and pre-emptive: solid `#ffd479` across a 140-cell early-game r1 is a board-spanning
gold ribbon that out-shouts a 29 px green tube, inverting the point of the feature. Alpha is a *uniform* stroke,
so it is still flat. **Expect to tune this number on the live host — it is the one value in this document chosen
without seeing it rendered.**

### 3.4 Fix the responsive board, and keep the stroke at exactly 5 CSS px

The board gets `width: min(480px, 100%)` + `aspect-ratio: 1`, matching the two phone-verified pages. The
renderer then **resizes its backing store to the real CSS width** (`tetris.ts:147–154` / `crazy-fruits.ts:106–113`
pattern), so drawing units are CSS pixels again and `lineWidth = 5` is literally 5 px with no compensation term.

*(Rejected, after being provisionally chosen and then reversed on evidence: keeping a frozen 480-unit board and
compensating the stroke as `5 * (480 / cssWidth)` via a `ResizeObserver`. It preserves the other two modes'
output down to the pixel on every screen size — but **every** responsive canvas in this repo resizes its backing
store, so it would have been the only one of its kind, and it keeps an arithmetic term that is the most likely
thing in the overlay to be subtly wrong.)*

**Accepted consequence:** the 5 px stroke is now the one thing on the board that does not scale with everything
else. On a phone it sits against a tube that shrank to ~23 px, so the hint line carries proportionally more
visual weight exactly where there is least room. That is inherent to "always exactly 5 px", not to the chosen
implementation. If it looks fat on the phone screenshot, the honest fix is abandoning the constant-px rule.

### 3.5 Visibility

On by default, with a **"Show planned route"** checkbox in the side panel to hide it. The overlay is the
explanation of the mode, so it must be on when the mode starts; the checkbox exists so the clean M35 board is
one click away. Rendered only while `mode() === 'watch-cycle'`. It does not persist across sessions, matching
every other game page.

## 4. What already exists (no new engine work)

The data is **already reachable from the frontend** — the overlay itself is a pure view-layer change.

| thing | where | note |
|---|---|---|
| `cycle: number[]` | `snake_solver.ts:106` (public field on `PgSnakeEnv`) | closed loop of cell indices, in travel order |
| `cycleIndex: number[]` | `snake_solver.ts:107` | cell → position on the cycle (−1 = off-cycle) |
| lazily built | `snake_solver.ts:605–608` (`initCycle` inside `chooseActionCycle`) | empty until the first cycle-mode move; `reset()` empties it again on a new game |
| rebuilt per food — **and retried on any tick** | `snake_solver.ts:609–613` → `tryRebuildCycle` (`:724`) | `this.cycle = newCycle` (`:864`) — a **new array reference**, a free change-detector. A *failed* rebuild leaves `cycleFoodTarget` stale and retries next tick |
| shortcut scoring | `snake_solver.ts:637–650` | `d * 1000 + Q * 50` ⇒ largest legal forward jump wins; see §3.1 |
| always full-board | `.pg:798` rejects partial coverage (`SNAKE_HAMILTONIAN_PRD.md` §4.3 step 5) | ⇒ `cycle.length === SIZE*SIZE` whenever it is non-empty |
| **closed** loop | `initCycle` / `tryRebuildCycle` | `cycle` stores positions only — the wrap segment `cycle[n-1] → cycle[0]` is implicit and **must be stroked**, or the loop reads as an open path |
| renderer seam | `snake-renderer.ts:64` `push(body, food, eaten)` | the one per-tick entry point |
| draw order | `snake-renderer.ts:127` `draw()` | `clear()` (bg + faint grid) → food → tube |
| frozen dimensions | `snake-renderer.ts:33–34` (`readonly cell`, `readonly boardPx`) | become mutable in §5.2 |
| cell → pixel | `snake-renderer.ts:98` `center(i)` | `((i % size + 0.5) * cell, (⌊i/size⌋ + 0.5) * cell)` |
| rounded-corner polyline | `snake-renderer.ts:174` `tubePath()` | reusable for the overlay at a smaller corner radius |
| **flat-fill house rule** | `lunar-lockout-render.ts:4` | *"Flat fills, no gradients, no shadows, no glow"* — see §3.3 |
| **closest overlay precedent** | `lunar-lockout-render.ts:371–403` `drawHints()` | dashed route lines between cell centres + landing rings, `ctx.save()` / `setLineDash` / `restore()` |
| **backing-store resize precedent** | `tetris.ts:147–154`, `crazy-fruits.ts:106–113` | `cssW = canvas.clientWidth` → `canvas.width = round(cssW * dpr)` — the §5.2 pattern |
| `ResizeObserver` precedent | `block-dude-render.ts:86`, `lunar-lockout-render.ts:139`, `rush-hour.ts:150` | all guard with `typeof ResizeObserver !== 'undefined'` |
| responsive board precedent | `crazy-fruits.scss:42`, `tetris.scss:34` | `min(Npx, 100%)` + `aspect-ratio` |
| planner-state-to-view precedent | `chess-director.ts:97–116` + `chess.ts:201–234` | planner state surfaced director → view; the seam shape M60.1 copies |
| checkbox precedent | `rush-hour.html:109`, `game-2048.html:112` | plain native `<input>` + signal setter, no forms module |

**No `.pg` edit, no transpile, no regeneration, no C# change, no new npm dependency.**

## 5. Design

### 5.1 Director seam — `snake-director.ts`

`SnakeAiFrame` (`:39`) gains two fields:

```ts
cycle: number[] | null;   // raw travel-order loop; null in 'search' mode
cycleEpoch: number;       // bumped whenever the cycle changes; drives the view's cache and the flash
```

The director keeps `lastCycleRef` **and `lastCycleLen`**, bumping `cycleEpoch` when either changes. Reference
identity alone fails in both directions: `initCycle` mutates in place via `length = 0` + `push` (so the first
build keeps the old reference) and `reset()` empties the same array on a new game. `tryRebuildCycle` *does*
assign a fresh array (`:864`), which the reference check catches — including on a non-food tick.

**The frame must hold a `slice()` copy, not the live array.** `this.core.cycle` is engine-owned and mutated in
place; handing the renderer a live reference would let a rebuild tear a frame mid-rAF. Copy once per epoch.

In `'search'` mode `cycle` is `null` and `cycleEpoch` stays `0`; the `chooseActionSearch` branch is untouched.

### 5.2 Responsive board + backing-store resize — `snake.scss`, `snake-renderer.ts`

- `snake.scss:20` → `width: min(480px, 100%); height: auto; aspect-ratio: 1;`
- `cell` and `boardPx` stop being `readonly`. A `syncSize()` reads `canvas.clientWidth`, and when
  `round(cssW * dpr)` differs from `canvas.width` it reassigns `canvas.width`/`height`, re-applies the transform
  with `setTransform(dpr, 0, 0, dpr, 0, 0)` (assigning `canvas.width` resets context state — the current
  `ctx.scale(dpr, dpr)` in the constructor would silently be lost), and recomputes `boardPx = cssW`,
  `cell = cssW / size`.
- Called from a `ResizeObserver` (guarded, per the three existing uses) and defensively at the top of `draw()`,
  mirroring `tetris.ts:153`.
- **The cached overlay `Path2D` (§5.3) is in board coordinates, so it must be keyed on `cell` as well as
  `cycleEpoch`** — otherwise a viewport change leaves stale geometry on screen at the wrong scale.

This is the one part of M60 that touches all three modes, and it is why §2's gate reads "identical drawing
logic", not "pixel-identical".

### 5.3 Overlay rendering — `snake-renderer.ts`

`push()` gains `cycle` and `cycleEpoch`; `Snapshot` carries them. The overlay is drawn in `draw()` **between
`clear()` and the food**, so it passes *behind* the tube.

**Why under the tube is correct, not just cheap:** by the cycle invariant the body occupies a contiguous run
*behind* the head in cycle order, so **r1 always lies on non-body cells and is never occluded**. r2 contains the
whole body and is progressively swallowed by the tube as the snake grows — which is exactly right: it is the
part of the loop already covered.

Drawn as full-loop-then-overdraw, which is the same picture as "r1 and r2" but keeps the expensive path cached:

1. **Full loop at r2's colour.** A `Path2D` built once per `(cycleEpoch, cell)`: the **closed** polyline through
   all `SIZE*SIZE` cell centres — including the implicit wrap segment `cycle[n-1] → cycle[0]` — corners rounded
   via the existing `tubePath` logic at `cell * 0.22`. Stroked flat at 5 px in `#3a4154`.
2. **r1 overdrawn.** Walk the cycle forward (**wrapping**) from the head's position to the food's, building the
   sub-path fresh each tick (≤ 144 points). Stroked flat at 5 px in `#ffd479` at the current flash alpha.

**Flash:** the renderer records `flashT0 = performance.now()` when the incoming `cycleEpoch` differs from the
last seen. `alpha = 0.45 + 0.55 * easeOut(1 − age / 300)` while `age < 300`, else `0.45`. The rAF loop's park
condition becomes `if (p < 1 || flashing)`. `stop()` clears `flashT0`.

Edge cases: `cycle` null/empty ⇒ draw nothing (search mode, and the first tick before `initCycle`); head or food
off-cycle (`index < 0`, not expected under full coverage) ⇒ skip r1 only, never throw.

### 5.4 Component & control — `snake.ts`, `snake.html`, `snake.scss`

- `showRoute = signal(true)`; `render()` (`snake.ts:114`) forwards `f.cycle`/`f.cycleEpoch`, passing `null` when
  `!showRoute()`, so hiding is a data decision and the renderer stays a pure function of its snapshot.
- The setter re-pushes the last snapshot, so toggling repaints immediately instead of waiting up to 120 ms.
- `snake.html`: under `.status`, inside `@if (mode() === 'watch-cycle')`, a native
  `<label><input type="checkbox" …> Show planned route</label>`.
- Intro copy gains one sentence, using §3.1's wording: the gold line is the **corridor the snake is allowed to
  move inside**, and it redraws whenever the plan changes.

## 6. Milestones

Single PR (`m60-snake-cycle-overlay`), four commits. Sizing lands **before** the overlay so the overlay is
written against final coordinates.

- **M60.1 — Director seam.** `SnakeAiFrame.cycle` / `.cycleEpoch`, epoch detection (ref **and** length),
  defensive `slice()`. *Gate:* cycle mode plays identically; search-mode frames carry `cycle: null`.
- **M60.2 — Responsive board + backing-store resize.** `snake.scss` `min(480px, 100%)` + `aspect-ratio`;
  mutable `cell`/`boardPx`, `syncSize()`, `setTransform`, guarded `ResizeObserver`. No overlay yet.
  *Gate:* all three modes render correctly and undistorted at 1280×800 and 390×844; no page-level horizontal
  scroll on the phone width; drawing logic untouched.
- **M60.3 — Overlay rendering + rebuild flash.** Both strokes, `(cycleEpoch, cell)`-keyed `Path2D`, wrap
  segment, bounded flash keepalive. *Gate:* r1/r2 visible and correctly coloured; the loop visibly redraws on
  rebuild; **search mode and human play still run the unchanged tube/food/grid paths**; rAF parks within
  ~300 ms of the last rebuild.
- **M60.4 — Control + live verification.** Checkbox, immediate repaint, intro copy. *Gate:* `playwright_node`
  MCP against the user's already-running host at 1280×800 and 390×844 — overlay on/off screenshots, ≥30 s
  watching at 60 fps with no long tasks; `docs/ARCHITECTURE.md` Snake section updated.

## 7. Non-goals

- **Any AI, engine, `.pg`, C#, checkpoint or training change** — hard gate, §2.
- An overlay for the **search** mode: there is no cycle there. Visualizing the M34 look-ahead's principal
  variation is a different feature.
- Human play (`snake-logic.ts`) — it has no plan to draw.
- Server work — Snake has been fully client-side since M33.
- Capping or truncating r1 (§3.1), cross-fading the rebuild (§3.2), gradients or dashes (§3.3).
- Step-through / scrubbing of the cycle rebuild (a `game-2048`-style playback control).

## 8. Risks

- **"It isn't following the line."** r1 is a corridor the snake shortcuts *within* (§3.1); a viewer told it is a
  *route* will read every shortcut as a bug. Mitigated by copy, and by the fact that the shortcut is visibly
  bounded by the corridor. This is the single most likely piece of negative feedback.
- **Degenerate r1.** Right after `initCycle` — the first thing a visitor sees — r1 can be ~97% of the loop and
  r2 nearly nothing. Accepted (§3.1); the flash and the subsequent rebuild resolve it within a food or two.
- **Gold out-shouting the snake.** Mitigated by alpha `0.45` chosen up front; tune live, first thing in M60.3.
- **Stale `Path2D` after a resize.** Keyed on `(cycleEpoch, cell)` (§5.2) — the specific bug this avoids is a
  full-board loop drawn at the old scale after a rotate or window drag.
- **Lost context transform.** Assigning `canvas.width` resets the 2D context; §5.2's `setTransform` is the fix.
  Missing it produces a board that renders at 1× on a hi-DPI screen after the first resize only.
- **Scope creep into the engine.** The temptation is to expose "the path the search *would* take" too. Out of
  scope (§7); the value of this milestone is that it is view-only and cannot regress the shipped AI.

## 9. Untouched

`snake_solver.pg`, `snake_solver.ts`, `snake-net.ts`, `snake-net.ckpt`, `snake-logic.ts`, every C# file under
`src/MintPlayer.AI.ReinforcementLearning.Environments/Snake/`, the Lab, all training/campaign code.
