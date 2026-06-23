# FruitCake — Product Requirements Document

> A "Suika"-style drop-and-merge physics game, now a **client-side game in the MintPlayer.AI
> playground** (the `RLDemo.Web` Angular SPA). It runs entirely in the browser — no backend, no AI.

- **Status:** Ported into RLDemo.Web · 2026-06-23
- **Originally:** Draft v1.0, 2026-06-22 (standalone .NET MAUI + Blazor app at `C:\Repos\FruitCake`)
- **Author:** Pieterjan (with Claude Code)
- **Genre:** Drop-merge physics puzzle (Suika / Watermelon Game clone)

> **What changed from the original PRD (corrections).** The original targeted **Android (primary),
> Windows, and web** from a shared C# core (`.NET MAUI` + `Blazor WebAssembly`, `SkiaSharp` rendering,
> `Aether.Physics2D`). That plan is **superseded**: FruitCake is now incorporated into the existing
> MintPlayer.AI web playground as **one more game alongside Rush Hour / 2048 / Cube / Snake /
> Mountain Car**, implemented natively for that stack. The native targets (MAUI/Android/Windows) and
> the shared-C#-core thesis are **dropped**. The gameplay spec (§5), feel (§6), and most tuning
> values carry over unchanged — only the platform/architecture and delivery decisions are rewritten.

---

## 1. Summary & Vision

FruitCake is a single-player casual physics puzzle. The player drops fruit into an open-top
container. Two fruits of the same kind that touch **merge** into the next-bigger fruit. The chain
runs from a tiny cherry up to a watermelon. The board fills as you play; you lose when fruit stacks
above the danger line. The hook is the satisfying *merge cascade* and the chase for a personal
high score.

The product goal is a polished casual game that lives at a URL inside the **MintPlayer.AI
playground**, requiring no install and no account. It is **the playground's first non-RL game** —
proof the site can host hand-built games as well as reinforcement-learning demos. An AI player is a
*possible future* addition (see §10), deliberately **out of scope for this port**.

---

## 2. Goals & Non-Goals

### Goals
- Faithfully reproduce the canonical Suika game loop and feel.
- Ship it as a **standalone Angular game component** in `RLDemo.Web`, following the playground's
  existing conventions (lazy route, nav link, home card).
- Smooth 60fps physics with stable stacking of ~50–100 fruit, **entirely client-side**.
- Persistent high score (and theme / music preference, and an in-progress game) in `localStorage`.
- A **fullscreen mode** that expands just the game stage to fill the screen.
- Be genuinely fun — the merge must feel good.

### Non-Goals (this port)
- **Any AI / RL player** — explicitly deferred (the rest of the playground is RL demos; this one
  isn't, *yet*).
- **Native platforms** — no Android, Windows, iOS, or macOS; no .NET MAUI host. (The original's
  primary target was Android; that is cancelled.)
- **A server/backend for FruitCake** — no controller, no model service, no WebSocket. Unlike the
  RL games, FruitCake needs nothing from the server beyond static file serving.
- Online multiplayer, accounts, monetization, level editor, alternate modes.
- **Online leaderboard** (the original's polish-tier idea). With no backend it is out of scope;
  high score stays device-local. The original's anti-cheat seed/replay machinery is therefore also
  dropped.

---

## 3. Target Platform

| Platform | Host | Distribution |
|---|---|---|
| Web (desktop + mobile browser) | `RLDemo.Web` Angular SPA | Served by the existing ASP.NET host (static SPA), deployed with the rest of the playground (Docker → Hetzner) |

Input model: **single pointer** (touch, mouse, or pen). Move horizontally to aim, tap/click to
drop. Plus an HTML **Fullscreen** button.

---

## 4. Architecture (web port)

FruitCake is a self-contained, client-only Angular feature under
`src/RLDemo.Web/ClientApp/src/app/fruit-cake/`. There is **no backend code** for it.

### 4.1 One standalone component, plain TypeScript modules
A signal-based standalone component (`fruit-cake.ts/.html/.scss`) drives a fixed-timestep
`requestAnimationFrame` loop (outside the Angular zone — the game renders itself, so per-frame
change detection would be waste) over a set of small, single-purpose modules:

- `fruit-cake-fruits.ts` — the 11-fruit chain (radii, colors, scores), merge rule, themes, color helpers.
- `fruit-cake-physics.ts` — the circle-collision world (see §4.3).
- `fruit-cake-effects.ts` — particles, score popups, screen shake, flash.
- `fruit-cake-art.ts` — the 11 vector fruit, drawn once into cached offscreen canvases.
- `fruit-cake-render.ts` — the draw pass + the letterbox transform + HUD/toolbar hit-testing.
- `fruit-cake-audio.ts` — synthesized Web Audio (pop / thud / music).
- `fruit-cake-game.ts` — the rules/state machine (drop queue, cooldown, scoring, game-over,
  `localStorage` persistence).

### 4.2 Rendering: HTML5 Canvas 2D
**Corrected from SkiaSharp.** The original rendered everything to one SkiaSharp surface shared by
the MAUI and Blazor hosts. The web port draws to a single **`<canvas>` 2D context**. The fruit
vector art (gradients, stems, leaves, seeds, net/stripe patterns) was ported 1:1 from the SkiaSharp
drawing code to Canvas 2D path/gradient calls. The container is fit into the canvas with a
letterbox transform whose inverse maps a pointer back to a drop position; the backing store is
scaled by `devicePixelRatio` for crisp output.

### 4.3 Physics: a purpose-built circle solver (no engine dependency)
**Corrected from Aether.Physics2D.** The original used the pure-managed Aether engine; bringing a
physics engine into the Angular bundle is unwarranted because **every collider is a circle and the
walls are static**. The port ships a compact **sequential-impulse solver** (`fruit-cake-physics.ts`):
gravity integration, circle–circle and circle–wall contacts, low-restitution velocity resolution
with Coulomb friction, and a non-linear position-correction pass for stable stacking — reproducing
the qualities the game depends on (fast settling, stable piles, contact-driven merges) in far less
code than a general engine.

The mandatory **deferred-merge pattern** is preserved: same-tier touching pairs are recorded during
contact detection and the bodies are added/removed only **after** the solve, so a merge never
mutates the body set mid-iteration. Per-body `pendingMerge` + a removed-guard prevent
double-merges.

### 4.4 Game loop
A fixed-timestep accumulator (`1/60 s`) inside a `requestAnimationFrame` callback, with the frame
delta clamped (≤ 0.25 s) to avoid a death spiral after a stall. The loop runs **outside the Angular
zone**. (Corrected from the original's MAUI `SKGLView.HasRenderLoop` / Blazor `SKCanvasView` paths.)

### 4.5 Persistence: `localStorage`
High score (`fruitcake:score`), theme index (`fruitcake:theme`), music preference
(`fruitcake:music`), and an in-progress game snapshot (`fruitcake:snapshot`, saved on each drop and
on teardown, resumed on load). Best-effort: failures (private mode) are swallowed.

---

## 5. Gameplay Requirements (canonical spec — unchanged)

### 5.1 The fruit chain (11 tiers)

Two fruits of the same tier touching → one fruit of **tier + 1** spawns at their midpoint. One-way
escalator: no skipping, no downgrade. Only **tiers 1–5 are player-droppable**; tiers 6–11 exist
only as merge products.

| # | Fruit | Droppable | Radius (px) | Merge pts |
|---|---|---|---|---|
| 1 | Cherry | ✓ | 24 | 1 |
| 2 | Strawberry | ✓ | 32 | 3 |
| 3 | Grape | ✓ | 40 | 6 |
| 4 | Dekopon | ✓ | 56 | 10 |
| 5 | Persimmon | ✓ | 64 | 15 |
| 6 | Apple | merge only | 72 | 21 |
| 7 | Pear | merge only | 84 | 28 |
| 8 | Peach | merge only | 96 | 36 |
| 9 | Pineapple | merge only | 128 | 45 |
| 10 | Melon | merge only | 160 | 55 |
| 11 | Watermelon | merge only | 192 | 66 |

- **Naming:** Fruit #4 is **Dekopon**, #5 is **Persimmon** — not generic oranges.
- **Score series** `1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66` — triangular numbers.
- **Radii are intentionally irregular** (not geometric). The big jump at pineapple (96→128) is a
  deliberate tension escalator — do not normalize it.
- **Watermelon + Watermelon → both disappear** (score awarded, board space freed). No tier 12,
  no win; the session continues.

### 5.2 Core loop

| Element | Requirement |
|---|---|
| **Queue** | One *current* fruit (held at top) + one *next* preview. Both uniformly random from **tiers 1–5**. |
| **Aiming** | A vertical drop-guide line follows the pointer's X across the container width. |
| **Dropping** | Tap/click releases the held fruit; it falls straight down under gravity. No control after release. |
| **Drop cooldown** | **500 ms** hard lock — input ignored; next fruit shown but not droppable (dimmed). |
| **Merging** | On contact of two same-tier bodies: remove both, spawn tier+1 at midpoint with zero velocity. Merges **chain**. |
| **Scoring** | Per merge: award the produced fruit's merge points. No combo/time multiplier. Accumulates. |
| **Game over** | A fruit **resting** (settled, low velocity) above the danger line for a ~1.5 s grace period ends the game. A fruit ejected over the rim ends it immediately. Momentary bounces above the line do **not** trigger loss. |

### 5.3 Tuning values (as shipped)

| Parameter | Value | Note |
|---|---|---|
| Container | 620 × 850 px | ratio ≈ 1:1.37; letterboxed to the canvas |
| Danger line Y | 150 px (~17% down) | |
| Gravity | 9.8 m/s² × 64 px/m = **627.2 px/s²** | PPM = 64, the original's pixels-per-meter constant |
| Restitution | 0.1 | very low; settle fast (no restitution below ~64 px/s approach, to kill jitter) |
| Friction | 0.006 | low; fruit slides/rolls naturally |
| Drop cooldown | 500 ms | hard input lock |
| Grace period above line | 1.5 s | reset if the pile drops back below |
| "Settled" speed threshold | 40 px/s | |
| Density | uniform (mass ∝ disc area) | keeps mass ratios smooth → stable stacks |
| Solver | 8 velocity iters, 4 position iters, 0.5 px slop, 0.8 correction | |
| Sim step | 1/60 s fixed | |

---

## 6. Feel / "Juice" Requirements (unchanged)

Suika's juice is **understated and precise** — each effect earns its place.

| Effect | Spec |
|---|---|
| Merge pop | Result fruit scales 0→full with `ease-out-back` overshoot (~150 ms). |
| Score popup | Floating `+N` rises from the merge point, fades ~600 ms; larger tiers slightly bigger. |
| Merge sound | Bright pop, pitch-shifted up for bigger fruit. The single most important sound. |
| Landing thud | Distinct from the merge pop; volume scales with impact speed. |
| Danger signal | Always-visible faint line; pulses red when a fruit rests above it. Never a surprise loss. |
| Next-fruit preview | Rendered at true relative size (clamped) in the corner. |
| Particle burst | 4–8 small fruit-colored dots, ~320 ms. |
| Screen shake | Only on pineapple+ merges; scales with tier. |
| Drop guide line | Faint vertical dashed line. **Not** a trajectory predictor. |
| Watermelon special | The climax — bigger burst (18 particles), screen shake + a brief white flash. |
| Ambient music + mute | Soft two-oscillator loop with a toggle; SFX mute toggle. |
| Accessibility | `prefers-reduced-motion` suppresses shake/flash; a colorblind tier-number toggle. |

The HUD toolbar (sound · tier labels · theme · music) is drawn on the canvas as compact **icon**
buttons and tapped, faithful to the original. The **Fullscreen** control is a small icon button
overlaid **inside the stage** (so it stays reachable in fullscreen on touch devices, which have no
Esc key); it fullscreens just the game stage, and the renderer letterboxes the play area to fill the
screen.

---

## 7. Scope

### Shipped in this port
- Full 11-fruit chain, correct radii + merge logic, triangular scoring.
- The custom circle-physics world (gravity, restitution, friction, contact-driven merges, stable stacks).
- Drop queue (current + 1 preview), 500 ms cooldown, danger line with red pulse.
- Game-over on above-line rest or out-of-bounds eject; restart by tapping the game-over screen.
- All "juice": merge pop, score popups, particle burst, screen shake, watermelon flash.
- Full audio: pitch-scaled merge pop, impact-scaled landing thud, ambient music loop; mute + music toggles.
- The 11 bespoke vector fruit, ported to Canvas 2D.
- Themes (Classic / Candy / Mono), colorblind tier labels, reduce-motion support.
- High score + settings + in-progress game persisted in `localStorage`.
- **Fullscreen** button.

### Explicitly not in this port
- AI player (§10) · native platforms · backend/leaderboard · social share.

---

## 8. Success Criteria

- **Functional:** the canonical loop is faithfully reproduced; a full cherry→watermelon run is
  achievable; no double-merge/ghost-body bugs.
- **Performance:** sustained ~60fps with 80+ fruit; stable settled stacks (no jitter creep).
- **Integration:** reachable from the nav + home page; lazy-loaded; doesn't regress the rest of the
  SPA; no backend changes required.
- **Feel:** the merge "feels good"; the danger signal is never a surprise.

---

## 9. Key Risks

| Risk | Mitigation |
|---|---|
| Custom solver less stable than a real engine for deep piles | Uniform density, low restitution + sub-threshold inelasticity, NGS position correction, 60 Hz step. If piles jitter/sink: raise position iterations or correction percent. |
| Double-merge / mid-iteration mutation | Deferred-merge queue drained after the solve; per-body `pendingMerge` + removed-guard. |
| Canvas perf with many fruit | Fruit art is pre-rendered once into cached offscreen canvases and blitted; per-frame draw is just textured quads. |
| Fullscreen quirks across browsers | Standard Fullscreen API on the stage element; renderer is resolution-independent (letterbox from canvas client size × DPR). |
| Audio autoplay policy | Web Audio context created/resumed lazily on first sound (always after a user gesture). |

---

## 10. Open Questions / Future

1. **An AI player** — the natural next step to fit the playground's RL theme. Suika is a hard
   continuous-aim, long-horizon environment; would need an environment wrapper + a trained policy
   (and likely a server endpoint to stream a watch-AI episode, like Snake/Mountain Car). Deferred.
2. **Art direction** — currently the ported vector fruit; emoji/flat-vector skins remain possible.
3. **Difficulty / modes** — alternate danger-line heights or chaos modifiers, if desired.
