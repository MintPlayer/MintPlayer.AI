# FruitCake "Watch AI" — the per-drop freeze (PRD & Plan)

> **Goal (owner report, 2026-08-02):** watching the AI play FruitCake at
> `https://ai.mintplayer.com/fruitcake` freezes for ~3 s every time a fruit lands or two fruits merge.
> Manual play is smooth. This PRD establishes the measured root cause — the depth-3 search runs
> **synchronously inside the `requestAnimationFrame` callback**, blocking the main thread for
> **0.97–5.7 s per drop** — and fixes it so the watch view animates continuously.

- **Status:** 🔍 **INVESTIGATED (root cause measured, 2026-08-02).** Implementation not started.
- **Author:** Pieterjan (with Claude Code)
- **Branch:** `m53-fruitcake-ai-stall` (off `master`)
- **Supersedes the open risk in** `FRUITCAKE_CLIENT_SIDE_AI_PRD.md:283` — *"Search cost too slow
  in-browser … if needed throttle to ~1 drop/s or lower TopK2"*. That risk was logged when the AI moved
  client-side (M32) and **the measurement was never taken**. This PRD takes it.

---

## 1. Symptom vs. reality

| owner's observation | what is actually happening |
|---|---|
| "freezes when a fruit lands **or two fruits merge**" | freezes **once per drop**. Landing/merging is what *ends* the settle phase; the search fires `BETWEEN_S = 0.25 s` later, so the freeze always *trails* a land or merge |
| "~3 seconds" | 0.97 s on an empty board → 5.7 s at 24 fruit. **~3 s ≈ 9–12 fruit**, an ordinary mid-game board |
| "only in Watch AI, not manual" | correct — manual mode never enters the `think` phase |

The reframe matters: **the trigger is the drop cycle, not the merge event.** Any fix keyed to merges would
miss.

---

## 2. Root cause (measured, not inferred)

`fruit-cake-director.ts:60-70`, reached from the rAF callback at `fruit-cake.ts:98`:

```ts
case 'think': {
  // One synchronous search per drop (a brief "thinking" hitch — acceptable for a watch view).
  const col = this.net
    ? this.world.chooseColumn(this.net, this.current, this.next, DEPTH, TOPK, TOPK2)
    : this.fallbackColumn();
```

The comment concedes a "brief hitch". It is not brief.

### 2.1 The main thread is fully blocked

Three independent probes agree to within 1–2 ms on **every** event (e.g. all three read 1871 ms on the
same stall):

- rAF inter-frame gap
- `PerformanceObserver('longtask')`
- a `MessageChannel` starvation probe

rAF callbacks **stop firing entirely**. A paused game clock would keep rAF at full rate — categorically
excluded. This is synchronous compute.

The gap histogram is **bimodal with nothing in between** — localhost, 51 s, 7828 frames:

| bucket | frames |
|---|---|
| < 50 ms | 7813 |
| 50–500 ms | **0** |
| 500–1000 ms | 1 |
| 1000–2000 ms | 14 |

Frames are either perfect or catastrophic — the signature of a single blocking call. Across two prod
windows the tab was blocked **36.8 % and 46.1 % of wall-clock time**.

### 2.2 What one decision costs

Measured by wrapping `PgFruitCakeWorld.prototype` on the live instance:

| call | measured | note |
|---|---|---|
| `dropAndScore` | **784** | = `14 + 5×(14 + 2×5×14)` |
| `clone` / `settleAfterDrop` / `leafValue` | 784 each | one per rollout |
| `net.forward` | **3 920** | 5 per `leafValue` (chance node over 5 droppable tiers) |
| `world.step` | ~78 000 | |
| `buildContacts` | ~390 000 | O(n²), run **5× per substep** |

One forward = 92 160 MACs (89→256→256→{1,14}), so **≤361 M MACs per decision**.

**The width is fixed.** These counts are byte-identical on every decision regardless of board state — a
`new Set(...)` over all observed counts returned a single element for each. Branch count never varies.

CDP CPU profile (prod, 20.8 s, 35 192 samples, inclusive):

```
frame               33.1%  6869ms
└─ update           30.7%  6376ms
   └─ chooseColumn  30.7%  6368ms
      └─ dropAndScore    30.7%  6365ms
         └─ bestContinuation 30.1%  6253ms
            ├─ settleAfterDrop 15.1%  3140ms   ← physics
            └─ leafValue       15.0%  3116ms   ← net
draw                 2.1%   445ms
(idle)              62.6%
```

Excluding idle, `chooseColumn` is **82 % of all main-thread work**. Rendering is 2.1 % — irrelevant.
Physics and net split the cost almost exactly evenly, so **neither alone is the lever**.

### 2.3 Why it grows to ~3 s and beyond

Controlled probe (fresh world, one fruit added at a time, one uninstrumented `chooseColumn` per state):

| fruit on board | median | min | max |
|---|---|---|---|
| 1–2 | 1240 ms | 968 | 1240 |
| 3–5 | 1304 | 1218 | 1340 |
| 6–8 | 1883 | 1264 | 2106 |
| **9–12** | **3306** | 2830 | 4669 |
| 13–16 | 4738 | 4333 | 4951 |
| 17–20 | 5096 | 4626 | 6703 |
| 21–24 | **5695** | 5159 | 6134 |

Linear fit: **+240 ms per fruit**, intercept 586 ms, R² = 0.888 — a **5.8× spread** across a game.

Mechanism confirmed: `world.step` **count fell** as fruit accumulated (79 619 → 75 789) while wall time
doubled. Same number of substeps, each costing more ⇒ **O(n²) `buildContacts`**, not longer simulations.

### 2.4 The browser runs the most expensive config in the repo

The retired C# serving path shipped `MaxDepth = 2, TopK = 10, TopK2 = 3` (`FruitCakeSearch.cs:21,25,28`)
= **154 rollouts**. The browser hardcodes `3/5/2` = **784** — 5× more. When this ran server-side, viewers
never paid the cost; M32 moved the decision into the browser and kept the *expensive* config.

### 2.5 Ruled out

Every alternative was checked and excluded: no fixed delay constant (only `BETWEEN_S = 0.25` /
`GAMEOVER_S = 1.8`); no `setTimeout`/`setInterval` anywhere in the folder; checkpoint fetched **once** in
the director constructor and explicitly kept across `reset()`; no server round-trip (**zero** per-move
requests, no WebSocket, no `/api/fruitcake` — the controller no longer exists); no Angular change-detection
storm (the loop runs in `zone.runOutsideAngular`); no audio/effects in watch mode (`renderFrame` is
*cheaper* than the human `render`); rAF never cancelled outside `teardown()`; **no Web Worker anywhere in
the app** (0 `Worker` constructions observed); console clean (0 errors on prod and localhost).

**Not a regression.** `DEPTH/TOPK/TOPK2 = 3/5/2` landed in the original client-side-AI commit `fcd2562`
and was never touched.

---

## 3. Design

### 3.1 The two levers, and why only one is a fix

**Lever A — get it off the main thread (Web Worker).** Removes the freeze *as a freeze*: the render loop
keeps running, the page stays responsive, the tab stops hanging.

**Lever B — reduce the work (`DEPTH`/`TOPK`/`TOPK2`).** Width is fixed at 784, so cuts scale predictably;
the chance-node average over 5 tiers (`fruitcake_solver.ts:355-361`) alone is a 5× factor.

**The measurement decides between them.** At **+240 ms per fruit**, even a 5× width cut still leaves
~1.1 s at 24 fruit. **Lever B alone does not fix this — it only shifts the curve down.** Lever A is the
fix; Lever B is complementary tuning.

But Lever A alone is also **not sufficient for the owner's actual complaint**. A worker makes the *page*
responsive; the *board* still stands still for 1–5.7 s while the AI thinks. "The game seems to get stuck
before the animations are resumed" would remain true. **Both the block and the wait must go.**

### 3.2 Design it twice — how to remove the *wait*

**Option 1 — reduce the search until it fits the existing 0.25 s `BETWEEN_S` gap.** `DEPTH = 2` is 14
rollouts (56× cheaper) ⇒ ~17 ms empty, ~100 ms at 24 fruit. Simple, one line. Costs playing strength by an
unmeasured amount.

**Option 2 — speculative pipelining: think for drop N+1 *during* drop N's settle animation.** The settle
animation runs ~1–2 s of real time on a nearly idle main thread; overlapping hides most of the think.
Requires a board to think *from*, and the real one doesn't exist yet — but the search already computed its
own prediction of it (`PgPlyResult.world`, the settled clone for the chosen column). Starting the next
search there is no less principled than the search's own internal lookahead, which already trusts those
worlds.

**The catch, and it is subtle: the prediction is not exact.** Search clones are rotation-off
(`dropAndScore` → `clone(false)`) while the live world is rotation-on (`fruit-cake-director.ts:22`). That
flag gates only the angular *damping* (`fruitcake_solver.ts:153-156`) — but `angularVel` is written by
`applyImpulse` **regardless** of the flag, and feeds back into linear velocity through the tangential
friction impulse (`:476-479`, `:493-495`, `:513-524`). So rotation-on and rotation-off worlds **diverge in
position**. Rotation is not cosmetic here.

⇒ Speculation must be **validated**, not trusted: when the live board really settles, compare it to the
prediction; on a match use the precomputed column instantly, on a mismatch fall back to a fresh search.
Cheap to check, and it degrades to today's behaviour in the worst case.

**Decision: Option 2 built on Lever A, with Option 1 held as the tuning knob and phone fallback.** The
worker is what makes speculation free — the speculative search costs the main thread nothing, so a wasted
speculation is invisible rather than a double stall.

### 3.3 What must NOT change

**`fruitcake_solver.pg` stays untouched.** It is single-sourced to C# and TS with a bitwise-parity
guarantee, and `PolyglotNetParityTests.cs:70-107`
(`CoreSearch_MatchesCsFruitCakeSearch_SameColumn`) pins TS and C# to the same column **at exactly the
browser's 3/5/2 config**. Changing the generated hot loops (e.g. `Float64Array` weights, contact pooling)
would alter the C# training path and risk that parity — high risk, low leverage. **Rejected.**

This is affordable because the generated `fruitcake_solver.ts` is **already worker-safe**: 655 lines, no
imports, and zero occurrences of `document`/`window`/`globalThis`/`performance`/`navigator`/`fetch` — it
touches only `Math` and `Array`. `fruitcake-net.ts` uses only `fetch`/`DataView`/`Uint8Array`/`TextDecoder`,
all present in `DedicatedWorkerGlobalScope`. Both move into a worker **unmodified**.

### 3.4 Build support — verified, not assumed

Spiked end-to-end against the running dev server on 2026-08-02:

- Builder is **`@angular/build:application` 22.0.6** (esbuild 0.28.1). `new Worker(new URL('./x.worker',
  import.meta.url), { type: 'module' })` is handled natively by
  `tools/angular/transformers/web-worker-transformer.js`, registered **unconditionally** in the AOT
  pipeline.
- Spike result: the emitted chunk contained the rewritten call site
  (`new URL("/worker-Y4FJZU4R.js?worker_file&type=module", …)`), and `GET /worker-Y4FJZU4R.js` returned
  **HTTP 200, `text/javascript`, marker present** — the dev server serves worker bundles as first-class
  output. **Worker delivery is not a risk.**
- **No `tsconfig.worker.json`, no `angular.json` change, no new dependency.** `webWorkerTsConfig` is in the
  schema but **inert** for this builder (only the karma paths read it) — adding it would be a no-op.
- Two gotchas for the implementer: the URL argument **must be a string literal** (a computed path is
  silently left untransformed), and `lib` here is DOM-only with no `webworker`, so use the existing
  house idiom from `screen-wake-lock.ts:3-11` — a minimal local shape + cast — rather than adding
  `"webworker"` to the shared `lib` (DOM and WebWorker declare conflicting globals in one program).

---

## 4. Milestones

### M53.0 — Baseline ✅ *(done, 2026-08-02 — this document)*
Root cause measured on prod **and** localhost; call counts, growth curve, CPU profile, and the
blocked-main-thread verdict all recorded above. Worker delivery spiked and confirmed.

### M53.1 — Move the search off the main thread
New `fruit-cake-ai.worker.ts` owning its own `PgDuelingNet` (via `loadFruitCakeNet`) and reconstructing a
`PgFruitCakeWorld` from a posted body snapshot (`tier, x, y, vx, vy` — exactly `clone()`'s semantics,
`fruitcake_solver.ts:184-194`). `fruit-cake-director.ts` becomes the only changed file: a new `thinking`
phase with a `pending` guard so `update()` posts once, not every frame; `onmessage` does the `spawnFruit`
+ `phase = 'settle'`. `fallbackColumn()` (14 blocking rollouts when the net is missing) moves into the
worker too. Payload is ~40 small objects — microseconds.

**Gates:**
- No main-thread task > **50 ms** attributable to the search, over a ≥60 s watch run (`PerformanceObserver('longtask')`).
- rAF gap histogram: **zero** gaps > 200 ms.
- Column choice **identical** to today's synchronous path for a fixed board + `(current, next)` pair —
  the worker runs the same generated code, so this must hold exactly.
- Net-missing path still plays (worker-side fallback).

### M53.2 — Remove the visible wait (speculative pipelining)
Start the search for drop N+1 from the search's own predicted settled world the moment drop N is
committed, overlapping the fall/settle animation. **Validate on arrival**: compare the live settled board
to the prediction; match ⇒ use the precomputed column, mismatch ⇒ fresh search. Instrument the
match rate — it is the number that decides whether this design earns its complexity.

**Gates:**
- Visible gap between a board coming to rest and the next fruit spawning ≤ **BETWEEN_S + 150 ms** at the
  95th percentile over a ≥60 s run.
- Speculation **match rate reported** (not gated — it is a measurement; a low rate sends us to M53.3).
- No change to the drop sequence when the prediction matches.

### M53.3 — Search-config A/B *(tuning; run only if M53.2's match rate is poor or phones still lag)*
Price the width cut before spending it. Harness is
`tools/MintPlayer.AI.ReinforcementLearning.Lab/FruitCake/FruitCakeSearchEval.cs` — the search config **is
a CLI flag**:

```
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- \
  --game fruitcake --search-eval --data C:\Repos\MintPlayer.AI\data\fruitcake-bigfruit \
  --ab-episodes 40 --depth 3 --topk 5 --topk2 2
```

Compare `3/5/2` (browser today) vs `2/10/3` (the shipped C# serving default) vs `2/5/3`.

Four traps, all verified: episode count is **`--ab-episodes`** (default 200), *not* `--episodes`;
`--seed` is ignored (`seedBase: 20_000` is hardcoded — which makes the greedy arm a free, bit-identical
shared control across runs); **use an absolute `--data` path** (there are two different
`fruitcake.dqn.ckpt` files in the tree); and the browser ships **`data/fruitcake-bigfruit`** —
`sha256` of `wwwroot/models/fruitcake-net.ckpt` matches it exactly, while `src/RLDemo.Web/data/fruitcake.dqn.ckpt`
is a *different*, stale net from the retired server path. A/B against the wrong one and the numbers
describe a net nobody plays. Leave `MergeWeight` at its 1.0 default — the parity test passes precisely
because it is untouched. Calibrate runtime with `--ab-episodes 8` first (far too few for a verdict — the
harness's own docs note a 10-episode eval bounced 750–971 on the *same* net).

**Gate:** ship a reduced config only if mean score is within **1 SE** of `3/5/2`, or if the owner accepts
the measured strength cost explicitly.

### M53.4 — Ship
Docs synced; stale "server-streamed" docstrings at `fruit-cake-render.ts:126` and `fruit-cake-frame.ts:1-3`
corrected (M32 leftovers); the "brief thinking hitch" comment at `fruit-cake-director.ts:61` replaced with
the real numbers. Consider deleting the stale `src/RLDemo.Web/data/fruitcake.dqn.ckpt`.

---

## 5. Rejected up front

- **Micro-optimizing the generated hot loops** (`Float64Array` weights, `PgContact` pooling, spatial
  partitioning for `buildContacts`) — requires `.pg` edits, alters the C# training path, risks bitwise
  parity. The O(n²) contact build at ~390 k calls/decision is the hottest *structural* cost and is the
  right long-term optimization, but it is optimization, not the fix.
- **Width reduction as the sole fix** — the growth curve refutes it (§3.1).
- **Throttling to ~1 drop/s** (the original M32 suggestion) — treats the symptom as a pacing choice; the
  tab would still hang.
- **`webWorkerTsConfig` in `angular.json`** — inert for this builder.
- **Adding `"webworker"` to the shared `lib`** — conflicts with DOM in one program.
