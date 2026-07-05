# FruitCake AI — Fully Client-Side Inference (PRD & Plan)

> **Goal (user's steer, cost-driven):** the "watch the AI play FruitCake" mode currently runs the
> **entire** AI on the server (net forward pass + depth-3 search) and streams every frame over a
> WebSocket. On a single Hetzner VPS this makes server CPU **and** bandwidth scale **linearly with
> concurrent viewers** — 100 viewers ≈ 100× the compute + traffic, and a real monthly-bill risk.
> This PRD moves the **whole** AI into the browser so per-viewer server cost drops to **zero** (a
> one-time, CDN-cacheable weights download). Rather than hand-port to TypeScript, it **single-sources the
> inference path (observation + net forward + search) in the same MintPlayer.Polyglot `.pg` as the
> physics** (user's steer, 2026-07-05 — see §3 D0) → C# **and** TS from one source, byte-identical by
> construction. It supersedes the deferred PG4 idea (column-only streaming), which kept the server in the
> loop.

- **Status:** Draft v1.1 · 2026-07-05 (4-agent analysis complete; pivoted to Polyglot single-source per
  user steer; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:**
  - A **valid 89-dim net** — i.e. **M30/G4** must ship a `fruitcake.dqn.ckpt` matching the current
    `FruitCakeEnv.ObservationSize` (89). The committed `models/fruitcake.dqn.ckpt` is still **83-dim
    (stale)**; client-side inference ships whatever net M30 lands. This PRD is net-agnostic — it
    reproduces *a* net faithfully, whichever one is current.
  - The **single-source physics** (M31 / `POLYGLOT_FRUITCAKE_PRD.md`) — the browser already runs the
    exact C# physics via the generated `fruitcake_solver.ts`. That removed the *physics* half of the
    stream; this PRD removes the *decision* half.
  - The net/search/observation internals as documented by the 4-agent analysis in §2.

---

## 1. Why this is possible (the misconception it corrects)

There is **no fundamental reason** JavaScript can't run this AI. "The AI is C#-only" was a statement
about *where the code was written*, never a capability limit. A trained model answering "best move"
is just arithmetic:

1. **The checkpoint** is a bag of float32 weight matrices + bias vectors.
2. **A move query** is a forward pass — `obs → matmul → ReLU → matmul → dueling recombine → argmax`
   — a few hundred multiply-adds. Trivial on any device.
3. **The depth-3 search** is deterministic game-tree logic (clone the world, drop, settle, evaluate
   leaf with the net, expectimax over the unknown 3rd fruit). **No RNG.**
4. **The physics** the search looks ahead with is **already in the browser** (single-source solver).

The only reason it ran server-side is that the forward pass, the observation builder, and the search
were written once, in C#. Porting them to TS is bounded work, not a rewrite of the SDK. We explicitly
do **not** port the SDK's `Tensor`/GEMM backends — we hand-write this one small net's forward pass.

---

## 2. What the analysis confirmed (4-agent sweep, 2026-07-05)

### 2.1 Net (`DuelingQNet`) — `Core/Nn/DuelingQNet.cs`
- **Architecture:** trunk = `Linear(89→256)→ReLU→Linear(256→256)→ReLU`; **value head** `Linear(256→1)`;
  **advantage head** `Linear(256→14)`. Widths come from the checkpoint (`HiddenSizes`), **not** hard-coded.
- **Dueling recombine:** `Q(s,a) = A(s,a) + (V(s) − mean_a A(s,a))` (Wang 2016 mean form,
  `DuelingQNet.cs:62-74`).
- **Only activation is ReLU** (`max(0,x)`); heads have none. No LayerNorm, no residual, no input norm
  (obs is already `[0,1]` from the env).
- **NoisyNets:** if the checkpoint is noisy the heads are `NoisyLinear`, but **noise is OFF at
  inference** (`NoiseEnabled` defaults false; serving never enables it) — so a loaded net computes
  `x·MeanWeight + MeanBias`. **The browser needs only the mean weights**; sigma tensors are ignored.
- **Weights are float32, row-major `[in,out]`.** For byte-faithful parity the TS forward pass emulates
  float32 via `Math.fround` per op (JS is float64-only) and rounds the observation to float32 before the
  first matmul (C# obs is a `float[]`). This makes golden Q-vectors reproducible and argmax exact.

### 2.2 Checkpoint format — `Core/Checkpoints/{CheckpointFormat,DuelingQNetCheckpoint}.cs`
Self-describing little-endian binary, trivially parseable in TS with a `DataView`:
```
magic  uint32  = "RLNC" (0x434E4C52, on disk 52 4C 4E 43)
kind   string  = "dueling-q"   (BinaryWriter 7-bit-varint length prefix + UTF-8)
ver    int32   = 2
InputSize   int32           (= 89 for the current net)
HiddenSizes int32 count + int32[]   (= [256,256])
Actions     int32           (= 14)
Noisy       bool  (1 byte; absent in v1 ⇒ plain)
<per parameter, in Parameters() order>  WriteFloats = int32 count + count×float32
```
Parameter order (`DuelingQNet.Parameters()`): `trunk[i].W, trunk[i].b` per layer, then `value.W,
value.b`, then `advantage.W, advantage.b`. Noisy heads emit `Mean*, Sigma*` — TS reads the `Noisy`
flag and **skips Sigma**. Verified against `models/fruitcake.dqn.ckpt` on disk (though that file is the
stale 83-dim net).

### 2.3 Observation — `Environments/FruitCake/FruitCakeEnv.cs:148` (`BuildObservation`, 89 dims)
Pure function of `(world, current, next)`; RNG-free. Field-by-field spec (write order):
`28` per-column interleaved `[surfaceHeight=(H−topY)/H, topTier/11]` · `14` danger margin
`(topY−150)/700` · `14` merge-with-current one-hot · `14` adjacent-equal-pair · `5` current one-hot
`(t=1..5)` · `5` next one-hot · `3` globals `[count/100, fillArea/(620·850), minTop/850]` · `6`
big-fruit block (top-2 by tier→lower-Y→leftmost-X; sentinel `(0.5, 1, 0)`). All `Math.Clamp`ed to
`[0,1]`. (Full normalizer list in the port checklist, §8.)

### 2.4 Search — `Environments/FruitCake/FruitCakeSearch.cs`
- **Serving config:** `MaxDepth=3, TopK=5, TopK2=2, MergeWeight=1.0, LosePenalty=1e9`
  (`FruitCakeController.cs:78`). **No RNG — fully deterministic.**
- **Ply primitive** `DropAndScore`: `clone(enableRotation:false)` → `SpawnFruit(tier, ColumnX(col,tier),
  HeldY(tier))` → `SettleAfterDrop(30, 8, 600, dt=1/60)` (returns merge points) → loss = `AnyEjected()
  || AnyRestingAboveDangerLine(40)` → value = `lost ? points−1e9 : points + leafValue(sim)`.
- **Structure at depth 3:** max over 14 first columns (current known) → top-5 → max over 14 next
  columns (next known) → top-2 → **uniform average over droppable tiers 1..5** (the unknown 3rd fruit)
  → max over 14 columns → leaf. Argmax uses strict `>` (lowest index wins ties); default column 7.
- **Leaf value** (`FruitCakeController.cs:69-77`): `(1/5)·Σ_{d=1..5} max_a Q(BuildObservation(w, d, d))`
  — 5 single-obs forward passes per leaf, averaged. Fallback with no net: `−pileHeight`.
- **Geometry:** `ColumnX(col,tier) = clamp((col+0.5)·(620/14), r+6, 620−r−6)`, `HeldY = 150−r−4`.

### 2.5 Serving path today — `Controllers/FruitCakeController.cs`, `Services/FruitCakeModelService.cs`
- `GET/CONNECT /api/fruitcake/live` = a **WebSocket**; the server owns physics + clock and streams one
  JSON frame per ~2 substeps (~30 fps), server-authoritative, infinite loop, **client never sends
  moves**. This is what scales with viewers.
- Net loaded at startup by `FruitCakeModelService` from `data/fruitcake.dqn.ckpt` (seeded from
  `models/` via Git LFS); rejects any net whose `InputSize != 89`.
- Front-end (`ClientApp/src/app/fruit-cake/`): a `mode` signal `'human' | 'watch'`. Human play already
  runs the **local** physics (`FruitWorld` over `fruitcake_solver.ts`) on a JS RAF loop; watch mode
  just renders server frames. **No TS neural-net / matmul code exists yet.**
- Static assets: `ClientApp/public/**` is copied to the site root at build → the natural home for a
  fetched weights file. Dev uses `UseAngularCliServer` (**never run `ng build`/`serve`/`test`**).

### 2.6 What the TS physics still needs
The solver has `spawnFruit/step/settleAfterDrop/clear/maxSpeed` but **no `clone`** and none of the
search's end-state queries. The C# **facade** has `Clone/AnyEjected/AnyRestingAboveDangerLine/
PileHeight` — but these are **pure functions of body state**, not host-only concerns. The clean move
(single-source principle) is to push them **down into the `.pg`** so both the C# facade and the TS
adapter get them from one source (see CS3), rather than hand-mirroring them in TS and reintroducing a
sync burden.

---

## 3. Design decisions

> **Headline decision (D0), user's steer 2026-07-05:** don't hand-port the inference logic to TS and
> babysit golden fixtures — **single-source the whole inference path (observation + net forward + search)
> in the same `fruitcake_solver.pg`**, exactly as the physics already is. One source → C# **and** TS,
> **byte-identical by construction.** This is the same win that justified single-sourcing the physics,
> extended to the decision half. It also **dissolves the float-parity problem** (see D2). The rest of the
> decisions follow from D0.

### D1 — What moves into the `.pg`, what stays per-platform
**Into the `.pg` (single-sourced, f64, C#↔TS byte-identical):**
- `buildObservation(world, current, next) -> f64[89]` (the exact fields in §2.3).
- A standalone **inference** forward pass: `dueling MLP` over flat f64 weight arrays
  (`forward(weights, obs) -> f64[14]`, `Q = A + (V − meanA)`, ReLU trunk). This is a *new inference
  core*, distinct from the SDK's training forward pass (which keeps autograd/GEMM/GPU).
- `chooseColumn(world, weights, current, next) -> int` — the depth-3 expectimax search, with the leaf
  (`(1/5)Σ_d max_a forward(weights, obs(w,d,d))`) **inlined** (no injected delegate).
- The search's world-queries — `clone(enableRotation)`, `anyEjected`, `anyRestingAboveDangerLine`,
  `pileHeight` — which are pure position/tier math (today hand-written in the C# facade; move them down).

**Stays per-platform (cannot / need not be Polyglot):**
- **Training** — autograd, GEMM, GPU backends, Adam. Polyglot has no backprop or device kernels; training
  stays C#/SDK-only and *produces* the weights. Only inference is single-sourced.
- **Checkpoint parsing** — reading the `.ckpt` binary is I/O (byte layout / varint), not pure math. Small
  hand-written parser per platform: C# already has `DuelingQNetCheckpoint`; TS gets a ~40-line
  `DataView` parser that yields the ordered f64 weight arrays the inference core consumes.
- **Host glue** — TS rendering/audio/DOM and the C# DTO/controller. Unchanged.

### D2 — f64 inference on both sides ⇒ parity is exact by construction (no `Math.fround`)
The SDK net is **float32**; JS is float64. Hand-porting would have needed `Math.fround` emulation + golden
tolerance to make argmax match. With the forward pass **authored in Polyglot as f64 on both sides**, C#
inference and TS inference are **byte-identical** — the emulation hack disappears and parity stops being a
tolerance test. The deliberate consequence: **serving inference moves from the SDK's f32 path to the
Polyglot f64 path** (D5). That is a *good* trade — f64 is strictly more precise, the physics already made
the same f32→f64 move in PG1 (net re-validated fine), and it makes *what the Lab A/B judges* == *what C#
serves* == *what the browser runs*, all identical.

### D3 — Weight delivery: **parse the existing `.ckpt` directly** (no new format)
The `.ckpt` is a clean, self-describing float32 binary; parse it directly (C# already does; TS gets the
small parser in D1) — weights stay single-sourced (one artifact for training, serving, browser), no JSON
blow-up, no second exporter. **Delivery:** an MSBuild `Target` in `RLDemo.Web.csproj` copies
`models/fruitcake.dqn.ckpt` → `ClientApp/public/fruitcake-net.ckpt` at build (single source = the LFS
model; no committed duplicate); the browser `fetch`es it once (~370 KB, CDN/browser-cached). **Validation:**
the TS parser asserts `magic`/`kind`/`InputSize===89`; on mismatch the front-end falls back to the
heuristic leaf (`−pileHeight`) — same as the server with no net, game still plays.

### D4 — Collapse `watch` mode into a local "director" loop
Watch mode becomes structurally identical to human play: the same local `FruitWorld` + renderer, driven by
a **director** that per drop calls the generated `chooseColumn` → spawns → settles → animates the settle
frames locally on the RAF loop. No socket, no server frames. The `mode` signal, renderer split, and RAF
loop stay; only the "who decides + who steps" seam changes.

### D5 — Switch C# serving + Lab eval to the Polyglot inference core, then retire the server path
Point C# serving (`FruitCakeController`/`FruitCakeModelService`) and the Lab's search-eval at the generated
f64 inference core, so the A/B judges exactly what's served. Confirm served quality holds with **one A/B**
vs the ~50%-watermelon / ~2505 bar (low risk; f64 ≥ f32 precision). Then, once the front-end is client-side,
**remove** the WebSocket loop (`FruitCakeController.Live`), the model-service net loading, and the
`FruitCakeApi` socket — keeping the checkpoint as a static asset. Payoff: **zero** server inference and
**zero** streaming bandwidth for FruitCake. (Snake/MountainCar keep their streamers; out of scope.)

### D6 — Polyglot rewrites the search needs
The current `FruitCakeSearch` uses two things that don't transpile: a `Func<world,double>` leaf delegate
and LINQ (`Enumerable…OrderByDescending…Take` for top-k). In the `.pg`: **inline the leaf** (obs+net are
now in the same source) and **write top-k as an explicit selection loop** (don't rely on a stdlib sort
transpiling). Recursion, `int?`, tuples, `.filter`, and lambdas already transpile (visible in the generated
solver). New Polyglot codegen gaps get the same handoff loop as the four prior releases — and since
FruitCake is Polyglot's north-star conformance sample, this *advances* Polyglot too.

---

## 4. Cost impact (the actual point)

| | **Today (server-side)** | **After (client-side)** |
|---|---|---|
| Server CPU per viewer | depth-3 search @ ~1 drop/s + ~30 fps settle stepping | **0** |
| Server bandwidth per viewer | ~30 JSON frames/s × N fruit, continuous | **0** |
| Scaling with N viewers | **linear** (N× CPU + N× traffic) | **flat** (server does nothing) |
| One-time client cost | — | ~370 KB weights (CDN/browser-cached) + local CPU |
| Server → client after ship | net inference + physics stream | one static file, once |

This directly answers the Hetzner concern: 100 concurrent viewers cost the VPS **the same as zero**.

---

## 5. Plan — phases CS0–CS7 (each ends on a passing gate + commit)

The `.pg` grows the inference core bottom-up; each block is proven **Polyglot-C# == SDK-C#** once, and
**C#↔TS byte-identity** comes free from the transpiler.

- **CS0 — World-queries into the `.pg`.** Add `clone(enableRotation)`, `anyEjected`,
  `anyRestingAboveDangerLine(restSpeed)`, `pileHeight` to `fruitcake_solver.pg`; regenerate C#+TS; the C#
  `FruitCakeWorld` facade delegates (drops its hand-written copies), the TS `FruitWorld` adapter exposes
  them. **Gate:** full C# build + the 17 FruitCake/parity tests green (facade behaviour unchanged); a TS
  clone deep-copy parity test.
- **CS1 — Observation in the `.pg`.** Add `buildObservation(world, current, next) -> f64[89]` (all fields
  §2.3). C# `FruitCakeEnv.BuildObservation` becomes a facade over it (f64→f32 cast for the training net).
  **Gate:** a C# test asserts the core obs == the legacy obs on N boards (exact after cast); C#↔TS
  byte-identical (transpiler).
- **CS2 — Inference forward pass in the `.pg`.** Add `duelingForward(weights, obs) -> f64[14]` over flat
  f64 weight arrays (ReLU trunk, `Q = A + (V − meanA)`). **Gate:** matches the SDK's
  `DuelingQNet.Forward` on N observations — **argmax exact**, values within ~1e-5 (f32-weights→f64 math);
  C#↔TS byte-identical.
- **CS3 — Checkpoint delivery + parsers (not Polyglot — I/O).** MSBuild target copies
  `models/fruitcake.dqn.ckpt` → `ClientApp/public/fruitcake-net.ckpt`; write the ~40-line TS `.ckpt`
  parser (magic/kind/version, varint string, ints, `Noisy`, float32 blocks; skip Sigma) yielding the
  ordered f64 weight arrays `duelingForward` wants. **Gate:** TS parses the current net; asserts arch
  `(89,[256,256],14)`; a TS test runs `duelingForward(parsedWeights, obs)` and matches the CS2 values.
- **CS4 — Search in the `.pg`.** Add `chooseColumn(world, weights, current, next) -> int` — depth-3
  expectimax (`MaxDepth=3, TopK=5, TopK2=2`, strict-`>` argmax, uniform-1/5 chance node), leaf inlined
  (`(1/5)Σ_d max_a duelingForward(weights, obs(w,d,d))`; heuristic `−pileHeight` when no weights), top-k
  as an explicit selection loop (D6). **Gate:** matches the existing SDK-backed `FruitCakeSearch`
  `ChooseColumn` on N `(world,current,next)` — **same column**; C#↔TS byte-identical.
- **CS5 — Adopt the f64 inference core server-side + A/B.** Point C# serving
  (`FruitCakeController`/`FruitCakeModelService`) and the Lab's `--search-eval` at the generated
  `chooseColumn`/`duelingForward` (f64) instead of the SDK forward. **Gate:** one 100-game A/B holds the
  served quality vs the ~50%-watermelon / ~2505 bar (f64 ≥ f32 precision, low risk); build + tests green.
- **CS6 — Front-end director (client-side watch).** Collapse `watch` mode into the local director loop
  (D4): fetch+parse weights (CS3), per drop call the generated `chooseColumn`, spawn, settle, animate
  locally; stop using the socket. **Gate:** host + Playwright — watch mode plays a full game in-browser,
  merges/score/game-over correct, **0 network frames**, 0 console errors.
- **CS7 — Retire the server path + measure.** Remove `Live` WebSocket, `FruitCakeModelService` net load,
  `FruitCakeApi` socket; keep the static weights asset. **Gate:** build + tests green; Playwright watch-AI
  still works; confirm no `/api/fruitcake/live` traffic; note the cost delta in this PRD.

**Ordering vs M30:** CS3's weights asset ships whatever `models/fruitcake.dqn.ckpt` is current, so
**M30/G4 must land an 89-dim net first** (or the client falls back to the heuristic leaf until it does).
CS0–CS2 and CS4 are net-agnostic and can be built + equivalence-tested against any 89-dim net in parallel
with M30.

---

## 6. Correctness gate (the crux)

Client-side AI is only trustworthy if the browser reproduces the server's decisions. With the inference
path single-sourced in Polyglot, the gate has **two** cheaper parts than a hand-port's golden fixtures:

1. **Polyglot-C# == SDK-C# equivalence** (one-time, per block): the generated obs/forward/search match
   the existing SDK implementations — CS1 obs equality, CS2 argmax-exact forward, CS4 same-column search.
   These are ordinary C# unit tests over the same in-process net.
2. **C#↔TS byte-identity** comes **free from the transpiler** (the property PG0–PG2 already established
   for the physics) — no committed golden fixtures to maintain, no `Math.fround` tolerance.

The only *behavioural* change to validate end-to-end is D5's f32→f64 serving switch: the CS5 A/B
(100 games vs the ~50%/2505 bar). A drift there is the one thing that would make served play differ, so
CS5 gates CS6/CS7.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Polyglot can't express part of the search (sort, delegate) | Inline the leaf; write top-k as an explicit selection loop (D6). Recursion/`int?`/tuples/`.filter`/lambdas already transpile. New codegen gaps → the established Polyglot handoff loop. |
| f32→f64 serving switch changes played quality | CS5 A/B (100 games) vs the ~50%/2505 bar before cutover; f64 ≥ f32 precision and PG1 already validated the net on f64 physics — low risk. |
| Search cost too slow in-browser (14×… settle sims/drop) | Physics is the same f64 code already running human play at 60 fps; depth-3 TopK=5/TopK2=2 is bounded. Measure in CS4/CS6; if needed throttle to ~1 drop/s (viewers watch, don't race) or lower TopK2. |
| Weights asset drifts from the served net | MSBuild copy from the single LFS source at build (D3); TS validates `InputSize` and falls back to the heuristic leaf on mismatch. |
| Removing the WebSocket breaks a shared abstraction | FruitCake uses a **bespoke** intra-drop streamer, not the shared `EpisodeStreamer` (Snake/MountainCar) — removal is localized (CS7). |
| Ships before M30's 89-dim net exists | Client falls back to the heuristic leaf (same as the server today with no net); flips to the real net the moment M30 lands it. |

---

## 8. Behavioural invariants (must hold in the `.pg` inference core; C#↔TS byte-identical)

1. Net: `Q = A + (V − mean_a A)`; ReLU trunk; widths from checkpoint; **mean weights only**; **all math
   f64 both sides** (no float32 emulation — parity is byte-identical by construction); single-obs forward
   `89→14`.
2. Checkpoint (the one per-platform, non-Polyglot part): `RLNC`/`dueling-q`/v2; read `Noisy`, skip Sigma;
   float32 LE, row-major `[in,out]`; parsed to f64 weight arrays for the core.
3. Observation 89-dim in the exact block order (§2.3) with normalizers `/850, /11, /700, /100,
   /(620·850), /620`; big-fruit order tier→lower-Y→leftmost-X; sentinel `(0.5,1,0)`; clamp `[0,1]`.
4. Search: `MaxDepth=3, TopK=5, TopK2=2, MergeWeight=1, LosePenalty=1e9`; strict-`>` argmax, default
   col 7; top-k excludes losing cols, ordered by one-drop value desc, min 1 kept; non-kept cols keep
   one-drop value; chance node = uniform 1/5 over tiers 1..5 (unknown ply only); `next` is known/max.
5. Ply: `clone(rotation off)` → spawn at `ColumnX/HeldY` → `SettleAfterDrop(30,8,600,1/60)` → loss =
   ejected `y<0` or resting `y<150 & speed<40`; value = `lost ? pts−1e9 : pts + leaf`.
6. Leaf = `(1/5)Σ_{d=1..5} max_a Q(obs(w,d,d))`; fallback `−pileHeight`.
7. Everything RNG-free except the fruit-sequence generator (uniform over tiers 1..5 — the browser's to
   own; the current per-connection server RNG goes away with the socket).

---

## 9. Out of scope

- Snake / MountainCar / cube streamers (separate `EpisodeStreamer` path; keep as-is).
- **Training** in Polyglot/the browser — autograd/GEMM/GPU/Adam stay C#/SDK-only (D1). Only *inference*
  is single-sourced.
- Retraining the net (that's M30/G4). This PRD ships whatever net is current.
- Any change to human-play mode (already fully client-side).

---

## 10. Supersedes PG4

`POLYGLOT_FRUITCAKE_PRD.md` §PG4 proposed streaming only the chosen column + animating client-side to cut
**bandwidth** while keeping the server's compute (net+search) and the WebSocket. Full client-side
inference is strictly better — it removes the compute, the bandwidth, **and** the socket — so **PG4 is
retired in favour of this milestone (M32).**
