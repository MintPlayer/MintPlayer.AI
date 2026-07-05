# FruitCake AI — Fully Client-Side Inference (PRD & Plan)

> **Goal (user's steer, cost-driven):** the "watch the AI play FruitCake" mode currently runs the
> **entire** AI on the server (net forward pass + depth-3 search) and streams every frame over a
> WebSocket. On a single Hetzner VPS this makes server CPU **and** bandwidth scale **linearly with
> concurrent viewers** — 100 viewers ≈ 100× the compute + traffic, and a real monthly-bill risk.
> This PRD moves the **whole** AI into the browser (TypeScript), so per-viewer server cost drops to
> **zero** (a one-time, CDN-cacheable weights download). It supersedes the deferred PG4 idea
> (column-only streaming), which kept the server in the loop.

- **Status:** Draft v1.0 · 2026-07-05 (4-agent analysis complete; not started)
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

### D1 — Weight delivery: **parse the existing `.ckpt` directly in TS** (no new format)
The `.ckpt` is already a clean, self-describing float32 binary. Parsing it in TS keeps weights
**single-sourced** (one artifact for training, serving, and the browser) — no JSON blow-up (float32
JSON ≈ 1.7× the bytes and gzips poorly), no second exporter to keep in sync.
- **Delivery:** an MSBuild `Target` in `RLDemo.Web.csproj` copies `models/fruitcake.dqn.ckpt` →
  `ClientApp/public/fruitcake-net.ckpt` at build (single source = the LFS model; no committed
  duplicate). The browser `fetch('/fruitcake-net.ckpt')` once; the CDN/browser caches it (~370 KB).
- **Validation:** the TS parser asserts `magic`, `kind==="dueling-q"`, and `InputSize===89`; on
  mismatch it logs and the front-end falls back to the heuristic board-value (`−pileHeight`), exactly
  as the server does with a missing net — the game still plays, just weaker.
- *Rejected:* a `--export-weights` JSON exporter in `FruitCakeLab` — a second weight format and a
  second thing to keep faithful. (Kept only as a note in case a non-.NET consumer ever needs JSON.)

### D2 — Float32 emulation in TS (parity)
The net is float32 in C#; JS is float64. The TS forward pass wraps each op in `Math.fround` and rounds
the observation to float32 before the first matmul. This makes the golden Q-vectors reproduce closely
enough that **argmax matches C# exactly** on the fixtures. (Physics stays f64 both sides — already
byte-identical from M31.)

### D3 — Push the search's world-queries into the `.pg` (single source), don't hand-mirror
`clone(enableRotation)`, `anyEjected`, `anyRestingAboveDangerLine(restSpeed)`, `pileHeight`, `count`
are pure position/tier math. Adding them to `fruitcake_solver.pg` regenerates both the C# core and the
committed TS core; the C# **facade** then delegates to the core (dropping its hand-written copies), and
the TS **adapter** exposes them. One source, no drift. (Host-only concerns — merge/land events, audio,
save/restore — stay in the facades, unchanged.)

### D4 — Collapse `watch` mode into a local "director" loop
Watch mode becomes structurally identical to human play: the same local `FruitWorld` + renderer, driven
by a **director** that per drop calls the TS search → spawns → settles → animates the settle frames
locally on the RAF loop (reusing the existing intra-drop cadence). No socket, no server frames. The
`mode` signal, renderer split, and RAF loop stay; only the "who decides + who steps" seam changes.

### D5 — Retire the server serving path
Once parity is proven and the front-end is client-side, **remove** the WebSocket loop
(`FruitCakeController.Live`), the `FruitCakeModelService` net loading, and the `FruitCakeApi` socket
client. Keep the checkpoint (now shipped as a static asset). This is the payoff — **zero** server
inference and **zero** streaming bandwidth for FruitCake. (Snake/MountainCar keep their streamers; those
are separate and out of scope.)

### D6 — Search/observation stay hand-ported (for now), gated by golden parity
Porting the observation + search via Polyglot too is attractive (the observation is pure arithmetic),
but the search takes a `Func<world,double>` net-leaf delegate and the observation reads facade body
views — not a clean `.pg` fit at v0.1.x. We hand-port both and **gate on C#-generated golden fixtures**
(§6). Single-sourcing them via Polyglot is a documented future option, not this milestone.

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

## 5. Plan — phases CS0–CS6 (each ends on a passing gate + commit)

- **CS0 — TS `.ckpt` parser + weights delivery.** Add the MSBuild copy target
  (`models/fruitcake.dqn.ckpt` → `ClientApp/public/fruitcake-net.ckpt`). Write `fruitcake-net.ts`
  parsing the `.ckpt` (magic/kind/version, varint string, ints, `Noisy`, float32 blocks; skip Sigma).
  **Gate:** parses the current net; asserts arch `(89,[256,256],14)`; exposes the ordered mean tensors.
- **CS1 — TS `DuelingQNet` forward pass.** Implement trunk+ReLU+heads+dueling recombine with `Math.fround`
  float32 emulation. **Gate:** matches a committed **golden Q-vector fixture** (obs → 14 Q-values)
  exported from C# — argmax exact, Q within ~1e-4.
- **CS2 — TS `BuildObservation` port.** Reproduce the 89-dim vector (all normalizers + big-fruit
  selection/sentinel). **Gate:** matches a committed **golden observation fixture** (world state →
  89 floats) from C#, exact to float32.
- **CS3 — World-queries into the `.pg` (single source).** Add `clone(enableRotation)`, `anyEjected`,
  `anyRestingAboveDangerLine`, `pileHeight`, `count` to `fruitcake_solver.pg`; regenerate C#+TS; C#
  facade delegates (drops hand-written copies); TS adapter exposes them. **Gate:** full C# build + the
  17 FruitCake/parity tests green (facade behaviour unchanged); TS clone deep-copies (parity test).
- **CS4 — TS `FruitCakeSearch`.** Port depth-3 expectimax (`TopK=5/TopK2=2`, strict-`>` argmax,
  uniform-1/5 chance node, leaf = mean-of-5 net max-Q). **Gate:** matches a committed **golden
  chosen-column fixture** — for a set of `(world, current, next)`, TS picks the **same column** as C#.
- **CS5 — Front-end director (client-side watch).** Collapse `watch` mode into the local director loop
  (D4); render locally; stop using the socket. **Gate:** host + Playwright — watch mode plays a full
  game in-browser, merges/score/game-over correct, **0 network frames**, 0 console errors.
- **CS6 — Retire the server path + measure.** Remove `Live` WebSocket, `FruitCakeModelService` net
  load, `FruitCakeApi` socket; keep the static weights asset. **Gate:** build + tests green; Playwright
  watch-AI still works; confirm no `/api/fruitcake/live` traffic; note the cost delta in this PRD.

**Ordering vs M30:** CS0's weights asset ships whatever `models/fruitcake.dqn.ckpt` is current, so
**M30/G4 must land an 89-dim net first** (or CS0 ships the heuristic-fallback path until it does). CS1–CS4
are net-agnostic and can be built + golden-tested against any 89-dim net in parallel with M30.

---

## 6. Correctness gate (the crux)

Client-side AI is only trustworthy if the browser reproduces the server's decisions. We commit three
**golden fixtures generated from C#** and assert TS parity:
1. **Q-vector golden** — N observations → 14 Q-values (CS1).
2. **Observation golden** — N `(world,current,next)` → 89 floats (CS2).
3. **Chosen-column golden** — N `(world,current,next)` → the depth-3 column (CS4).

A tiny C#-side helper (a `FruitCakeLab --emit-goldens <dir>` sub-mode, following the existing
`--search-eval`/`--ab` return-pattern) serializes these; a TS test (Node or Playwright `evaluate`)
loads and checks them. Float policy: argmax/column **exact**; raw Q within ~1e-4 after `Math.fround`.
A drift here is the one thing that would make client-side play *visibly* differ, so this gate is
non-negotiable before CS5/CS6.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| float32↔float64 drift flips a near-tie column | `Math.fround` per-op emulation + float32 obs; golden fixtures gate argmax exactness (§6). Worst case is a different but equally-valid move in *watch* mode — not a correctness bug. |
| Search cost too slow in-browser (14×… settle sims/drop) | Physics is the same f64 code already running human play at 60 fps; depth-3 with TopK=5/TopK2=2 is bounded. Measure in CS4/CS5; if needed, throttle to 1 drop/s (viewers watch, don't race) or lower TopK2. |
| Weights asset drifts from the served net | MSBuild copy from the single LFS source at build (D1); TS validates `InputSize` and falls back to heuristic on mismatch. |
| Removing the WebSocket breaks a shared abstraction | FruitCake uses a **bespoke** intra-drop streamer, not the shared `EpisodeStreamer` (Snake/MountainCar) — removal is localized (CS6). |
| Ships before M30's 89-dim net exists | CS0 fallback = heuristic board-value (same as server today with no net); flips to the real net the moment M30 lands it. |

---

## 8. Port checklist (behavioural invariants — must hold in TS)

1. Net: `Q = A + (V − mean_a A)`; ReLU trunk; widths from checkpoint; **mean weights only**; float32
   via `Math.fround`; single-obs forward `[1,89]→[1,14]`.
2. Checkpoint: `RLNC`/`dueling-q`/v2; read `Noisy`, skip Sigma; float32 LE, row-major `[in,out]`.
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
- Single-sourcing observation/search via Polyglot (future; D6).
- Retraining the net (that's M30/G4). This PRD ships whatever net is current.
- Any change to human-play mode (already fully client-side).

---

## 10. Supersedes PG4

`POLYGLOT_FRUITCAKE_PRD.md` §PG4 proposed streaming only the chosen column + animating client-side to cut
**bandwidth** while keeping the server's compute (net+search) and the WebSocket. Full client-side
inference is strictly better — it removes the compute, the bandwidth, **and** the socket — so **PG4 is
retired in favour of this milestone (M32).**
