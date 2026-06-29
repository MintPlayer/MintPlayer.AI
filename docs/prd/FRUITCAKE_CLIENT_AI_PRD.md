# FruitCake — Move the "Watch AI" to the Client (PRD & Plan)

> Today FruitCake's "Watch AI" runs **entirely on the server**: the C# physics, the `DuelingQNet`,
> and a depth-3 forward-model search all run in a per-connection WebSocket loop on the CPU-only
> Hetzner VPS, streaming ~30 fps JSON frames to a browser that only renders. This PRD moves that
> whole agent **into the browser** — net inference, observation, physics, and search — collapsing the
> server's role to serving static assets + the (cached) checkpoint. The server-side engine is
> **preserved** (it still trains the net and serves as the parity oracle); it just leaves the serving
> hot path.

- **Status:** Draft v1.0 · 2026-06-28 (investigation complete; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** the shipped FruitCake AI (`FRUITCAKE_AI_PRD.md`) and its plateau work
  (`FRUITCAKE_IMPROVE_PRD.md` — the depth-3 `FruitCakeSearch` + the 83-dim net this ports), the web's
  load-only model path (`Program.cs`, `FruitCakeModelService`), and the existing **client-side**
  TS physics (`fruit-cake-physics.ts`) used today for human play. Companion to [PLAN.md](PLAN.md) (M30).

---

## 1. Motivation — per-viewer server load on a CPU-only box

A three-angle investigation (model portability, physics+search portability, server-load profile)
converged on one conclusion: **FruitCake "Watch AI" is the playground's heaviest per-viewer standing
cost, and its agent is the easiest to move client-side because the hard part — the physics — is already
ported to TypeScript.**

- **The current serving model is a persistent per-connection workload.** `FruitCakeController.Live`
  (`FruitCakeController.cs:49-132`) accepts a WebSocket and runs a real-time loop for the tab's whole
  lifetime: ~60 fps `world.Step` (`:105`), serialize+send every 2nd sub-step ≈30 fps (`:107-111`), and
  a **depth-3 expectimax search per drop** (`:78,99`). The frame *sends* are paced with `Task.Delay`
  (thread-friendly), but the search is uninterrupted CPU (~45 ms/drop, more with the net leaf, `:64`).
- **It scales by wall-clock presence, not request rate.** N viewers = N simultaneous physics+search
  loops. There is **no CPU limit, no concurrency cap, and no WebSocket admission control**
  (`docker-compose.yml` declares no `deploy.resources`; `Program.cs` wires only `UseWebSockets()` + DI).
  Once the few cores saturate, `Task.Delay(33ms)` deadlines slip **for everyone** — degradation is
  global and graceless rather than capped.
- **The concern generalizes.** Snake and MountainCar use the same per-connection streaming loop via
  `EpisodeStreamer` ("the backend owns the episode loop and the clock… one env per connection",
  `EpisodeStreamer.cs:8-60`). They are lighter, but architecturally identical. The request/response
  solvers (cube, 2048, Rush Hour) are **not** affected — they're one-shot per click and scale with
  click rate, not viewers.
- **CPU-only is confirmed** (`Program.cs:34-37`, `docs/OPTIMIZATIONS.md`): no GPU in prod. The cube
  beam search is the one workload with a real reason to stay server-side; FruitCake's is not.

**Why FruitCake first:** its "AI" is pure portable arithmetic (a 91k-param MLP + a deterministic
physics search), and — uniquely — **a faithful TypeScript twin of its physics already exists**
(`fruit-cake-physics.ts` ↔ `FruitCakeWorld.cs`, each file declaring the other its canonical mirror).
So we are not porting a physics engine from scratch; we are extending one that is already there.

---

## 2. This reverses the original serving decision — deliberately

The original FruitCake AI PRD chose **server-authoritative serving** (`FRUITCAKE_AI_PRD.md` §4.6) and
explicitly **rejected** "client simulates in TS, asks the server per drop," and judged single-source
physics not worth it (§4.8). That was the right call **then**: the goal was correctness with minimal
risk, and server-authoritative serving made C#↔TS physics divergence *irrelevant*.

This PRD makes the opposite trade **on purpose**, because the optimization target has changed from
"ship it correctly" to "serve it cheaply at concurrency." The consequence is explicit and load-bearing:

> **Moving the search client-side makes C#↔TS physics parity a correctness requirement, not a polish
> item.** The net's strength comes from the depth-3 search valuing *simulated future boards* with the
> net's leaf Q. The net was trained against **C# `FruitCakeWorld`** physics; the client search will
> simulate with **TS `FruitWorld`**. If the two settle boards differently, the search feeds the net
> off-distribution leaves and play silently weakens.

The good news from the investigation: the two engines were **built as mirrors** — identical constants
(gravity, restitution 0.1, friction 0.3, 12 velocity / 4 position iterations, slop 0.5, correction
0.8, angular damping 0.995) and an identical solver structure (`FruitCakeWorld.cs:30-44,97-123` ↔
`fruit-cake-physics.ts:65-79,124-150`). The original PRD §4.8 already *recommended* a cross-language
golden-vector conformance test to catch drift — **it was never built.** This PRD builds it and makes
it the central de-risking item (§4.G3). Parity is testable, not fundamental.

---

## 3. Goals & Non-Goals

### Goals
- **Run the FruitCake "Watch AI" entirely in the browser** — net inference, observation, physics, and
  depth-3 search — so the server holds **zero per-viewer standing CPU** for it.
- **Match the current server agent's strength** within the env's A/B noise (judged on the ≥200-game
  max-tier distribution, never a 10-episode eval — the M28 seed trap).
- **Keep the UI smooth**: the visible drop animates at the same ~30 fps; the search never blocks the
  render thread (run it in a Web Worker).
- **Preserve the server-side engine** (the owner's explicit ask): `FruitCakeWorld`, `FruitCakeEnv`,
  `FruitCakeSearch`, and `FruitCakeController` stay in the repo — they still train the net and serve as
  the **parity oracle** for the conformance test. They simply leave the serving hot path.
- **Establish a reusable pattern** for moving the other streamed demos (Snake, MountainCar) client-side.

### Non-Goals (v1)
- A NN framework in the browser (ONNX Runtime Web / TensorFlow.js) — the net is 91k params; a flat
  hand-written matvec is smaller and faster for this call shape (§4.G0).
- Bit-identical C#/TS floating-point. We require **behavioural** parity (same merges, same settled
  layout within tolerance, and — the gate that matters — the **same column choice**), not bit-equality.
- Moving the cube / 2048 / Rush Hour solvers (they're request/response — not a per-viewer cost).
- Retraining the net. This ships the **current** checkpoint, run in a new place. (It composes with any
  future M29 retrain — the client just loads whatever `.ckpt` is shipped.)
- Removing the server endpoint's *code*. We preserve it (possibly behind a flag); we only stop the
  client from using it.

---

## 4. Design

### 4.G0 The net — hand-port the forward pass to TypeScript
The shipped net (`DuelingQNet`, `DuelingQNet.cs`) is small and rigidly structured: input **83** →
`Linear`+ReLU **256** → `Linear`+ReLU **256** → value head `Linear 256→1` + advantage head
`Linear 256→14`, recombined `Q = A + (V − mean_a A)` (`DuelingQNet.cs:62-74`). **91,151 parameters
(~356 KB)**, `noisy=false` ⇒ deterministic. The forward pass is ~30 lines of TS over `Float32Array`.

- **Do not** use ORT-Web/TF.js. The cost shape is thousands of tiny **batch-1** calls per drop (§4.G4),
  not one big op; a framework adds a multi-MB payload + per-call dispatch to "accelerate" a 91k-param
  net — a net loss. A flat preloaded-weights matvec runs each forward pass in microseconds.
- **Weight delivery:** add a `GET /api/fruitcake/weights` endpoint that streams the loaded net's
  parameters as a flat little-endian `float32` blob + a tiny JSON header (`inputSize`, `hidden[]`,
  `actions`), in `DuelingQNet.Parameters()` order (`DuelingQNet.cs:76-82`). One source of truth (the
  shipped `.ckpt`, `FruitCakeModelService`), ~356 KB, fetched once and HTTP-cached. (Alternative: a
  build-time export committed to `assets/` — rejected as a second copy to keep in sync.) The browser
  parser is trivial and never sees the C# `DuelingQNetCheckpoint` binary format.
- **Fallback parity:** when no net is available the C# serving lambda falls back to
  `HeuristicBoardValue = −PileHeight` (`FruitCakeSearch.cs:37`). The client mirrors this if the weights
  fetch fails, so the demo always plays.

### 4.G1 The observation — port `BuildObservation` to TypeScript
`FruitCakeEnv.BuildObservation` (`FruitCakeEnv.cs:144-202`) is already **static and pure** (designed
for a non-env caller) — a direct TS port: 14 surface-height + 14 top-tier + 14 danger-margin + 14
merge-with-current + 14 adjacent-equal-pair + current one-hot(5) + next one-hot(5) + 3 globals = **83**.
Port `ColumnX`/`HeldY` (`:252-260`) alongside. Locked by a unit test vs C# golden observations (§4.G3).

### 4.G2 The physics — extend the existing TS `FruitWorld` with the search surface
`fruit-cake-physics.ts`'s `FruitWorld` already implements the full solver. It is **missing only the
search surface** the C# `FruitCakeWorld` exposes:
- `clone(rotationOff)` — deep copy that, with rotation off, **zeroes `invI`, `angle`, `angularVel`**
  (the TS world always sets `invI`, `fruit-cake-physics.ts:113`; the C# clone zeroes them,
  `FruitCakeWorld.cs:187-202`). The search must plan rotation-off to match the trained-on physics.
- `settleAfterDrop(settleSpeed, minSubsteps, maxSubsteps)` (↔ `FruitCakeWorld.cs:169-179`),
  `anyEjected` (`:133`), `anyRestingAboveDangerLine` (`:140`), `maxSpeed` (`:125`), `pileHeight` (`:155`).

~40 lines, mechanical. The **visible** drop still uses the existing **rotation-on** `FruitWorld` so
fruit roll (cosmetic, exactly as the server serves today); only the **search clones** run rotation-off.

### 4.G3 Physics conformance test — the central de-risk (NEW)
A cross-language golden-vector test (the one §4.8 recommended but never shipped):
- A C# tool/test emits golden traces: a **fixed seed + scripted drop columns** → for each drop, the
  settled board (sorted `{tier, x, y}` rounded to a tolerance) + merge points, **rotation-off**.
- The TS test replays the same script through `FruitWorld` (rotation-off) and asserts agreement within
  tolerance. Cross-runtime float math isn't bit-exact, so compare with a position epsilon and exact
  tier/merge-count equality.
- **The gate that actually matters** (more forgiving, and what we ship on): run the *full search* on
  both sides over a set of golden boards and assert the **same column choice**. The net's argmax over
  14 columns tolerates small position noise; equal column choice ⇒ equal play.

### 4.G4 The search — port `FruitCakeSearch` to TypeScript
A direct port of `FruitCakeSearch` (`FruitCakeSearch.cs`): `MaxDepth=3`, `TopK`, `TopK2`, lose-pruning
(`LosePenalty`), and the **expectimax chance node** over the unknown 3rd fruit (`:76-117`). Leaf value =
the serving lambda (`FruitCakeController.cs:69-77`): the net's max-Q **averaged over all 5 droppable
tiers** (so each leaf = 5 forward passes), heuristic fallback otherwise.

**Cost:** ≈780 settle-sims/decision, ≈5 net passes each (~3,900 tiny forward passes/drop). ~45 ms in C#
⇒ ~150–450 ms in JS. Mitigations, in order: (a) **Web Worker** (§4.G5) so it never blocks rendering;
(b) **depth-2** fallback if a device is slow (28% vs 50% watermelon, ~8× cheaper); (c) typed-array body
storage; (d) WASM SIMD only if ever needed (overkill for this size).

### 4.G5 Serving — client-side worker + the existing renderer
```
main thread: open game → post {board, current, next} to worker → await chosen column
           → spawn current at that column in the rotation-ON FruitWorld
           → step at ~30 fps, rendering (the existing render path), until settled
           → advance current/next → repeat; restart on game-over
worker:      run the depth-3 rotation-OFF search → post back the column
```
- The renderer is already frame-shaped (`{x,y,angle,tier}` + `score/held/next/danger`,
  `FRUITCAKE_AI_PRD.md` §4.6), so "Watch AI" reuses it — now reading from the **local** world instead of
  a streamed array. Human play is untouched.
- Replace `FruitCakeApi.connectLive` (the WebSocket, `fruit-cake-api.ts:37-44`) with a local
  worker-driven loop. The net weights are fetched once via `/api/fruitcake/weights`.
- With the worker, even ~300 ms think-time is invisible (the previous drop is still animating).

### 4.G6 Preserve the server engine; cut the client over
- Keep `FruitCakeWorld` / `FruitCakeEnv` / `FruitCakeSearch` (training + parity oracle) and
  `FruitCakeModelService` (now also feeds `/api/fruitcake/weights`).
- `FruitCakeController.Live` (the WS) is **preserved** but no longer used by the client — leave it in
  place (optionally behind a config flag / as a debug fallback). No deletion, per the owner's ask.
- `/api/fruitcake/status` stays (the client still gates "ready" on it).

---

## 5. Milestone plan

- **G0 — TS net + weight endpoint.** `/api/fruitcake/weights` (flat float32 + JSON header) + a TS
  `Float32Array` forward pass. *Gate: TS Q-values match C# `GreedyQAgent.QValues` on fixed observations
  within 1e-4; argmax identical.*
- **G1 — TS observation.** Port `BuildObservation` + `ColumnX`/`HeldY`. *Gate: TS obs matches C# golden
  observations element-wise within 1e-5.*
- **G2 — TS physics search surface.** Extend `FruitWorld` with `clone(rotationOff)` / `settleAfterDrop`
  / `anyEjected` / `anyRestingAboveDangerLine` / `maxSpeed` / `pileHeight`. *Gate: builds; clone is
  rotation-off-correct (invI/angle/angularVel zeroed).*
- **G3 — Conformance test (de-risk gate).** C# golden-trace emitter + TS replay; position-tolerant board
  equality + exact tier/merge equality; **plus** same-column-choice over golden boards. *Gate: both
  agree; this gate must pass before G6 cutover is trusted.*
- **G4 — TS search.** Port `FruitCakeSearch` (depth-3, TopK/TopK2, chance node, lose-prune) + the
  net-leaf lambda + heuristic fallback. *Gate: same column choice as C# on the golden boards (folds into
  G3).*
- **G5 — Worker + UI cutover.** Web Worker runs the search; main thread plays the drop in the rotation-on
  `FruitWorld` and renders; AI-mode toggle reuses the renderer; weights fetched once. *Gate: "Watch AI"
  plays end-to-end in-browser with no WebSocket; UI stays ~30 fps.*
- **G6 — Validation + relief.** ≥200-game distribution of the **client** agent vs the **server** agent
  (same net) — max-tier histogram + mean score within A/B noise. Measure per-drop worker time on
  mid/low hardware; enable the depth-2 fallback threshold if needed. Confirm the server holds no
  standing per-viewer CPU. *Gate: client strength ≈ server within noise; smooth on a mid laptop.*
- **G7 — Stretch: generalize.** Apply the same client-side pattern to Snake and MountainCar (tiny nets,
  light envs) — retire their per-connection `EpisodeStreamer` loops. Gated on G0–G6.

**Recommended first session:** G0 (net+weights, fully testable headless) → G1 → G3's conformance
harness early (it's the risk) → G2/G4 → G5 worker cutover → G6 validation.

---

## 6. Measurement
- **Strength:** the **≥200-game max-tier distribution + mean score** of the client agent vs the current
  server agent on the same net — must match within the env's A/B noise (never a 10-episode eval; M28).
  Reuse the `FruitCakeAb` methodology.
- **Smoothness:** visible drop sustains ~30 fps; worker think-time per drop measured on mid + low-end
  hardware (target: invisible behind the previous drop's animation; depth-2 fallback if a device
  exceeds budget).
- **Server relief:** per-viewer standing CPU for FruitCake → ~0 after cutover; the server serves only
  the SPA bundle + the cached weights blob.

## 7. Risks
| Risk | Mitigation |
|---|---|
| **C#↔TS physics drift weakens play** (now load-bearing) | The conformance test (G3) is the gate; ship on **same-column-choice** parity (forgiving of float noise), not bit-equality; the engines are already deliberate mirrors. |
| Search too slow on low-end devices | Web Worker (off the render thread) + depth-2 fallback (~8× cheaper) + typed-array bodies; WASM SIMD only if ever needed. |
| Float nondeterminism across runtimes | Compare with tolerance + behavioural (merge/column) equality; never assert bit-equality. |
| Weight blob staleness / caching | Single source (`/api/fruitcake/weights` from the loaded `.ckpt`); version/etag the blob; client refetches on net change. |
| Net leaf off-distribution from rotation mismatch | Search clones run **rotation-off** (as the net was trained); only the visible world is rotation-on (cosmetic), matching today's server behaviour. |
| Regression vs the proven server path | Server controller is **preserved** (optionally flag-gated) as a fallback + the parity oracle; cutover is reversible. |

## 8. Open questions
1. **Depth on the client:** keep depth-3 everywhere, or auto-drop to depth-2 below a measured per-drop
   time budget? (Lean: ship depth-3, add a runtime fallback in G6.)
2. **Server endpoint fate:** keep `Live` behind a config flag as a debug fallback, or leave it dead but
   present? (Owner asked to preserve; default to flag-gated-off.)
3. **Snake/MountainCar now or later?** G7 is the natural follow-up; do them in this milestone or a
   separate one?
4. **WASM SIMD** for the matvec — worth it for the lowest-end devices, or is plain `Float32Array` enough?
   (Measure in G6 before deciding.)

## 9. References
- Current server serving + search + pacing: `src/RLDemo.Web/Controllers/FruitCakeController.cs:49-132`
  (search `:78,99`; net leaf `:69-77`; pacing `:36,107-111`).
- Shared streamer (Snake/MountainCar per-connection loop): `src/RLDemo.Web/Services/EpisodeStreamer.cs:8-60`.
- Net + checkpoint: `src/…Core/Nn/DuelingQNet.cs:62-82`; `src/…Core/Checkpoints/DuelingQNetCheckpoint.cs`;
  load path `src/RLDemo.Web/Services/FruitCakeModelService.cs`.
- Observation + helpers: `src/…Environments/FruitCake/FruitCakeEnv.cs:144-260`.
- Search to port: `src/…Environments/FruitCake/FruitCakeSearch.cs` (full).
- Physics twins: `src/…Environments/FruitCake/FruitCakeWorld.cs` ↔
  `src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-physics.ts` (mirror headers at `:3` / `:1`).
- Client API to replace: `src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-api.ts:37-44`.
- Hosting / no limits: `Program.cs`, `docker-compose.yml`; CPU-only `Program.cs:34-37`, `docs/OPTIMIZATIONS.md`.
- Prior decisions reversed here: `FRUITCAKE_AI_PRD.md` §4.6 (server-authoritative) + §4.8 (single-source
  physics + the recommended-but-unbuilt conformance test).
