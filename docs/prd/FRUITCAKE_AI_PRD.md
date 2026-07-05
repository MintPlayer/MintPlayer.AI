# FruitCake AI — Product Requirements Document

> Train an agent to play **FruitCake** (the Suika-style drop-and-merge game) and surface it in the
> playground as a "Watch the AI play" mode alongside human play — making FruitCake the playground's
> first game that is **both** hand-built **and** has a learned agent.

- **Status:** Draft v1.0 · 2026-06-23 (investigation complete; not yet started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** the shipped FruitCake game (PR #12) + rolling-physics (PR #13). Companion to
  [PRD.md](PRD.md) (the RL library) and the add-a-game checklist in [../ADDING_A_GAME.md](../ADDING_A_GAME.md).

---

## 1. Summary & Vision

FruitCake is a continuous-aim, long-horizon, stochastic physics puzzle. We want an agent —
trained with the from-scratch C# `MintPlayer.AI.ReinforcementLearning` library — that plays it
competently, viewable live in the browser the same way Snake and Mountain Car are: the server runs
the game and the agent and streams the play; the page renders it.

The decisive insight from the investigation: **every prior Suika-RL effort was bottlenecked by
environment speed** — they screen-scraped a browser at real time (~0.5 s per drop), making data
collection the dominant cost. We avoid that entirely with a **headless, deterministic, native C#
simulation that runs as fast as the CPU allows**. That single fact is what makes from-scratch RL
here realistic. (Prior art: [suika_rl](https://github.com/edwhu/suika_rl),
[LIACS thesis 2024–25](https://theses.liacs.nl/pdf/2024-2025-PoelsmaJJulian.pdf).)

---

## 2. Goals & Non-Goals

### Goals
- A trained agent that plays FruitCake clearly better than random/naïve play, viewable live in the
  browser ("Watch AI" mode) next to the existing "Play yourself".
- A **headless C# FruitCake environment** that runs entirely in simulated time — training never waits
  on real-time physics (see §4.2, the hard requirement).
- Reuse the library's existing, proven machinery (Snake's Double+Dueling DQN path) rather than new
  algorithms or a new network type.
- A **greedy heuristic baseline** shipped first: an honest comparison bar, an immediately-watchable
  "AI", and a source of demonstration data to warm-start RL.
- Reproducible, resumable training via the Lab campaign harness; deployable checkpoint shipped via
  Git LFS; the web stays **load-only** (never trains).

### Non-Goals (v1)
- Superhuman or competition-grade play. Target is "competent and clearly learned" (see §7).
- A convolutional / image-based agent. The library has **no conv layer** (§4.4); adding one is a
  large separate effort and unnecessary if the engineered-feature observation works.
- Continuous-control (SAC). `SacTrainer` isn't in this tree; the action space is discretized (§4.3).
- Changing human play. The existing client-side TS game is untouched; AI mode is additive.
- Bit-identical C#/TS physics. Server-authoritative serving (§4.6) makes exact parity unnecessary.

---

## 3. Background & the central constraint

Unlike Rush Hour / 2048 / Cube / Snake / Mountain Car, **FruitCake's game logic and physics live
only in TypeScript** in the web client (`fruit-cake-physics.ts`, `fruit-cake-game.ts`). Training
runs in **C#** (the RL library + the Lab/Console). Therefore the **single biggest build item, and a
hard prerequisite for everything else, is a C# port of the FruitCake physics + rules** as a headless
environment. This document treats that port as central.

---

## 4. Architecture

### 4.1 Headless C# environment

A `FruitCakeEnv : IEnvironment<float[], int>` (+ `IStatefulEnvironment` for bitwise-resumable
training) under `src/MintPlayer.AI.ReinforcementLearning.Environments/FruitCake/`, following the
exact conventions of `SnakeEnv` / `MountainCarEnv`:

- A C# port of the circle solver + merge/scoring/game-over (~450–600 lines, mechanical translation).
  **Everything UX-only is dropped: the 500 ms drop cooldown, effects/particles, audio, rendering.**
- `Step(action)` = **one drop**: place the current fruit at the chosen column's x, then simulate the
  physics to rest, and return the new observation + reward. (One decision per `Step` — correct for RL.)
- **Deterministic & seeded:** the fruit sequence uses the library's `Xoshiro256StarStar` (seeded in
  `Reset`, serialized in `SaveState`), replacing `Math.random()`. The solver is array-ordered and
  single-threaded, so contact/merge order is deterministic given spawn order. (Float results are
  deterministic within one runtime; cross-runtime bit-equality is neither needed nor assumed — §4.6.)
- **Terminated vs truncated** kept strictly distinct: terminated = danger-line/eject game-over;
  truncated = an optional max-drops cap.
- The ported solver is a **standalone `FruitWorld` class shared by both the env and the live WS
  handler** (§4.6) — written once, used for training (step-to-rest) and serving (tick-to-snapshot).
- **Training physics is linear-only (rotation off).** Merges depend only on positions and tiers, not
  orientation, so the trainer runs the cheaper, fully-deterministic linear solver; rotation is a
  rendering toggle re-enabled only for the live stream (§4.6). The agent transfers unchanged.

### 4.2 Throughput — simulated time, never real time *(hard requirement)*

**Training must never be gated to wall-clock.** A fruit taking "~1 second to fall" is ~60 simulated
60 Hz sub-steps run in a tight compute loop with **no sleeps, no timers, no frame pacing**. This is
the project's key advantage over all prior Suika-RL work and is non-negotiable.

Physics-in-the-loop is **CPU-bound** (the policy net is tiny; the RTX 3060 buys little here) and is
the dominant cost. The throughput plan:

- **Early-settle detection** (biggest win): stop sub-stepping the instant max body speed < threshold
  and the merge queue is empty, instead of a fixed horizon. Most drops settle in well under a second
  of sim-time.
- **Spatial-hash broadphase** to replace the O(n²) pair scan (~5 rebuilds/step × up to ~100 bodies)
  with ~O(n) — directly attacks the late-game cost.
- **Parallel `VectorEnv`** (already in Core, `parallel: true`; each env owns its RNG so N-across-cores
  equals sequential) → ~6–7× on the 8-core dev box.
- Lean training sim: fewer solver iterations / optional fruit-count cap during training, restored for
  eval. Rotation off (§4.1).

The only place real-time pacing remains is the **live "Watch AI" stream** (humans watch at ~60 fps);
that is the serving layer's concern (§4.6), not training's.

### 4.3 Observation, action, reward

- **Observation — a compact fixed feature vector (~40 dims), not an image** (no CNN available):
  - Per-column **surface** over `Ncol = 14` columns (≈44 px ≈ smallest-fruit diameter): normalized
    surface height + top-fruit tier → 28 dims.
  - **Current** tier one-hot(5) + **next** tier one-hot(5) → 10 dims (the game pivots on these).
  - Globals: normalized fruit count, board fill ratio, max-height-vs-danger-line margin → ~3 dims.
  - A surface encoding hides pockets under overhangs, but Suika piles are near-convex so it's
    acceptable; fall back to a flattened coarse occupancy grid (still MLP) only if it plateaus.
- **Action — discrete drop column:** `DiscreteSpace(14)`; action → x = column center, clamped to
  `[r + inset, W − r − inset]` for the current fruit's radius. No action mask needed (every column is
  always legal). Discretizing matches the library's strongest path and avoids continuous-control
  variance; continuous-x PPO is a fallback only if 14 columns prove too coarse.
- **Reward:** per step, the normalized `Δscore` the drop produced (e.g. `/10` or `log1p`); small
  terminal penalty (≈ −1) to value survival; `γ = 0.99` (long horizon). Optional shaping **only if it
  won't learn to survive**: a small penalty ∝ pile height / fill ratio, and/or a small bonus for
  producing a higher tier (counteracts sparse high-tier payoff). Start without shaping.

### 4.4 Algorithm & network

- **Double + Dueling DQN** (`DqnTrainer` + `DuelingQNet`) on a **`ResidualMlp` trunk** (~256-wide, a
  few residual blocks), ε-greedy, `ReplayBuffer`, target net — mirroring the Snake setup that already
  works in this codebase. Verified library gaps: **no convolutional layer** and **no `SacTrainer`** in
  this tree, which is exactly why the design is feature-vector + discrete DQN.

### 4.5 Heuristic baseline + demonstration warm-start

Build a **greedy/1-ply heuristic first** and ship it alongside the learned agent:

- For each candidate column, cheaply simulate the current drop and score the outcome (immediate merges
  + resulting cascade, minus peak-height / danger-line proximity). Pick the best. With the fast
  headless sim, optional shallow lookahead over the *known* next fruit is cheap.
- Value: a strong, cheap **v1 "AI"**; an honest **comparison bar** for the RL agent; and a generator
  of **demonstration data** to warm-start DQN (DQfD-style — the SDK already supports demonstration
  datasets, [PRD.md](PRD.md) §8/asset-portability), de-risking the learner.

### 4.6 Serving — server-authoritative stream

**Approach A (server-authoritative), with a bespoke WebSocket handler.** The server's C# physics is
authoritative; the browser only renders. This makes C#↔TS physics divergence irrelevant: in AI mode
the watched game *is* the C# simulation. (Rejected: "client simulates in TS, asks the server for each
drop-x" — the agent was trained against C# physics and would act on a board that resolves differently
in the TS solver.)

- **Renderer fit is excellent.** The existing canvas renderer already draws each fruit purely from
  `{x, y, angle, tier}` plus `score / heldTier / nextTier / dangerActive`. A streamed frame is exactly
  that payload; AI mode reuses `render()` essentially unchanged, reading fruit from a streamed array
  instead of the local `FruitWorld`. Human play keeps the local TS physics.
- **Do NOT reuse `EpisodeStreamer`.** Snake/Mountain Car stream one frame per `env.Step` (one action
  per tick). FruitCake has **two timescales**: the agent decides once per *drop*, but viewers watch
  ~60 Hz of falling/rolling/merging *between* decisions. The custom handler owns the C# `FruitWorld` +
  agent and loops `{ agent picks x → spawn → tick at 60 Hz, emitting a frame each tick (or every 2nd)
  until settled → next decision }`, restarting on game-over — same 503/connection/`RequestAborted`
  conventions as `EpisodeStreamer`, but streaming intra-drop frames it can't.
- **Rotation on for the stream:** the shared `FruitWorld` runs with rotation enabled when serving (for
  visual parity with human play), off when training (for speed). The agent is unaffected.

### 4.7 Training pipeline

- **Where:** the **Lab campaign harness** (not the Console one-shot, not the web). A
  `FruitCakeCampaign : ITrainingCampaign` cloned from `SnakeDqnCampaign`: `Environment="fruitcake"`,
  save-best the deployable net under `fruitcake/dqn` + full `DqnTrainingState` under
  `fruitcake/dqn-state` for bitwise resume; eval metric = **mean score / mean max-tier reached**
  (score-maximizing paradigm, `IsComplete=false`, run to the wall-clock budget). Wire `--game
  fruitcake` in `Lab/Program.cs`; add a `CampaignContractTests` case (fresh → `TrainChunk` advances →
  `Checkpoint` → fresh instance `Resume`s).
- **Checkpoint + shipping:** `DuelingQNetCheckpoint` (the format Snake/the web already load). Commit
  the deployable `.ckpt` to repo `models/` via **Git LFS** (`*.ckpt` tracked); `Program.cs` copies
  `SeedModelsDirectory` → `/data` at startup; CI must `git lfs pull` before `docker build`.
- **Web is load-only:** `FruitCakeModelService.Initialize` loads the shipped checkpoint or sets
  `Status=Failed`; `status` reports `loading|ready|failed`; the WS `live` returns **503 while not
  ready**; the frontend polls `status` and connects when ready — identical to Snake/Mountain Car.

### 4.8 Single-source physics — can the solver be written once?

> **UPDATE 2026-07-04 — the premise below changed.** This section's "no maintained C#↔TS transpiler exists"
> conclusion is now obsolete: **MintPlayer.Polyglot** ships (v0.1.0), and the FruitCake solver is its north-star
> conformance sample (one `.pg` → byte-identical C#/TS). A 3-agent investigation confirmed the solver core is a
> clean fit (pure `+ − × ÷ √`, no transcendentals, the two twins are already 1:1). Single-source is now viable and
> planned — see **[`POLYGLOT_FRUITCAKE_PRD.md`](POLYGLOT_FRUITCAKE_PRD.md)** (PLAN M31). The analysis below is kept
> for historical context (it was correct for the transpiler landscape at the time).

Short answer (historical): **not cleanly via transpilation, and with server-authoritative serving (§4.6) it isn't
required.**

- **Generic C#↔TS transpilation is not a maintained, viable path.** C#→JS/TS tools (Bridge.NET,
  JSIL, SharpKit) are discontinued; h5 (a Bridge fork) is niche and undependable; there is no mature
  TS→C# transpiler. Depending on an unmaintained transpiler for a core system is a worse risk than
  two small hand-ports.
- **The two realistic single-source routes, and why neither fits v1:**
  - **C# → WebAssembly** (C# as source of truth — training needs it natively fast; the browser runs
    the same compiled logic). Cost: the .NET WASM runtime (~MBs) + a build step bolted onto an Angular
    SPA that ships no .NET today, plus JS↔WASM marshalling each frame (mitigable via a shared buffer).
    Heavy for a casual game; the only route worth revisiting if exact parity ever becomes mandatory.
  - **Shared native core (Rust/C++) → wasm for web + P/Invoke for .NET.** Best raw perf both sides,
    but introduces **native dependencies into the .NET side — against the library's from-scratch,
    zero-native-deps pillar** ([PRD.md](PRD.md) §1/§4). Rejected for this repo.
- **Why it isn't needed (v1):** server-authoritative serving makes the C# sim authoritative for AI
  mode; the TS solver is used **only for human play**. The two must merely *feel* consistent, not be
  identical — so single-source buys consistency polish, not correctness.
- **Recommended instead — one reference spec, two small ports, locked by a conformance test:** keep
  the TS solver as the reference, hand-port it to C# once (milestone A0, ~300 lines), and add a
  **cross-language golden-vector conformance test**: a fixed seed + scripted drop sequence produces a
  sequence of board-state hashes; assert the TS and C# solvers agree (exact where floating-point
  allows, else within tolerance) so drift is caught in CI. Cheap, no runtime dependency, and it
  targets the drift risk directly.

---

## 5. Add-a-game layer mapping

| Layer | FruitCake specifics |
|---|---|
| **1. Environment** | `FruitCakeEnv : IEnvironment<float[],int>` (+ `IStatefulEnvironment`) wrapping the ported `FruitWorld`; `DiscreteSpace(14)` action, ~40-dim `BoxSpace` obs. **The dominant work item.** |
| **2. Model service** | `FruitCakeModelService : IModelStartupService`, near-verbatim `SnakeModelService` (lazy `TryLoadFromStore` → `DuelingQNetCheckpoint.Load` → `GreedyQAgent`, stale-checkpoint input-width guard, `Status`/`Error`). `EnvironmentId="fruitcake"`, `AlgorithmId="dqn"`. |
| **3. Controller / WS** | `FruitCakeController`: `status` + a **custom drop-loop streamer** (not `EpisodeStreamer`); new `FruitCakeFrameDto { Fruit[]{x,y,angle,tier}, HeldTier, NextTier, Score, Danger, Done }`. |
| **4. DI / startup** | Two lines in `Program.cs` mirroring Snake; `UseWebSockets()` already present; seed-copy is game-agnostic. |
| **5. Frontend** | *Component already exists.* Add an **AI-mode toggle** ("Watch AI" vs "Play yourself") to `fruit-cake.ts` + a `FruitCakeApi` (copy `snake-api.ts`): in AI mode stop the local rAF/physics and render streamed frames. Human mode untouched; no new route. |
| **6. Console / Lab** | `FruitCakeCampaign` + `--game fruitcake` dispatch in `Lab/Program.cs`; campaign contract test. |

---

## 6. Phased plan (proposed milestones)

- **A0 — C# `FruitWorld` port + `FruitCakeEnv`.** Deterministic, seeded, headless, fast-forwarded
  (§4.1–4.2), linear-only training mode. Parity smoke-test vs the TS game on a few seeded sequences.
  *Gate: env steps at thousands of drops/sec single-thread with early-settle; deterministic replays.*
- **A1 — Heuristic baseline.** Greedy 1-ply (+ optional 1-fruit lookahead) policy (§4.5). Wire serving
  (§4.6) so it's watchable immediately. *Gate: clearly beats random; reaches mid-chain fruit.*
- **A2 — DQN training (Lab).** Surface obs, 14-col discrete, Double+Dueling DQN on `ResidualMlp`;
  optionally warm-start from A1 demonstrations. Throughput work (spatial hash, parallel envs) as
  needed. *Gate: matches/beats the heuristic on mean score / max tier.*
- **A3 — Ship.** Best checkpoint to `models/` (LFS), `FruitCakeModelService` load-only, AI-mode
  frontend toggle, status/503 contract, tests. *Gate: live "Watch AI" plays end-to-end.*

---

## 7. Success criteria / targets

- **Minimum viable (high confidence):** clearly beats random and the naïve baseline; reliably reaches
  mid-chain fruit (apple→peach, tiers 6–8) with occasional pineapple; survives meaningfully longer
  than untrained play.
- **Ambitious (plausible, more tuning):** routinely reaches melon/watermelon; mean score rivaling a
  decent human.
- **Engineering:** training runs entirely in simulated time (no real-time waits); env is deterministic
  & resumable; live AI mode streams end-to-end with the existing renderer; web never trains.

---

## 8. Key risks

| Risk | Mitigation |
|---|---|
| **C# physics port is the gating cost** (game logic is TS-only) | Treat A0 as the central milestone; share one `FruitWorld` between env and serving; parity is only "behaviourally plausible," not bit-exact (server-authoritative serving removes the parity requirement). |
| **Physics-in-the-loop throughput** (CPU-bound, O(n²), settle-per-drop) | Early-settle + spatial-hash broadphase + parallel `VectorEnv` + lean training sim (rotation off, fewer iters). §4.2. |
| **Learnability** (long horizon, stochastic next-fruit, sparse high-tier reward) | current+next tier in obs; γ=0.99; reward shaping if needed; heuristic-demonstration warm-start; the heuristic baseline guarantees a shippable "AI" regardless. |
| **No CNN / no SAC in the library** | Engineered-feature obs + discrete DQN by design — needs neither. |
| **Two timescales break the generic streamer** | Bespoke WS handler emitting intra-drop frames (§4.6). |
| **C#/TS physics drift over time** | Server-authoritative AI mode renders the C# sim, so AI play needs no parity with TS. Single-source via transpilation isn't viable; keep two small ports locked by a cross-language golden-vector conformance test in CI (§4.8). |

---

## 9. Open questions

1. **Observation fidelity:** does the 14-column surface vector suffice, or is a flattened coarse
   occupancy grid (or eventually a real conv layer in the library) needed for strong play?
2. **Discrete granularity:** is 14 columns enough aim resolution, or do we need more bins / continuous
   PPO?
3. **Heuristic depth:** is greedy 1-ply enough to be a satisfying shipped baseline, or is 1-fruit
   lookahead worth it?
4. **Shaping:** can it learn survival from `Δscore` + terminal penalty alone, or is height-penalty
   shaping required?
5. **Compute budget:** target wall-clock for A2 on the dev box (CPU-bound) — how many hours buys the
   "minimum viable" bar?
6. **Single-source physics (§4.8):** is the two-ports-plus-conformance-test approach enough long-term,
   or is C#→WASM worth revisiting later to guarantee exact human/AI parity?

## 10. Follow-up — NoisyNets experiment + A/B harness (2026-06-26, PR #15)

After the shipped DQN, two exploration paths were tried to push it further (detail in `NOISYNETS_PRD.md`):

- **NoisyNets** (learned exploration instead of ε-greedy) — built as a library capability (N0–N3) and run
  on FruitCake. Warm-starting the strong net into full σ₀=0.5 **degraded** it (too aggressive for a refined
  policy); a **from-scratch** noisy run was the right use. A **200-game paired A/B** (`FruitCakeAb`,
  `--game fruitcake --ab --baseline <dir>`) then showed it **matched but did not beat** ε-greedy (702.1 vs
  714.4, Δ −12.3 ± 29.8 SE). **Not shipped**; `models/fruitcake.dqn.ckpt` unchanged.
- **Methodology correction (answers a latent assumption in §7):** this env's eval is **extremely
  seed-sensitive** — a 10-episode eval bounced 750–971 on the *same* net, and the campaign's "886" was
  seed-luck. Robustly, nets sit ~700–714. **Use the multi-seed paired A/B (`--ab`), never a 10-episode
  eval, to compare nets.** This supersedes any single-eval "score" cited earlier in this PRD.
- **Continued ε-greedy — the caveat that matters.** The historical 741→886 gain was a **full-state resume**
  (preserved replay buffer + annealed optimizer + ε≈0.05) over a long run; that state is ephemeral and gone.
  A short **net-only warm-start** (fresh buffer/optimizer, ε=0.2) instead **degrades the strong net first** —
  a 20-min/134k-drop run dipped 882→557→571 and never beat the 886 gate. So continued ε-greedy only helps
  with full-state resume **or** a *long* run at *near-greedy* ε (~0.05); a short moderate-ε warm-start is the
  wrong shape. The shipped net (~700–714 robust) appears **plateaued for cheap continuation**. (A vectorized-
  envs speedup was also tried and reverted — the backend already parallelizes data-gen; `docs/OPTIMIZATIONS.md`.)
