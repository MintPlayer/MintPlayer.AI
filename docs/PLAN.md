# RL.NET — Implementation Plan

Companion to [PRD.md](PRD.md). Each milestone ends in a **git commit on a passing gate**;
revert-friendly by design. Order is chosen so each milestone adds at most 2–3 genuinely new
components (the CleanRL/SB3 lesson: localize bugs by construction).

> **Status (2026-06-10): M0–M8 complete, every pre-registered gate passed.**
> The **interactive web playground** (PRD §7) is live for Rush Hour: draw on a canvas,
> play it yourself, AI-solve with back/forward playback, persisted model. Next is
> **M9** (2048 page + training-on-demand + gallery), then M10 (Docker). M11 = stretch.

## M0 — Skeleton + core contracts  *(part of the quick demo)* ✅

- Solution restructure: `src/RL.NET.Core`, `src/RL.NET.Environments`, `src/RL.NET.Demo`
  (existing console project), `tests/RL.NET.Tests` (xUnit).
- `IEnvironment<TObs,TAct>`, `StepResult<TObs>`, `EnvInfo`, `Space<T>` (`DiscreteSpace`, `BoxSpace`).
- `Xoshiro256StarStar` RNG + SplitMix64 `SeedSequence` fan-out.
- `MetricsLogger` (CSV + console), greedy evaluation loop, console renderer abstraction.
- Environments: deterministic **GridWorld 4×4**, **FrozenLake** (slippery ⅓-⅓-⅓).
- **Gate:** env dynamics unit tests pass (incl. FrozenLake slip distribution); RNG
  determinism test (same seed → identical sequences).

## M1 — Tabular agents + the quick demo  *(deliverable: watchable demo)* ✅
**Result:** GridWorld policy exactly optimal in 16/16 states (33 ms of training);
FrozenLake 74.2% success (≈ the VI-optimal policy's own rate under the 100-step cap).

- Q-learning + SARSA, epsilon schedule (linear decay), double-precision Q-tables.
- Value iteration (oracle for tests), policy/value console visualization (arrow map).
- Demo CLI: `train` → live metrics → animated greedy playback + policy map.
- **Gate:** Q-learning greedy policy == value-iteration policy on GridWorld (exact);
  FrozenLake success ≥ 0.70/100 episodes (median of 3 seeds); bitwise seed-determinism test.

## M2 — Tensors, autograd, NN  *(the from-scratch heart)* ✅
**Result:** managed GEMM 18–22 GFLOP/s single-thread; 3,441 Adam steps/s on the PRD
config (target ≥ 1,000); gradient checks caught a real transposed-GEMM argument bug.
*Deferred:* tensor/tape pooling (the zero-steady-state-allocation goal) — revisit if
profiling ever shows GC pressure; current throughput met targets without it.

- Spike first: benchmark hand-rolled GEMM (`TensorPrimitives.Dot` per row → tiled
  `Vector256<float>`) against the ≥1k Adam-steps/sec target before building everything else.
- `Tensor` (flat `float[]` + shape/strides, pooled), ~15-op tape autograd
  (matmul, add, mul, relu, tanh, exp, log, sum, mean, gather, broadcast…),
  `Linear`, ReLU/Tanh, softmax/log-softmax (log-sum-exp stable), MSE + Huber,
  Categorical distribution (sample/log-prob/entropy), Adam, global-norm grad clipping,
  He/Xavier init. `IComputeBackend` seam defined here.
- **Gate:** finite-difference gradient checks on every op + composed losses;
  GEMM benchmark target met; zero steady-state allocations in the training inner loop.

## M3 — CartPole + REINFORCE + DQN ✅
**Result:** CartPole port matches Gymnasium golden trajectories bit-for-bit (float32);
Double DQN solves CartPole in ~15k steps / 6.5 s with a perfect 500.0 final eval;
REINFORCE gate ≥ 400 passed. *Deferred:* full-resume checkpointing and the Dueling
head (Double DQN landed; neither was needed for any gate so far).

- **CartPole-v1 faithful port** (exact constants/update order from PRD §6) validated against
  committed golden trajectories from Python Gymnasium
  (`tools/generate_goldens.py` → `tests/RL.NET.Tests/Fixtures/cartpole_golden.json`).
- REINFORCE (reward-to-go, return normalization). Gate: CartPole ≥ 400 median/3 seeds +
  policy-gradient direction unit test (log-prob of rewarded action increases).
- DQN: circular replay buffer (**stores `terminated` only**), target network (hard sync),
  step-based epsilon decay, Huber loss, full-resume checkpointing. Then Double + Dueling as
  small deltas.
- **Gate:** DQN CartPole ≥ 475 median/3 seeds; overfit-one-transition test (loss→0, Q→r);
  truncation test (truncated transition bootstraps); replay wraparound unit test.

## M4 — PPO + vectorized environments  *(the scale-out milestone)* ✅
**Result:** PPO solves CartPole in ~20k env steps / 2.1 s (final eval 494/500).
Built as `VectorEnv` (one class, `parallel` flag); since each env owns its RNG,
parallel mode reproduces sequential **bitwise**, not just within tolerance.

- `IVectorEnv`: sequential-deterministic (default) + parallel (Tasks) modes, autoreset with
  `final_observation` passthrough in `EnvInfo`.
- Rollout buffer (steps × envs), GAE(λ) with the two distinct masks
  (`1−terminated` inside δ, `1−done_any` on the recursive term), values recorded pre-step,
  advantage normalization per minibatch, lr annealing, grad-norm clip 0.5, orthogonal init,
  approx-KL / clip-fraction / explained-variance logging.
- **Gate:** PPO CartPole ≥ 475 median/3 seeds; hand-computed 3-step GAE unit test;
  parallel mode reproduces sequential results at metric-level tolerance.

## M5 — 2048  *(owner's game #1)* ✅

- Env: 4×4 board, log2-encoded observation, action masking for invalid moves,
  spawn 90% 2 / 10% 4; console renderer.
- Action-mask infrastructure (`IActionMaskProvider`): masked exploration/argmax in
  DQN + GreedyQAgent, masked TD-target max via masks stored in the replay buffer,
  masked evaluation. (Categorical/PPO masking deferred to when a game needs it.)
- Afterstate TD(0) n-tuple learner (Szubert & Jaśkowski): **gate passed** — 84%
  2048-rate after 100k self-play games (168 s), vs the pre-registered ≥ 10% target;
  the ≥ 80% stretch criterion is met as well. Best tile observed: 8192.
- Generic masked Double DQN runs on the same env via `2048dqn` demo section
  (demonstrates the framework path; n-tuple remains the strong 2048 agent).

## M6 — Rush Hour  *(owner's game #2 — sparse-reward planning)* ✅

- Board logic in `RL.NET.Environments/RushHour` (6×6, vehicles len 2–3, action =
  vehicle·2+direction over a masked 32-action space); BFS optimal solver as oracle
  (also returns the optimal action sequence for future imitation use).
- Puzzle sets are generated deterministically from a seed (random layout + BFS filter
  into difficulty bands) instead of imported data files.
- **Gate passed:** masked Double DQN solves **30/30 (100%)** of the easy set
  (optimal 4–10) within 2× optimal after 40k steps (~1 min) — with the pure sparse
  −1/+100 reward; the potential-based shaped variant exists but wasn't needed.
- Still open for later: medium/hard curriculum and imitation warm-start from BFS
  solutions (M11). The interactive front-end is now the **M8 web playground page**;
  the existing `C:\Repos\Spelletjes\Rush Hour` app remains a possible puzzle-data
  source (M11 — request a clean checkout when that starts).

## M7 — Checkpointing + model store  *(prerequisite for the web app)* ✅
**Result:** all gates passed (11 new tests, 106 total green). The resume test
serializes a DQN run interrupted at 2k steps to bytes, deserializes, resumes on a
fresh env, and lands bitwise-identical to an uninterrupted 4k-step run — weights,
target net, both RNG streams and the env snapshot. Demo round-trip: CartPole trains
in ~6 s, `--save` writes an 18 KB checkpoint, `--load` skips training and reproduces
the 500.0 eval exactly.

- Checkpoint formats per the PRD decisions (`RLNet.Core.Checkpoints`): JSON for
  tabular Q-tables (`TabularCheckpoint`); versioned little-endian binary for MLPs
  (`MlpCheckpoint`), Adam moments+step (`AdamCheckpoint`) and the 2048 n-tuple tables
  (`NTuple2048Agent.Save/Load`, 17×65 536 floats ≈ 4.5 MB).
- Full DQN training resume: `DqnTrainingState` (nets, optimizer, replay buffer, RNG
  streams, current obs, env snapshot) + `DqnTrainer.Train(..., resume:)`. New
  `IStatefulEnvironment` (snapshot/restore complete env state incl. RNG) — implemented
  by CartPole; envs without it resume with a fresh episode (functional, not bitwise).
- `IModelStore` / `FileModelStore`: one *current* checkpoint per (environment,
  algorithm) as `<root>/<env>.<algo>.ckpt`; atomic save (temp + rename, old checkpoint
  survives a failed write); List/Delete for the web app's status pages.
- Demo: `--save` / `--load` / `--data <dir>` (default `./data`) on the cartpole, 2048,
  2048dqn and rushhour sections; "persisted model" launch profiles added.
- **Gate (passed):** round-trip tests — reloaded agents bitwise-identical (MLP forward
  pass, n-tuple eval games, tabular Q exact); interrupted-and-resumed DQN
  bitwise-matches uninterrupted; atomic-save failure test.

## M8 — Web host + Rush Hour page  *(first end-to-end playground slice)* ✅
**Result: both gates passed.** Playwright e2e against the running dev host: drew the
hand-verified optimal-7 puzzle on the canvas, played it manually to a win (7 moves),
reset to the drawing, hit "Solve with AI" — **the DQN solved it in 7 moves (optimal)**
— and stepped the playback back/forward to the red-car-at-exit end state. API gate
(xUnit, `Category=Slow`): a generated easy puzzle returns a trajectory that is verified
move-by-move legal, matches the server-reported states, ends solved, within 2× optimal.

- `src/RL.NET.Web`: ASP.NET Core host + Angular 22 ClientApp (zoneless, signals) wired
  through **MintPlayer.AspNetCore.SpaServices** (`UseSpaImproved` + `UseAngularCliServer`
  with the `Local:` cliRegex — running the host is all that's needed; never start
  `ng serve` separately). Landing page lists games.
- Rush Hour page: HTML5-canvas board editor (red car/car/truck/erase tools, overlap +
  exit-row validation, live BFS feedback "solvable — optimal N"), **manual play** with
  real rules (click to select, arrow keys/buttons), **reset to drawn state**.
- Solve API: `POST /api/rushhour/solve` → stored model (M7) + BFS oracle → **trajectory**
  (action + resulting positions per step) for both the AI and the optimal solution, plus
  metadata. Also `analyze` (validation + BFS, no model) and `status`.
- `RushHourModelService`: lazy-loads the checkpoint from the model store; if absent, a
  hosted service trains it at startup (progress streamed to the UI banner; the demo run
  stopped at 40k steps, eval 95.7) and saves it — restarts load instantly.
- Playback UI: ⏮ ◀ ▶ ⏭ + play/pause + scrubber, AI/optimal trajectory toggle,
  last-moved-vehicle highlight; honest "AI did not solve this one" path when it fails.

**Post-gate additions (same day, user feedback):**
- Red vehicle is fully user-configurable: position along the exit row AND length —
  new "Red truck (3)" tool next to "Red car (2)". Tool glyphs switched to
  universally-rendered ↔/↕ arrows (the ▭/▯ rectangles were font-dependent tofu).
- `RushHourGenerator` gained `varyRedLength` (off by default — existing seeds stay
  bitwise-identical, M6 reproducibility intact).
- **Generalization rework:** a model trained on a fixed 30-puzzle set memorizes it and
  fails on arbitrary drawn boards. The web model now trains on **2,000 generated
  puzzles** (optimal 2–12, 2–9 vehicles, both red lengths, 256×256 net) — reached the
  eval-92 threshold at 300k steps (92.3). Browser-verified: drawn boards with shifted
  red cars and red trucks now solve (mostly optimally); one truck-blocks-truck layout
  remains an honest-finding failure (M11 imitation/curriculum is the principled fix).
  The training recipe lives in `RushHourModelService.TrainingPuzzles()/TrainingOptions()`
  and the Slow API gate consumes the same statics, so test and production can't drift.
- Solve rollout budget now scales with difficulty: `max(60, 2 × optimal)` moves.
- **Slide compaction** (`RushHourSolver.CompactSolution`): BFS returns an arbitrary
  order among the equally-optimal solutions, which can split one fluid slide around
  unrelated moves ("R left 1 … R left 1" instead of "R left 2"). A greedy run-reordering
  pass groups commutable same-vehicle moves — identical move count, fewer visible
  slides (card 40: 62 → 53 slides over the same 81 cell-moves; the official card
  solution has 51). Used for the playground's optimal trajectory.
- **Official ThinkFun cards 38/39/40** encoded as solver regression tests with their
  published solutions replayed move-by-move on our board (legality asserted per
  single-cell slide): BFS optima 77/82/81 single-cell moves, and all three printed
  solutions turn out to be single-cell optimal once the final drive-out through the
  exit is discounted. Card 40 was drawn vehicle-by-vehicle in the browser e2e: analyze
  reported 81, the AI failed honestly within its 162-move budget, and the 81-step
  optimal playback scrubbed to the red-car-at-exit frame.

## M9 — 2048 page + training-on-demand + gallery ✅
**Result: gates passed (browser-verified).** With an empty store the page shows live
training progress (10k/100k games, avg score climbing) while drawing/manual play stay
usable, and solve returns 503 + status until ready; warm store solves instantly. The
freshly-trained n-tuple played the starter board for **2,491 moves, 55,480 points,
best tile 4096 — reached 2048** — scrubbed instantly in the browser from the compact
trajectory. The playout landed in the gallery; clicking the entry replays it.

- 2048 page: canvas tile editor (click/right-click cycles values), manual play with
  the real merge/spawn rules + arrow keys, reset to drawn state. The solve response is
  **compact**: per step (action, spawn cell, spawn value, score gained) — 2048 states
  are derivable deterministically, so per-step boards are omitted and `finalCells`
  serves as the replay checksum (a test replays client-side rules and must land
  exactly there). Spawns are seeded from a board hash → same drawing, same playout.
- Training at startup, not per-request: `ITrainableModelService` (Rush Hour + 2048)
  run in parallel by one hosted service; progress polled by the UI banner; checkpoint
  saved on finish (restarts load instantly). Functionally equivalent to the planned
  enqueue-on-solve (training is general, independent of the submitted state).
- **Public gallery:** every solve is persisted (JSON per entry under `data/gallery`,
  atomic write, corrupt entries skipped); `/gallery` lists newest-first and links
  `/rushhour?replay=<id>` / `/2048?replay=<id>`, which load the entry straight into
  playback. Unit-tested across store re-instantiation (restart survival).

## M10 — Docker

- Multi-stage Dockerfile: Node + .NET SDK build stage (Angular production build +
  `dotnet publish`) → ASP.NET runtime image; data directory at `/data` (model store +
  gallery) declared as a **volume**; configuration via environment variables.
- docker-compose example with a named volume; README run instructions.
- **Gate:** `docker build` + run, solve a drawn puzzle in the browser; `docker restart`
  — model store and gallery still there (volume), no retraining needed.

## M11 — Stretch (unordered, not started)

MountainCar (exploration stress test) · Snake (demo gif) · TorchSharp `IComputeBackend`
implementation · TensorBoard event writer · self-play scaffolding (TicTacToe + minimax oracle)
· NuGet packaging · Dueling DQN head (deferred from M3) · tensor/tape pooling (deferred
from M2) · Categorical/PPO action masking (deferred from M5) · harder Rush Hour sets with
curriculum + imitation from BFS solutions · watch-only playground pages for CartPole/2048
self-play · importing puzzles from `C:\Repos\Spelletjes\Rush Hour` as gallery data
(ask for a clean checkout first).

## Testing strategy (cross-cutting, from research)

1. **Known-solved thresholds** as integration tests (median over ≥3 seeds) — slow bucket.
2. **Bitwise seed-determinism** traces per algorithm (sequential mode).
3. **Hand-computed unit tests**: discounted returns, 3-step GAE, buffer wraparound,
   schedule endpoints, terminated-vs-truncated target masking.
4. **Gradient sanity**: finite differences; overfit-one-transition; PG direction test.
5. **Probe environments**: one-step bandit (exploration bugs), constant-reward env
   (V must converge to r/(1−γ)) — isolate value vs policy vs exploration failures.
6. **Golden trajectories** from Python Gymnasium for ported envs.

## Gate results at a glance

| Milestone | Gate | Result |
|---|---|---|
| M1 tabular | FrozenLake success ≥ 70% | 74.2% (≈ theoretical optimum) |
| M2 numerics | ≥ 1k Adam steps/s (batch 64, 4→64→64→2) | 3,441/s; GEMM 18–22 GFLOP/s |
| M3 DQN | CartPole ≥ 475 (median/3 seeds) | 500.0 in ~15k steps / 6.5 s |
| M3 REINFORCE | CartPole ≥ 400 (median/3 seeds) | passed |
| M4 PPO | CartPole ≥ 475 (median/3 seeds) | 494.1 in ~20k steps / 2.1 s |
| M5 2048 | 2048-tile rate ≥ 10% (stretch 80%) | **84%** after 100k games / 168 s |
| M6 Rush Hour | ≥ 90% of easy set within 2× optimal | **100%** (30/30) after 40k steps / ~1 min |
| M7 checkpoints | resumed DQN bitwise == uninterrupted | passed (+ all round-trips bitwise/exact) |
| M8 web playground | e2e draw→solve→playback; API trajectory ≤ 2× optimal | passed (AI solved the e2e puzzle optimally, 7/7) |

## Immediate next step

**Build M9 (2048 page + training-on-demand + gallery)** — 2048 canvas editor, n-tuple
playout trajectories (move + spawned tile per step), background training jobs with
browser-visible progress, persisted public game gallery.
Run the playground: `dotnet run --project src/RL.NET.Web` (Development spawns + proxies
the Angular dev server itself — do not run `ng serve`). Console demos:
`dotnet run --project src/RL.NET.Demo -c Release -- [grid|lake|cartpole|ppo|2048|2048dqn|rushhour]
[seed] [--load] [--save] [--data <dir>]`. Tests: `dotnet test` (`Category=Slow` for gates).
