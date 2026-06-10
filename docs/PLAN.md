# RL.NET — Implementation Plan

Companion to [PRD.md](PRD.md). Each milestone ends in a **git commit on a passing gate**;
revert-friendly by design. Order is chosen so each milestone adds at most 2–3 genuinely new
components (the CleanRL/SB3 lesson: localize bugs by construction).

> **Status (2026-06-10): M0–M6 (the library) all complete, every pre-registered gate
> passed.** New requirement inserted the same day: the **interactive web playground**
> (PRD §7) — planned below as **M7–M10**, none started. The former stretch list is now
> M11; checkpointing moved out of it into M7 as a hard requirement.

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

## M7 — Checkpointing + model store  *(prerequisite for the web app)*

- Checkpoint formats per the PRD decisions: JSON for tabular Q-tables; versioned
  little-endian binary for NN weights + Adam state + RNG state (full training resume);
  versioned binary for the 2048 n-tuple weight tables (17×65 536 floats — by far the
  largest artifact, ~4.4 MB).
- `IModelStore`: one *current* checkpoint per (environment, algorithm) under a
  configurable data directory; atomic save (write-temp-then-rename); load-or-null.
- Demo sections gain `--save`/`--load` so trained agents survive between runs.
- **Gate:** round-trip tests — reloaded agent produces bitwise-identical greedy
  evaluation; an interrupted-and-resumed DQN run (weights + optimizer + RNG restored)
  bitwise-matches an uninterrupted run with the same master seed.

## M8 — Web host + Rush Hour page  *(first end-to-end playground slice)*

- `src/RL.NET.Web`: ASP.NET Core host + Angular ClientApp wired through
  **MintPlayer.AspNetCore.SpaServices** (`UseAngularCliServer` in dev — running the host
  is all that's needed; never start `ng serve` separately). Landing page lists games.
- Rush Hour page: HTML5-canvas board editor (place/drag vehicles, validation:
  overlaps, red car on exit row), **manual play** with real rules, **reset to drawn
  state**.
- Solve API: `POST /api/rushhour/solve` with the drawn state → runs the stored trained
  model (M7) and the BFS oracle → returns a **trajectory** (action + resulting state per
  step) + metadata (solved, AI move count, BFS-optimal count).
- Playback UI: back/forward step buttons + play/pause over the trajectory; always
  recoverable to the drawn state.
- **Gate:** e2e (Playwright against the running host) — draw a known puzzle, submit,
  step through a returned solution that actually solves it; API integration test: a
  generated easy puzzle returns a solving trajectory within 2× BFS-optimal.

## M9 — 2048 page + training-on-demand + gallery

- 2048 page: canvas tile editor, manual play (real merge/spawn rules), reset to drawn
  state; solve returns an n-tuple-agent playout *from the drawn state* as a trajectory
  (move + spawned tile per step) for deterministic browser replay.
- Background training jobs: if the model store has no model for the requested
  environment, the solve request enqueues a general training job (independent of the
  submitted state) with progress (games played / current eval score) streamed or polled
  by the browser; the solve completes when training does. Checkpoint saved on finish.
- **Public game gallery:** submitted states + returned solutions persisted (JSON files
  under the data directory next to the model store) and listed/replayable on the site.
- **Gate:** with an empty model store, submitting a 2048 board visibly trains first and
  then solves; with a warm store it solves immediately; gallery entries survive an app
  restart.

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

## Immediate next step

**Build M7 (checkpointing + model store)** — it unblocks everything in the web
playground (M8–M10). Demo entry points: `dotnet run --project src/RL.NET.Demo -c Release
-- [grid|lake|cartpole|ppo|2048|2048dqn|rushhour] [seed]` (launch profiles exist for
each). Tests: `dotnet test` (statistical gates carry `Category=Slow`).
