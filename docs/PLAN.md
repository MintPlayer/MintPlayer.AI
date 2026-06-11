# MintPlayer.AI.ReinforcementLearning — Implementation Plan

Companion to [PRD.md](PRD.md). Each milestone ends in a **git commit on a passing gate**;
revert-friendly by design. Order is chosen so each milestone adds at most 2–3 genuinely new
components (the CleanRL/SB3 lesson: localize bugs by construction).

> **Status (2026-06-11): M0–M10 complete, every pre-registered gate passed — and the
> project is published and LIVE.** Repo: github.com/MintPlayer/MintPlayer.AI · NuGet
> packages 0.1.0 on nuget.org · Docker image on GHCR · playground deployed at
> **https://ai.mintplayer.com** (see "Shipped" below). M11 is underway with its headline
> already landed: imitation learning + policy-guided A* solves every official ThinkFun
> card tested **optimally**, including expert card 40 (81 moves, ~2.6k node expansions)
> — drawn, solved and replayed in the browser.

## M0 — Skeleton + core contracts  *(part of the quick demo)* ✅

- Solution restructure: `src/MintPlayer.AI.ReinforcementLearning.Core`, `src/MintPlayer.AI.ReinforcementLearning.Environments`, `src/RLDemo.Console`
  (existing console project), `tests/MintPlayer.AI.ReinforcementLearning.Tests` (xUnit).
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
  (`tools/generate_goldens.py` → `tests/MintPlayer.AI.ReinforcementLearning.Tests/Fixtures/cartpole_golden.json`).
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

- Board logic in `MintPlayer.AI.ReinforcementLearning.Environments/RushHour` (6×6, vehicles len 2–3, action =
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

- Checkpoint formats per the PRD decisions (`MintPlayer.AI.ReinforcementLearning.Core.Checkpoints`): JSON for
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

- `src/RLDemo.Web`: ASP.NET Core host + Angular 22 ClientApp (zoneless, signals) wired
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

## M10 — Docker ✅
**Result: gate passed.** `docker build` → run with a seeded `rlnet-data` volume →
instantly ready (no retraining), production SPA served, **card 40 solved 81/81 by the
AI (lookahead)** through the containerized API; `docker restart` → model store and
gallery intact. Cold volumes train their own models at startup (banners show progress).

- Multi-stage Dockerfile: Node 22 stage builds the Angular bundle; SDK stage publishes
  with `-p:SkipAngularPublish=true -p:EnableSpaBuilder=false` (both npm hooks disabled —
  the Node stage already did that work); slim `aspnet:10.0` runtime image.
  `DataDirectory=/data` env + `VOLUME /data`, port 8080.
- `docker compose up` → http://localhost:8080 with a named volume; README documents
  seeding the volume with existing `*.ckpt` files to skip first-run training.

## M11 — In progress: imitation learning + policy-guided search  *(overnight campaign 2026-06-10→11)*

The "harder Rush Hour" stretch item turned out to be the headline result. A reactive
DQN cannot crack expert cards (81-move solutions compound per-step error: 0.99⁸¹ ≈ 44%
success even at 99% accuracy), so the night built the principled ladder instead:

- **`RushHourOracle`** — for any config, enumerate the ENTIRE reachable state graph and
  label every state with its exact distance-to-goal (multi-source backward BFS; sliding
  moves are reversible) + one optimal action. Deep STATES are plentiful even though deep
  START states can't be generated randomly — supervision for free, at any depth.
- **`RushHourPolicyNet`** — shared trunk + 32-way policy head + distance-to-goal value
  head, trained with masked cross-entropy + Huber on the from-scratch autograd.
- **`RushHourPolicySearch`** — cycle-avoiding greedy rollout, and **A\* with the value
  head as heuristic**: search turns a 91%-accurate policy into an exact solver.
- **`tools/MintPlayer.AI.ReinforcementLearning.Lab`** — resumable long-running trainer: streams random configs through
  the oracle (stratified by distance), evals held-out official cards every 10 min,
  checkpoints to the model store, logs CSV (`data/logs/imitation.csv`).

**Overnight run (~5.6 h, single thread, pure managed .NET): 412,913 configs,
224.8 M labeled samples, policy accuracy 76% → 91.4%.** A day-2 incremental
continuation (the Lab resumes net + Adam state from the model store) added another
180.7 M samples → **405 M total, accuracy 92.3%**. Held-out official ThinkFun cards,
every 10-minute eval, via policy-guided A*:

| Card | Optimal | AI result | Node expansions |
|---|---|---|---|
| Level 1 | 16 | **16 (optimal)** | ~0.7–1k |
| Card 38 | 77 | **77 (optimal)** | ~3.1k |
| Card 39 | 82 | **82 (optimal)** | ~3.7k |
| Card 40 (hardest) | 81 | **81 (optimal)** | ~2.4–2.9k |

(Blind BFS explores hundreds of thousands of states on card 40; the learned heuristic
cuts that ~100×.) Random 30-puzzle eval: greedy 30/30, search 30/30. The playground
prefers the policy net (5-min store TTL → improving checkpoints picked up live):
reactive rollout first, A* fallback, honest `aiMode` labeling — card 40 drawn in the
browser shows "AI (with lookahead) solved it in 81 moves (optimal 81)"
(docs/screenshots/card40-ai-solved.png). The wide-band DQN (optimal 2–20 band,
threshold 90 reached at 480k steps, eval 92.85) remains the fallback when no policy
checkpoint exists. Known gap: the REACTIVE policy still fails level 1 specifically
(search covers it at ~1k expansions) — a candidate for AlphaZero-style fine-tuning.

## M12 — GPU/CUDA backend  *(planned 2026-06-11, deliberately parked)*

Not scheduled — to be picked up **when the workload justifies it**: the imitation
accuracy plateau demands a wider net, a new environment needs CNN-scale compute, or
training campaigns become throughput-bound beyond what overnight CPU runs deliver.
The assessment below was made with the dev machine's RTX 3060 Laptop GPU
(6 GB, compute 8.6, driver current) so the plan is ready to execute.

**The constraint that shapes the design:** the existing `IComputeBackend` seam passes
host `float[]`s, so a naive CUDA backend would LOSE to the CPU at today's sizes — a
256×384 batch-GEMM is ~75 MFLOPs ≈ 7 µs of GPU compute, less than one kernel launch,
and PCIe transfer costs more than the math. The ~500× raw-compute headroom
(~10 TFLOP/s FP32 vs our 20 GFLOP/s managed GEMM) is only reachable with
**device-resident tensors** and bigger batches/nets. GPU work must therefore start
with an honest benchmark and an API evolution, not a drop-in backend.

**Library choice: ILGPU** (JIT-compiles C# kernels to PTX) over TorchSharp — MIT,
megabytes not gigabytes, no native payload, keeps the from-scratch identity (we write
our own tiled GEMM kernel; 1–3 TFLOP/s realistic ≈ 50–150× current CPU), and its CPU
accelerator keeps CI and GPU-less machines green. TorchSharp remains the documented
alternative if ILGPU ever falls short.

- **M12a — benchmark first:** extend the Bench tool with an ILGPU GEMM sweep
  (128²→4096² plus our real training shapes), measured **with and without transfer
  costs**, and full training-step comparisons across batch/hidden sizes.
  **Gate:** a CPU↔GPU crossover table committed to the docs.
- **M12b — device-resident backend:** evolve `IComputeBackend` to a device-tensor API
  (allocate/upload/download + ops on device handles); `ManagedBackend` stays trivial;
  `IlgpuBackend` implements the real thing; port the training hot loop. First consumer:
  the imitation Lab (infinite oracle data → batch 4096+, wider nets).
  **Gate:** Lab samples/hour ≥ 5× the CPU baseline (~40 M/h) at equal model quality.
- **M12c — the payoff campaign:** overnight GPU imitation run with a wider net, aiming
  past the 92.3% accuracy plateau; results added to the M11 table.

## M11 — Stretch (unordered, not started)

MountainCar (exploration stress test) · Snake (demo gif) · TorchSharp `IComputeBackend`
implementation · TensorBoard event writer · self-play scaffolding (TicTacToe + minimax oracle)
· NuGet packaging · Dueling DQN head (deferred from M3) · tensor/tape pooling (deferred
from M2) · Categorical/PPO action masking (deferred from M5) · AlphaZero-style fine-tuning
of the Rush Hour policy (close the reactive level-1 gap; shrink search expansions)
· watch-only playground pages for CartPole/2048 self-play · importing puzzles from
`C:\Repos\Spelletjes\Rush Hour` as gallery data (ask for a clean checkout first).

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
| M9 2048 + gallery | train-first visible; warm solve instant; gallery survives restart | passed (n-tuple playout: 2,491 moves, 55,480 pts, reached 2048; gallery replay verified) |
| M10 Docker | build → run → solve → restart keeps models + gallery | passed (cold volume self-seeds; card 40 solved 81/81 in the container) |
| M11 imitation (in progress) | held-out official cards via policy-guided A* | **all optimal**: level 1 = 16, card 38 = 77, card 39 = 82, card 40 = 81 moves |

## Shipped (2026-06-11) — release engineering

Beyond the milestones, the project is published and deployed:

- **Renamed** to `MintPlayer.AI.ReinforcementLearning` (libraries) / `RLDemo.*` (apps).
- **Pre-trained models committed** in `models/` (with provenance README); empty model
  stores self-seed from them — fresh clones, volumes and containers start trained.
- **NuGet**: `MintPlayer.AI.ReinforcementLearning.Core` + `.Environments` 0.1.0 on
  nuget.org (published by `build-master` on every master push, `--skip-duplicate`).
- **GitHub**: https://github.com/MintPlayer/MintPlayer.AI — CI workflows for branches,
  PRs and master (Slow-bucket tests excluded in CI; `EnableSpaBuilder=false`).
- **Docker/GHCR**: `ghcr.io/mintplayer/mintplayer.ai/playground:master` with provenance
  attestation, auto-flipped public.
- **Deployed**: `playground-docker` SSH-deploys to the VPS (ng-bootstrap convention) —
  **live at https://ai.mintplayer.com**, verified by solving card 40 (81/81, aiMode
  search) over the public internet.

## Immediate next step

All planned milestones are done and the playground is in production. Candidates, in
suggested order:

1. **AlphaZero-style fine-tune** (overnight Lab run): train the policy on
   search-corrected play to close the reactive level-1 gap and shrink A* expansions.
2. **Slide-optimal AI answers**: search/compaction that minimizes official piece-moves,
   not just single-cell moves.
3. **Stretch list (M11)**: MountainCar, Snake, TorchSharp backend, TensorBoard writer,
   self-play scaffolding, Dueling head, tensor pooling, PPO masking, importing puzzles
   from the owner's original Rush Hour app.
4. **GPU/CUDA (M12)** — fully planned above, parked until the workload justifies it
   (wider nets past the imitation plateau, CNN-scale envs, or throughput-bound
   campaigns).

Run the playground: `dotnet run --project src/RLDemo.Web` (Development spawns + proxies
the Angular dev server itself — do not run `ng serve`). Console demos:
`dotnet run --project src/RLDemo.Console -c Release -- [grid|lake|cartpole|ppo|2048|2048dqn|rushhour]
[seed] [--load] [--save] [--data <dir>]`. Tests: `dotnet test` (`Category=Slow` for gates).
Training campaigns: `dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab --
--hours N --data src/RLDemo.Web/data` (resumes net + Adam from the model store).
