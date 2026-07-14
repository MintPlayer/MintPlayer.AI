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

### Fine-tune round (2026-06-11 evening, paused — resumable)

Two changes attacked the reactive level-1 gap:

1. **DAgger-style on-policy mix** — per config, the Lab now rolls the CURRENT net out
   from the start plus seven deep states, relabels every visited state via the oracle
   dictionary (the whole reachable graph is labeled, so it's a lookup), double-weights
   states from failed rollouts, and fills ~12% of the sample budget with them
   (stratified sampling fills the rest).
2. **Multi-label supervision** — a greedy-failure trace on level 1 showed near-tied
   logits at every wrong move: most states have 3–4 equally-optimal actions, and
   single-label CE actively penalized the rest of the optimal set, flattening the
   policy. `RushHourOracle` now emits an `OptimalActionsMask`; the Lab trains CE
   against a uniform soft target over the set, and accuracy counts any optimal argmax.

After ~1 h on the new objective (+34 M samples, 439 M total): accuracy 91.4% *under the
new any-optimal metric* and still climbing (CE near its new ~log k floor), **level-1
greedy now flickers solved (44–60 mv) instead of always-fail**, search stays optimal
(card 38 flickers 77/78). Paused mid-campaign on request; the Lab resumes net + Adam
from the store, so continuing is just:

```
tools/.../MintPlayer.AI.ReinforcementLearning.Lab.exe --hours H --data src/RLDemo.Web/data --seed <new>
```

Ship criteria before copying checkpoints into `models/`: level-1 greedy solves
consistently across evals, accuracy plateaus, official cards stay search-optimal.

## M12 — GPU/CUDA backend  *(first-class SDK pillar, planned 2026-06-11)*

**This is core SDK capability, not a demo accelerator.** The project's goal is a
high-end, open-source, .NET RL SDK; the games are showcases. A serious RL SDK that
can't use the GPU isn't competitive, so M12 is built **on its own merits**, not gated on
any demo needing it. (Earlier drafts parked it "until the workload justifies it" — that
was demo-quality logic, superseded 2026-06-13 once the SDK was named as the deliverable.
The cube plateau is now just one *beneficiary*, not the justification.) The assessment
below was made with the dev machine's RTX 3060 Laptop GPU (6 GB, compute 8.6, driver
current) so the plan is ready to execute.

**SDK lens — the deliverable is the *API*, not any one trained model.** The compute
backend is public surface: users build on it and it's expensive to change post-release.
So the device-tensor abstraction must hide all device internals (PTX, kernel launches,
host↔device transfers), stay **general** across every env/algorithm (don't over-fit it
to the cube's `324×1024` shapes), and ship with GPU↔`ManagedBackend` correctness parity
(finite-diff grad checks on device) and documented extension points. This is the
"design it twice" interface where the discipline actually pays off.

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

- **M12a — multithreaded CPU GEMM (ships for its own sake)** ✅ *(kernel done
  2026-06-13, `dafebc4`)*: the managed GEMM now parallelizes large products across cores
  by partitioning disjoint output rows — **bitwise-identical** to the sequential path at
  any worker count (determinism preserved), threshold-gated so small classic-control nets
  keep the thin sequential path. `BackendTests` assert dop-1-vs-8 byte parity for all
  three kernels + correctness vs a naive reference; GradCheck/NN/DeepRl stay green.
  Pinning needs `AllowUnsafeBlocks` but stays pure managed (no P/Invoke). *Bitwise-parity
  half of the gate met; the throughput-scaling bench row is captured by M12b on an
  uncontended machine.* (Lab gen/train double-buffering is still open — a follow-up.)
- **M12b — benchmark ✅ *(done 2026-06-13)*:** the Bench tool has a **CPU thread-scaling
  sweep** (GEMM GFLOP/s + end-to-end training-step speedup vs worker count, on the real cube
  shapes) **plus a GPU column** running `IlgpuBackend` on the same shapes (host-span, incl.
  host↔device transfer; skipped on a GPU-less machine, and prefers the discrete CUDA card
  over an integrated iGPU). Both tables measured + committed below.
  **Gate (met):** CPU↔GPU crossover + CPU thread-scaling tables committed (see "Measured 2026-06-13").
- **M12c — ILGPU backend ✅ *(host-span done 2026-06-13, `de362c3`)*:** a separate
  `MintPlayer.AI.ReinforcementLearning.Ilgpu` package (Core stays dependency-free)
  implements `IlgpuBackend : IComputeBackend` with three JIT-compiled GEMM kernels (one
  output element per thread, accumulating per the interface contract). Selects the CUDA
  GPU when present, else ILGPU's CPU accelerator (keeps CI/GPU-less machines green).
  Correctness validated against `ManagedBackend` via the CPU accelerator (`IlgpuBackendTests`,
  relative tolerance — cross-backend equality is approximate, FMA vs separate mul+add).
  **`AdaptiveBackend` ✅ (`74a77e5`)** wraps CPU+GPU and auto-routes each GEMM by MAC count
  (small → multithreaded CPU, large → CUDA), pure CPU when no GPU — `Backend.Current = new
  AdaptiveBackend()`, no knobs.
  - **M12c-perf — device-resident tensors:** the host-span backend transfers every call, so
    it LOSES at small sizes and only wins for large GEMMs (PRD §10). **Scoped version ✅
    (`a30d7a0`):** `IlgpuBackend.MlpForwardScalar` runs a whole scalar-MLP forward resident on
    the GPU — ~2× DAVI throughput. **But it re-uploads weights every call** (the 8192-wide wall).
    The full removal of both GPU bottlenecks is now planned concretely as **M19 (tiled GEMM) +
    M20 (device-resident tensors, staged)** below — investigated 2026-06-13.
  **Gate (met):** `IlgpuBackend` computes all three GEMMs correctly with `ManagedBackend`
  parity tests; scoped resident forward matches the autograd forward.
- **M12d — showcase campaigns *(in progress)*:** GPU unlocks the demos CPU can't reach. The
  teacher-free **value-iteration (DAVI) cube campaign (M18)** runs on the GPU device-resident
  forward — `Lab --game cube-davi` on the RTX 3060. Bigger Rush Hour nets past the 92.3%
  plateau and the 2048-wide cube remain candidates. Showcases of the SDK's GPU path.

### Measured 2026-06-13 (dev machine: 8 logical cores, RTX 3060 Laptop + Intel Iris Xe)

**CPU thread-scaling (M12a) — near-linear on the GEMM, Amdahl-limited end-to-end:**

| Shape | dop 1 | dop 8 | speedup |
|---|---|---|---|
| GEMM 256×324×1024 (trunk1) | 31 GFLOP/s | **123** | 3.95× |
| GEMM 256×1024×1024 (trunk2) | 23 | 79 | 3.41× |
| Full cube-1024 Adam step | 14 steps/s | **36** (9,240 samples/s) | 2.52× |

**CPU↔GPU crossover (M12b) — host-span ILGPU, *includes* host↔device transfer:**

| Shape | CPU 8-thread | RTX 3060 (host-span) | winner |
|---|---|---|---|
| 256×324×1024 | 123 GFLOP/s | 49 | **CPU** (transfer-bound) |
| 256×1024×1024 | 79 | 91 | GPU (1.15×) |
| 1024×1024×1024 | ~80 | **146** | GPU (~1.8×) |

**Finding — the host-span GPU path leaves ~98% of the 3060 on the table** (146 of ~10,000
GFLOP/s): per-call transfer + allocation dominate, exactly as PRD §10 predicted. So:
- For the **current 1024-wide imitation net, 8-thread CPU (9,240 samples/s) wins or ties** —
  do NOT switch the default to GPU; the device-selection picks CUDA over the iGPU now.
- The GPU only pulls ahead at **large square GEMMs** (1024²+), and even then modestly *until*
  device-resident tensors remove the per-call transfer. **So M12c-perf pays off specifically
  in the big-net regime (the 2048-wide / value-iteration showcase), and should be built
  *with* M12d — not retrofitted onto the small imitation net, where it would regress.**

## M11 — Stretch (unordered, not started)

MountainCar (exploration stress test) · Snake (demo gif) · TorchSharp `IComputeBackend`
implementation · TensorBoard event writer · self-play scaffolding (TicTacToe + minimax oracle)
· NuGet packaging · ✅ Dueling DQN head *(done 2026-06-15 — `DuelingQNet` (shared trunk → value+advantage,
mean-centered) behind `DqnOptions.Dueling`; reuses the `IValueNet` contract so it drops into the trainer +
target-sync + type-tagged resume; solves CartPole median-of-3)* · tensor/tape pooling (deferred
from M2) · ✅ Categorical/PPO action masking *(done 2026-06-15 — additive logit-bias mask shared by
`PolicyAgent` inference + PPO rollout/update/eval; `VectorEnv.CurrentActionMasks`; unlocks PPO on the
masked games)* · AlphaZero-style fine-tuning
of the Rush Hour policy (close the reactive level-1 gap; shrink search expansions)
· watch-only playground pages for CartPole/2048 self-play · importing puzzles from
`C:\Repos\Spelletjes\Rush Hour` as gallery data (ask for a clean checkout first).

## M13 — Rubik's Cube page + Kociemba solve  *(owner's game #3 — the port, PRD §11)* ✅
**Result: gate passed** (browser-verified 2026-06-12): full scramble → Solve (algorithm)
→ 21-move solution in 116 ms → playback ends on a solved cube; gallery entry replays.
The Kociemba port's disk-table variant (K_Search/K_CoordCube — its deserialization
already threw NotSupportedException upstream) was dropped; the runtime-table path
(SearchRunTime + CoordCubeBuildTables) is warmed once at startup (~12 s, in-memory).

Port of `C:\Repos\WebGames\Rubiksolver` into the playground (source repo stays untouched).

1. **Cube core** in `MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/`:
   facelet cube (54 stickers, byte-encoded) with quarter/half-turn move application
   (port the sticker-cycle tables from `rubiksCube.ts`/`K_CubieCube`), scramble
   generation (seeded, no trivially-cancelling consecutive moves), `IsSolved`, and
   converters: 6×9-color DTO ↔ facelets ↔ 54-char Kociemba string. Port the detailed
   edge/corner validation from `Rubiksolver/Models/CubeState.cs` (missing/duplicate/
   invalid piece diagnostics).
2. **Kociemba two-phase solver**: copy `Rubiksolver/Kociemba/` (8 files, ~2,500 LOC,
   pure C#) under the cube folder, adjust namespaces, keep runtime in-memory table
   generation (`buildTables: false`). Keep the `SearchRunTime` path used by the source
   app; drop the table-writing variant if unused.
3. **Web API** (`CubeController`): `POST /api/cube/solve` (validate → Kociemba → move
   list + move count + solve time, error mapping ported from `CubeSolver.cs`),
   `GET /api/cube/status` (model/training status, M14-ready), gallery entries
   (`gallery.Add("cube", …)`). Warm the pruning tables once in the background at
   startup so the first solve isn't the one paying the 2–5 s build.
4. **Front-end**: add `three` + `@types/three` to the Angular workspace (npm dep, not
   the CDN import map); new lazy route `/cube` + home card. One Angular component
   porting `rubiksCube.ts` (rendering + client-side state tracking) and `main.ts`
   (orchestration): orbit controls, 18 move buttons, scramble / easy-scramble / reset,
   speed slider, move history, solve → solution playback (prev/play/next), status
   text. SCSS adapted from `Styles/style.scss` to the playground theme.
5. **Tests**: move-semantics units (each face move ×4 = identity, F B' sequences against
   known sticker layouts, scramble→inverse→solved); DTO/Kociemba conversion
   round-trips; validation diagnostics; `[Slow]` gate: 20 random depth-20 scrambles →
   Kociemba solution applied → solved, ≤ 22 moves each; `WebBackendTests`-style API
   tests (invalid cube → 400 with diagnostic message).

**Gate:** e2e (Playwright against the dev host): scramble → Solve (algorithm) →
playback ends on a solved cube; API returns ≤ 22 moves for a full scramble.

## M14 — Rubik's Cube RL  *(AI solve button + training, PRD §11)* ✅
**Result: gate passed — 600/600 (100%)** depth-1–6 scrambles solved within 20 moves
(greedy alone 77.8%: 100/100/99/86/58/24 per depth — the rest via Q-guided best-first
lookahead, reported as `aiMode: search`). Two training findings: (1) a plain greedy DQN
oscillates (A A′ A A′), so the env masks the inverse of the previous move
(`IActionMaskProvider`, same machinery as Rush Hour) — undoing can never shorten a
solution; (2) the first recipe (γ 0.98, 400k steps, ε→150k) was unstable early and
plateaued ≈ 75%; the shipped one (γ 0.99, 600k steps, buffer 200k, ε→200k) trains in
~65 min and plateaus ≈ 70 eval return, which lookahead converts to 100% on the band.
`models/cube.dqn.ckpt` committed with provenance.

1. **`RubiksCubeEnv : IEnvironment<float[], int>`**: obs Box(324) one-hot stickers;
   Discrete(12) quarter-turns; `Reset` scrambles to depth d ~ U[1..MaxDepth] (constructor
   parameter — curriculum = growing MaxDepth); reward −1/step +100 solved; cap 20 moves
   (truncated, not terminated). Console renderer (`RenderString` = flattened net).
2. **`CubeModelService : ITrainableModelService`** mirroring `RushHourModelService`:
   load `cube.dqn.ckpt` from the store or train once at startup (masked DQN recipe,
   depth band 1–6, target a few minutes' wall-clock like the others), thread-safe
   progress snapshot for `/api/cube/status`.
3. **`POST /api/cube/solve-ai`**: greedy rollout (≤ 20 moves), response = move list +
   `solved` + the Kociemba reference move count for honest comparison; gallery entry
   either way. UI: second button "Solve (AI)" with the training-status banner pattern.
4. **Console + Lab**: `cube` section in RLDemo.Console (`--load/--save/--data`);
   Lab campaign support for longer/deeper training runs.
5. **Ship a pre-trained checkpoint** `models/cube.dqn.ckpt` (+ provenance in
   `models/README.md`); the existing seed-on-startup copies it into fresh stores.

**Gate (pre-registered, PRD §11):** ≥ 90% of 100 eval scrambles (depths 1–6) solved
within 20 moves. Deep scrambles failing is expected and shown honestly.

**Stretch (the M11 recipe, oracle = Kociemba):** imitation-learn a policy/value net on
Kociemba solutions (unlimited labeled data; expand half-turns to two quarter-turns),
greedy + policy-guided search fallback, `aiMode` reporting — lifts the AI toward full
scrambles without pretending DQN got there.

## M15 — 2048 classic play feel  *(swap the merge experience, PRD §12)* ✅
**Result: gate passed** (browser-verified 2026-06-12): DOM tiles with the original
palette and animations (a merging move produced two sources sliding under a popping
`tile-merged` 4 plus a `tile-new` spawn); the AI playout animates through the classic
engine; the 2,491-move replay reconstruction matched `FinalCells` exactly (no parity
warning) — the classic Cirulli engine and the server's `Board2048` agree move-for-move.

Source: `C:\Repos\WebGames\Game2048` (Cirulli-architecture TS port: GameManager/Grid/
Tile/HTMLActuator, ~620 LOC TS + 616 LOC SCSS). Only the front-end game experience
changes; API, n-tuple agent, `Board2048`, gallery and edit mode stay as they are.

1. **Classic engine** (`game-2048-classic.ts`): port Grid/Tile/move from
   `Scripts/game_manager.ts` — traversal-order processing, `mergedFrom` double-merge
   prevention, farthest-position slide, 90/10 spawn — keeping tile *values* internally;
   add the two boundary mappings (values ↔ server exponents; classic directions
   0=up 1=right 2=down 3=left ↔ server actions 0=left 1=down 2=right 3=up) and a
   deterministic-spawn entry point (`applyStep(action, spawnIndex, spawnValue)`) for
   AI playback.
2. **Classic board rendering**: replace the canvas in `game-2048.html`/`.ts` with the
   DOM structure from `Game2048/wwwroot/index.html` (grid background + tile container);
   SCSS adapted from `Styles/main.scss` (tile-position transition classes, `appear`/
   `pop` keyframes, score-addition float, tile palette) restyled for the playground's
   dark theme; Angular renders tiles from the engine state (the HTMLActuator role).
3. **Manual play** drives the classic engine (keep the existing keyboard handling; add
   the historic touch-swipe support); score + best-tile readouts as today.
4. **Edit mode** renders through the same DOM tiles (click to cycle up, right-click
   down — unchanged interaction, no animation classes while editing).
5. **AI playback** applies each `PlayoutStepDto` through `applyStep` so playback gets
   the classic slide/pop/appear animations; scrubber seeks rebuild the board
   unanimated; keep verifying against `FinalCells`.
6. **Out of scope:** `game-2048-api.ts`, `Game2048Controller`, `Game2048ModelService`,
   `Env2048`/`Board2048`, gallery replay format.

**Gate:** Playwright e2e — manual moves show sliding/merging tiles (DOM tiles with
transition classes present, no canvas); AI playback completes with the playback board
equal to `FinalCells`; existing 2048 API tests untouched and green.

## M16 — Cube imitation from Kociemba  *(the PRD §11 stretch, M11 recipe)* ✅
**Result: gate passed — 96/100 (96%)** random scrambles across depths 1–10 solved
within 40 quarter-turns after a 2 h campaign (two resumable 1 h runs, 7.7 M
oracle-labeled states from ~370k Kociemba solves; action accuracy 73.5%, plateauing —
the MLP's ceiling, as with Rush Hour). Greedy alone 54%; per-depth: 10/10 through
depth 7, then 9, 10, 7 — `aiMode: search` does the heavy lifting from depth 6 on.
Campaign finding: per-eval search budgets must stay small (2k expansions) — the first
smoke run spent 40 of its 43 minutes inside failed full-budget eval searches.

Lift the cube AI past the DQN's depth-6 band: imitation-learn a policy/value net on
Kociemba solution paths (unlimited oracle-labeled data — every random scramble solved
once yields ~25–35 labeled states), then greedy + policy-guided A* at inference.

1. **`CubeOracle`**: scramble at a random depth (1–22), solve with Kociemba, expand
   half-turns to quarter-turns, walk the solution path — each state labeled with the
   next quarter-turn action and the quarter-turn distance-to-go (the value target).
2. **`CubePolicyNet`** mirroring `RushHourPolicyNet`: shared ReLU trunk (324 → 512 →
   512), 12-logit policy head, scalar distance head (`DistanceScale` 30); versioned
   checkpoint `cube.policy.ckpt` (+ `cube.policy-adam.ckpt` for campaign resume).
3. **`CubePolicySearch`**: greedy rollout (visited-set cycle avoidance, no-undo mask)
   and A* with the value head as heuristic, solution cap 40 quarter-turns.
4. **Lab `--game cube`**: streaming campaign — generate, solve, train (CE + Huber,
   Adam), checkpoint every 10 min, CSV log, per-depth eval (greedy/search at depths
   2–20). Resumable like the Rush Hour campaign.
5. **Web**: `CubeModelService.PolicyNet` (refreshing, preferred over the DQN);
   `/api/cube/solve-ai` tries policy greedy → policy search → DQN fallback, `aiMode`
   reported as before.

**Gate (pre-registered):** ≥ 90% of 100 random scrambles across depths 1–10 solved
within 40 quarter-turns (greedy or search) after a ~1 h campaign; deeper-band rates
reported honestly (longer campaigns keep improving it — the net is resumable).

## M17 — Wider cube policy net  *(lift the imitation ceiling; triggers M12)*

The overnight campaign (2026-06-12→13, two resumed stints, ~236 M cumulative
oracle-labeled states) confirmed the 512-wide MLP has **plateaued**: greedy action
accuracy moved only 54% → 64% and the gate sat at 96→97/100. More of the same data no
longer buys capability — the wall is **network capacity**, the same lesson M11 hit at
Rush Hour's 92.3%. (Read the **greedy %**, not the gate total: A* already mops up greedy
misses near the ceiling, so the 100-scramble total is a poor discriminator above ~96.)

**Approach — a width *ladder*, not a single guess.** Width is the obvious capacity
lever, but it may not be the binding constraint (the depth-≥8 greedy stall smells more
like distribution coverage than capacity — see step 2). So make trunk width a config
parameter and climb the ladder, reading the signal off each rung before paying for the
next. Hidden size is baked into the checkpoint, so each width is a **fresh net under its
own id** (e.g. `cube.policy-w1024`) — it cannot resume a narrower net's weights; the
shipped 512 net stays in place until a wider one beats it on the same gate seeds.

| Trunk | Params | MACs/sample | vs 512 | Samples in an 8 h CPU overnight | Verdict |
|---|---|---|---|---|---|
| 512→512 (shipped) | ~0.44 M | ~435 k | 1× | ~134 M (last night) | plateaued |
| **1024→1024** | ~1.40 M | ~1.39 M | 3.2× | ~59 M | **rung 1 — CPU-trainable** |
| 2048→2048 | ~4.89 M | ~4.88 M | 11.2× | ~19 M (under-trained) | rung 2 — **GPU-gated** |

1. **Rung 1 — `324 → 1024 → 1024`.** The largest net that still trains to convergence on
   CPU overnight. This is the experiment that tells us whether width is even the right
   lever, *before* committing to the GPU work a bigger net needs.
2. **DAgger-style on-policy mix** (the M11 lesson): relabel the states the net actually
   visits during greedy rollout with the Kociemba oracle, not only random scrambles —
   targets the distribution mismatch that likely caps greedy from depth ≥ 8. Apply at
   rung 1; if it lifts greedy sharply, the wall was coverage, not capacity.
3. **Weighted A\*** at inference (`h ← w·h`, `w > 1`): no retraining, cuts depth-10+
   search latency; report any move-count inflation honestly.
4. **Rung 2 (2048-wide) is a GPU showcase, no longer a CPU decision gate.** Since M12
   (the GPU backend) is now built on its own merits as an SDK pillar — not gated on the
   cube needing it — 2048-wide becomes a natural *demo of the GPU path* once M12d lands:
   it can't be trained to convergence on CPU (~19 M samples/night, and a train-bound loop
   defeats double-buffering), but on GPU it's cheap, very possibly paired with a
   value-iteration (DAVI-style) objective (how the serious cube solvers use nets this
   size). Rung 1's converged result still *informs* it — if greedy plateaus near the 512's
   ceiling, the wall is the imitation **algorithm** not capacity, so the GPU showcase
   should lead with DAgger/value-iteration rather than raw width — but rung 1 is no longer
   a blocking gate, just evidence for which showcase to build.

**Gate (pre-registered):** ≥ 98/100 across depths 1–10 within 40 quarter-turns **and**
greedy alone ≥ 70% (the metric that actually moves), evaluated on the same fixed gate
seeds as M16; a wider net replaces the shipped one only if it beats it on both.

**Performance coupling — this milestone fires the M12 trigger.** Even rung 1's 3.2×
wider trunk, on the serial generate-then-train loop (`CubeLab.cs` alternates
`Parallel.For` data-gen with a single-threaded train pass — neither saturates its
resource), makes the training GEMM the campaign bottleneck for the first time, and that
GEMM runs at ~1.4 of ~20 GFLOP/s. Cheapest-first order: **(a)** double-buffer generation
against training (fills the idle halves of the current loop, pure-CPU, ~up-to-2×
wall-clock — helps rung 1, but *not* a train-bound rung-2 net), **(b)** multithread the
managed GEMM, **(c)** the full GPU path (M12), which rung 2 (2048) requires outright.
(a)/(b) precede (c) because they're days, not weeks, and may suffice for rung 1.

## M18 — Value iteration (DAVI)  *(teacher-free self-improvement — the SDK's "beat the teacher" path)*

M16/M17 imitate Kociemba, so they are **capped by the teacher** (and Kociemba isn't even
quarter-turn optimal). M18 adds the paradigm that can *exceed* a teacher: deep approximate
value iteration (DAVI, à la DeepCubeA), bounded only by the cost objective (fewest moves),
not a demonstrator. It's a general, reusable trainer — a third paradigm alongside RL
(DQN/PPO) and imitation — distinct from the exact tabular `Solvers.ValueIteration` (this is
the function-approximation counterpart for non-enumerable state spaces).

- **Planning foundation ✅ (`25fe036`):** `IDeterministicModel<TState>` (Core.Planning — the
  pure forward model: actions / apply / goal-test / state-key, distinct from the RL
  `IEnvironment` loop) + `BreadthFirstPlanner` (provably-optimal, the validation oracle) +
  `CubeModel`. Tests: BFS solves shallow cubes optimally, teacher-free.
- **DAVI trainer ✅ (`db2027e`):** `ValueIterationTrainer<TState>` learns a cost-to-go value
  net by bootstrapping each target from a one-step lookahead over the model
  (`target = min_a [1 + (IsGoal(s′) ? 0 : V_target(s′))]`, anchored `V(goal)=0`), with a
  periodically-synced target net; generic (inject featurize + state sampler). `GreedyValuePlanner`
  is the greedy inference policy. **Finding: predict RAW cost-to-go (`DistanceScale=1`)** —
  squashing targets to ~0.1 starved the gradients and the argmin couldn't separate distances.
- **Value-guided A\* ✅ (`15f8fa3`):** `ValueGuidedSearch` (weighted A*, `f = g + w·h`) +
  `trainer.SolveWithSearch` — reaches states greedy gets stuck on; the inference-time
  ceiling-raiser (use for post-run eval + the web demo).
- **GPU campaign ✅ (`c41fc39`, resume-hardened `9b79ba6`):** `Lab --game cube-davi` — resumable
  teacher-free campaign on the `AdaptiveBackend`, solve-rate curriculum (deepen at ≥95%),
  the GPU device-resident successor forward (M12c-perf, ~2× throughput), configurable net
  depth (`--layers`). Persists the FULL training state (Adam + curriculum + iterations + RNG)
  so a restart continues losslessly.

**Validated:** fast deterministic test — greedy descends *optimally* under an exact (BFS)
value; `[Slow]` test — the full DAVI loop learns to solve ≥ 80% of shallow (depth ≤ 3) cubes
**teacher-free**, checked against the BFS optimum.

**Campaign result (2026-06-13, 1024×3 net, ~30.5k iters, stopped early):** reached **curriculum
depth 9 teacher-free** — greedy d1–4 100%, d5–6 ~95%, d7 70%, d8 55%, d9 30%. Two findings:
(1) the original **0.95 advance gate was the limiter** — the run parked at depth 5 (~9k iters)
then depth 7 (~13.5k iters); the **stall-fallback curriculum** (`22ee724`: advance at ≥0.6 OR
force-advance every 3k iters) immediately unstuck it (7→8→9 in ~4k iters). (2) **Greedy solve
rate degrades with depth** (70→55→30% at d7→9) — the myopic-greedy + 1024×3-MLP capacity wall.
This is the evidence base for M19–M21: going deeper/optimal needs a **residual net** and the
**GPU bottlenecks removed** to train it.

## M19 — GPU compute: tiled GEMM kernel  *(bottleneck #1)*  — ✅ DONE 2026-06-13

The naive ILGPU kernel (one thread per output, no reuse) was replaced with a **shared-memory tiled
GEMM**: each thread group cooperatively stages a tile of each operand into shared memory (load →
`Group.Barrier` → multiply-accumulate → barrier), via an explicitly-grouped kernel (`LoadStreamKernel`
+ `KernelConfig(grid, group)`, `SharedMemory.Allocate`). **One generic tiled core** parameterized by a
`GemmDims` struct (rows/cols/reduction + per-operand row/col strides + accumulate-vs-write) serves all
three layouts (A·B, Aᵀ·B, A·Bᵀ) and the resident-inference write — the inner multiply loop is
layout-agnostic. Boundary guards write 0 for out-of-range tile loads so the tail (the cube's k=324,
not a multiple of 16) stays correct and divergence-free. The **tile edge is adaptive**: 16 on a GPU
(256 threads/group), capped to ≤ logical-core-count on ILGPU's CPU accelerator (whose group limit is
the core count) — shared memory is allocated at the compile-time max (16²) and only a tile² sub-region
is used, so one compiled kernel serves every device.

**Measured (RTX 3060, operands resident, transfer excluded — isolates kernel compute):** naive→tiled
256³-class **562→669** (1.2×), 1024³ **444→626** (1.4×), **2048³ 268→620 GFLOP/s (2.3×)**. The gain
**grows with GEMM size** (exactly where wide nets live) because the naive kernel falls off the memory
wall as the working set grows while the tiled kernel reuses staged tiles. **Honest shortfall** vs the
5–10× / 1–3 TFLOP/s estimate: this is shared-memory tiling *only*. The remaining lever — **M19b:
register-blocked micro-tiles (each thread computes a 4×4/8×8 output block) + vectorized loads** — is
what pushes from ~0.6 to multi-TFLOP, and is the documented next step. cuBLAS/ILGPU.Algorithms remain
a documented escape hatch (native dependency) but the SDK stays from-scratch.
**Files:** `Ilgpu/IlgpuBackend.cs` (tiled kernel + `GemmDims` + launch sites; naive kept bench-only via
`BenchGemmGflops`), `Bench/Program.cs` (larger shapes + transfer-excluded naive-vs-tiled table),
`IlgpuBackendTests` (exact-tile / row/col/k-tail / rectangular / cube-shape, CPU accelerator).
**Gate:** ✅ correctness vs `ManagedBackend` within tolerance (212/212); committed naive-vs-tiled table.

## M20 — GPU residency: device-resident tensors  *(bottleneck #2 — realizes "M12c-perf")*

The scoped `MlpForwardScalar` keeps activations resident but **re-uploads weights every call** (~67 MB/
layer at 8192-wide → ~570 MB/step). Fix: weights resident, re-synced only when they change. **Staged:**
- **Stage 1 — resident-weight inference (unblocks wide-net DAVI):** ✅ **DONE 2026-06-13.** `DeviceMlp`
  (`IlgpuBackend.CreateResidentForward`) holds each layer's weights/biases resident on the device,
  uploaded once on construction and re-uploaded only on `OnTargetSynced`; `Forward` chains the tiled
  GEMM + bias/ReLU on-device, transferring only the input batch up and the scalar outputs down. The
  trainer's `batchForward` Func was replaced by a Core-side **`ITargetForward`** interface (`Forward` +
  `OnTargetSynced`); the trainer calls `OnTargetSynced` exactly when it refreshes the target net (start
  + every `TargetUpdateInterval` steps), so weight upload drops from **per-step to per-sync (~200×
  fewer)**. Default `AutogradTargetForward` keeps CPU-only machines on the autograd path. Lock-shared
  (a sync racing a forward would read half-updated weights). Wired into `cube-davi`.
- **Stage 2 — device-resident training fwd/bwd + on-device Adam** (scoped MLP path): the full DAVI step
  resident, not just inference. **~4–6 d.**
- **Stage 3 — full port:** evolve `IComputeBackend` to a device-handle API (allocate/upload/download/free
  + ops on opaque handles); `Tensor.Data/Grad` become device-backed; port the ~20 autograd ops (11 trivial
  elementwise, 5 reductions/softmax need real kernels, Gather=scatter-add) + Adam; `ManagedBackend` wraps
  `float[]` so CPU/CI stay green. The general SDK-wide GPU capability. **~10–15 d.**
  **Memory (8192×3 on 6 GB 3060):** weights ~0.55 GB/net, Adam ~1.1 GB, activations ~0.15 GB → Stage 1
  ~0.7 GB, full ~2.2 GB — **fits**; throughput (M19), not memory, is the constraint. **Lock-shared sync
  is mandatory** (a `Sync` racing a `Forward` reads half-updated weights).

## M21 — Shortest-move cube solver  *(the capability M19+M20 unlock; QTM-optimal showcase)*

**✅ BUILT + MEASURED 2026-06-14.** All four pieces shipped (`c381389`): `ResidualMlp` + LayerNorm autograd
op, `IValueNet` abstraction, **BWAS** (batched weighted A*), and the Kociemba-QTM gate eval. **Measured
capability after the 236k-iter campaign** (residual 1024×4, BWAS **w=1.5, ≤100k exp, 12 cubes/depth**):
**QTM-optimal (solution length = scramble depth) on all 12/12 cubes through depth 15**, d16 10/12, d17 5/12,
**every solve ~2–2.5× shorter than Kociemba's QTM** — see the gate table. **Key finding:** the live greedy
eval (collapses ~d10) and the light in-loop probe (8k exp — looked plateaued at d14-partial) both badly
**undersold** the net; only a real search budget reveals the true reach. The "plateau" was a *search-budget*
artifact, **not** a network-capacity ceiling (full analysis: `OPTIMIZATIONS.md` → Capability findings).
**Still open:** a provably-optimal Tier-1 claim (weight=1 + BFS verification); pushing past d15 (where the
net finally softens — god's-number 26 QTM remains out of reach on one 3060, as stated). **✅ Wired into the
web cube page** (2026-06-14): "Solve (self-taught AI)" runs BWAS via a resident GPU forward (CPU fallback for
GPU-less hosts) — see `OPTIMIZATIONS.md` → web solver backend.

Make the SDK solve a cube in the **fewest quarter-turns** (god's number 26 QTM), teacher-free, beating
Kociemba (which isn't QTM-optimal). Depends on M19+M20 (a residual net at depth needs the GPU port to train).
- **Residual value net** (`Core.Nn` `ResidualMlp`): 324→4096→2048 + 3–4 residual blocks (width 2048,
  LayerNorm — *not* BatchNorm, which fights the target-net bootstrap), scalar out, ~8–14M params. Reuses the
  scalar-output/checkpoint contract so the trainer/forward are untouched. Depth-with-residuals is the untried
  lever (M17 showed width alone is diminishing).
- **Deepening curriculum + loss-threshold target sync:** raise the cap toward 26; past the greedy-stall
  depth, advance on **loss convergence / time-per-rung** (greedy never hits 0.6 at depth 15, but the value
  still learns from exposure); optional ε-loss target sync (sync only when online-net loss < ε — DeepCubeA's
  stability trick).
- **Batched weighted A\* (BWAS):** batched overload of `ValueGuidedSearch` (expand top-N frontier, score all
  successors in ONE forward — IDA* is wrong here, per-node forwards kill it). λ knob: λ=1 optimal iff the
  value is admissible, λ>1 faster/suboptimal; near-optimal is the honest claim.
- **Two-tier gate (honest):** **Tier 1** — depths 1–7 (BFS-tractable), ≥95% solved **provably QTM-optimal**
  (verified vs `BreadthFirstPlanner`). **Tier 2** — depths 8–20, ≥90% solved with **mean QTM length ≤
  Kociemba's** (beats the teacher). Full god's-number / ~60%-optimal-everywhere is **NOT** reachable on one
  3060 (DeepCubeA used billions of states on multi-GPU for days) — stated plainly.
- Then wire `value-davi` into the web cube page as the third **"self-taught AI"** solver. **Effort:** net
  ~2–4 d, curriculum ~2 d, BWAS ~2–3 d, then the multi-day GPU campaign (gated on M19+M20).

## M22 — MountainCar + Snake  *(SDK breadth: classic control + masked grid game; designed 2026-06-15 by a 4-agent investigation)*

Two new games that exercise different corners of the SDK. Prereqs already shipped this cycle: PPO action masking +
Dueling DQN (PR #2). **Both games ship two modes** (PRD §7.1): **watch-AI = principle B** (server-authoritative
WebSocket stream — backend owns the loop + clock, frontend is a pure renderer, no timer/race) and **human play =
client-side** (a JS timer ticks a local TS engine + keyboard, no backend in the loop). A reusable server-side
episode-streamer (Reset → loop{policy → Step → send frame} until done) backs both watch-AI modes.

**MountainCarEnv** (`Environments/MountainCarEnv.cs`, mirrors `CartPoleEnv`). ✅ **BUILT + MEASURED 2026-06-15.**
- Classic Gymnasium MountainCar-v0: `[position, velocity]`, 3 actions, reward −1/step, `terminated` at x ≥ 0.5,
  `truncated` at 200 (distinct — GAE bootstraps the truncation). `v += (a−1)·0.001 + cos(3·x)·(−0.0025)`,
  clamp v∈±0.07, x∈[−1.2,0.6], left-wall inelastic; start x∼U[−0.6,−0.4]. `IStatefulEnvironment`.
- **Observation NORMALISED to ~[-1,1]** (position centred/9, velocity/MaxSpeed). *Measured finding — the key fix:*
  raw velocity (~0.07) is ~14× smaller than raw position, so the dense net couldn't see velocity (the signal it
  must pump on) and PPO reached the goal **0/100**. Normalising both → solved.
- **Algorithm: PPO** (DQN's ε-greedy can't do the swing-up) trained against an **extended horizon (1000)** so a
  fresh policy ever reaches the goal + a **speed-bonus reward shaping** (`+13·|velocity|`, training only); eval/gate
  on the standard 200-step unshaped env. `NumEnvs 16, RolloutSteps 256, EntropyCoef 0.01, SolveThreshold −110`.
  **Measured:** early-stops at ~120k steps (**< 1 min CPU**), greedy eval **mean return −107.9, reached goal
  100/100** — past the official −110 "solved" bar. Seed shipped to `models/mountaincar.ppo.ckpt` (Git LFS).
- **Viz:** 2-D `<canvas>` hill `y = sin(3x)` + flag at 0.5 + car. **Watch-AI (B):** `WS /api/mountaincar/live`
  streams `{position, velocity, action, reward, done}` per tick, server owns the episode, client renders. **Human
  play:** client-side TS physics (the same dynamics) on a JS timer, ←/→ = push left/right (no key = no push).
  Needs `app.UseWebSockets()` + Traefik `Upgrade` pass-through (the infra delta, shared with Snake).

**SnakeEnv** (`Environments/Snake/SnakeEnv.cs`, mirrors `Env2048`/`RushHourEnv` masked-env style). ✅ **BUILT + MEASURED 2026-06-15.**
- Configurable `Size`×`Size` grid; 4 absolute-direction actions; **`IActionMaskProvider` masks only the 180°
  reversal** (no new masking code — reuses the PPO/DQN masking). Reward +1 food / −0.01 step / −1 death;
  `terminated` on wall/self collision (board-full = win), `truncated` at 1000 steps; seeded food respawn;
  `IStatefulEnvironment` (bitwise resume).
- **Observation: 12 compact engineered features** (danger one-step ×4, food-direction ×4, heading one-hot ×4) —
  **not** a raw grid. *Measured finding:* the raw 12×12 grid (`float[432]`) into a dense MLP learned to
  **survive but not hunt** (~1.5 food, 44 min) — no conv prior. The compact features fix food-seeking AND, being
  **grid-size-invariant**, enable a **grid-size curriculum**: train fast on a small grid, transfer to a larger one.
- **Algorithm: masked Double+Dueling DQN**, `Hidden [128,128], lr 5e-4, buffer 100k, batch 128, ε→0.05 over 30k,
  MaxSteps 100k`, **trained on a 6×6 grid (~10–15 min CPU)**. **Measured:** eats **14.7 food on 6×6** and —
  same net, never trained there — **22.1 food on the 12×12 demo grid** (transfer); far past the **≥5-food gate**.
  Honest framing stays "eats a lot, then eventually self-traps" (no true endgame lookahead). Seed checkpoint
  shipped to `models/snake.dqn.ckpt` (Git LFS).
- **Viz:** DOM/CSS grid like 2048. **Watch-AI (B):** `WS /api/snake/live` streams each move
  `{body, food, action, reward, done}`, server owns the episode (incl. the food-respawn RNG → fully consistent),
  client renders. **Human play:** pure client-side TS engine (`snake-logic.ts`) on a JS timer, keyboard steers,
  reversal guard mirroring the env mask.

**Build path (per the add-a-game checklist — `docs/ADDING_A_GAME.md`, 6 layers each):** env (+ console training to
produce the seed `.ckpt` in `models/`, shipped via Git LFS) → `*ModelService : ITrainableModelService` → a shared
**WebSocket episode-streamer** handler (watch-AI, both games) + `Program.cs` DI & `app.UseWebSockets()` → Angular
page with two modes (watch-AI WS client + human-play TS engine on a JS timer) + route/nav → gallery label.
**Effort:** the first WS streamer is the new infra (~1 d, reused by both); then Snake ~1–2 d, MountainCar ~2–3 d
(env + PPO). Land the reusable streamer + Snake first (simpler env), then MountainCar.

## M24 — Cube efficiency: time-bounded solver + curriculum autopilot  *(2026-06-16, branch `cube-davi-efficiency`)*

Follows the M18/M21 DAVI line. Two fronts; full detail + measurements in `docs/OPTIMIZATIONS.md`.

**Inference (deployed solver).**
- Buffer-pooled `DeviceResidualMlp.Forward` (reuse GPU activation buffers across calls; the per-step successor-eval
  hot path no longer churns the allocator) + state-key caching in `ValueGuidedSearch.SolveBatched`.
- **Time-bounded search:** `ValueGuidedSearch.Solve/SolveBatched` gained an optional `TimeSpan maxTime`; the web
  `CubeController` ships a **15 s deadline** (expansions kept only as a memory-safety ceiling). A budget sweep
  showed 15 s solves the same cubes as 20 s with lower latency — so the budget is *dynamic per cube* (effort scales
  with difficulty) under a fixed latency cap. Diagnosis: the solver was **search-bound through ~d15**, so this is
  the cheap reach lever; verified live on the RTX 3060.

**Training (capability autopilot).** Diagnosed (via the new `--value-curve` probe) that the value is
**accuracy-bound past ~d14** — predicted `V(start)` saturates ~13.7 and flatlines past d18 (a DAVI bootstrap fixed
point, worsened by the old curriculum force-advancing to d26 onto unmastered shells). Rebuilt the recipe into a
run that's safe to leave training unattended:
- **Value-accuracy advance gate** (`--advance-ratio`, default 0.9) replaces the old `greedy ≥0.6 OR force-advance`
  rule — deepen a shell only when `V(d)/d` shows it's mastered, so bootstrap targets are always trustworthy.
- **Auto-widen-on-plateau** (`--auto-widen`/`--max-width`/`--widen-stall-samples`) — on a frontier loss plateau
  (capacity-bound), auto-`WidenTo` the trunk 2× (function-preserving warm start) and continue; trigger is
  loss-plateau, not a timer.
- Supporting: progressive `ResidualMlp.WidenTo` (Net2WiderNet, exact at integer-multiple widths),
  `--set-curriculum-depth`, and tuning flags (`--target-sync-interval`, `--beta2`, `--checkpoint-every`,
  `--frontier-bias`).

Together: the run **deepens when accurate, grows capacity when stuck, and never advances onto inaccurate targets**.

**Result (2026-06-16):** ran the autopilot on the 690M-sample net to push past d14. It stuck at d14, and the loss
floor (~0.10) proved **invariant to lr (2e-3/1e-3/5e-4), width (1024→2048 widen), and sample count** — so the wall
is the **DAVI bootstrap fixed point / sample-scale gap (~7% of DeepCubeA's 10¹⁰)**, *not* capacity or hyperparameters.
**Laptop-scale further-training does not deepen past ~d14-15; the remaining lever is DeepCubeA-scale compute (weeks
on one 3060), which is out of scope.** Capacity ruled out. Full analysis in `docs/OPTIMIZATIONS.md`. The autopilot
itself is sound (the right machinery for such a run); the cheap real win was inference-side (the 15 s time-bounded
solver). Status: branch open (PR #7); deployed net unchanged.

## M25 — Reusable training-campaign harness  *(SDK breadth; inserted 2026-06-21; PRD §14)* — ✅ DONE 2026-06-21 (branch unpushed)

The four campaign harnesses (`Program.cs` Rush Hour, `CubeLab`, `CubeDaviLab`, `CubePolicyLab`) copy-paste one
loop. A 3-agent investigation (2026-06-21) confirmed two paradigms that must not share an interface, and a
migration order easy→hard so the runner is validated before it meets CubeDavi (never design the interface around
CubeDavi first, or it absorbs CubeDavi's specifics and stops being reusable).

**Status (2026-06-21):** ✅ **all deliverables done** on branch `training-campaign-harness` (off master, NOT pushed;
many small commits, one squashed PR — push/PR when ready). **DONE + verified:** the Core abstraction (deliverable 1);
**all four goal-reaching games migrated** (deliverable 2 — CubeLab, CubePolicy, RushHour, CubeDavi); the **Snake**
score-maximizing campaign (deliverable 3 — the original goal); the **hosting/DI layer**; **GPU-backend DI**; and the
**cube `TrainStep`/`Shuffle` dedup**. 264 tests green. **Design note:** the "two paradigms" live at the *eval* level
(solve-rate vs mean-return) over one `ITrainingCampaign` interface — there is intentionally **no `GoalReachingCampaign`
/`ScoreMaximizingCampaign` base class**: every game implements the interface directly, and future score games (2048
n-tuple, SAC/PPO) use different trainers a DQN-shaped base wouldn't fit. **RLDemo.Web is now also wired onto the
shared RL-runtime DI** (`71ed1ae` — `AddReinforcementLearning` + `AddGpuBackend` replace its hand-rolled store/backend
registrations). **Possible follow-ups (not blocking):** share the generic `Shuffle` with RushHour; a
`ScoreMaximizingCampaign` base only if a 2nd score game reveals real shared shape.

**Commits so far:** `ada3d76` docs · `ed1490e` Core `CampaignRunner`+`ITrainingCampaign` (+3 tests) · `102d64a`
CubeLab migrated (eval-only gate 96/100) · `126ed72` CubePolicy migrated (smoke clean) · `2cc4673`
`CampaignRunner`→instance+`TimeProvider` · `43ce002` AIHost + DI + `[Register]` source-gen · `4c60e1f` docs ·
`5e39f8d` RushHour migrated (eval-only parity vs shipped net: cards 16/77/82/81, random30 29/30 g + 30/30 s) ·
`07bb2ed` CubeDavi migrated (GPU stack + curriculum/auto-widen/grow + 5 eval-only modes + two CSVs; value-curve +
fresh-net training smoke verified) · `24e2741` GPU-backend DI (new `Ilgpu.Hosting` `AddGpuBackend()`; 264 tests pass) ·
`04b965c` docs · `982ca9d` Snake DQN campaign (score-maximizing; fresh→20k food@12 ~19.7, resume 20k→35k verified) ·
`d658a9a` cube `TrainStep`/`Shuffle` dedup (`CubePolicyTraining`; gate 97/100, training smoke verified).

**Hosting & DI (decided + built 2026-06-21).** `CampaignRunner` is an **instance** class (not static), taking an
injected **`TimeProvider`** (BCL; system clock by default, a fake clock drives the deterministic tests). A new
package **`MintPlayer.AI.ReinforcementLearning.Hosting`** ships **`AIHost.CreateBuilder(string dataDirectory)`** →
`HostApplicationBuilder` (the AI counterpart to `WebApplication.CreateBuilder`) plus
`services.AddReinforcementLearning(dataDir)` (registers `IModelStore`=`FileModelStore`, `TimeProvider.System`, and
`CampaignRunner`). `CampaignRunner` is DI-registered via **MintPlayer.SourceGenerators**
`[Register(ServiceLifetime.Singleton, "ReinforcementLearningCore")]` — the generator emits
`…Core.DependencyInjectionExtensionMethods.AddReinforcementLearningCore(IServiceCollection)` (which
`AddReinforcementLearning` calls); DI resolves `TimeProvider` through the ctor. Core now references
`MintPlayer.SourceGenerators`(`.Attributes`) `10.20.0`; all hosting / `Microsoft.Extensions.*` deps live in the
Hosting package so Core stays lean. The Lab cube/cube-policy entry points resolve `IModelStore` + `CampaignRunner`
from the host (DI all the way). **Why `dataDirectory`, not raw `args`:** the MS command-line config provider throws
on bare flags like `--eval-only`, so each game parses its own args and passes the dir.

**GPU-backend DI (done 2026-06-21, `24e2741`).** A new package **`MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting`**
ships **`services.AddGpuBackend()`**, registering the shared **`AdaptiveBackend`** (CPU + CUDA-by-GEMM-size routing;
pure CPU when no GPU) as a container-owned singleton. Kept **separate from the lean Ilgpu compute package** so that
backend carries no DI dependency — the same Core↔Hosting split. The two GPU-benefiting cube campaigns
(`CubeDaviCampaign`, `CubeEfficientCampaign`) now take the `AdaptiveBackend` by **constructor injection** and set it as
`Backend.Current`, instead of each `new`-ing + disposing its own; the container owns its lifetime (disposed with the
host), while the campaigns still own their per-eval / per-widen device-resident stacks. **No GPU added to
RushHour/cube-imitation/Snake/2048** — their nets are too small to beat the multithreaded CPU at the routing
threshold (it's ILGPU via the CUDA *driver*/PTX JIT, not the CUDA toolkit; ILGPU also targets OpenCL/CPU).

1. **Core abstraction** — `Core/Training/`: `CampaignRunner` (+ injectable clock) drives `ITrainingCampaign :
   IDisposable` (`Resume`/`TrainChunk`/`IsComplete`/`Evaluate`/`Checkpoint`/`TryRunStandaloneEval`); minimal
   `CampaignEval`; `Action<CampaignProgress>` callback (all Console/CSV IO stays in the Lab). Unit-tested with a
   fake campaign + fake clock + in-memory `IModelStore` (fast bucket, sub-second, no wall-clock).
2. **`GoalReachingCampaign`** base + migrate easy→hard: **CubeLab → CubePolicyLab → RushHour → CubeDaviLab**, one
   commit each, re-verifying tests + CI and keeping each game's existing checkpoint ids + CSV columns. CubeDavi
   exercises every interface addition at once (dual stop via `IsComplete`; curriculum + auto-widen inside
   `TrainChunk`; GPU resident stack via `IDisposable`; five eval-only modes via `TryRunStandaloneEval`; its two
   CSVs stay campaign-owned).
3. **`ScoreMaximizingCampaign`** base + **Snake** campaign — resumable DQN (chunk `DqnTrainer` by raising
   `MaxSteps`, persist the full `DqnTrainingState`), eval = food eaten on the 12×12 grid. This is the `--game
   snake` campaign that started this thread, now in its correct paradigm instead of bent onto the solver interface.

**Gate (pre-registered):** each migrated game behaves identically (existing tests + CI green, same checkpoints/CSV);
the `CampaignRunner` unit test passes deterministically; the Snake campaign resumes bitwise (reload
`DqnTrainingState`) and reaches **≥ the shipped ~22 food / 12×12 baseline**.

**Risks:** `CampaignEval` becoming a god-struct (keep it minimal — the investigation's single biggest design risk);
CubeDavi exercising every addition at once (migrate it LAST, after the runner is proven on the simple loops).

**Stretch:** migrate MountainCar / Pendulum / CartPole / 2048 onto campaigns; a DQN auto-grow hook
only if a Snake plateau proves capacity-bound. *(→ M26.)*

## M26 — Single training path: make RLDemo.Web load-only  — ✅ DONE 2026-06-22 (branch unpushed)

**Problem.** After M25 there were two definitions of "how to train game X": the Lab's resumable
`ITrainingCampaign`s, and the web's one-shot `EnsureModel` in each `*ModelService` (`DqnTrainer.Train(...)` etc.).

**Key finding (reframed the milestone).** That web training was **vestigial**: every game's checkpoint
(`snake.dqn`, `mountaincar.ppo`, … — 11 files) is committed to `models/` via **Git LFS** and seeded into the store
at startup, so `EnsureModel`'s train branch never fired in production. So the fix isn't "make the web run campaigns"
(my earlier plan) — it's **remove the web's training entirely**. Single path = train on a dev machine (Lab/Console),
commit the checkpoint (LFS), the web loads it.

**Done (`9a15264`):**
- `ITrainableModelService.EnsureModel` → **`IModelStartupService.Initialize`** (load checkpoint / warm caches, never
  train); `ModelTrainingHostedService` → `ModelStartupHostedService`.
- The 5 game services dropped their trainer calls, `TrainingOptions`, training consts and progress fields; a missing
  checkpoint now reports `Failed` ("no model in the store") rather than training in-process.
- `ModelStatus` lost `Training`; `StatusResponse`/`Status2048Response` slimmed to `(status, error)`; controllers
  updated. Frontend: the now-impossible `'training'` state removed across all 5 games (api unions, progress fields,
  the 503 `'training'` solve-variant → `'loading'`, banners + polling). Angular prod build clean.
- The RushHour solve gate (which mirrored the service recipe) now **inlines** its own DQN recipe — a self-contained
  recipe-regression check. **264 tests green**; the `WebApplicationFactory` API tests still boot the real `Program.cs`.

**Not needed (vs the earlier plan):** no shared `Campaigns` library and no web↔`CampaignRunner` wiring — the web
doesn't train at all, so the campaigns stay `internal` to the Lab (the contract tests keep the `extern alias` reach).
**Net −284 lines.** If a contributor needs to (re)produce a checkpoint, that's a dev-side Lab/Console run + an LFS
commit, documented in `ADDING_A_GAME.md`.

## M27 — Snake: spatial observation + anti-self-trap shield  *(branch `snake-spatial-obs`, stacked on M25/M26)* ⏳

**Problem.** "Improve the Snake AI" (it expects >100 food). The shipped net scored **~21 food on 12×12** and was
stuck there structurally, not for lack of training: (1) the observation was **12 local features** (danger one cell
out + a food compass) — the agent was *blind to its own body*, so a long snake inevitably trapped itself; (2)
`MaxEpisodeSteps = 1000` was a flat truncation that *hard-capped* how much food was even reachable per episode.

**Fixes (this milestone).** A grid-size-invariant **egocentric observation** — a 9×9 obstacle+food patch centred on
the head + food/tail direction & distance + heading + length + a **flood-fill of reachable open space per move**
(the anti-trap signal a fixed window can't give) — and a **starvation episode limit** (`StarveLimit ≈ 2·cells` with no
food) replacing the flat cap. Plus an opt-in **`safeMask`**: the action mask also forbids moves that flood-fill into
a region too small for the body (a reactive 1-ply shield), used in training and the live web demo.

**Experiment ledger (food@12, 50-ep eval unless noted; all >> the old 21):**
| config | food |
|---|---|
| egocentric obs, `[128,128]` | 39.9 |
| `[256,256]` (capacity) | 39.4 — capacity is *not* the bottleneck |
| + tail feature + γ0.995 | 43.1 |
| + step-penalty 0 + γ0.997 | 42.6 |
| same net + post-hoc shield | 46.6 |
| **shield-*trained*** (γ0.997, pen 0) | **52.3 peak / 50.7 @200-ep** |

Shipping the shield-trained net (**~50 food, ~2.4× the old 21**): `models/snake.dqn.ckpt` overwritten,
`SnakeController` live demo uses `safeMask`. **In progress:** a small→large **grid curriculum** (7×7 → 9×9 → 12×12,
warm-started) — train space-management where trapping is unavoidable-to-confront, then transfer up; may beat ~50.

**Honest ceiling.** ~50 is the **pure-learned (DQN + 1-ply shield)** plateau across capacity/feature/horizon/reward
experiments. Reliable **100** (filling ~70% of the board) needs **multi-ply planning/search** — the EfficientCube
pattern (learned net guiding a beam/lookahead, already in this repo) or a Hamiltonian planner — out of scope for
this learning-only milestone, recommended as the follow-up.

## M28 — 2048 expectimax · NoisyNets exploration · FruitCake A/B  *(PR #15, branch `fruitcake-more-training`)* ✅

Three game-playing improvements plus a reusable exploration capability. Detail lives in
`NOISYNETS_PRD.md` and `FRUITCAKE_AI_PRD.md`; this is the ledger entry.

- **2048 expectimax** (`Expectimax2048`) — averages the n-tuple **afterstate value over the random spawn**,
  fixing 1-ply greedy's spawn-blindness (it would slide the big tile out of a corner and let a stray 2 wreck
  it). 100-game A/B on the shipped net: **avg ~44k → ~84k (~1.9×), best tile 4096 → 8192**, *no retrain*;
  default depth-1 (~1.2 s/full playout — serving-viable; deeper is far slower for no reliable gain). 2048 is
  scored as a high-score game, so it keeps optimizing past 2048. `Game2048Controller` serves it.
- **NoisyNets (N0–N3)** — learned, state-dependent exploration (Fortunato 2017): `NoisyLinear` (factorized
  Gaussian, noise sampled as autograd *constants* → grads reach μ/σ, never the noise — no new op),
  `DuelingQNet` noisy heads + checkpoint **v2** (v1 files load as plain — every shipped `*.dqn.ckpt` keeps
  working), `DqnTrainer` NoisyNets path (resample cadence, ε=0, eval noise-off, serialized `NoiseRng`),
  FruitCake `--noisy` + `ToNoisy` promote-plain→noisy. **Serving unchanged** (noise off default →
  deterministic means). 293/293 non-slow tests; noisy resume bitwise-identical.
- **FruitCake N4 verdict — NoisyNets MATCHED ε-greedy, did NOT beat it → not shipped.** A 200-game paired
  A/B (`FruitCakeAb`, `--game fruitcake --ab --baseline <dir>`) is a statistical tie: candidate **702.1** vs
  baseline **714.4** (Δ **−12.3 ± 29.8 SE**, 49% wins), noisy net higher-variance. The eye-catching 886/971
  single-evals were **seed-luck** — over 200 games both nets sit ~700–714. `models/fruitcake.dqn.ckpt`
  unchanged. **Methodology lesson: judge this stochastic env with the multi-seed paired A/B, never a
  10-episode eval.** Continued **ε-greedy** training remains the proven lever for real FruitCake gains.
- **Vectorized DQN envs — tried & reverted** (~3× *slower*): the compute backend already parallelizes
  data-gen (multithreaded GEMM / CUDA), so a second parallelism layer just oversubscribes the cores. To use
  more cores, run N independent runs with the backend single-threaded. Writeup: `docs/OPTIMIZATIONS.md`.

## M29 — FruitCake: break the pineapple plateau  *(planned — see `FRUITCAKE_IMPROVE_PRD.md`)* 🔜

The shipped FruitCake DQN plateaus at a **pineapple**, almost never a **watermelon**. A four-angle
investigation (observation, reward/algorithm, serving-side search, external Suika SOTA) **all converged**:
the plateau is **reward-design + perception bound, NOT capacity or exploration** (continued ε-greedy *and*
NoisyNets both failed — M28), and the game **rewards planning with a forward model far more than a reactive
net** (a 2025 Leiden thesis: DQN ≈ random + mode-collapse, while a Monte-Carlo forward-model planner *beat the
human*). Two independent angles fingered the same gap — **adjacency of equal tiers**: perception can't see it,
reward doesn't value it. Three levers, prioritized:

- **A. Serving-side forward-model search (no retrain)** — `FruitCakeSearch`, depth-2 over the deterministic
  physics (both pieces known ⇒ no chance node, *cleaner* than 2048's expectimax), leaf = merge points + the
  trained net's max-Q. The literature's #1 lever; reuses the current net; ship first.
- **B. Richer relational inputs (retrain)** — mergeable-adjacency per column, per-column danger margin, next-next,
  then a tier-occupancy grid (the Snake-M27 "inputs > training" pattern; the user's steer).
- **C. Dense reward shaping + n-step (retrain)** — max-tier-reached bonus + **potential-based** shaping toward
  stackable adjacency (policy-invariant), + **n-step returns** (new `DqnOptions.NStep`) + γ→~0.997 for the
  long-horizon credit assignment.

Plan F0–F6 in the PRD; judge on the **≥200-game paired max-tier distribution** (never a 10-ep eval — M28 seed
trap). Honest ceiling: even SOTA Suika agents reach strong-human, not "always watermelon." Recommended first
session: F0 baseline → F1 search (no-retrain win) → training session F2+F3+F4 → F5 A/B + conditional ship.

> **B4 tier-occupancy grid — measured 2026-07-06, NULL result (commit `d8e6893`, not shipped).** The
> lever-B escalation ("do B4 if B1–B3 underdeliver"): appended a 14×10 dominant-tier grid to the observation
> (obs 89 → 229, single-sourced in `fruitcake_solver.pg`), trained a fresh net to 321k drops. Depth-3 net+search
> = **2519 / 52% watermelon** — a tie on the ~2505/~50% bar. The reactive net is **saturated**; richer perception
> doesn't move the deployed search ceiling (same verdict as F2 inputs, M30 big-fruit, M28 NoisyNets, F6 distill,
> curriculum). Code preserved at commit `d8e6893` (tag `archive/fruitcake-tier-grid-b4`); **never merged**
> (229-dim obs would break the live 89-dim net). See `FRUITCAKE_IMPROVE_PRD.md` §4.B B4 + `docs/OPTIMIZATIONS.md`.

## M30 — FruitCake: big-fruit position inputs  *(planned — see `FRUITCAKE_BIGFRUIT_INPUTS_PRD.md`)* 🔜

A focused input experiment (the user's steer): the current 83-dim observation is a per-column **skyline with
no absolute fruit positions**, so the net can't see *where* its biggest fruit sits. Add the **(x, y, tier) of the
two biggest fruits** to `BuildObservation` (→ ~89-dim). **Checkpoint reuse: no** — width change ⇒ the shipped
83-dim `fruitcake.dqn.ckpt` won't load (strict shape check + `FruitCakeModelService` width guard; no
weight-transfer util exists), so this is a **fresh retrain** (F5 precedent), not a resume — an optional ~50-line
weight-pad warm-start util is the only alternative and isn't on the critical path. **Honest prior:** the reactive
net is documented as **saturated** (F2 richer inputs already gave +36% score but 0 watermelon; the watermelon
breakthrough was **search**, not the net; F6 distillation and curriculum both failed). So this is built to
**falsify cheaply** and is judged on **net + forward-search** (the deployed system), not the greedy net. Plan
G0–G4 in the PRD; ship `fruitcake.dqn.ckpt` **only if net+search beats the depth-3 ~50%-watermelon / ~2505 bar**
by a seed-noise-beating margin — else record a negative result (NoisyNets/F6 style) and keep the capability.

## M31 — FruitCake: single-source physics via MintPlayer.Polyglot  *(planned — see `POLYGLOT_FRUITCAKE_PRD.md`)* 🔜

The FruitCake circle-physics solver is duplicated by hand in C# (`FruitCakeWorld`, training + serving) and TS
(`fruit-cake-physics.ts`, human play) — kept in sync only by discipline (the risk `FRUITCAKE_AI_PRD.md` §4.8 flagged).
**MintPlayer.Polyglot 0.1.0 now exists** (a maintained C#↔TS transpiler; FruitCake is its north-star conformance
sample), reopening §4.8's "not viable" verdict. A 3-agent investigation confirmed the fit: both cores are pure
`+ − × ÷ √` with **no transcendentals** (rotation math is pure arithmetic; trig lives only in render/audio glue) and
are already 1:1 ports — squarely inside Polyglot's byte-identical-safe op set. **Output dirs:** TS is configurable
(CLI `--out` → `ClientApp/.../fruit-cake/`); C# is **not** (fixed `obj/.../polyglot/`, compiled in-assembly — so the
`.pg` must live in the Environments project). **Precision:** the pilot uses `f32` in the `.pg` → reproduces today's
C# float32 / TS float64 split exactly (**no retrain**, net stays valid); `f64`-everywhere byte-identity is a deferred
optional upgrade (needs net re-validation). **v0.1.0 constraints to design around:** generated C# is *internal-by-default*
(FruitCakeWorld is public + cross-assembly → facade/`InternalsVisibleTo` until Polyglot ships public-emission) and there's
*no npm/Linux CLI* yet (commit the generated `.ts` for Linux CI). Plan **PG0–PG3**: PG0 = zero-risk validation → PG1 C# cutover → PG2 TS cutover → PG3 optional f64
byte-identity. **PG0 PASSED (on Polyglot 0.1.1), 2026-07-04.** The real solver ports to a clean type-checking `.pg`.
0.1.0 surfaced a blocker (a codegen precedence bug — dropped parens around `??` under `+` → generated TS went all-NaN);
it was root-caused (handoff in `docs/prd/polyglot-pilot/`) and **fixed in Polyglot 0.1.1**. On 0.1.1 the pilot is
**byte-identical C#↔TS and NaN-free** (7-drop trace + 28-drop varied cascade + float checksum) — so byte-identity holds
for this solver. `f32` proved impractical (cast per literal) → `f64` (so the C# solver moves float32→double, gated by
the PG1 net A/B). **PG0 + PG1 DONE.** **PG1 cutover (2026-07-04):** `FruitCakeWorld` is now a thin **public facade**
over the transpiled internal core (`PgFruitCakeWorld`, committed generated C# — not built at runtime, so Linux CI is
unaffected); consumers unchanged; physics moved **float32→double**; env Save/Restore delegates at double precision
(bitwise resume kept). Full build + 17 FruitCake/parity tests green; the trained net re-validated on double physics
(30-game: greedy 895, net+search 2317 / 33% watermelon — within noise, **no retrain**). **PG2 DONE + browser-verified
(2026-07-05):** Polyglot **0.1.4** added TS `export` emission (the 4th fix — 0.1.2 Linux CI, 0.1.3 nullable, 0.1.4
exports); switched C# to **build-time transpilation** (MSBuild PackageReference 0.1.4, no committed `.cs`); the web
client's `fruit-cake-physics.ts` is now a thin `FruitWorld` facade over the committed generated TS core; verified via
host + Playwright (drops/merges/`onMerged`-exact/`mergeBorn`/score, 0 console errors). **PG3 (byte-identity) done as a
side-effect** (f64 both sides). **⇒ PG0–PG2 COMPLETE: FruitCake physics is fully single-source (one `.pg` → C#
training/serving + TS human-play).** Optional **PG4** (not built): watch-AI could stream only the chosen column + animate
client-side (byte-identical physics enables it) to cut streaming bandwidth — but the depth-3 search + net stay C#-only.
**⇒ PG4 is SUPERSEDED by M32** (full client-side inference removes the compute, the bandwidth, *and* the WebSocket).

## M32 — FruitCake: fully client-side AI (zero server inference)  *(✅ CS0–CS7 DONE 2026-07-05 — see `FRUITCAKE_CLIENT_SIDE_AI_PRD.md`)*

> **✅ DONE (2026-07-05) — the FruitCake AI runs entirely in the browser.** CS0 `e429cdf` (world-queries →
> `.pg` + fixed a leaked-`float` timestep) · CS1 `19bd0b8` (`buildObservation` → `.pg`, obs reproduces legacy
> float32 within 1e-5) · CS2 `e8e3c6b` (`PgDuelingNet.forward` → `.pg`, argmax-exact vs SDK) · CS4 `63213db`
> (`chooseColumn` depth-3 → `.pg`, **same column** as C# `FruitCakeSearch`) · CS3+CS6 `d458311` (ship the 89-dim
> net as `ClientApp/public/fruitcake-net.ckpt` (LFS) + TS `.ckpt` parser; client-side `FruitCakeDirector`
> replaces the WebSocket watch mode) · CS7 `b54bfc5` (retire `FruitCakeController` / `FruitCakeModelService` /
> `FruitCakeApi` / stale 83-dim `models/fruitcake.dqn.ckpt`). **Verified host + Playwright:** watch mode plays
> (fruit fall/roll/merge, score climbs), **0 console errors**, weights fetched once from `/fruitcake-net.ckpt`,
> **0 `/api/fruitcake` requests**. 22 FruitCake/Polyglot tests green. Three Polyglot 0.1.4 codegen bugs found +
> worked around (filed **MintPlayer.Polyglot#9**; handoff `docs/prd/polyglot-pilot/POLYGLOT_BUG_HANDOFF_M32.md`).
> **CS5 (quality A/B) — DONE, ship-as-is:** depth-3 net+search on the shipped G3 89-dim net (100 paired games)
> = **2493 mean / 49% watermelon / meanTier 10.48**, wins 100/100 vs greedy — **on par with the prior live AI**
> (83-dim was ~2505 / 50%, within noise). Browser f64 inference is argmax/column-equivalent (CS2/CS4), so this
> is representative. No retrain needed. (Also = the **M30/G4 verdict**: big-fruit inputs are a *null result* for
> net+search quality — the saturated-net prior holds; kept because they don't hurt.) **The canonical FruitCake
> net now lives once at `ClientApp/public/fruitcake-net.ckpt`** (browser); the server loads no FruitCake net.
> **Shipped in PR #23** (single PR: NetTransfer + M30 + M31 + M32).



**Cost-driven (user's steer).** "Watch the AI play" runs the **entire** AI server-side (net forward pass + depth-3
search) and streams every intra-drop frame over the `/api/fruitcake/live` WebSocket — so both server **CPU and
bandwidth scale linearly with concurrent viewers** (100 viewers ≈ 100× load on the single Hetzner VPS, a real
monthly-bill risk). A 4-agent analysis (2026-07-05) confirmed the AI is fully portable: the net is plain float32
matmuls (**noise is off at inference ⇒ mean weights only**), the depth-3 search is **RNG-free deterministic**
game-tree logic, the **physics is already in the browser** (M31 single-source solver), and `BuildObservation` is a
pure function. Nothing is C#-specific — it was only *written* in C#. **Design pivot (user steer 2026-07-05):** rather
than hand-port to TS + babysit golden fixtures, **single-source the inference path (observation + net forward + search)
in the same `fruitcake_solver.pg` as the physics** → C# **and** TS byte-identical by construction. This also
*dissolves* the float-parity problem: with the forward pass authored as **f64 on both sides**, no `Math.fround`
emulation is needed — the deliberate consequence is serving inference moves from the SDK's f32 path to the Polyglot f64
path (strictly more precise; PG1 already validated the net on f64 physics). Two hard boundaries stay per-platform:
**training** (autograd/GEMM/GPU/Adam — Polyglot has no backprop, stays C#/SDK and *produces* the weights) and
**checkpoint parsing** (binary I/O — a ~40-line TS `.ckpt` parser; C# already has one). **Plan CS0–CS7:** CS0 push
`clone`/`anyEjected`/`anyRestingAboveDangerLine`/`pileHeight` into the `.pg` → CS1 `buildObservation` in the `.pg`
(C# env delegates) → CS2 `duelingForward` in the `.pg` (f64 dense MLP) → CS3 checkpoint delivery (MSBuild copy to
`ClientApp/public/`) + TS `.ckpt` parser → CS4 `chooseColumn` depth-3 expectimax in the `.pg` (inline leaf, top-k as an
explicit loop) → CS5 point C# serving + Lab `--search-eval` at the f64 core + one A/B vs the ~50%/2505 bar → CS6
collapse `watch` into a **local director loop** (drop the socket) → CS7 **retire** the server path (`Live` WebSocket +
`FruitCakeModelService` net load + `FruitCakeApi` socket) and measure. **Correctness crux (cheaper than a hand-port):**
per-block **Polyglot-C# == SDK-C#** equivalence tests (obs equality, argmax-exact forward, same-column search) + C#↔TS
byte-identity **free from the transpiler** (no committed goldens, no tolerance); only D5's f32→f64 serving switch needs
an end-to-end A/B. **Payoff:** per-viewer server cost → **0** (one ~370 KB CDN-cacheable weights download). **Depends
on M30/G4** for a valid 89-dim net (client falls back to the heuristic leaf until it lands; CS0–CS2/CS4 are net-agnostic
in parallel). **Supersedes PG4.** Also advances MintPlayer.Polyglot (FruitCake is its north-star conformance sample).

> **Branch `net-transfer-input-grow` — merge readiness (updated 2026-07-05).** This branch carries **four** initiatives:
> M-series NetTransfer/`GrowInput` (SDK, self-contained), **M30** FruitCake big-fruit inputs (obs 83→89), **M31**
> Polyglot single-source physics, and **M32** fully client-side FruitCake AI. **The old merge blocker is RESOLVED:**
> the stale 83-dim server model no longer matters — M32 removed the server's FruitCake net entirely; the browser now
> runs the AI from the 89-dim `ClientApp/public/fruitcake-net.ckpt`. **Remaining pre-master judgement (quality, not
> architecture):** the shipped browser net is the **G3** net; if its net+search play quality (G4 A/B vs the
> ~50%-watermelon / ~2505 bar) is unsatisfying, retrain (M30/G4) and replace `ClientApp/public/fruitcake-net.ckpt`.
> Still recommended: **split into separate PRs** (NetTransfer is independently mergeable; M30/M31/M32 are entangled).
> **UPDATE: merged to master via PR #23 (single PR) 2026-07-05.** A/B ship-as-is (2493 / 49% watermelon, on par).

## M33 — Client-side AI for Snake & MountainCar  *(planned — see `CLIENTSIDE_AI_SNAKE_MOUNTAINCAR_PRD.md`)* 🔜

Extend M32 to the **only two remaining WebSocket-AI games** (2048/RushHour/Cube are request/response REST → out of
scope). A 3-agent investigation + Polyglot **0.2.0** probes (2026-07-05) confirmed: 0.2.0 **fixed all three M32
codegen bugs** (#9 closed — casts / null-typed-locals / nested-generics now work); std.math still has **no
transcendentals** (only `abs`/`floor`/`sqrt`/`min`/`max`); **user `.pg`→`.pg` imports work but *inline* the imported
symbols** (a standalone library `.pg` would double-define under the glob-all build) → **duplicate the net-forward per
game** (simpler than the glob-exclusion a shared `nn.pg` needs); the `.ckpt` parser generalizes to one shared `ckpt.ts`
(`dueling-q` + a new `mlp` branch);
`EpisodeStreamer` is used only by these two → deletable when both migrate. **Two independent PRs, Snake first.**
- **Snake — ✅ DONE (client-side, verified in-browser).** obs **177** (9×9×2 patch + 15 scalars; the "8-ray" memory is stale); net =
  plain `DuelingQNet 177→[256,256]→4` → **reuses `PgDuelingNet` + the parser verbatim**; **no search** (one greedy
  masked Q-step). Real work: port the **action mask** (reversal + anti-self-trap **flood-fill shield**, load-bearing —
  net trained with it) and re-express `LinkedList`/`HashSet`/`Queue` as flat `List`s. Pure integer math → byte-identity
  free. Plan SN0–SN6.
- **MountainCar — ✅ DONE (client-side, verified; uniform Polyglot — 0.3.0 added cos/tanh).** Two transcendentals on the client path (`cos(3·pos)` in the env, **`tanh`**
  in the PPO `Mlp` net) — **neither writable in a `.pg`**. **Recommended: Option B** — pragmatic client-side hand-port
  (reuse `mountaincar-logic.ts` dynamics, ~30-line TS `Mlp`+`tanh` forward, new `parseMlp`, ship ckpt, delete socket);
  no net perturbation, keeps a trivial ~15-line env twin. **Option A** (uniform Polyglot with byte-identical `cos`/`tanh`
  polynomial approximations) only if uniform single-source is a hard requirement, gated on an argmax-parity spike (MC0).
  Plan MC0–MC5. **Payoff:** the last two per-viewer AI sockets → zero server inference.

## M34 — Snake: look-ahead search (client-side, single-source)  *(branch `m34-snake-search`, off `master`; see `SNAKE_SEARCH_PRD.md`)* ⏳

**Problem.** Shipped client-side Snake (M33) plays masked-greedy **one-step** over the 177-dim net and is stuck at the
**~50 food@12** reactive plateau M27 already charted (capacity/features/reward/horizon all swept → ~50; a reactive net
can't avoid a trap that forms several moves ahead). The reachable-free-space input + the 1-ply flood-fill shield M27
added are *already shipped* — the missing lever is **planning more than one ply**, not more training.

**Fix.** Port PR #11's proven idea — net-guided multi-ply look-ahead (food@12 ~50 → ~78.6) — into the **single-source
`snake_solver.pg`**. (PR #11 itself is unmergeable: it predates M32/M33's Polyglot + client-side rewrite — server-side C#
`SnakeSearchAgent` + 39-dim ray obs; `CONFLICTING`.) `chooseActionSearch` runs a receding-horizon **beam search** over
cloned envs with **pure-survival leaf scoring** (board-full win ≫ everything; a death *delayed* beats a death now; else
`food·w − trapPenalty·[reachable<len] + freeSpaceAhead·w − headFoodDist·w`), reusing the shipped `reachableFreeSpace`
flood-fill. The trained net breaks ties between **equally-safe root moves** (one forward per move, not per node). No
retrain, no obs change; runs in C# eval AND the browser director byte-identically.

**Experiment ledger (food@12, 12×12, shipped 177-dim net, no retrain; greedy ≈ 50; d12/b16, net-tiebreak 50):**
| config | food | latency | verdict |
|---|---|---|---|
| survival only (`SpaceRatioWeight` 0) | 70.3 / 72.6 (seed1/100) | 10.7 ms/move | prior baseline |
| **+ anti-fragmentation ratio 100k (SHIPPED)** | **81.3 / 80.6** (max 106–108) | ~11 ms/move | **shipped** — +14%, robust across seeds |
| ratio 200k / 400k | 82.2 / 76.0 | ~10 ms/move | 200k ~ties 100k; 400k over-weights → under-eats |
| net-tiebreak net 500 (ratio 0) | 67.8 | 10.6 ms/move | heavy net weight overrides survival → worse |
| net-guided *per node* d10/b16 | 74.0 | **89 ms/move** | rejected: ~9× cost, no strength gain |
| d20/b32 (ratio 0) | 66.2 | 38 ms/move | rejected: deeper/wider misranks under beam pruning |

**Key findings.** (1) **The anti-fragmentation term is the biggest single lever** — scoring the *fraction of free cells
still reachable* (the user's original reachability-ratio idea, applied in the **search leaf score**, not as a net input)
lifts food@12 ~71 → ~81 (+14%, robust on a second seed base; single games now reach a near-full board, 106–108). It
catches fragmentation the absolute `reachable < length` trap test misses. (2) Depth has a **sweet spot ~12** — deeper+wider
*misranks* under beam pruning. (3) Evaluating the net at every node buys **no** strength for ~9× the latency — the search
carries it — so the net is a cheap **root-move tiebreak**. "Make the Snake net strong" resolves as: the net's reactive
ceiling is real (~50) and can't be trained away; the **agent** is made strong by search, the net a leaf/tiebreak evaluator.

**Client.** `snake-director.ts` swaps greedy `chooseAction` → `chooseActionSearch` (incl. `SpaceRatioWeight`), drives the
live env with `safeMask: false` (planner supersedes the 1-ply shield); `snake_solver.ts` regenerated from the `.pg`; the
shipped `snake-net.ckpt` reused verbatim.

**Gate.** food@12 ≈ **81** (markedly past ~50; high per-episode variance ~55–108), fully client-side, byte-identical
C#/TS, watchable in-browser cadence (~11 ms/move C#). **Honest ceiling.** ~81 mean (single games hit ~106); a reliable
clean 100+ needs a **tail-reachability invariant / Hamiltonian endgame** (PR #11's stretch) — next milestone.
**Transpiler note.** A multi-`.pg` **incremental**-rebuild codegen bug surfaced (stale out-dir → duplicate prelude /
non-`partial` PolyglotProgram); clean/CI builds unaffected. Handoff: `polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`.

## M35 — Snake: curved-tube rendering  *(view-only; planned 2026-07-10; branch `m35-snake-tube` off `master`; see `SNAKE_RENDER_PRD.md`)* 🔜

**Problem.** The board renders as flat coloured `<div>` squares on a CSS grid (`snake.html` `@for` + `cellClass()` +
`.cell/.body/.head/.food`), snapping one cell per `setInterval` tick (120 ms watch / 150 ms human — no rAF, no
interpolation). It reads as a blocky 1980s snake: disconnected squares, hard elbows, no head, no glide.

**Fix (view layer only — zero logic/AI change).** A 2-agent investigation (2026-07-10) mapped the renderer and the
technique space. Introduce a `<canvas>` and a new pure-view `SnakeTubeRenderer` (`snake-renderer.ts`); route the
existing `render(body, food, eaten)` writes into it. Draw the snake as a **curved tube**: a Catmull-Rom spline through
the (interpolated) cell centres → cubic Béziers (`cp1 = p1+(p2−p0)/6`, `cp2 = p2−(p3−p1)/6`; `tension` knob), which
**corners smoothly for free**; a **tapered tail** (ribbon polygon or stamped circles — stroke can't vary width); an
oriented **head** drawn on top (`rotate(atan2(dir))`, eyes, optional tongue); cheap 3-D via a **multi-pass stroke** +
highlight + cached gradient (no `shadowBlur`). The **biggest visual win is interpolation**: a `requestAnimationFrame`
loop animates `p∈[0,1]` per tick over two consecutive states (head lerps old→new head; tail lerps old→new tail
**unless growing**) so the tube glides instead of snapping — rAF only *reads* the latest snapshot, the game loop stays
on `setInterval`. Hi-DPI crisp via `devicePixelRatio` backing-store scaling.

**Decision.** Canvas 2D, not SVG (DOM/layout thrash, slower for a redrawn-every-frame loop, can't taper either) and
not WebGL (only wins at scales we don't have; costs shaders/mesh for no visible gain on one ≤144-segment snake).

**Scope.** Touches only `snake.html` (grid → `<canvas>`), `snake.scss` (drop `.cell*`; canvas is the board), `snake.ts`
(construct renderer, route `render()` → `renderer.push()`, stop the loop on `stop()`/destroy), and new `snake-renderer.ts`.
**Untouched:** `snake-logic.ts`, `snake-director.ts`, `snake_solver.ts`, `snake-net.ts`, `snake_solver.pg` — M34
strength (~81 food@12) and all tests stand as-is.

**Gate (visual, in-browser; no numeric metric).** One continuous rounded tube that curves through turns (no visible
squares/elbows), a head facing travel, a tapered tail, smooth glide between ticks, distinct food; both Watch-AI and
Play-yourself identical; steady 60 fps at full board; crisp on hi-DPI; rAF loop torn down cleanly with the game.
Verify by save-and-live-reload against the running host (**do not** run `ng serve`/`ng build`); attach a before/after
screenshot/clip to the PR. Single view-only PR.

## M36 — Network visualizer: see the net, and watch it learn  *(2026-07-12; branch `m36-network-visualizer` off `master`; see `NETWORK_VISUALIZER_PRD.md`)* ✅

**Problem.** A trained net is only ever visible as numbers — a `.ckpt` on disk, a CSV of eval scalars, a console line
per eval. You cannot *see* a network's shape or structure, and — the real gap — you cannot **watch it change as it
trains**. Learning itself is invisible.

**Fix (M36.1 — shipped; the priority "watch it evolve" gate).** A read-only **pull** telemetry seam plus a viewer
served by the training process itself. A 4-agent investigation (2026-07-12) established the constraint: the web app is
train-free (its server WS infra was removed in M32/M33), so live telemetry must come from the **Lab CLI** where the net
lives; and on the CPU path every parameter is host-resident `float[]`, so reading weights mid-training is free. Design:
- **Core seam** (`Core/Telemetry/NetworkTelemetry.cs`): `INetworkTelemetrySource` (`NetKind` + `SnapshotParameters()` +
  `Sample()`) — a **pull** model, so no trainer calls anything. `NetworkInspector` turns any net's parameter tensors
  into telemetry by pairing each rank-2 weight with the rank-1 bias after it — keying off `Parameters()` (not a
  concrete type) so even the non-`IModule` policy nets work. Frame = per-layer stats + a block-averaged ≤24² magnitude
  **heatmap** (payload bounded regardless of width).
- **All six games**: each `ITrainingCampaign` also implements the source in ~4 lines (return the live net's params +
  metrics it already tracks) — DQN, imitation-policy, EfficientCube-policy, and DAVI value nets alike. No trainer
  changes. Sampling on a background thread is a benign race with training writes and **provably harmless** (no writes/
  RNG/ordering) — **verified**: viz vs no-viz `snake.dqn.ckpt` + `snake.dqn-state.ckpt` are SHA256-identical.
- **Environment-aware tooltips (all games)**: the source optionally exposes `InputLabels`/`OutputLabels` +
  `SampleIo()` + `SampleActivations()` (default null → generic). Labels live on the envs — `FruitCakeEnv` (89
  features + 14 columns), `SnakeEnv` (177 egocentric-vision features + 4 directions), `RubiksCubeEnv` (12
  quarter-turns), `RushHourBoard` (32 vehicle×dir moves). **Every neuron shows a live value** in all five games — input feature
  values, output Q-values/scores, and **hidden-neuron activations** (each net type exposes `LayerActivations`). The
  DQN games forward their actual `CurrentObs`; the batch-trained nets (Cube, Rush Hour, DAVI) forward a **fixed probe
  state** (constant-seed scramble / level-1 puzzle), so you watch the net's opinion of one board evolve. Read-only
  forwards, CPU for a single row even under the GPU backend → still SHA256-identical with a viewer connected. The
  viewer attaches output labels to the column matching their count (a policy net's action head precedes its value
  head) and draws labeled columns in full.
- **Live viewer** (`tools/…Lab/VizServer.cs` + shared `VizLauncher.cs`): an `HttpListener` on localhost serving one
  self-contained HTML page (Canvas 2D node-link graph + weight-heatmap strip + loss sparkline + **beginner hover
  tooltips**; labeled input/output columns drawn in full so each neuron is hoverable) and a **WebSocket** (`GET /ws`). WebSocket (not
  SSE) is the owner's call — bidirectional-ready for future viewer→trainer controls. **Fully async** (per-viewer
  bounded `Channel` + async send pump + async sample loop; no blocking I/O; costs nothing with no viewer connected).
  **Gated to a Development host environment** (`VizLauncher` skips it in Production; the Lab defaults to Development).
  `--game <snake|fruitcake|rushhour|cube|cube-policy|cube-davi> --viz [port]` (bare `--viz` = 5250).

**Gate (met).** `--game snake --viz` → the net **visibly evolves while training** (weight heatmaps/edges shift,
per-layer |w| grows, eval climbs); **hover tooltips** explain each part for RL newcomers; a **Production** process
skips the socket (prints a note) while training proceeds; opening the page mid-run shows the graph instantly; and viz
vs no-viz checkpoints are **SHA256-identical**. 314 fast tests green. Screenshot (with tooltip):
`docs/screenshots/m36-network-visualizer.png`.

**Follow-ups (planned).** M36.2 — static `.ckpt` inspection as an Angular `/network` page reusing the browser
`.ckpt` parsers (client-side, no server). M36.3 — signed (diverging) heatmaps; viewer→trainer controls over the
existing WebSocket (pause/step, cadence, layer-select); continuous-control PPO/SAC once they train through the Lab's
`--game` dispatch.

## M37 — Progressive net growth (Net2WiderNet / Net2DeeperNet)  *(2026-07-12; branch `m36-network-visualizer`)* ✅

**Problem.** A net trains at a fixed architecture; the visualizer made it natural to *watch* a net grow, but the DQN
games had no growth to show.

**Fix.** Function-preserving architecture growth on the shared `DuelingQNet`: `WidenTo` (Net2WiderNet — new units
duplicate a random existing one, the next layer's incoming weights split evenly across copies) and `Deepen`
(Net2DeeperNet — an extra trunk layer initialized to identity, exact after a ReLU). Both produce a net computing the
**same function** (unit tests assert forward-equality). A shared `DqnGrowth` helper applies a staged schedule
(`[16]→[32]→[32,32]→[64,64]→[64,64,64]→[128,128,128]`, alternating wider/deeper) on a step cadence, rebuilding the
Adam optimizer (moments are keyed to the parameter set) via `DqnTrainingState.WithNetwork` (buffer/RNGs/n-step
accumulator/step-count carried over; obs & action dims unchanged so they stay valid). Wired into **both DQN games**:
`--game snake|fruitcake --grow [--grow-every N]` (starts from the tiny stage, grows mid-run).

**Coverage — every trainable net now grows.** The growth math is factored into `Net2Net` (`WidenTrunk` + `SetIdentity`)
and shared:
- **DQN games** (Snake, FruitCake) — `DuelingQNet` grows wider+deeper via `--grow` (`DqnGrowth`).
- **Imitation/EfficientCube policy nets** (Cube, Cube-policy, Rush Hour) — the two-headed nets were **refactored** from
  a fixed 2-layer trunk to a variable-depth `PolicyValueNet` core (shared by `CubePolicyNet`/`RushHourPolicyNet`).
  Checkpoint format bumped to **v2** (stores the trunk-widths array); **v1 shipped files still load** (one hidden width
  → a two-layer trunk), guarded by a test. They grow wider+deeper via `--grow` (`PolicyGrowth`, same schedule).
- **DAVI value net** (`ResidualMlp`, cube-davi) — already grew **width** (pre-existing `--auto-widen` / `--grow-to`).

**Gate (met).** `--game fruitcake --viz --grow` grows the DQN net `[16]`→`[128,128,128]` live (wider **and** deeper),
and `--game rushhour --viz --grow` grows the **policy** net the same way (`docs/screenshots/m37-policy-net-grows.png` —
note the identity diagonals in the just-deepened layers' heatmaps), both with **no loss spike** (function-preserving).
320 fast tests green (incl. Net2Net forward-equality for DuelingQNet + PolicyValueNet, and a v1-checkpoint-load test).
Screenshots: `docs/screenshots/m36-network-grows.png` (DQN), `m37-policy-net-grows.png` (policy).

## M38 — Reduce per-game boilerplate (campaigns · web services · frontend)  *(2026-07-12; branch `m38-reduce-boilerplate-plan`, PR #31; supersedes stale PR #27; see `BOILERPLATE_REDUCTION_PRD.md`)* ✅

**Status (2026-07-12):** B0–B5 complete on `m38-reduce-boilerplate-plan` (PR #31), each behaviour-preserving with its
own commit; solution builds 0/0 and the fast suite (now **326** tests) stays green after every step. **B2 is
SHA256-bitwise-verified** (fresh seed-1 `snake`/`fruitcake` produce byte-identical deployable + resume checkpoints vs
the pre-refactor build), and the whole branch was independently re-reviewed line-by-line (no behavioural divergence)
and **exercised live**: the web host loaded the real shipped checkpoints (`StartupCheckpoint`/`RefreshingCheckpoint`,
incl. the Cube `onReload` GPU-resident rebuild on a real RTX 3060), the Lab games ran through `LabHost` (CPU + GPU),
and the Snake page's Watch-AI was Playwright-verified (plays, survives a visibility toggle, zero console errors).
**B3 `CliArgs`** landed too (typed, bounds-safe, culture-invariant flag reads across five labs — the int/long/ulong
locale inconsistency fixed; `CubeDaviLab`'s bespoke config-precedence block left as-is; 6 new unit tests incl. a
comma-decimal culture test). **B5 P7** landed as a *removal*: the byte-identical per-component `onVisibilityChange`
was redundant with `ScreenWakeLock`'s own re-acquire, so it's deleted from all three components. The one item
**intentionally not done** is B5 P8 (a shared director watch-loop) — its shared part is a 3-line `setInterval`
wrapper around game-specific render/status, a shallow abstraction not worth extracting.


**Problem.** Adding each game left near-identical copy-paste in three layers, and several copies have already
**drifted** (one has a bug its siblings don't). PR #27 took a first cut on 2026-07-10 but is now stale/conflicting:
M36 (`INetworkTelemetrySource`) and M37 (`DqnGrowth`) edited exactly the campaign bodies/fields PR #27 relocates,
so a rebase would silently drop net-growth. This milestone re-scopes: keep what still applies, re-cut the campaign
refactor to **absorb** the M36/M37 duplication, and add the further duplication a three-agent audit surfaced. A
**behaviour-preserving** refactor — training stays SHA256-bitwise-identical.

**Audit (2026-07-12, 3 parallel agents; counts verified vs `master`).** Lab: two DQN campaign twins (~200 shared
lines), `INetworkTelemetrySource` copied **6×**, net-growth wiring per campaign, CLI flag parsing hand-rolled 6×
(with an ad-hoc-culture latent bug), host-bootstrap tail 6×, imitation/policy plumbing 3×. Web: the cadence-refresh
"keep-previous-on-corrupt" checkpoint getter **4×** (highest bug-risk), startup/readiness quartet 3×, `TryBuildPuzzle`
duplicated controller↔deck-store (drift bug), `/status`+503 3× (+ a needless `Status2048Response` fork). Frontend:
`pollStatus()` 3× (2 **missing the `try/catch`** cube has — a latent unhandled-rejection bug), `*-api.ts` status/503
3×, watch wake-lock scaffolding 3×, director loop 2×, atomic-write 2×.

**Phased plan (each step ends on a green build + the relevant test suite; ordered low-risk→high-risk so a regression
is bisectable).**

- **B0 — docs + standalone bug-fix extracts (no behaviour risk).** Land `Core/Checkpoints/README.md` (the
  when-to-use-which-checkpoint table, from PR #27, still accurate). Fix + de-dup the two drift bugs: one
  `RushHourPuzzleDto.TryBuild` on the Environments layer (controller + `RushHourDeckStore` both call it), and a
  frontend `pollModelStatus(api, set, ms)` with `try/catch` built in (fixes 2048 + rush-hour). Delete
  `Status2048Response` in favour of the shared `StatusResponse`. **Gate:** web API tests green; frontend builds.
- **B1 — web checkpoint concern (deep pair).** `RefreshingCheckpoint<T>` (`.Current` hides TTL + double-checked
  lock + keep-previous-on-corrupt + `onReload` hook) and `StartupCheckpoint<T>` (readiness holder, applied by
  **composition** — no shallow `ModelService<T>` base). Collapse the 4 refreshing getters + 3 startup quartets in
  Cube/Game2048/RushHour services; Cube's GPU resident-forward rebuild rides `onReload`. `ModelStatusResult`
  controller extension for the `/status` trio (solve endpoints stay bespoke). **Gate:** web service + API tests
  green; net negative diff.
- **B2 — re-cut `DqnScoreCampaign` base (the M36/M37 absorption).** Shared score-max DQN spine owning
  resume/train/save-best/checkpoint **plus** net-growth (`grow`/`growEvery` + `_growRng` + `DqnGrowth.Maybe` in its
  `TrainChunk`) **plus** `INetworkTelemetrySource` itself; Snake/FruitCake supply only env + `BaseOptions` +
  `EvaluateNet` + labels + FruitCake's `AdaptWarmNet` (plain→noisy + `GrowInput`). **Gate (the hard one):** a
  fixed-seed `--game snake`/`fruitcake` run yields a **SHA256-equal** `*.dqn.ckpt` + `*.dqn-state.ckpt` vs a
  pre-refactor baseline; `--viz` still streams both nets live.
- **B3 — Lab CLI + host plumbing.** `CliArgs` value-reader (hides bounds-check + `++i` + `InvariantCulture` — fixes
  the locale inconsistency) + `CommonLabArgs.Parse` for the shared 6 flags; `LabHost.Run(...)` owning
  DI+GPU+CSV+viz+viewer-lifetime (`WaitForViewer` moves off `SnakeLab`). Apply to all 6 labs. `LabLog.Line`,
  `GrowthRng(seed)` trivia. **Gate:** every `--game` still parses + runs identically (bitwise spot-check retained).
- **B4 — imitation/policy plumbing extract.** `SupervisedNetState` (net+Adam load-or-init/save + growth one-liner)
  + `WindowMean`, applied to RushHour/CubeImitation/CubeEfficient. **Campaigns stay separate** — only plumbing is
  shared (their `Evaluate`/data-gen are different algorithms). **Gate:** a short imitation/policy run is
  SHA256-equal to baseline.
- **B5 — frontend watch scaffolding (optional polish).** `WatchWakeLock` (mode-signal-driven; hides sentinel +
  `visibilitychange`) for snake/mountaincar/fruit-cake; `runDirectorLoop(...)` for the 2 `setInterval` directors;
  `fetchStatus`/`classifySolve` in `*-api.ts`; `AtomicFile.Write` for the 2 temp writes. **Gate:** frontend builds;
  manual watch-mode smoke.

**Deliberately left alone:** per-game *solve* controllers/DTOs (different algorithms — a shared base would be
shallow/leaky, PR #27's correct call), the 3 imitation/policy campaigns *as wholes*, `CubeDaviLab`'s 46-flag block,
`CampaignRunner`/`DqnGrowth`/`PolicyGrowth`/`VizServer` (already deep), fruit-cake's rAF loop.

**Gate (whole milestone):** full solution builds 0/0; `dotnet test --filter "Category!=Slow"` green; DQN + a
policy campaign SHA256-bitwise-identical vs baseline; `--viz` live for every game; net **negative** line diff with
every shared module carrying an interface comment. See `BOILERPLATE_REDUCTION_PRD.md`.

## M39 — Self-play training (Connect-4 → chess)  *(2026-07-12; branch `m39-chess-selfplay-plan`, PR #32; see `CHESS_SELFPLAY_PRD.md`)* — M39.1 + M39.2 SHIPPED

**Status (2026-07-12):** **M39.1 (rails on Connect-4) and M39.2 (chess) SHIPPED**, each its own commit on
`m39-chess-selfplay-plan`. The reusable stack — `IZeroSumGame<TState>` + `Core/Planning/Mcts.cs` (PUCT) +
`PolicyValueTraining` + `SelfPlayCampaign<TState>` — is in Core/Lab; Connect-4 and chess are both consumers, the
latter reusing the rails **unchanged**. Chess movegen is **perft-verified** (25/25 published counts, incl. startpos
depth 5 and Kiwipete depth 4); the 4672 move encoding round-trips (encode→decode→apply) with no collisions; a
`--game chess` run from random init plays legal chess and beats a random-legal opponent. A **robustness slice of
M39.3** also shipped: `--opponent-random` mixes in learner-vs-random games so the net trains on the off-distribution
positions an unexpected move reaches (the direct code answer to "a novel move disorients the AI"). 361 fast tests
green (perft, MCTS-vs-negamax, encoding round-trip, both self-play contracts). The rest of M39.3 (batched-leaf MCTS,
GPU/conv, league play, web page) remains optional future work. Honest scope holds: legal, steadily-improving play,
not engine strength.

**Problem.** The SDK trains via reward (DQN/PPO), a forward model (DAVI), or an exact oracle (cube/Rush Hour imitation). Chess
has no cheap oracle and a reactive net plateaus — the repo's own history says **search is the lever**. The missing paradigm is
**self-play**: the improving net plays itself and the games *are* the training signal (AlphaZero-style). Add it as a **reusable**
capability — chess is the headline, but the machinery is shared so the next two-player game plugs in by writing only its rules.

**Key decision (design-it-twice, 3-agent investigation 2026-07-12).** AlphaZero-style (MCTS-guided self-play + the two-headed
`PolicyValueNet`), **not** plain reactive self-play (tactically blind — the documented plateau). The chess rules engine is the
same irreducible cost either way, so the marginal cost of "real improvement" is just MCTS + visit-count targets. One new **deep**
seam — `IZeroSumGame<TState>` in `Core/Planning` (a *sibling* to `IDeterministicModel`: side-to-move + win/loss/draw + per-state
legal moves, which `IDeterministicModel` can't honestly express) — consumed by both MCTS and the self-play campaign. **Reuse-first:**
~70% of the outer loop already exists (`PolicyValueNet` unchanged, the soft-CE+value train step from the imitation campaigns,
`ITrainingCampaign`/`CampaignRunner`, model store + checkpoint format, the M38 `AdamState`/`TrainWindow`/`PolicyGrowth` plumbing,
action masking, RNG streams, `--viz`, the A/B harness → Elo gate). New code is confined to the seam, MCTS, the `(obs,π,z)`
self-play loop, and each game's rules.

**Phased plan (each step ends on a green build + its gate; ordered to de-risk the *novel* machinery before the huge rules surface).**

- **M39.1 — the rails, proven on Connect-4.** `IZeroSumGame<TState>` + `Core/Planning/Mcts.cs` (PUCT, sign-flipping value backup,
  Dirichlet root noise, temperature → root visit-count π) + `PolicyValueTraining.TrainStep` (generalized from
  `CubePolicyTraining.TrainStep`: soft-CE + value regression) + `SelfPlayCampaign : ITrainingCampaign` (reusing
  `AdamState`/`TrainWindow`/`PolicyGrowth`/telemetry) + `Connect4Game : IZeroSumGame` + a **negamax** oracle for tests. **Gate:**
  MCTS unit-tested vs negamax on forced-win/draw positions; from random init, self-play win-rate vs negamax/random **climbs**;
  `CampaignContractTests` resume roundtrip; fast suite green. This validates the self-play machinery cheaply, away from chess rules.
- **M39.2 — chess as consumer #2.** `ChessBoard` + full legal movegen + draw/mate detection; the 8×8×73 = 4672 move encoding;
  `ChessGame`/`ChessEnv` (`IEnvironment` + `IActionMaskProvider` + `IStatefulEnvironment`); flattened-plane observation (static,
  train==serve); a thin `ChessPolicyNet` wrapper over `PolicyValueNet` (policy head 4672, value `tanh` → WDL); `ChessSelfPlayCampaign`
  + `--game chess` Lab dispatch; an Elo eval (`ChessAb`, mirroring `FruitCakeAb`). **Gate (the hard one): perft** node-counts matched
  to published values (startpos, Kiwipete, …) to depth 5–6 *before any training*; move encode/decode round-trips; terminated-vs-truncated
  split correct; win-rate vs random-legal climbs; contract-test resume; ship `models/chess.az.ckpt` (LFS).
- **M39.3 — scale + robustness (optional).** Batched-leaf MCTS; GPU-resident forward *if* the net grows enough to clear the
  routing threshold (honest: a small chess MLP won't); conv-backend support for positional strength (a separate backend workstream);
  a web showcase page. **Anti-exploitation levers** (so a novel/weak human move can't disorient the net into losing — real even at
  superhuman level, cf. the KataGo cyclic-group exploit): opponent-**pool/league** play + an occasional random/weak mover, diverse
  randomized opening positions, and adversarial fine-tune on any discovered exploit. (Dirichlet root noise + temperature already ship
  in the M39.1 MCTS/campaign; search-from-the-actual-position is the primary defence. NoisyNets is *not* the right tool — it perturbs
  the policy globally rather than broadening position coverage.)

**Honest scope.** MLP-only net (no conv) over a flattened board → *legal, steadily-improving* play, not engine strength; self-play is
CPU-bound (the small MLP won't hit the GPU lever that helps the cube). Connect-4 is where the self-improvement curve is unmistakable;
chess is the headline consumer. **Left out of v1:** DQN/PPO for self-play (wrong fit), the DQN `ReplayBuffer` (uses an `(obs,π,z)`
window instead), bitboards (unless a bench forces it), superhuman strength. **Whole-milestone gate:** builds 0/0; fast suite green;
perft passes; both games' self-play win-rate rises vs a fixed baseline; resume roundtrips. See `CHESS_SELFPLAY_PRD.md`.

## M40 — Play the chess AI in the browser (single-source via MintPlayer.Polyglot)  *(2026-07-12; see `CHESS_WEB_POLYGLOT_PRD.md`)* — M40.1–M40.4 ✅ SHIPPED (net committed; conv-net strength upgrade tracked in M42)

**Goal.** Play the self-taught chess AI (M39) **in the browser**, client-side, with **zero server inference** — the
FruitCake pattern (ARCHITECTURE §10): write the inference path once in a `.pg`, transpile to C# (training/serving) +
TypeScript (browser); the browser downloads and parses the `.ckpt` and runs the net + MCTS locally. The wrong path
is a server `ChessController` (per-viewer CPU — the thing M32/M33 removed); the right path is single-source.

**Feasibility — VERIFIED (2026-07-12)** by transpiling a probe with the bundled `polyglot.exe`:
- `Math.exp`, `Math.tanh`, `Math.log`, `Math.sqrt` all transpile (need `import { Math } from "std.math"`), so MCTS's
  masked-softmax priors + `tanh` value + PUCT `sqrt` are expressible. Transcendentals are **not** bit-exact across
  C#/JS (only `+ - * / sqrt` are — the reason FruitCake avoided them) but chess **inference doesn't need bit-exactness**
  (the browser AI must play well, not match C# to the ULP), so `exp`/`tanh` are fine.
- Bitwise ops `& | << >>` transpile (castling flags; booleans also fine).
- **Polyglot itself is extendable:** the toolchain lives at `C:\Repos\MintPlayer.Polyglot` — a missing feature
  (`switch`/enum/Math fn) can be **added there and PR'd** (owner-authorized). Known limits to design around
  (from the FruitCake solver): **no nested-generic params** (use flat `List<f64>` + offsets like `PgDuelingNet`),
  prefer `i32` consts + `if/else` over enums/`switch`.

**Phases (each ends on a green build + its gate).**
- **M40.1 — single-source the engine. ✅ SHIPPED 2026-07-12.** Ported `ChessBoard`/`ChessRules` + `ChessMoveEncoding`
  into `Environments/Chess/polyglot/chess_solver.pg` (internal `Pg`-prefixed core: `PgChessState` + `PgChessMove` —
  board `List<i32>[64]`, promotion/castling as `i32`, bounded `for`+`continue` rays, `List<(i32,i32)>` delta tables;
  every construct proven by `fruitcake_solver.pg`). `MintPlayer.Polyglot.MSBuild` transpiles every `**/*.pg` to `obj/`
  before CoreCompile (bumped 0.3.1 → **0.6.0**, which bundles win-x64 + linux-x64 + linux-arm64 + osx-x64 + osx-arm64,
  so CI/ubuntu is unaffected and macOS dev no longer needs `$(PolyglotTool)`). `ChessState`/`ChessRules`/
  `ChessMoveEncoding` are now **thin C# facades** over the core (public API + both test files unchanged); `ChessGame`'s
  seam (`LegalMoves`/`Apply`/`Result`) delegates to the core's `legalMoveIndices`/`applyIndex`/`result` so training and
  the browser share one implementation; `perft` recurses entirely in-core. **Gate MET: perft 25/25 on the generated
  engine** (incl. startpos d5 = 4,865,609, Kiwipete d4 = 4,085,603, ~10 s), encoding round-trip + terminal-detection
  green, `SelfPlayCampaign` chess contract green, full suite 362/362 (no FruitCake/Snake/MountainCar regression from
  the shared re-transpile). **Polyglot MSBuild multi-`.pg` incremental bug — FIXED upstream in 0.6.0 (PR #26):** a
  single-`.pg` edit used to make MSBuild's partial-incremental build hand the CLI a subset → inline duplicate prelude
  clash (CS0101/CS0260). Diagnosed here, verified a stamp-`Outputs` + `RemoveDir` fix against the source `.targets`,
  and it shipped in 0.6.0; the temporary local `_PolyglotForceFullRetranspile` workaround has been removed
  (`docs/prd/polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`).
- **M40.2 — single-source the inference math + a TS `.ckpt` parser. ✅ SHIPPED 2026-07-13.** Added to the `.pg`:
  `writeObservation()` (18-plane × 64 = 1152, parity-tested vs `ChessGame.WriteObservation`), `PgPolicyValueNet.forward`
  (flat-array ReLU trunk → policy logits + linear value), and `PgChessMcts` (inference PUCT — masked-softmax priors,
  `sqrt`, value negated per ply, **no Dirichlet**). `ClientApp/src/app/chess/chess-net.ts` parses the `selfplay-pv`
  `.ckpt` (magic RLNC, trunk widths, per-layer W/b in `Parameters()` order; inputSize/actions supplied) into the
  generated `PgPolicyValueNet`; committed `chess_solver.ts` emitted. **Gate MET:** `ChessNetParityTests` — C#
  `PolicyValueNet.Forward` vs the generated net agree within f32 tol on the start position (round-tripped through the
  real `.ckpt` bytes), plus observation parity + an MCTS runtime smoke (valid legal-move distribution, `chooseMove`
  legal); 358/358 fast tests. Also did the `std.math` cleanup (`Math.abs`/`Math.max`, kept `isign`). The two TS-emitter
  gaps chess first hit (local `List` decls losing their annotation; `List<T?>` → `T | null[]`) were filed as
  **MintPlayer.Polyglot issue #27** with a verified two-part fix, **fixed upstream in 0.7.0 (PR #28)**; the Environments
  `.csproj` is on 0.7.0 and the regenerated `chess_solver.ts` is now **strictly typed** (`let d: [number,number][] = []`,
  `children: (PgMctsNode | null)[]`) — verified strict-`tsc`-clean. 365/365 tests on 0.7.0.
- **M40.3 — the browser page. ✅ SHIPPED 2026-07-13 (net committed).** `chess-director.ts` (runs the
  transpiled `PgChessMcts`+net over the loaded `.ckpt`) + a standalone Angular chess component: an 8×8 board (White at
  the bottom, square rows), click-to-move validated by the transpiled engine (legal-target dots, orange
  selected/last-move highlights, auto-queen), check/checkmate/stalemate/draw status, a **captured-pieces tray** (red ✕
  per taken piece), and **two modes — "Play the AI" and "Watch AI-vs-AI"** (self-restarting loop with a speed slider).
  Route `/chess` + a Home tile; `.ckpt` MIME mapping already in `Program.cs`. **Gate MET:** Angular builds the chess
  chunk clean; `/chess` served; `/api/chess/*` → 404 (zero server inference); director/net/solver strict-`tsc` clean;
  and the transpiled engine+net+MCTS played a full 80-ply legal game in Node over the real checkpoint. Playwright MCP
  was unavailable, so the in-browser click-through wasn't automated — verified structurally + functionally instead.
  **Net SHIPPED (commit `1dd734d`):** `wwwroot/models/chess.az.d1.ckpt` (LFS) + `chess-difficulties.json` committed, so a
  fresh deploy has a net to load; verified headless (loads + plays a full legal game). Honest caveat: this is the flat-MLP
  net → legal-but-weak chess; the conv-net upgrade is tracked in **M42** (replaces it with no page changes).
- **M40.4 — difficulty via an auto-captured net ladder (both modes). ✅ SHIPPED (investigation + owner refinement
  2026-07-13; see `CHESS_WEB_POLYGLOT_PRD.md` §9, esp. §9.6).** Training is offline-only (the Lab); the ladder is
  produced **hands-off by the training agent** — when the live net becomes *significantly stronger than the last
  promoted checkpoint*, the Lab auto-writes a new difficulty `.ckpt` into `src/RLDemo.Web/wwwroot/models/` + updates a
  manifest. The net ladder is the backbone (not a novelty) because promotion is gated on a **net-vs-net arena** margin,
  making tiers reliably ordered by construction (Level K+1 provably beats Level K).
  - **M40.4a — the Lab mechanism (the owner's ask).** In `SelfPlayCampaign<TState>` (generic; enabled by `--ladder`):
    a champion = last-promoted frozen net; on each eval, `ArenaVsNet(liveNet, champion, arenaGames)` (deterministic
    MCTS argmax, short randomized openings on a **separate arena RNG** → training stays bitwise-reproducible,
    alternating colours); if challenger score ≥ `--promote-margin` (~0.58) → save `chess.az.d{K}.ckpt` to
    `--difficulty-dir` (default the web models dir) + rewrite `chess-difficulties.json` (`{label, ckpt, sims,
    temperature, cpuct, winRateVsRandom}`) + champion := frozen copy. Flags in `ChessLab`. **Gate:** a short ladder
    run auto-produces ≥2 ordered tier ckpts + a valid manifest in `wwwroot/models`; a non-`--ladder` run's weights are
    unchanged for the same seed (reproducibility check).
  - **M40.4b — the web selector.** `chess-director.ts` loads `chess-difficulties.json` (hardcoded fallback), a
    `setDifficulty(d)` (re-fetch net only if the ckpt URL changed; cache by URL; `aiStep` uses `search`+optional
    temperature, `T=0`→argmax), and a Level-picker in both modes (Play = opponent; Watch = shared level, per-side is a
    follow-up). No `.pg` change (`PgChessMcts.search` already returns π). Honest labels ("Level K" / "Full strength",
    never "Grandmaster").

**Honest scope:** the M39 net is a small, briefly-CPU-trained MLP → legal, still-learning chess, beatable by a decent
human. **Non-goals:** engine strength, a server chess endpoint, bit-exact transcendentals, browser training. See
`CHESS_WEB_POLYGLOT_PRD.md` (incl. its reference appendix: files to port, checkpoint byte-format, commands).

## M41 — Reusable deterministic CPU-parallel data generation  *(2026-07-13; see `PARALLEL_SELFPLAY_PRD.md`)* — M41.1 + M41.2 + M41.3 ✅ SHIPPED

**Why:** AlphaZero self-play (`SelfPlayCampaign`) generates games **single-threaded** (`TrainChunk`'s `for … PlayGame()`),
and for chess the wall-time is dominated by CPU-bound MCTS **movegen** — so a multi-core box mostly idles while training
crawls (the M40.4 256-sim run bottleneck). The GPU is a poor fit here (tiny net, batch-1 inference); **CPU-core
parallelism is the real lever.**

**Finding (2-agent investigation):** the repo already parallelizes *cube* data-gen, but it's **hand-rolled + duplicated
in two Lab files** (`CubeImitationCampaign.cs:71`, `CubeEfficientCampaign.cs:93` — `Parallel.For` + per-worker lists +
per-worker seeded RNG); **nothing reusable exists in Core** (Core only has generic GEMM / `VectorEnv` / DAVI-featurize
parallelism). Self-play + DQN campaigns have no explicit parallelism. Self-play **is** safe to parallelize: shared
read-only net inference is concurrent-safe (fresh buffers, no static cache, `[ThreadStatic]` `NoGrad`, batch-1 forwards
don't nest-parallelize); the only blockers are the shared `_window`/counters and the shared mutable RNGs. The
bitwise-reproducibility invariant (M25/M26/**M36 SHA-verified**) is preservable exactly as Core already does it
(`VectorEnv`: per-unit RNG; GEMM: disjoint output rows → DOP-invariant).

- **M41.1 ✅ — Core primitive.** `DeterministicParallel.Generate<TItem>(count, seeds, stream, baseIndex, makeItem, parallel, dop)`
  in `Core/Training`: per-index-derived RNG (golden-ratio stride, à la `VectorEnv`), disjoint ordered slots, index-order
  results. 12 unit tests prove bitwise parallel==sequential across DOP 1/2/4/8/16 + edges (commit `bc0c48f`).
- **M41.2 ✅ — parallel self-play.** `SelfPlayCampaign` generation refactored onto it: each game a pure fn of its own
  index-derived RNG over the stable read-only net, returning samples; owner-thread merge in ascending index then train.
  Shuffle moved to the Buffer stream (no seed collision with game 0). `--parallel`/`--dop` on `ChessLab`. **Gate MET:**
  a Connect-4 run gives a **byte-identical checkpoint** at sequential vs parallel-dop-1 vs dop-8
  (`SelfPlayCampaignTests.ParallelGeneration_...AtAnyDop`). Eval/arena left sequential (not the bottleneck; can't affect
  weights). Commit `a9fa5c3`.
- **M41.3 ✅ — cube dedup.** `CubeImitationCampaign` + `CubeEfficientCampaign` migrated off their hand-rolled
  `Parallel.For` onto `DeterministicParallel.Generate` (new raw-`ulong baseSeed` overload; the `SeedSequence` overload
  now delegates to it). Passing `(baseSeed: roundBase, baseIndex: 1)` reproduces the old `roundBase + φ·(worker+1)`
  per-worker seeding **byte-for-byte** — verified by `DeterministicParallelTests.RawSeedOverload_ReproducesTheCube…`,
  so cube training output is unchanged. The shared `Interlocked` solve-counter is gone (each generator returns its own
  count, summed on the owner thread).

**Answers the owner's question:** the parallelism isn't in Core for no principled reason — it was hand-rolled per Lab
campaign; the pattern is generic and matches Core's existing determinism approach, so it should be (and now will be)
extracted. **Non-goals:** GPU/batched-MCTS (separate, larger effort); parallelizing the DQN campaigns (not the bottleneck).

## M42 — Convolutional residual net for chess (reusable in Core)  *(2026-07-13; see `RESIDUAL_CONV_NET_PRD.md`)* — M42.1 + M42.2 ✅ SHIPPED (merged to master, PR #31) · M42.3 🟡 partial (draw-collapse pathology diagnosed + fixed, commit `282c665`, but strength gains UNPROVEN — tiers don't beat baseline on a fair eval + metrics saturate; needs non-saturating eval + scale) · M42.4 🟡 steps 1+3 done (`c1c7d8e`), browser wiring (2+4) remains · M42.5 🟡 scale-readiness (batched leaf inference + `--gpu` + non-saturating strength eval + `SelfPlayOptions` de-ceiling refactor shipped; resident-conv-forward, distribution, WDL/history deferred). No conv tier shippable yet.

**Why:** chess self-play has **plateaued at ~random** (M40.4: winRate-vs-random ~50%→35%, material margin flat ~+0.1
of the +0.75 gate, no tier ever promotes) despite 256 sims + material-shaped targets. The honest bottleneck is the
**model**: a flat `[256,256]` MLP over a 1152-float vector throws away the 8×8 board structure. Owner's decision
(2026-07-13): a true **AlphaZero-style convolutional residual tower** over `[18,8,8]`. Stacked after M41 (which makes
the heavier training iterations affordable).

**Finding (2-agent investigation):** a residual net *class* (`ResidualMlp`) is **already in Core** but is the wrong
shape (single scalar head, residual **MLP** not conv, used only by cube DAVI) — so "move it to the library" is a no-op;
it's already there. The reusable two-headed net (`PolicyValueNet`) is a flat MLP. **No convolution exists anywhere**
(no Conv2D/im2col/pool/BatchNorm; LayerNorm does exist and is the repo's deliberate choice). `SelfPlayCampaign`/
`PolicyValueTraining` hardcode the concrete `PolicyValueNet` type — there's **no two-headed-net interface** — so a conv
net needs one introduced. The obs already reshapes cleanly to `[18,8,8]` (`plane*64+sq`). And the **browser-inference
twin** (`chess_solver.pg` `PgPolicyValueNet`) is a flat-MLP forward — a conv net breaks client-side play until the `.pg`
gains a conv forward (inference-only), so that's a first-class phase, not a follow-up.

- **M42.1 ✅ — Conv2D in Core.** `Tensor.Conv2D` via **im2col → existing GEMM → col2im** (reuses the tuned, GPU-routed
  GEMM; NO new backend/ILGPU kernel), rank-2 `[N, C·H·W]` representation so `CheckRank2` is never tripped, LayerNorm (not
  BatchNorm). **Gate MET:** three finite-difference gradient checks (3×3 SAME, stride-2 valid, 1×1) green
  (`GradCheckTests.Conv2D_*`). Commit `67806af`.
- **M42.2 ✅ — `IPolicyValueNet` + the conv net.** Interface introduced; `PolicyValueNet` implements it (self-play
  determinism gate still byte-identical ⇒ zero MLP behaviour change). `ConvResidualPolicyValueNet` (3×3 stem → N residual
  blocks → 1×1-conv policy + value heads; Save/Load/LayerActivations). `SelfPlayCampaign`/`PolicyValueTraining`
  generalized to the interface + an `IPolicyValueNetBuilder` (MlpNetBuilder default / ConvNetBuilder); `ChessLab --arch
  conv --filters --blocks`. **Gate MET** (`ConvResidualNetTests`): head shapes, exact save/load round-trip, loss falls
  under Adam. (De-risking residual-MLP step skipped — went straight to conv.) Commit `21b779d`.
- **M42.3 ⏳ — train chess with the conv net.** `--arch conv --filters 64 --blocks 6` + material shaping + ladder,
  running in the background (branch `m42-chess-conv-net`). **Gate:** beats the MLP baseline — material margin ≥ +0.75
  and/or winRate ≫ 50% with **≥1 ladder tier promoted** (on merit, not the automatic Level-1 baseline); determinism
  preserved. *(Long run; evaluated from the background training, not blocking.)*
  - **Perf fix (commit `71fe44c`) that unblocked useful throughput — parallelize eval + ladder arena.** The conv net's
    per-node cost exposed a bottleneck M41's analysis missed: not self-play generation but the **measurement** phase.
    `ArenaVsRandom` + `ArenaVsNet` ran **sequentially on the owner thread between chunks**, so at conv cost they *stalled
    training* (observed ~0.8 cores for ~24 min/cycle, one eval every ~30 min). Refactored both onto the same
    `DeterministicParallel` primitive (per-game RNG, inference-only). Trained weights untouched — DOP-invariance
    checkpoint test still **bitwise-identical**; all `SelfPlayCampaign` tests green. Also added **`--max-plies`** (a
    chunk's wall time is bounded by its slowest game, so the ply cap sets throughput). Tuned run:
    `--sims 64 --games 16 --max-plies 100 --eval-games 8 --arena-games 12` → ~4–5 min/chunk (was 20–30).
  - **Root-cause fix — DRAW-COLLAPSE (commit `282c665`), core goal MET.** With the arena noise removed (`--arena-games
    40`) the trustworthy signal showed the net *regressing* (material vs baseline −2→−9 pawns, value loss →0.03, winRate
    pinned 50%). Cause: a non-mating net + short ply cap ⇒ nearly all self-play games are ply-capped **draws (z=0)** ⇒
    the outcome signal vanishes ⇒ net collapses to passive, material-bleeding play. (My throughput tuning — low sims +
    short plies — *caused* it.) Fix: **material-adjudicate ply-capped games** (`GameResult.Ongoing` at cap + ≥1.5-pawn
    edge → win/loss z, else true draw; no-op for materialless games → Connect-4 determinism stays bitwise-green).
    **Result: collapse broken** — same config went −9 → **+3.78 pawns** in self-play, and the material regression
    stopped. Merit tiers promoted — but on **8-game-noisy** evals + the self-play **material** metric.
  - **Honest limit — strength gains UNPROVEN.** A fair **40-game winRate-vs-random** ranking of the captured tiers
    (`data/tier-ranking.txt`) came out ~50–59% and **did not beat the barely-trained baseline** (L1 58.8% ≥ L3 52.5%),
    so the conv net at 64f/64-sim after ~100–200 games doesn't demonstrably play stronger than its baseline. Caveats:
    winRate-vs-random **saturates** (any net that draws-but-can't-mate random ≈ 50%) so it can't cleanly rank them
    either; and the draw-collapse fix is genuinely real. **The real gap: no non-saturating strength metric.** Next:
    (a) add a non-saturating eval (vs a simple material-greedy / depth-2 minimax opponent) to *measure* strength;
    (b) then scale training volume (AlphaZero needs ≫200 games), net capacity, and sims (128 didn't help → capacity/
    volume is the limit). Overnight detail: `data/chess-conv-autorun-log.md`. **No conv tier shippable yet.**
- **M42.4 🟡 steps 1+3 DONE (commit `c1c7d8e`); steps 2+4 (browser wiring) remain.** The conv forward is single-sourced
  in `chess_solver.pg` (`PgConvNet`) with a C# parity test (`ChessNetParityTests`) green on real conv `.ckpt` bytes
  (<2e-3); dispatch via a nullable `PgPolicyValueNet.conv` field (no `.pg` interface feature — filed
  MintPlayer.Polyglot#29). **Remaining:** TS conv parser in `chess-net.ts` + regen `chess_solver.ts` (CLI at
  `C:\Repos\MintPlayer.Polyglot`) + wire `loadChessNet`/`chess-director.ts` + copy the chosen conv tier into
  `wwwroot/models`. Best done **interactively** (regenerating the committed `.ts` blind risks the live MLP `/chess`).
- **M42.5 🟡 scale-readiness (repo value = *prove the SDK can train a chess AI*, not our weak net).** Shipped:
  **batched leaf inference** (`Mcts.SearchBatched`, virtual loss → a wave of leaves per `net.Forward`; `--leaf-batch`;
  `leafBatch=1` bitwise-identical to sequential, proven) + **`--gpu`** wiring (commit `4801c98`); non-saturating
  **`--vs-minimax`** strength eval (`7526bd7`); **`--value-weight`** (`71366dd`); and the **`SelfPlayOptions`
  de-ceiling refactor** (`8c382ae`) — one options record replacing the ~20-param telescoping ctor, exposing
  `--window`/`--batch`/`--epochs`/`--clip`/`--temp-moves`/`--cpuct`/`--dirichlet-alpha`/`--root-noise` (defaults
  bitwise-identical). **Deferred (postponed, only pays off on a real GPU/cluster run — see PRD §8):** (a) a
  **GPU-resident batched forward for the conv net** (conv analogue of `Ilgpu/DeviceMlp`) — the one piece that unifies
  the chess-MCTS and cube-value GPU paths; (b) distributed actor→learner topology; (c) quality features (WDL head,
  aux targets, history planes).

**Non-goals:** removing `PolicyValueNet` (stays the connect-4/cube-policy/rush-hour net + fast baseline) or `ResidualMlp`
(stays the cube DAVI value net); spatial BatchNorm (reuse LayerNorm); Net2Net growth for the conv net (stays MLP-only).

## M43 — GPU-resident batched forward for the conv net  *(2026-07-14; see `GPU_RESIDENT_CONV_PRD.md`)* — ✅ BUILT + GPU-measured (M43.1 `852cf31` / M43.2 `b49a4c2` / M43.3 `f39cf2f`); resident **14.9× faster** than autograd on an RTX 3060 (leaf-batch 256), on-GPU parity ~1e-6

**Why:** `--gpu` + `--leaf-batch` (M42.5) batch the leaves, but the conv forward still routes through `Backend.Current`
(weights re-upload per GEMM; activations round-trip host↔device between the tower's ~14 convs). The cube's value net
already runs **GPU-resident** (`DeviceMlp`/`DeviceResidualMlp`). This is the piece that makes the chess GPU path as
efficient as the cube's and **unifies the two families' GPU inference** (ARCHITECTURE §4). Repo value = *prove the SDK
can train a chess AI (GPU and all)* — so it lives in the **library**, not the lab.

**Finding (3-agent analysis):** the resident pattern is **Core seam → Ilgpu impl → Lab wiring**; `DeviceResidualMlp` is a
near-exact scaffold; **only two new GPU kernels** are needed (device im2col + scatter/bias) — the tower/heads reuse
existing resident kernels, and the net's whole-row LayerNorm maps onto `LaunchLayerNorm` as-is. The existing
`ITargetForward` is scalar; the conv net is two-headed → a **new** two-headed seam. Everything generic
(`ConvResidualPolicyValueNet`, `Mcts`, `Conv2D`, `IPolicyValueNet`) is already in Core, so the resident path belongs in
the library; only CLI/DI selection is lab-specific. Inference-only (training stays autograd); no new determinism loss
beyond what `--gpu` already accepts; testable on ILGPU's CPU accelerator (no discrete GPU needed).

- **M43.1 ✅ (`852cf31`) — Core seam.** `IPolicyValueForward` (two-headed, inference-only, weight-sync lifecycle) + a
  bitwise-identical `AutogradPolicyValueForward` CPU default; conv-net shape exposed; `EvaluateBatch` routed through it.
- **M43.2 ✅ (`b49a4c2`) — Ilgpu impl + kernels.** `DeviceConvPolicyValueNet` + the two new kernels (`Im2Col_Kernel`,
  `ScatterBias_Kernel`) + `CreateResidentForward(ConvResidualPolicyValueNet)`. **Gate MET:** parity vs
  `ConvResidualPolicyValueNet.Forward` on the ILGPU CPU accelerator within f32 tol (`IlgpuBackendTests`).
- **M43.3 ✅ (`f39cf2f`) — Lab wiring.** Core-typed `forwardFactory` (campaign stays Ilgpu-free); `ChessLab` supplies the
  GPU-aware factory + per-chunk `OnWeightsSynced`; `--gpu` safe on GPU-less machines (autograd fallback). **On-GPU
  measured** (`--bench-forward`, RTX 3060): resident **14.9×** faster than autograd (109.9 vs 1634.7 ms/forward,
  leaf-batch 256; 2,329 vs 157 leaves/s), parity ~1e-6.

**Non-goals:** WDL/categorical value head; distributed actor→learner; browser conv perf (still deferred,
RESIDUAL_CONV_NET_PRD §8). The resident conv *trainer* is now designed as **M44** (below).

## M44 — GPU-resident training step for the conv net  *(2026-07-14; see `GPU_RESIDENT_CONV_TRAINER_PRD.md`)* — ✅ SHIPPED (M44.1 measured→GO, M44.2 seam, M44.3 resident trainer; ~24× train step on RTX 3060)

**Why:** with `--gpu`, M43 made self-play *inference* resident (~15×), but the **training step** still runs host-span
(weights re-upload per GEMM, CPU im2col/col2im). The cube already has the training-side answer (`DeviceResidualTrainer`
/ `IResidentTrainStep`); this is its two-headed conv analogue. **But measure first** — a self-play *chunk* is dominated
by *generation* (MCTS to the ply-cap straggler), not the owner-thread train step, so the resident trainer may buy little.

**Finding (3-agent analysis):** the cube trainer transfers mostly unchanged (`Param{W,G,M,V}`, backward GEMM-transposes,
on-device clip+Adam, `SyncToHost`, `BuildStack` wiring). The scalar `IResidentTrainStep` doesn't fit our two-headed
CE+MSE loss → a new two-headed `IPolicyValueTrainStep` seam (the training dual of `IPolicyValueForward`). Only **4 new
kernels**: `Col2Im` + `GatherNCHWToMOutC` (the transpose of M43's forward im2col/scatter) + `PolicyCeGrad` (softmax−π)/B
+ `ValueTanhMseGrad`; everything else (GEMM transposes, bias/ReLU/LayerNorm grads, clip, Adam) already exists, plus one
forward-caching change (`LaunchLayerNormTrain` + x̂/1σ caches — no new kernel). Generic → library; only CLI/factory in the lab.
Determinism: a resident *trainer* mutates weights (non-bitwise) → **opt-in**; the CPU autograd path stays the reference.

- **M44.1 ✅ MEASURED → GO.** Instrumented `TrainChunk` (gen-vs-train split behind env `CHESS_CHUNK_TIMING`, off by
  default). On an **RTX 3060** (`--gpu --arch conv --parallel --leaf-batch 128 --games 6 --sims 48 --max-plies 60`),
  train share rose **36.5 → 47.8 → 63.5 %** as the window filled (360 → 720 → 1080), gen ~**constant** at ~15 s. Each
  128-batch costs **~3 s** host-span; batches/chunk = `epochs·⌊window/batch⌋`, so at the **default 40 k window** the
  train step is **~98 %** of chunk wall-time. Generation was never the bottleneck (M43's resident forward already owns
  it). **BUILD M44.3.** (Caveat: the split is config-dependent — a tiny-window/huge-chunk run is gen-bound — but every
  serious/cluster run has a large window and pays the ~3 s/batch host-span cost M44.3 removes.)
- **M44.2 ✅ SHIPPED — Core seam + wiring (behaviour-preserving).** `Core/Nn/IPolicyValueTrainStep.cs` +
  `AutogradPolicyValueTrainStep` (inlines the former `PolicyValueTraining.TrainStep` verbatim); `SelfPlayCampaign` takes
  an optional Core-typed `Func<IPolicyValueNet, Adam, IPolicyValueTrainStep>` factory (null → autograd), routes the batch
  loop through `_trainStep.Step`, `SyncToHost` before the forward re-sync; the duplicate Lab `PolicyValueTraining` deleted.
  **Gate MET:** the DOP-invariance checkpoint-hash test still passes bitwise; all 3 SelfPlayCampaign tests green.
- **M44.3 ✅ SHIPPED — Ilgpu trainer + 2 kernels.** `DeviceConvResidualTrainer : IPolicyValueTrainStep` (resident
  forward caching x̂/σ + post-ReLU + im2col cols → two-headed backward → clip → Adam) + `CreateResidentTrainer` overload;
  `ChessLab` wires it for `--gpu --arch conv`. Only **2 new device kernels** (`Col2Im`, `GatherNCHWToMOutC` — the
  transposes of M43's im2col/scatter); the softmax−π / tanh-MSE head grads are computed on the **host** (the repo keeps
  softmax/tanh off the device — CUDA can't JIT `ExpF`/`TanhF` without ILGPU.Algorithms — and the heads are tiny), so the
  planned `PolicyCeGrad`/`ValueTanhMseGrad` kernels were dropped. **Gate MET:** gradient-parity vs autograd + SyncToHost
  round-trip (CPU accelerator) green. **On-GPU (RTX 3060):** train step ~3000 ms → **~122 ms per 128-batch (~24×)**;
  train share of a chunk 36→48→64 % → 3→5→8 %. **Adam-resume gap (P.2)** accepted (optimizer re-warms on `--gpu` resume).

**Non-goals:** resident Adam-state checkpointing (P.2 — shared cube/chess fix, later); WDL head; distribution; browser.

## M45 — Single-box multi-GPU self-play  *(2026-07-14; see `MULTI_GPU_SELFPLAY_PRD.md`)* — 🟢 M45.1 shipped (library: enumerate GPUs, `.Gpus`); M45.2/3 not built

**Why:** `--gpu` uses one GPU — `IlgpuBackend.SelectDevice` enumerates all devices but takes `.FirstOrDefault()` of the
CUDA ones (`IlgpuBackend.cs:191`). A multi-GPU box idles all but one. Since a chunk is generation-bound (M44.1), the win
is to run self-play **generation on every CUDA GPU** at once. Owner's flow: enumerate all CUDA GPUs → shard the dataflow
across them → CPU fallback (the CUDA↔CPU fallback already exists; it just never enumerates past the first device).

**Finding (3-agent analysis):** the single-GPU assumption is a handful of localized seams (`SelectDevice` `FirstOrDefault`;
one `Context`/`Accelerator`/`DeviceLock` per `IlgpuBackend`; `AdaptiveBackend` builds one; `AddSingleton<AdaptiveBackend>`;
the `Backend.Current` global; one `_forward`/`_trainStep`). But the enablers exist: `DeterministicParallel.Generate`
already shards games by **global index** bitwise-invariantly (the clean per-GPU axis); **N backends = N independent locks**
(parallel across GPUs); the M43/M44 device seams; and the per-chunk `SyncToHost → _net → OnWeightsSynced` weight lifecycle
(→ a fan-out). One box, one process, one campaign, one local store are all **kept** — this is why single-box ≪ cluster.

- **M45.1 ✅ SHIPPED — Library: enumerate + device-addressable backend.** `SelectDevices` (all CUDA, or CPU); device-
  pinning `IlgpuBackend(Context, Device)` ctor (shared context, caller-owned); `AdaptiveBackend` builds one backend per
  CUDA device on one shared context, exposes `Gpus` (list, empty on CPU-only) — **`.Gpu` removed**; the autograd GEMM
  router keeps a *private* primary `Gpus[0]`, so M43/M44 and `--gpu` are behaviourally unchanged. All 7 `.Gpu` sites
  migrated to `Gpus.FirstOrDefault()`. **Gate MET:** 2 new tests (pinned-device GEMM parity; `Gpus` consistency) + 37
  Ilgpu/cube/self-play tests green; web + Lab build clean.
- **M45.2 — Lab: `--gpus` flag + sharded generation + weight fan-out.** One resident forward per GPU; route each game's
  leaf batch to `_forwards[i % nGpus]` (index-deterministic); training stays on the primary GPU; `OnWeightsSynced` fans
  out to all forwards per chunk. **Gate:** `--gpus 1` ≡ `--gpu`; CPU DOP-invariance SHA test green; `--gpus all` runs at N=1.
- **M45.3 — Measure (needs ≥2 GPUs).** Generation throughput vs 1 GPU; near-linear expected until the owner-thread merge
  saturates. **Cannot be measured on the single RTX 3060** — M45.1/2 ship the capability + N=1 correctness; real scaling
  is validated on multi-GPU hardware. Stated honestly.

**Out of scope:** data-parallel *training* across GPUs (gradient all-reduce; ILGPU has no collectives, and training isn't
the bottleneck after M44); cross-machine distributed / actor-learner (network transport, cross-process replay buffer,
coordinator, fault tolerance — the model store is local-FS only; a separate systems project). The seams M45 builds
(per-device resident forwards, index-sharded generation, weight fan-out) are the ones such a harness would reuse.

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
| M13 cube port | e2e scramble → Kociemba solve → playback solved; ≤ 22 moves | passed (20-move scramble solved in 21 moves / 116 ms; playback ends solved; gallery replays) |
| M14 cube RL | ≥ 90% of depth-1–6 scrambles solved within 20 moves | **100%** (600/600; greedy alone 77.8%, rest via Q-guided lookahead) |
| M15 classic 2048 | classic animations live; replay reconstruction == `FinalCells` | passed (2,491-move replay bit-exact; slide/pop/appear verified in-browser) |
| M16 cube imitation | ≥ 90% of depth-1–10 scrambles within 40 quarter-turns | **96%** (96/100; greedy alone 54%) after a 2 h resumable campaign |
| M17 wider net (1024, rung 1) | beat the 512 net on the same gate seeds | **96/100, greedy 69%** (vs 512's 97/100, greedy 64%) at *half* the samples — modest greedy gain, accuracy plateaued ~79% like 512: width is diminishing, the algorithm is the lever |
| M12a CPU GEMM scaling | ~linear to core count; bitwise-identical | GEMM **3.95×** on 8 cores; full step 2.52× (Amdahl); byte-identical dop-1-vs-8 |
| M12c/perf device-resident forward | match autograd forward; speed up DAVI | matches within tol; **~2× DAVI throughput** (500 iters 20s vs 40s) |
| M18 DAVI (teacher-free) | learn to solve shallow cubes with no oracle | greedy-optimal under exact value; **≥80%** of depth ≤3 teacher-free; campaign reached **curriculum depth 9** (greedy d7 70%, d9 30%) — stall-fallback curriculum unstuck the 0.95 gate; plateau ⇒ need residual net + GPU port (M19–M21) |
| M19 tiled GEMM (bottleneck #1) | correctness vs ManagedBackend; naive→tiled GFLOP/s table | 212/212 green incl. exact-tile/k-tail/rectangular; on RTX 3060 (resident operands, transfer excluded): 256³-ish **562→669** (1.2×), 1024³ **444→626** (1.4×), **2048³ 268→620 GFLOP/s (2.3×)** — gain grows with size; adaptive tile (16 on GPU, ≤cores on CPU accel). Honest: short of the 5–10× estimate — shared-memory tiling only; **register-blocking is the next lever** toward multi-TFLOP (M19b) |
| M20 Stage 1 resident weights (bottleneck #2) | DeviceMlp matches autograd; weights upload per-sync not per-step | DeviceMlp + `ITargetForward` green (forward matches within tol; OnTargetSynced re-uploads); weight transfer dropped from per-step to per-target-sync (~200×); wired into `cube-davi` |
| M20 Stage 2 resident residual fwd (2026-06-14) | DeviceResidualMlp matches autograd | GPU LayerNorm + add kernels; residual successor-eval resident → **~2× iters/s** (4.5 vs 2.3, residual 1024×4) |
| M20 Stage 3 resident training (2026-06-14) | gradients match autograd; full step on-device | `DeviceResidualTrainer` (fwd-cache + backward + clip + on-device Adam), `IResidentTrainStep` seam; gradient parity verified; **~11 iters/s (4.8× host-span)** |
| M19b register-blocked GEMM (2026-06-14) | correctness vs ManagedBackend; faster than tiled | 4×4 micro-tile in explicit registers; **2.2× @256-class, 3.2× @1024³/2048³ → ~2.0 TFLOP/s** (7.4× naive @2048³); production path routes through it |
| Learning-curve levers P.7/P.8/P.9 (2026-06-14) | feed GPU, decouple pacing, per-update efficiency | parallel successor gen (GPU 0→95-100% util) + `--batch`; sample-paced curriculum + lighter eval; ε-loss target sync (freeze-proof) + `--lr`. Net: **~3,050 samples/s (~2× prior)**, GPU-bound at ~2 TFLOP/s |
| General `IComputeBackend` port Phase 1 (2026-06-14) | every autograd op behind the seam; CPU bitwise-identical | all ops (Map/Zip/reductions/LogSoftmax/Gather/Huber/LayerNorm) routed through the backend; 224/224 green. Phase 2 (device-backed Tensor) parked far-future (no measured win at our scale) |
| **M21 shortest-move solver — BWAS capability (2026-06-14, residual 1024×4, ~44k iters)** | **provably-optimal shallow; beat Kociemba's QTM mid-range** | batched A* (w=2.5, ≤40k exp, 12/depth): **12/12 & QTM-optimal through depth 10** (exactly d·qt), **100% solved through depth 12**, 83% d13, 75% d14; **every solve beats Kociemba's QTM, often ~2×** (d10 10 vs 19, d12 12.2 vs 29.3). Greedy (live curve) collapses ~d10-11 — search reads the net far deeper. Beats the earlier 1024×3 MLP at every deep level (d12 100% vs 80%). Full god's-number (26 QTM) NOT reached — honest two-tier story |
| M28 2048 expectimax (2026-06-26) | beat the shipped n-tuple greedy on score, serving-viable | **~84k vs ~44k (1.9×)**, best tile 4096→8192 over 100 games, *no retrain*; ~1.2 s/playout at depth 1 |
| M28 NoisyNets capability (2026-06-26) | learns, serializes, serves deterministically; no shipped checkpoint broken | 293/293; σ-grads flow; noisy resume bitwise; v1 ckpts load as plain; serving unchanged (noise off) |
| M28 FruitCake NoisyNets empirical (2026-06-26) | match or beat ε-greedy at equal budget (multi-seed) | **matched** — 200-game paired A/B tie (702.1 vs 714.4, Δ −12.3 ± 29.8 SE); single-evals were seed-luck; not shipped |
| M36.1 network visualizer — watch it train (2026-07-12) | see the net evolve live during training (all games); beginner-readable; zero training impact | **met** — pull-based seam; **all six `--game`s** stream topology + weight frames over a **WebSocket** to a self-contained page with **hover tooltips**; net visibly evolves (heatmaps/edges shift, eval 7.0→13.9); **Development-gated**; viz vs no-viz checkpoints **SHA256-identical**; 314 tests green |
| M37 progressive net growth (2026-07-12) | grow the net wider+deeper mid-training without a loss spike, everywhere possible | **met** — shared `Net2Net` (WidenTrunk/SetIdentity); `--grow` grows **all** trainable nets live: `DuelingQNet` (Snake, FruitCake) and the refactored variable-depth `PolicyValueNet` (Cube, Cube-policy, Rush Hour), `[16]`→`[128,128,128]`; DAVI `ResidualMlp` already grew width. Policy checkpoint → v2 with v1 back-compat (tested). 320 tests (4 new: widen/deepen forward-equality ×2, v1 load, grown round-trip) |

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

**M13–M18 are done** (2026-06-12→13). The Rubik's Cube game ships with the Kociemba button +
a gate-passing imitation AI; 2048 plays like the original. Then, with the SDK named as the
north star: the **GPU pillar (M12)** landed — multithreaded CPU GEMM, the ILGPU/CUDA backend,
the auto-routing `AdaptiveBackend`, and a scoped device-resident forward (~2× DAVI throughput),
all measured + committed. **M17** showed width is diminishing (1024 net: greedy 64%→69%, accuracy
plateaued ~79% — the *algorithm* is the lever). So **M18** added the lever: a teacher-free
**value-iteration (DAVI) trainer** + value-guided A* + a GPU campaign that learns to solve cubes
with no Kociemba.

*The north star is the **SDK**, not any demo's score (owner, 2026-06-13). Order favors
SDK capability over squeezing a showcase.*

*Owner's focus (2026-06-13): remove the two GPU bottlenecks, toward a shortest-move
(quarter-turn-optimal) cube solver. M19+M20 are the priority; M21 is the capability they unlock.*

**CURRENT STATUS (2026-06-14): M19, M20 (Stages 1–3), M21, the general-port Phase 1, and all
learning-curve levers (P.1/P.7/P.8/P.9) are DONE** — see the gate table. The residual DAVI campaign runs
fully device-resident at **~3,050 samples/s** (GPU-bound at ~2 TFLOP/s via the register-blocked GEMM),
curriculum at the depth-20 cap (campaign stopped at 236k iters). **Measured solver capability (heavy BWAS,
w=1.5, ≤100k exp): QTM-OPTIMAL through depth 15 (12/12, solution length = scramble depth), d16 10/12,
d17 5/12, beating Kociemba's QTM ~2–2.5×.** A 2026-06-14 heavy-search diagnostic proved the apparent
plateau was a **search-budget artifact** — the greedy live eval (~d10) and the light in-loop probe (8k exp,
~d14-partial) both grossly understated the net; the net heuristic is accurate to ~d15 and degrades
*gradually* past it. **The net is not the bottleneck through d15 — eval-time search is. → The next step is
P.10 (tune BWAS expansions / weight / frontier), not a wider net** (deferred: `OPTIMIZATIONS.md` F.2).
**What remains:** P.10 eval-time search lever; heavier in-loop eval so the live curve reflects real capability;
resident Adam-state checkpointing
(P.2, lossless resume); deeper-than-15 reach (more capacity/training; full god's-number 26 QTM out of reach
on one 3060).

**Further-training findings (2026-06-15).** Resuming the campaign from the depth-20 / 236k checkpoint and
letting the curriculum climb to its depth-26 cap (force-advanced past the greedy-stall gate) over ~80k more
iters left the deep frontier **flat**: the light in-loop BWAS probe held d15/d16 within sampling noise and
the training loss never descended below ~0.10 (DAVI bootstraps, so flat loss ≠ failure — but the capability
curve confirmed no real movement). This is **not** an architecture ceiling: the 1024×4 residual net already
matches DeepCubeA's class (M17 settled that width is not the lever). It is a **training-scale gap** — the net
has seen ~36.5M states (≈285k iters × batch 128) vs DeepCubeA's **~10 billion** (≈0.4%). At the observed
~1.5–3k states/s on one RTX 3060 Laptop (≈130–260M states/day), reaching 10B is ~5–11 **weeks** — infeasible.
So deeper-cube capability has **two complementary levers, neither a wider net**: (a) **eval-time search budget**
(P.10 — already proven to turn the strong ≤d15 heuristic into deeper solves; free, no training), and
(b) **longer training** for genuinely deeper *heuristic* reach (bounded by laptop wall-clock — several days
buys ~10–20× more states, pushing the optimal frontier somewhat past d15, **not** to god's number).
**Decision (user opted into multi-day training, 2026-06-15):** run a multi-day DAVI campaign from the current
depth-26 checkpoint (uniform 1–26 scramble sampling — finally the regime that builds deep states; the prior
run had only just reached depth 26), monitored with periodic *heavy* BWAS probes (the light in-loop probe is
blind past its 8k budget), and combined with the P.10 search-budget lever. Full analysis: `OPTIMIZATIONS.md`.

**Result (2026-06-16).** Overnight run **313k → 615k iters (+38.7M states, ~doubling total training to ~79M)**,
lr 1e-3 throughout (heavy-probe A/B confirmed it was still learning, so no LR decay). Heavy-BWAS deep-solve
rate (w2.5, 200k exp, same cubes, 5/depth, d14–22) rose **21/25 (313k) → 24/25 (415k) → 24/25 (615k)** — the
jump was the first half; the second half consolidated (d22 **3/5 → 5/5**, d16 4/5 → 5/5; d18/d20 swap within
n=5 noise). Gains are in **reach/robustness**, not optimality (solution lengths held/crept up slightly at
weight 2.5; every solve still ~19–24 qt vs Kociemba's ~29–31). Confirms the call: incremental sharpening, not
a leap to "any cube" — that needs DeepCubeA-scale (~10B) compute. The 615k net is promoted to
`models/cube.value-davi-res.ckpt` (LFS) so the web "Solve (self-taught AI)" ships it.

**Learning-curve findings (measured 2026-06-14).** The campaign is **GPU-bound** after the resident
stages. **Batch size is throughput-neutral** (+16% samples/s 128→512 — it only moves the bottleneck
CPU→GPU; an iter-paced curriculum then advanced ~3.4× slower, since fixed by sample-pacing); **net width is
not a lever** (M17 diminishing + quadratic GFLOP). The throughput lever was the GEMM kernel itself —
**✅ P.1 register-blocked GEMM (2.2–3.2× → ~2 TFLOP/s)** — plus the cheap **✅ P.8** (sample-paced
curriculum + lighter eval) and **✅ P.9** (LR scaling + ε-loss target sync). Full analysis: `OPTIMIZATIONS.md`.

0. **M19 — tiled GEMM kernel** (bottleneck #1, compute). ✅ **DONE 2026-06-13.** Shared-memory
   tiled ILGPU GEMM (one generic `GemmDims`-parameterized core for A·B / Aᵀ·B / A·Bᵀ + write,
   adaptive tile). Measured **1.2–2.3× the naive kernel** (up to 620 GFLOP/s resident, gain grows
   with size) — honest shortfall vs the 5–10× estimate: tiling-only, no register-blocking. **M19b
   (register-blocked micro-tiles + vectorized loads)** is the open lever toward multi-TFLOP.
1. **M20 — device-resident tensors** (bottleneck #2, transfer), staged: Stage 1 resident-weight
   inference (`DeviceMlp` + `ITargetForward`) ✅ **DONE 2026-06-13** (weights upload per-target-sync,
   not per-step — wired into `cube-davi`) → Stage 2 device-resident
   training → Stage 3 full `IComputeBackend` device-handle port. Spec'd above.
2. **M21 — shortest-move solver** (the capability): residual value net + deepening curriculum +
   batched weighted A*, two-tier optimality gate (provably-optimal to depth ~7, beats Kociemba
   to ~15–20). Depends on M19+M20. Then wire `value-davi` into the web page as the third solver.
3. **AlphaZero-style fine-tune** — *started 2026-06-11, paused mid-campaign; see the
   "Fine-tune round" section under M11 for results so far and the resume command.*
3. **SDK breadth** (the demos exist to show range): algorithm coverage (✅ PPO action
   masking *(done)*, ✅ dueling head *(done)*, maybe SAC), more envs (**MountainCar + Snake — designed, see M22**),
   a TensorBoard writer, public-API stability/semver discipline, reproducibility guarantees.

   **"Switch algorithm, keep the work" — make the three reusable assets first-class.**
   When an algorithm plateaus, what carries over is the env (already portable via
   `IEnvironment`), the data, and the learned representation. Today the data/weights
   transfer is ad-hoc; these three features make it a clean SDK capability:
   - **Algorithm-agnostic transition store** ✅ *(done 2026-06-13)* — `ReplayBufferCheckpoint`
     serializes a replay buffer of `(s, a, r, s′, terminated, next-mask)` to a
     self-describing file, factored out of `DqnTrainingState` (which now delegates to it,
     byte-identically — shipped DQN checkpoints still load). Off-policy → off-policy
     switches (DQN ↔ SAC ↔ DDPG) can now reload the same buffer across a restart, not just
     within one process. (On-policy methods still need fresh rollouts; such a buffer feeds
     them only as *demonstrations*, not drop-in data.)
   - **Trunk/head-separated checkpoints** — a format that lets you load a trained trunk
     and reinitialize heads, so transferring a feature extractor across algorithms (whose
     heads differ in meaning — Q vs policy-logits vs V) is a one-liner, not manual surgery.
   - **Demonstration-dataset abstraction** — a uniform way for any algorithm to be seeded
     from oracle/expert data (DQfD-style), so the Kociemba/BFS oracles become reusable
     warm-start sources, not per-demo glue.
   - **Function-preserving net transfer** ✅ *(done 2026-06-30 — see `NET_TRANSFER_PRD.md`)* —
     `IValueNet.GrowInput(n)` grows a trained net's input dimension when an env's observation
     gains features (new in-weights zero-init ⇒ identical output on the old features), so
     enriching an observation warm-continues instead of retraining from scratch. Unifies the
     generic transfer mechanics in `NetTransfer` (collapsing the duplicated `CopyFrom`), with
     the structure-specific transforms staying on their nets (`ResidualMlp.WidenTo`,
     `DuelingQNet.ToNoisy`, `CubePolicyNet.PolicyAsMlp`). Follow-up: trainer auto-grow on a
     wider-obs resume (T4).
4. **Slide-optimal AI answers**: search/compaction that minimizes official piece-moves,
   not just single-cell moves.
5. **Stretch list (M11), once everything above is complete** — extras, not on the
   critical path: MountainCar, Snake, TorchSharp backend, TensorBoard writer, self-play
   scaffolding, Dueling head, tensor pooling, PPO masking, importing puzzles from the
   owner's original Rush Hour app.

Run the playground: `dotnet run --project src/RLDemo.Web` (Development spawns + proxies
the Angular dev server itself — do not run `ng serve`). Console demos:
`dotnet run --project src/RLDemo.Console -c Release -- [grid|lake|cartpole|ppo|2048|2048dqn|rushhour|cube]
[seed] [--load] [--save] [--data <dir>]`. Tests: `dotnet test` (`Category=Slow` for gates).
Training campaigns (resume net + Adam + full training state from the model store):
`dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release --
[--game rushhour|cube|cube-davi] --hours N --data src/RLDemo.Web/data` — `--game cube`
(Kociemba imitation) and `cube-davi` (teacher-free value iteration) take `--eval-only` for
the gate report; `cube-davi` also takes `--width`, `--layers` and `--max-depth`, runs on the
`AdaptiveBackend` (GPU device-resident forward), and logs `models/logs/cube-davi.csv`. Use
`--data models` to refresh the shipped seeds.
