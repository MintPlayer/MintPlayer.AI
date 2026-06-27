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
