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

**M13–M16 are done** (2026-06-12): the Rubik's Cube game is in the playground with both
the Kociemba button and a gate-passing AI (imitation policy net + lookahead, DQN
fallback), and 2048 plays like the original again. Next candidates, in suggested order:

0. **Deeper cube AI**: resume `Lab --game cube` (checkpoints in `models/`) — the MLP is
   plateauing at ~73.5% action accuracy, so the bigger wins are a wider trunk and/or a
   DAgger-style on-policy mix (relabel the states the net actually visits — the M11
   lesson), and a weighted-A* variant to cut search latency at depth 10+.

1. **AlphaZero-style fine-tune** — *started 2026-06-11, paused mid-campaign; see the
   "Fine-tune round" section under M11 for results so far and the resume command.*
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
`dotnet run --project src/RLDemo.Console -c Release -- [grid|lake|cartpole|ppo|2048|2048dqn|rushhour|cube]
[seed] [--load] [--save] [--data <dir>]`. Tests: `dotnet test` (`Category=Slow` for gates).
Training campaigns (both resume net + Adam from the model store):
`dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release --
[--game rushhour|cube] --hours N --data src/RLDemo.Web/data` — `--game cube` also takes
`--eval-only` for the per-depth gate report; use `--data models` to refresh the shipped seeds.
