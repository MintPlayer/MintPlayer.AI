# MintPlayer.AI.ReinforcementLearning — Product Requirements Document

**Status:** shipped · 2026-06-11 — all M0–M10 gates passed; published to nuget.org
(`MintPlayer.AI.ReinforcementLearning.Core` / `.Environments` 0.1.0) and **live at
https://ai.mintplayer.com**. M11 (imitation learning + policy-guided search) underway —
every official Rush Hour card tested is solved optimally. See [PLAN.md](PLAN.md) for
per-milestone results.
**Owner:** Pieterjan
**Repo:** https://github.com/MintPlayer/MintPlayer.AI (net10.0, blank solution at start)

## 1. Vision

A reinforcement-learning library written **from scratch in C#/.NET**, because .NET has no
serious, stable, dependency-light RL story:

- **ML.NET** has no RL support at all (feature request open since 2018, not on the roadmap);
  Microsoft's only RL products (Azure Personalizer, Project Bonsai) are retired/retiring.
- **Gym.NET / SciSharp** is dormant (last real commit April 2024, ports the deprecated
  pre-2022 Gym API; TensorFlow.NET is explicitly unmaintained).
- **RLMatrix** is the only living library, but it is single-maintainer, has churned its
  license (MIT → proprietary dual → MIT, NuGet metadata still dual-license), and hard-depends
  on TorchSharp's multi-hundred-MB native payload (CUDA-Windows ~3 GB).
- There is **no Gymnasium-equivalent environment standard for .NET** at all.

MintPlayer.AI.ReinforcementLearning fills that gap as an *educational-but-usable* library: every component (tensor math,
autograd, replay buffer, algorithms) implemented in readable managed C#, verified against
literature benchmarks, with an architecture that scales up (pluggable compute backend) and
out (vectorized environments) without rewrites.

## 2. Goals

1. **Quick demo first** — a watchable console demo (agent visibly learning) within the first
   milestone, on a framework whose interfaces don't need to change as algorithms get bigger.
2. **Gymnasium-faithful environment API** for .NET — typed `Reset`/`Step` with separate
   `Terminated`/`Truncated`, spaces, seeding — so results are comparable to literature.
3. **Algorithm ladder**: tabular Q-learning/SARSA → REINFORCE → DQN (+Double/Dueling) → PPO
   (vectorized envs, GAE). Each gated by a known-solved threshold before the next starts.
4. **From-scratch numerics**: own tensor type, SIMD matmul, tape-based autograd, Adam —
   zero native dependencies in the core.
5. **Real games as benchmarks**: CartPole (literature-comparable flagship), **2048** and
   **Rush Hour** (the owner's own games) as showcase environments.
6. **Reproducibility as a feature**: fixed RNG implementation, master-seed fan-out,
   deterministic single-threaded mode, seeded learning-curve regression tests.
7. **Interactive web playground** *(added 2026-06-10, after the v1 gates passed)*: an
   ASP.NET Core + Angular web app where anyone can draw a game state, play it, submit it
   to a trained model, and step through the AI's solution — with persisted training
   results and Docker deployment. Spec in [§7](#7-interactive-web-app-mintplayerai-playground).

## 3. Non-goals (v1)

Explicitly out of scope to prevent the scope creep every research thread warned about:

- GPU execution, CNNs / Atari-scale nets (the `IComputeBackend` seam exists, nothing
  more) — *now fully planned as M12, see [§10](#10-planned-gpucuda-backend-m12)*
- TorchSharp backend (optional *later* package, never a core dependency)
- Multi-agent / self-play frameworks (TicTacToe/Connect-4 deferred until single-agent API is stable)
- Distributed training, ONNX export, model-based / offline RL
- Unity/Godot adapters, netstandard multi-targeting
  *(NuGet publishing was originally out of scope here but shipped on 2026-06-11)*
- LunarLander or anything needing a physics engine (Box2D port ≈ a project in itself)

## 4. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Compute backend v1 | **Pure managed**: `float[]`/`Span<float>` + `TensorPrimitives` (stable in .NET 10) + hand-rolled tiled SIMD GEMM (BCL ships no matmul) | Classic-control nets (4→64→64→2 ≈ 1.8M FLOPs/batch-64 step) run at thousands of Adam steps/sec on one core; SB3 itself recommends CPU for MLP policies. ONNX Runtime training is deprecated; TorchSharp is a ~GB-scale native dependency that buys nothing at this size. |
| Differentiation | **Tape-based reverse-mode autograd at tensor granularity** (~15 ops), verified by finite-difference gradient checks; hand-derived fused fast paths (Linear, softmax+CE) allowed later as optimizations | Educational, composes PPO's clipped loss naturally, negligible overhead at these sizes. |
| Scale-up seam | `IComputeBackend` boundary defined before v1 ships; TorchSharp/ILGPU can implement it later | One-way door identified in research: retrofitting is a rewrite. |
| Scale-out seam | Batched forward passes from day one; `IVectorEnv` with **sequential-deterministic** mode (default, reproducible) and parallel mode (Tasks — no GIL in .NET) | Resolves the determinism-vs-parallelism conflict explicitly: determinism wins by default, parallelism is opt-in; seed tests run sequential mode. |
| Environment API | Gymnasium-faithful, **generic** `IEnvironment<TObs, TAct>`: `Reset(seed) → (obs, info)`, `Step(act) → (obs, reward, terminated, truncated, info)` | The `terminated`/`truncated` split is load-bearing (bootstrap iff NOT terminated); conflating them is the most common silent RL bug. Info channel carries `final_observation` for autoreset/vec envs. |
| Precision | `float` (Single) throughout the tensor/NN stack; `double` for tabular Q-tables and reward accumulation | float32 matches literature/PyTorch; doubles make tabular exact-value tests clean. |
| RNG | Own **xoshiro256\*\*** implementation + SplitMix64 master-seed fan-out (env / policy / init / buffer streams) | `System.Random`'s algorithm is not contractually stable across .NET versions — a reproducibility hazard. |
| Solution layout | `src/MintPlayer.AI.ReinforcementLearning.Core` (tensors, autograd, NN, spaces, agents, training), `src/MintPlayer.AI.ReinforcementLearning.Environments`, `src/RLDemo.Console` (console CLI — repurposed existing project), `tests/MintPlayer.AI.ReinforcementLearning.Tests` | Small enough to move fast, split along the natural package seams for later. Root namespace `MintPlayer.AI.ReinforcementLearning`. |
| Test framework | **xUnit**. Fast unit tests always; statistical solve-threshold tests (3 seeds, median) in an opt-in `[Trait("Category","Slow")]` bucket | RL bugs are statistical; single-seed pass/fail lies. |
| Logging | CSV per run + live console metrics. TensorBoard event files deferred (needs a protobuf writer) | Keep v1 dependency-free. |
| Checkpoints | JSON for tabular Q-tables; versioned little-endian binary for NN weights + optimizer state + RNG state (full resume) | Cross-version stability not promised in v1. |
| License | **MIT**, stated up front, never churned | RLMatrix's license instability is precisely the gap being filled. |
| Web stack | ASP.NET Core host + Angular SPA via **MintPlayer.AspNetCore.SpaServices** (`UseAngularCliServer` in dev — the host spawns and proxies the Angular dev server; built static assets in production) | Owner's standard hosting model; one process to run in dev, one container in prod. |
| Model store | One *current* checkpoint per (environment, algorithm) in a configurable data directory (env var / appsettings), using the PRD checkpoint formats (JSON tabular / versioned binary NN, n-tuple tables as versioned binary too) | Training results survive restarts; the web app never trains from scratch unless the store is empty. |
| Solve API contract | Solve endpoint returns a **trajectory** — per step: action + resulting board state (+ any stochastic event, e.g. the 2048 tile spawn) | 2048 is stochastic, so a bare move list is not replayable; returning states makes browser playback trivial and game-agnostic. |
| Deployment | Multi-stage Dockerfile (Node + .NET SDK build stage → ASP.NET runtime); volume `/data` for the model store + submitted games | Anyone can pull/run it and see persisted models and the public game gallery. |
| Reference fixtures | One-time Python/Gymnasium script generates golden trajectories (CartPole, FrozenLake) committed as test fixtures | A physics transcription error is indistinguishable from an algorithm bug. |

## 5. Public API sketch

```csharp
// Environments (Gymnasium-faithful)
public interface IEnvironment<TObs, TAct>
{
    Space<TObs> ObservationSpace { get; }
    Space<TAct> ActionSpace { get; }
    (TObs Observation, EnvInfo Info) Reset(ulong? seed = null);
    StepResult<TObs> Step(TAct action);
    void Render(IRenderer renderer);   // console renderer in v1
}
public readonly record struct StepResult<TObs>(
    TObs Observation, double Reward, bool Terminated, bool Truncated, EnvInfo Info);

// Spaces
public sealed class DiscreteSpace : Space<int>          // n actions / states
public sealed class BoxSpace : Space<float[]>           // low/high bounds, shape

// Agents
public interface IAgent<TObs, TAct>
{
    TAct Act(TObs observation, bool greedy = false);
}
// Trainers own the learning loop: TabularTrainer, DqnTrainer, PpoTrainer…
// configured by plain options records (QLearningOptions { Alpha, Gamma, EpsilonSchedule … })
```

Conventions: reward `double`; observations `float[]` for deep RL, `int` for tabular;
one master seed fans out to all RNG streams; `Render` never sits inside the training hot loop.

## 6. Environments & success criteria

Pre-registered, objective "solved" definitions (fixed now so benchmarks stay meaningful):

| Env | Spaces (obs / act) | Solved criterion | Role |
|---|---|---|---|
| **GridWorld 4×4** (deterministic, step −0.04, goal +1) | Discrete(16) / Discrete(4) | Greedy policy == value-iteration optimal policy (exact unit test) | First demo; tabular correctness oracle |
| **FrozenLake 4×4** (slippery ⅓-⅓-⅓, +1 goal, 100-step cap) | Discrete(16) / Discrete(4) | Success rate ≥ 0.70 over 100 episodes | Stochasticity test (Gymnasium-comparable) |
| **CartPole-v1** (faithful port: gravity 9.8, masscart 1.0, masspole 0.1, **length 0.5 = half-pole**, force 10.0, τ 0.02, explicit-Euler source order, terminate \|x\|>2.4 or \|θ\|>0.2095 rad, +1/step incl. terminal, truncate@500, init U(−0.05,0.05)⁴) | Box(4) / Discrete(2) | Mean return ≥ 475 over 100 consecutive episodes | Deep-RL flagship; literature-comparable |
| **2048** (4×4, spawn 90% 2 / 10% 4, invalid moves masked) | Box(16) log2-encoded / Discrete(4) | Pre-registered: reach 2048 tile in ≥ 10% of 100 eval games (DQN target); stretch: TD/n-tuple agent ≥ 80% | Owner's game #1; stochastic, showy |
| **Rush Hour** (6×6 sliding-block; action = (vehicle, ±1 slide); reward −1/step, +100 exit; per-difficulty puzzle sets) | Box(72): vehicle-identity plane + red-car plane / Discrete(32 masked) | ≥ 90% of *easy* puzzle set solved within 2× optimal moves (optimal via built-in BFS solver) | Owner's game #2; sparse-reward planning, curriculum learning |
| MountainCar-v0 (optional) | Box(2) / Discrete(3) | Mean ≥ −110 over 100 episodes | Exploration stress test (vanilla agents *legitimately* fail — that's a finding, not a bug) |

Rush Hour note: board logic is reimplemented inside `MintPlayer.AI.ReinforcementLearning.Environments` (~150 LOC) rather
than referencing `C:\Repos\Spelletjes\Rush Hour` (currently being modified by another
session); the existing app can later consume MintPlayer.AI.ReinforcementLearning for visualization, and its puzzle
definitions can be imported as data.

## 7. Interactive web app ("MintPlayer.AI Playground")

*Requirement inserted 2026-06-10, after the library v1 (M0–M6) gates passed. Delivered as
milestones M7–M10 in [PLAN.md](PLAN.md).*

A new project `src/RLDemo.Web`: an **ASP.NET Core** application hosting an **Angular**
front-end through **MintPlayer.AspNetCore.SpaServices** (the host runs and proxies the
Angular CLI dev server in development; serves the built bundle in production).

1. **One page per game/environment.** A landing page lists the environments; each game
   gets its own page. v1 of the playground covers the two drawable board games —
   **Rush Hour** and **2048**; watch-only pages for CartPole etc. can come later.
2. **Draw + play + reset.** Each game page has an HTML5-canvas board editor: the user
   draws a game state (places Rush Hour vehicles / sets 2048 tiles), can **play it
   himself** with the real environment rules, and can **reset back to the drawn state**
   at any time. The drawn state is validated (e.g. Rush Hour overlap/exit-row rules)
   before play or submission.
3. **Solve API.** A button posts the drawn game state to the backend:
   - If a trained model for that environment exists in the **model store**, the backend
     immediately runs it on the posted state and returns the solution.
   - If not, the backend first **trains** — a background job, *independent of the
     submitted state* (the model is general for the environment, not fitted per-puzzle) —
     with progress observable from the browser; the solve runs when training completes.
   - The response is a **trajectory**: per step, the action taken and the resulting board
     state (plus stochastic events such as 2048 tile spawns), with metadata — solved or
     not, move count, and for Rush Hour the BFS-optimal move count for comparison.
4. **Solution playback.** The browser animates the returned trajectory with
   **back/forward step buttons** (plus play/pause), and can always return to the user's
   drawn initial state.
5. **Persistent training results.** Trained models (n-tuple weight tables, NN weights +
   optimizer + RNG state) are checkpointed to the model store on disk, so training never
   restarts from scratch. *This promotes the checkpointing item deferred from M3 into a
   hard requirement.*
6. **Public game gallery.** Submitted game states (and the solutions returned for them)
   are persisted and listed on the site for anyone to browse and replay.
7. **Dockerized.** Multi-stage Dockerfile (Node + .NET SDK build → ASP.NET runtime
   image); a **volume** (`/data`) persists the model store and the submitted-games
   gallery across container restarts.

## 8. Performance targets (measured with BenchmarkDotNet on dev machine)

- Tabular: ≥ 500k env-steps/sec on GridWorld (training loop included)
- NN: ≥ 1,000 Adam steps/sec, batch 64, net 4→64→64→2, single thread
- DQN solves CartPole (≥475) in ≤ 5 min wall-clock, single core
- Training allocations: zero per-step gen0 allocations in steady state (pooled tensors/tape)
- An early **spike benchmark validates the managed-GEMM assumption before M2 is built out**

## 9. Risks

| Risk | Mitigation |
|---|---|
| Terminated/truncated mishandling (silent, partially-learning agents) | Dedicated unit test: truncated transition MUST bootstrap; store `terminated` only in replay buffer |
| Hand-written GEMM slow/wrong | Spike benchmark + finite-difference gradient checks from day one |
| Env physics transcription errors | Golden trajectories recorded from Python Gymnasium, committed as fixtures |
| GC pressure from tape autograd | `ArrayPool<float>`, tape reuse, in-place ops; allocation budget in perf tests |
| Statistical flakiness in CI | Median over ≥3 seeds, generous budgets, slow tests opt-in/nightly |
| Premature abstraction (the SB3/CleanRL lesson) | Single-file reference per algorithm first; abstractions extracted only after gates pass and must reproduce seeded curves |
| Rush Hour sparse reward too hard for vanilla DQN/PPO | Curriculum (easy→hard puzzle sets), shaped reward variant, BFS solver as oracle/imitation source; treat failure as a documented finding |
| Hyperparameter fragility across envs | Known-good per-env configs committed with the test suite |
| User-drawn Rush Hour puzzles are out-of-distribution for a model trained on generated sets | BFS oracle always produces a reference solution; UI reports both AI and optimal move counts; failures shown honestly as findings; harder-curriculum/imitation items raise generality later |
| Long training jobs inside a web request | Training always runs as a tracked background job with polled/streamed progress; solve requests queue behind it rather than time out |

## 10. GPU/CUDA backend (M12) — a first-class SDK capability

*Assessed 2026-06-11 against the dev machine (NVIDIA RTX 3060 Laptop, 6 GB, compute
8.6). Full phase plan in [PLAN.md](PLAN.md) §M12.*

> **Reframed 2026-06-13.** The project's deliverable is a high-end, open-source .NET RL
> **SDK**; the games are showcases. A serious RL SDK that can't use the GPU isn't
> competitive, so M12 is **core capability built on its own merits — not parked until a
> demo needs it** (the earlier "trigger conditions" framing was demo-quality logic). The
> owner is taking on the CUDA work directly. The deliverable that matters most is the
> **public device-tensor API**, which must stay general across every env/algorithm and
> ship with `ManagedBackend` correctness parity — not any one trained model. The cube
> width ladder (PLAN §M17) becomes a *beneficiary and benchmark*: rung 1 (1024-wide)
> trains on CPU now; rung 2 (2048-wide) is a natural **GPU showcase** once the backend
> lands, since it can't converge on CPU. A standalone SDK-quality win lands first
> regardless of GPU: a **multithreaded CPU GEMM** (the baseline every GPU-less user gets).

**Key findings from the assessment:**

| Question | Answer |
|---|---|
| Raw headroom | ~500× (GPU ~10 TFLOP/s FP32 vs 20 GFLOP/s managed GEMM) |
| Why not a drop-in `IComputeBackend`? | The seam passes host `float[]`s; at today's sizes one batch-GEMM (~75 MFLOPs ≈ 7 µs) costs less than a kernel launch, and PCIe transfers cost more than the math — a naive CUDA backend would **lose** to the CPU. Device-resident tensors are required. |
| Library | **ILGPU** (C# kernels JIT-compiled to PTX): MIT, megabytes not gigabytes, keeps the from-scratch identity (own tiled GEMM kernel, 1–3 TFLOP/s realistic), CPU accelerator keeps CI green without a GPU. TorchSharp stays the fallback option. |
| First consumer | The imitation Lab — oracle data is infinite, so batch 4096+ and wider nets are pure wins; expected 5–20× campaign throughput. |
| Order of work | Benchmark first (CPU↔GPU crossover table, with/without transfer costs), then evolve the backend seam to device tensors, then a payoff campaign past the 92.3% plateau. |

## 11. Rubik's Cube page — owner's game #3 (M13–M14)

*Requirement inserted 2026-06-12. The owner's existing standalone app
`C:\Repos\WebGames\Rubiksolver` (ASP.NET Core + Three.js + a C# port of Kociemba's
two-phase solver) is **ported into** the playground; the WebGames repo is read-only
source material and stays untouched.*

A third game page at `/cube`, following the §7 playground contract (draw/scramble →
solve → trajectory playback → gallery), with one deliberate twist: the cube keeps
**two** solve buttons.

1. **3D cube + manual play.** Three.js-rendered 3×3×3 cube (ported from
   `Rubiksolver/Scripts/rubiksCube.ts` + `main.ts`, ~800 LOC): orbit camera, 18
   face-move buttons (U U' U2 … B2) with animated 90°/180° rotations, move history,
   animation-speed slider, scramble and reset. Three.js becomes a proper npm
   dependency of the Angular workspace (the original loads it from a CDN import map).
2. **Solve (algorithm) — kept as-is.** Posts the cube state to the backend, which runs
   the ported Kociemba two-phase solver (`Rubiksolver/Kociemba/`, ~2,500 LOC, pure C#,
   pruning tables generated in memory on first use) and returns a move list (≤ 22
   moves, ~10 s timeout). This button always works on any valid cube — it is the
   oracle, exactly as the BFS solver is for Rush Hour.
3. **Solve (AI).** Posts the same state; the backend runs the trained RL agent —
   greedy rollout first, falling back to a Q-guided best-first search by the same net
   (the M11 Rush Hour pattern: still the AI, now with lookahead) — and returns its move
   trajectory, the mode that produced it (`aiMode: greedy|search`) and the Kociemba
   reference move count. Failures are reported honestly (`solved: false`).
4. **RL environment + training** per the §7.3 contract: a `RubiksCubeEnv` in
   `MintPlayer.AI.ReinforcementLearning.Environments`, a model service that loads from
   the model store or trains once at startup, a pre-trained checkpoint committed in
   `models/`, console demo section, and Lab support for longer campaigns.
5. **Honest scope for the AI (pre-registered).** Solving arbitrary 20-move scrambles
   with RL is DeepCubeA-scale work and is **out of scope for v1**; the v1 agent is
   trained on a shallow-scramble curriculum and gated on that band. The UI offers both
   a full **Scramble** (~20 moves — where the algorithm button shines and the AI will
   usually fail, shown honestly) and an **Easy scramble** (≤ 6 quarter-turns — the AI's
   home turf). Deeper capability (imitation from Kociemba solutions + policy-guided
   search, the M11 Rush Hour recipe) is the designated stretch.

Environment spec (extends the §6 table):

| Env | Spaces (obs / act) | Solved criterion | Role |
|---|---|---|---|
| **Rubik's Cube 3×3×3** (episode = invert a depth-*d* scramble, d ~ U[1..6]; reward −1/step, +100 solved; cap 20 moves) | Box(324): 54 stickers one-hot over 6 colors / Discrete(12): quarter-turns U U' D D' L L' R R' F F' B B' | Pre-registered: ≥ 90% of 100 eval scrambles (depths 1–6) solved within 20 moves | Owner's game #3; huge state space, curriculum learning, oracle-checked |

Key decisions:

| Decision | Choice | Rationale |
|---|---|---|
| Where the Kociemba port lives | `MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/Kociemba` (namespace adjusted, otherwise verbatim) | It is the cube's oracle — same role and home as `RushHourSolver`; also the future imitation-data source. Testable without the web host. |
| Cube state encoding | One canonical facelet cube (54 stickers) in C# with move tables; converters to the 6×9-color DTO (wire format, ported validation incl. detailed edge/corner diagnostics) and the 54-char Kociemba string | One source of truth for move semantics; the DTO stays human-readable and matches the existing front-end state tracker. |
| AI action space | 12 quarter-turns (no half-turn actions) | Smaller action space learns faster; half-turns are two actions. The algorithm button still plays Kociemba's half-turn notation directly. |
| Pruning-table cost | Tables built in memory on first solve (2–5 s), `buildTables: false` (no disk writes) | Matches the source app's behavior; a startup warm-up keeps the first user request fast. No new Docker volume content. |
| WeatherForecast template cruft | Not ported | Dead code in the source app. |

Risks (extends §9): RL on the cube is famously hard — vanilla DQN may plateau below the
gate even on depth ≤ 6. Mitigations: curriculum over scramble depth, action masking is
not applicable (all 12 moves always legal) but the inverse-of-last-move can be masked to
halve trivial cycles; the Kociemba oracle provides unlimited imitation data if DQN falls
short (promote the M11 recipe from stretch to plan). Failure on the deep band is a
documented finding, not a blocker — the algorithm button always answers.

## 12. 2048 — restore the classic play feel (M15)

*Requirement inserted 2026-06-12. The owner dislikes the playground's canvas-rendered
2048 and prefers the original game feel of `C:\Repos\WebGames\Game2048` (a faithful
TypeScript port of Gabriele Cirulli's 2048: DOM tiles, 100 ms slide transitions, pop-on-
merge and appear-on-spawn keyframes, score-addition float). That repo is read-only
source material.*

Swap **only the "how tiles merge" experience** — rendering, animation and the in-browser
move engine — for the historic code. Everything else is explicitly out of scope and must
not change:

- **Unchanged:** the n-tuple agent and its checkpoint, `Game2048Controller` and the
  solve/status API contract (exponent cells, `PlayoutStepDto(action, spawnIndex,
  spawnValue, scoreGained)`, `FinalCells` checksum), `Board2048`/`Env2048`, the gallery,
  the edit-mode concept (set up an arbitrary board, then play or let the AI play).
- **Replaced:** the canvas board in `ClientApp/src/app/game-2048` becomes the classic
  DOM/CSS board (grid background + absolutely-positioned tiles with
  `tile-position-x-y` transition classes, `tile-new`/`tile-merged` animations, SCSS
  adapted from `Game2048/Styles/main.scss` to the playground's dark theme); manual play
  runs the historic engine (Cirulli traversal order + `mergedFrom` double-merge
  prevention — same merge *semantics* as the server, different *presentation*).
- **AI playback animates through the same classic board**: each `PlayoutStepDto` is
  applied via the classic engine with the server-provided spawn injected instead of a
  random one, so the AI's playout gets the same slide/pop/appear feel; scrubber seeks
  re-derive the board without animation, and the existing `FinalCells` checksum still
  verifies reconstruction.

Boundary mappings (the two implementations disagree on conventions; convert at the API
seam only): server cells are **exponents** (0–15), classic tiles are **values**
(2–32768+); server action ids are 0=left 1=down 2=right 3=up, classic directions are
0=up 1=right 2=down 3=left. The classic engine's merge results must stay bit-identical
to `Board2048.SlideLine`/`applyMove` (existing parity: both are standard 2048 rules with
double-merge prevention); this parity is asserted by replaying AI trajectories against
the `FinalCells` checksum and by the existing Playwright e2e.
