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
8. **Asset portability — "switch algorithm, keep the work"** *(added 2026-06-13, SDK
   capability)*: when an algorithm plateaus, the reusable assets — the environment, the
   collected data, the learned representation — should carry over to a different
   algorithm as first-class SDK features rather than ad-hoc glue. Three deliverables make
   this real: (a) an **algorithm-agnostic transition store** (serialized
   `(s, a, r, s′, done)` replay reusable across off-policy algorithms, or as
   demonstrations for on-policy ones); (b) **trunk/head-separated checkpoints** so a
   trained feature extractor transfers across algorithms whose heads differ in meaning
   (Q vs policy-logits vs V); (c) a **demonstration-dataset abstraction** so any algorithm
   can be warm-started from oracle/expert data (DQfD-style) — making the Kociemba/BFS
   oracles reusable seeds, not per-demo code. Roadmap detail in [PLAN.md](PLAN.md)
   "Immediate next step" §3.

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
| Deployment | Multi-stage Dockerfile (Node + .NET SDK build stage → ASP.NET runtime); volume `/data` for the model store + submitted games. Pre-trained checkpoints ship in `models/` (seeded into `/data` at startup) and are stored via **Git LFS** (`*.ckpt`) — the CI build checks out with `lfs: true` so the image gets real nets, not pointers. The host is **CPU-only** (e.g. Hetzner): ILGPU degrades to CPU, so the self-taught cube solver runs its bounded CPU path there. | Anyone can pull/run it and see persisted models and the public game gallery; the 35 MB self-taught net stays out of the regular git history. |
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

### 7.1 Interaction models — two architectural principles *(added 2026-06-15)*

A game's frontend↔backend connection is **not one-size-fits-all**; the right shape follows the game's
*temporal nature*. The playground supports two distinct interaction principles, and each game declares which
it uses:

| Principle | What it is | When to use | Transport | State |
|---|---|---|---|---|
| **A. Compute-and-return** | The frontend sends a request; the backend computes the *entire* answer (a solution / full playout) and returns it in one response. The browser renders it. | A **solve** (one fixed optimal answer) or where the **client legitimately drives** (human play; simple per-move query). | Plain HTTP `POST` (the existing `fetch` + `await json()` pattern). | Stateless; survives deploys; retriable. |
| **B. Server-authoritative live stream** | The **backend owns the episode loop *and* the clock**: it `Reset`s an env, then on each tick runs the policy + `Step`s and **streams the frame** `(state, action, reward, done)`. The frontend is a **pure renderer** — it draws frames as they arrive and has **no timer of its own**. | Watching an **agent play a game live** — whether real-time continuous-control (MountainCar) or step-paced turn-based (Snake). | **WebSocket** (`app.UseWebSockets()`; Traefik passes the `Upgrade` through the existing `websecure` router). | Stateful per connection; a dropped socket just restarts the (cheap) episode. |

This is a genuine architectural fork, not a one-off. Principle A is "ask a question, get the answer." Principle
B is "watch an agent *play*, live" — and crucially it is **server-authoritative**: putting the game loop and the
clock on the backend means there is **no frontend timer to drift, and no race** between a client tick and in-flight
data, so behavior is consistent across clients and runs. The decision rule: **if the user is *watching the AI
play*, it's B (the backend drives); if the user is asking for an answer or driving the game themselves, it's A.**

Per-game assignment:

| Game | Principle | Endpoint(s) |
|---|---|---|
| Rubik's Cube | A — one request → full solution | `POST /api/cube/solve{,-ai,-davi}` |
| 2048 | A — per-move query or full playout *(shipped pre-§7.1)* | `POST /api/2048/solve` |
| Rush Hour | A — one request → full trajectory + BFS-optimal *(shipped pre-§7.1)* | `POST /api/rushhour/{analyze,solve}` |
| **Snake — watch-AI** | **B — server streams each move `{body, food, action, reward, done}`** | `WS /api/snake/live` |
| **Snake — human play** | client-side only; a **JS timer** ticks the local TS engine, keyboard steers | — |
| **MountainCar — watch-AI** | **B — server streams each tick `{position, velocity, action, reward, done}`** | `WS /api/mountaincar/live` |
| **MountainCar — human play** | client-side only; a **JS timer** ticks the local TS physics, ←/→ push | — |

Each new game ships **both modes**, which nicely demonstrates the two principles side by side:
- **Watch-AI = principle B** — *backend* owns the loop + clock, streams frames, frontend is a pure renderer with
  no timer (no drift, no race with in-flight data).
- **Human play = client-side** — the *human* is the clock, so a **JavaScript timer** drives a local TS engine
  (Snake grid logic / MountainCar physics) with keyboard input and **no backend in the loop** (no round-trip per
  tick, no race). Using a client timer here is correct precisely because nothing on the server is authoritative.

WebSocket is the app's first realtime infrastructure. The existing solve-style games (cube, Rush Hour) and the
already-shipped 2048 stay on principle A. A reusable server-side episode-streamer (Reset → loop{policy → Step →
send frame} until done) backs both watch-AI modes; a small client engine backs both human-play modes — only the
env/agent/state-serialization differ.

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
> cube width ladder (PLAN §M17) becomes a *beneficiary and benchmark*: rung 1 (1024-wide)
> trains on CPU now; rung 2 (2048-wide) is a natural **GPU showcase** once the perf path
> lands, since it can't converge on CPU.
>
> **Status 2026-06-13 — the pillar landed (PLAN §M12, gate table):**
> - **M12a multithreaded CPU GEMM** — bitwise-identical, ~3.95× on 8 cores; the baseline every
>   GPU-less user gets.
> - **M12b benchmarks** — CPU thread-scaling + CPU↔GPU crossover tables measured & committed.
> - **M12c ILGPU backend** — separate `…Ilgpu` package implementing `IComputeBackend` (CUDA, else
>   CPU accelerator); plus **`AdaptiveBackend`** that auto-routes each GEMM CPU-vs-GPU by size, no
>   knobs. Correctness validated vs `ManagedBackend`.
> - **M12c-perf (scoped) device-resident forward** — `IlgpuBackend.MlpForwardScalar` keeps an MLP
>   forward resident on the GPU (no per-layer transfer); **~2× DAVI throughput**, used by the
>   value-iteration campaign (§13).
>
> **The two GPU bottlenecks (investigated 2026-06-13; PLAN §M19–M20 — the owner's focus):**
> 1. **Compute — the GEMM kernel was naive** (one thread per output element, no reuse). ✅ **M19 done
>    (2026-06-13):** replaced with a **shared-memory tiled GEMM** (adaptive tile, one `GemmDims`-generic
>    core for all three layouts + write). Measured **1.2–2.3× the naive kernel** (RTX 3060, resident
>    operands: up to **620 GFLOP/s** at 2048³, gain grows with size). Honest shortfall vs the 5–10×
>    estimate — tiling only; **register-blocked micro-tiles (M19b)** is the open lever toward multi-TFLOP.
>    From-scratch; cuBLAS documented as a native-dependency escape hatch only.
> 2. **Transfer — weights re-uploaded every call.** The scoped resident forward kept activations on the
>    GPU but re-uploaded weights each call (~570 MB/step at 8192-wide). ✅ **M20 Stage 1 done
>    (2026-06-13):** a **`DeviceMlp`** holds weights resident and re-uploads only on the trainer's
>    target-net sync, via a Core-side **`ITargetForward`** seam (`Forward` + `OnTargetSynced`) — weight
>    upload drops **per-step → per-sync (~200×)**, wired into `cube-davi`. Remaining stages: (2) device-
>    resident training fwd/bwd + on-device Adam; (3) the full `IComputeBackend` device-handle redesign
>    with `Tensor` device-backed (the general SDK-wide GPU capability). Memory fits a 6 GB 3060 (~2.2 GB
>    for an 8192×3 net); throughput, not memory, is the constraint.
>
> Together these unlock training the residual nets the **shortest-move solver (§13.1, PLAN §M21)** needs.
>
> **Measured 2026-06-14 (after Stages 1–3 + parallel successor gen):** the residual DAVI campaign is now
> **GPU-bound at ~620 GFLOP/s** (3060 at 95–100%). Two consequences for the *learning curve*: (1) **batch
> size and net width are NOT levers** — batch is throughput-neutral (just moves the bottleneck CPU→GPU; an
> iter-paced curriculum then climbs ~3.4× slower), width is diminishing (M17) + quadratically more GFLOP on
> a slow kernel; (2) **the throughput lever is the GEMM kernel — register-blocked micro-tiles (P.1/M19b),
> targeting ~3–5×.** Cheap per-update/pacing wins (LR scaling, ε-loss target sync, sample-paced curriculum,
> lighter eval) stack on top. Full analysis: `docs/OPTIMIZATIONS.md` ("Learning-curve levers").

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

## 13. Teacher-free value iteration — DAVI (M18)

*Added 2026-06-13. Imitation (M16/M17) is capped by its teacher (Kociemba — itself not
quarter-turn optimal), so the SDK needs a paradigm that can **exceed** a teacher. This is
that paradigm: deep approximate value iteration (DAVI, à la DeepCubeA), bounded only by the
cost objective (fewest moves), not a demonstrator. It's a general, reusable trainer — a third
kind alongside RL (DQN/PPO) and imitation — and the SDK's "beat the teacher" capability.*

Distinct from the exact tabular `Solvers.ValueIteration` (FrozenLake-scale, enumerated states):
this is the **function-approximation** counterpart for state spaces too large to enumerate
(the cube has ~4.3×10¹⁹ states), where a net generalizes a cost-to-go it never tabulates.

| Decision | Choice | Rationale |
|---|---|---|
| Model seam | `IDeterministicModel<TState>` (actions / apply / goal-test / state-key) in `Core.Planning` | The pure forward model classical search and model-based learning share — distinct from the RL `IEnvironment` Reset/Step loop. Minimal, general; the cube implements it as `CubeModel`. |
| Learning rule | DAVI: `target(s) = min_a [1 + (IsGoal(s′) ? 0 : V_target(s′))]`, anchored `V(goal)=0`, with a periodically-synced target net | The value-iteration update under function approximation; the signal propagates outward from the goal. No oracle/teacher — bounded by the move-cost objective. |
| Value target scaling | Predict **raw** cost-to-go (`DistanceScale = 1`) | Squashing targets to ~0.1 starves the gradients so the greedy `argmin` can't separate distances (a real finding — it then only solves the depth-1 freebies). |
| Inference | Greedy descent (`GreedyValuePlanner`) for speed; **weighted A\*** (`ValueGuidedSearch`, `f = g + w·h`) to reach deeper than greedy with no retraining | Greedy is the fast path; the search is the inference-time ceiling-raiser (mirrors the policy-guided A* used for the imitation nets). |
| Generality | Trainer is generic over `TState`; env-specifics (featurize, state sampler) are injected | The DAVI algorithm is reusable across goal-directed envs (cube, Rush Hour, future), not cube-specific. |
| Campaign / compute | `Lab --game cube-davi`: solve-rate curriculum (deepen at ≥95%), runs on the `AdaptiveBackend` with the device-resident GPU forward (§10), configurable net (`--width`/`--layers`), persists the full training state (Adam + curriculum + iterations + RNG) for lossless resume | DAVI evaluates ActionCount× successors per state, so the value-net forwards are large enough to win on the GPU; the campaign is the M12d GPU showcase. |

**Validation / gate:** a fast deterministic test proves the greedy policy descends *optimally*
under an exact (BFS) value; a `[Slow]` test proves the full DAVI loop learns to solve ≥ 80% of
shallow (depth ≤ 3) cubes **teacher-free**, checked against the `BreadthFirstPlanner` optimum.
**First campaign result (2026-06-13):** reached **curriculum depth 9 teacher-free** (greedy d7 70%,
d9 30%); the stall-fallback curriculum unstuck the old 0.95 gate; the per-depth greedy fall-off is
the 1024×3-MLP capacity wall — evidence for §13.1.

### 13.1 Shortest-move (quarter-turn-optimal) solver — the flagship goal (PLAN §M21)

*Owner's goal (2026-06-13): solve the cube in the **fewest quarter-turns** (god's number 26 QTM),
teacher-free, beating Kociemba (which is fast but not QTM-optimal). Investigated 2026-06-13; depends
on the GPU bottleneck removal (§10 / PLAN §M19–M20) — a residual net at depth can't train in tolerable
wall-clock until then.*

> **✅ BUILT + MEASURED 2026-06-14.** All pieces shipped; the GPU port (M19/M20 + register-blocked GEMM)
> made the residual net trainable. **BWAS result after the 236k-iter campaign** (residual 1024×4, weight
> **1.5, ≤100k expansions**, 12 cubes/depth): **QTM-OPTIMAL through depth 15** — 12/12 at every depth, each
> solution exactly *depth* quarter-turns — then d16 10/12 (16.2 qt), d17 5/12 (17.0 qt); **every solved cube
> beats Kociemba's QTM ~2–2.5×** (d12 12 vs 29.3, d15 15 vs 30.2 — Kociemba minimizes half-turns, so its QTM
> balloons). Tier 1 (≤ depth 7) met empirically; a *provable* claim needs weight=1 + BFS verification (open).
> Tier 2 (beat Kociemba's QTM) met comfortably through ~depth 16. **Key finding:** an earlier light read
> (44k net, ≤40k exp) and the live greedy eval (collapses ~d10) and the in-loop 8k-exp probe (looked flat at
> d14-partial) all **undersold** the net — the apparent plateau was a *search-budget* artifact, not a
> capacity ceiling. The net heuristic is accurate to ~d15 and degrades gradually past it. **So the next
> capability lever is eval-time search (more expansions / weight→1 / wider frontier), not a wider net.**
> **✅ Wired into the web cube page** (2026-06-14): the "Solve (self-taught AI)" button runs BWAS via a
> resident GPU forward where a CUDA device is present, CPU fallback otherwise (the host-span GPU path is
> transfer-bound and barely beats CPU — resident is 7–10×; see `OPTIMIZATIONS.md`). **Still open:** the
> eval-time-search lever (P.10); heavier in-loop eval readout; depth-16+ (more capacity/training; full
> god's-number 26 QTM remains out of reach on one 3060).
>
> **Update 2026-06-15 — the deeper-than-15 lever is training *scale*, not architecture.** A fresh resume
> (236k → ~313k iters, curriculum force-climbed to its depth-26 cap) left the deep frontier **flat** (light
> in-loop probe held d15/d16 within sampling noise; loss steady ~0.10 — DAVI bootstraps, so flat loss ≠
> failure). Diagnosis: the net has trained on **~36.5M states vs DeepCubeA's ~10B (≈0.4%)** — a *training-scale
> gap, not a capacity ceiling* (M17 settled that width is not the lever). On one RTX 3060 Laptop (~1.5–3k
> states/s ≈ 130–260M/day), 10B is ~5–11 **weeks** — infeasible. So depth past 15 comes from two complementary
> levers, **neither a wider net**: **(a) eval-time search budget** (P.10; free, no training) and **(b) longer
> training** (several days ≈ 10–20× more states → optimal frontier somewhat past d15, **not** god's number).
> Per the owner (2026-06-15), a multi-day campaign is running from the depth-26 checkpoint (uniform 1–26
> scramble sampling — the regime that actually builds deep states), tracked with periodic *heavy* BWAS probes
> (the light 8k-exp in-loop probe is blind past its budget). See PLAN.md → "Further-training findings (2026-06-15)".
>
> **Result (2026-06-16).** Overnight run 313k → **615k iters (+38.7M states, ~doubling total training to ~79M)**,
> lr 1e-3 throughout. Heavy-BWAS deep-solve rate (w2.5, 200k exp, same cubes, 5/depth, d14–22): **21/25 → 24/25**
> (d22 **3/5 → 5/5**, d16 4/5 → 5/5; gain landed in the first half, then consolidated). The improvement is in
> **reach/robustness**, not optimality (solves still ~19–24 qt vs Kociemba ~29–31). Incremental, as predicted —
> not "any cube" (that needs ~10B-state DeepCubeA-scale compute). The stronger 615k net is now the shipped
> `models/cube.value-davi-res.ckpt` (LFS).

| Decision | Choice | Rationale |
|---|---|---|
| Net | **Residual MLP** (`Core.Nn` `ResidualMlp`): 324→4096→2048 + 3–4 residual blocks (width 2048, **LayerNorm** not BatchNorm), scalar out, ~8–14M params | Depth-with-residuals is the untried lever (M17: width alone is diminishing); the identity path makes 6–10 effective layers trainable for the non-smooth cost surface. LayerNorm avoids the BatchNorm-vs-target-net-bootstrap interaction. Reuses the scalar-output/checkpoint contract. |
| Curriculum | Deepen toward depth 26; past the greedy-stall depth, advance on **loss convergence / time-per-rung** (not greedy solve-rate); optional **ε-loss target sync** (sync only when online loss < ε) | Greedy solve-rate never hits a mastery threshold at deep levels, but the value still learns from exposure; ε-sync is DeepCubeA's stability trick. |
| Shortest-path inference | **Batch-weighted A\* (BWAS)** — batched `ValueGuidedSearch` (expand top-N frontier, score all successors in one forward); λ knob | IDA* re-evaluates the net per node (kills throughput); BWAS amortizes the GPU forward. λ=1 optimal iff the value is admissible; λ>1 faster but suboptimal — **near-optimal is the honest claim**. |
| Optimality gate (two-tier) | **Tier 1:** depths 1–7 (BFS-tractable), ≥95% solved **provably QTM-optimal** (vs `BreadthFirstPlanner`). **Tier 2:** depths 8–20, ≥90% solved with **mean QTM length ≤ Kociemba's** | Tier 1 is the rigorous, falsifiable core ("optimal where we can prove it"); Tier 2 is the honest "beats the teacher" claim where proof is intractable. |
| Honest non-goal | Full god's-number coverage (~60% optimal everywhere, à la DeepCubeA) is **NOT** reachable on a single RTX 3060 | DeepCubeA used ~billions of states on multi-GPU for days. State the laptop-GPU ceiling plainly. |

**Realistic target on one 3060 (after M19+M20):** "Solves any cube; provably QTM-optimal to depth ~7,
near-optimal and shorter than Kociemba to ~depth 15–20, teacher-free." Then it becomes the third
**"self-taught AI"** solver on the web cube page, beside the Kociemba button and the imitation AI.

## 14. Reusable training-campaign harness  *(PRD §14; inserted 2026-06-21; SDK breadth, see [PLAN.md](PLAN.md) M25)*

The four long-running training campaigns — Rush Hour (`Program.cs`), `CubeLab`, `CubeDaviLab`, `CubePolicyLab` —
copy-paste the same scaffold: flag parsing, `FileModelStore` + `logs/` + CSV-header-if-missing, resume-on-start,
a `while (now < deadline)` loop with periodic eval + checkpoint, and an identical `Log`/`Shuffle`. The SDK goal
("make the reusable assets first-class", PLAN → Immediate next step) wants this loop as first-class, *tested* Core
surface, not per-game glue. A 3-agent investigation (2026-06-21) mapped the duplication and found the campaigns
split into **two paradigms that must NOT share one interface** — forcing the self-driving DQN onto the cube
"solve" shape, or unioning the five eval outputs into one struct, merely re-leaks the per-game complexity the
harness is meant to hide.

1. **Shared core** — `CampaignRunner` drives any `ITrainingCampaign` over a resumable, wall-clock-budgeted loop;
   it owns eval/checkpoint cadence and emits results through an `Action<CampaignProgress>` callback. IO-agnostic:
   Core does no `Console`/file writes (the Lab does CSV + console); `IModelStore` exposes no root dir.
   `ITrainingCampaign : IDisposable` = `{ Environment; Resume(store); TrainChunk()→long; IsComplete;
   Evaluate()→CampaignEval; Checkpoint(store); TryRunStandaloneEval(store) }`. The runner takes an **injectable
   clock** (it gates on time → unit-testable). `CampaignEval` stays minimal (metric dict + report line + CSV cells).

2. **`GoalReachingCampaign`** (final goal — reach a terminal "solved" state; eval = **solve-rate** on held-out
   instances): the cube family (Kociemba imitation, DAVI value-iteration, EfficientCube policy) + Rush Hour.

3. **`ScoreMaximizingCampaign`** (infinite goal — maximize cumulative return, no terminus; eval = **mean
   return/score**): Snake (DQN), extensible to MountainCar / Pendulum / CartPole / 2048. Wraps a self-driving
   trainer (`DqnTrainer`…) in resumable chunks (raise `MaxSteps`, persist `DqnTrainingState`).

| Decision | Choice | Rationale |
|---|---|---|
| One harness vs two | **Two** over one shared runner | Solve vs reward-maximize differ in training-driving and eval shape; one interface would bend the self-driving DQN or become a god-struct (the investigation's biggest risk). |
| `CampaignEval` shape | **Minimal**; campaigns own auxiliary CSVs | CubeDavi writes two CSVs on two cadences + five disjoint eval-only formats — a union type re-leaks that into the runner. |
| CubeDavi eval-only modes | `TryRunStandaloneEval(store)` escape hatch, run before training | value-curve / vs-kociemba / time-budget / search differ in *output structure*, not just metric values. |
| Snake / DQN home | `ScoreMaximizingCampaign`, NOT the solver interface | `DqnTrainer` is self-driving with fat-state resume; the cube shape fights it. |
| Location | `Core/Training/` (`…Core.Training`), IO-agnostic callback | Sibling to `DqnTrainer`/`Evaluator`; keeps Core packagable + free of Console/file IO. |
| Runner shape | **Instance class** (not static) with an injected `TimeProvider` (BCL) | DI-composable; the `TimeProvider` makes the time-budgeted loop deterministically testable and the system clock the default. |
| DI registration | **MintPlayer.SourceGenerators `[Register]`** on `CampaignRunner` (dogfood) | The generator emits `AddReinforcementLearningCore()`; DI resolves `TimeProvider` via the ctor — no hand-written registration. |
| Host builder | **`AIHost.CreateBuilder(dataDir)`** in a **new `…Hosting` package** | The AI counterpart to `WebApplication.CreateBuilder`; keeps `Microsoft.Extensions.*` deps out of Core. Takes a data dir (not raw `args` — the command-line config provider chokes on bare flags like `--eval-only`). |

**Hosting & DI.** Training tools compose like an ASP.NET host: `AIHost.CreateBuilder(dataDir).Build()` →
`services.GetRequiredService<CampaignRunner>()` + `<IModelStore>`. `AddReinforcementLearning(dataDir)` registers the
`FileModelStore`, `TimeProvider.System`, and (via the Core source-generated `AddReinforcementLearningCore()`) the
runner. The GPU compute backend registers separately via `services.AddGpuBackend()` (built 2026-06-21) — shipped in
its **own** package `MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting`, kept apart from the lean Ilgpu *compute*
package so that backend carries no DI dependency, and so Core stays backend-agnostic. It registers the shared
`AdaptiveBackend` as a container-owned singleton; the two GPU-benefiting cube campaigns inject it instead of
constructing their own. `CampaignRunner` + `ITrainingCampaign` live in Core (BCL `TimeProvider` only); `AIHost` + the
DI extensions live in the Hosting package.

**Gate (✅ all met, 2026-06-21):** (1) each migrated campaign's existing tests + CI stay green and its run behaves
identically (same checkpoint ids + CSV columns) — verified per game by eval-only parity (RushHour cards 16/77/82/81;
cube gate 97/100) + 264 tests; (2) a deterministic `CampaignRunner` unit test (fake campaign + fake clock + in-memory
`IModelStore`) asserts the loop, eval cadence, resume, and checkpoint calls — sub-second, no wall-clock; (3) the Snake
score-maximizing campaign resumes bitwise (reload `DqnTrainingState` — verified 20k→35k) and reaches the shipped
baseline (~20 food on the 12×12 grid by 20k steps, climbing to ~22).

**Paradigm decision (settled 2026-06-21).** The two paradigms (goal-reaching vs score-maximizing) are NOT two
interfaces or two base classes — they share one **minimal, paradigm-agnostic** `ITrainingCampaign`
(`Resume`/`TrainChunk`/`IsComplete`/`Evaluate`/`Checkpoint`). The distinction lives only in each campaign's
`Evaluate` (solve-rate vs mean episodic return, both expressed as generic `CampaignEval` metrics) and in whether
`IsComplete` ever fires. A `…Campaign` base class per paradigm was considered and rejected: it would be a shallow
wrapper (every goal-reaching game already implements the interface directly), and the obvious score-maximizing base
shaped around `DqnTrainer` wouldn't fit future score games on different trainers (2048 n-tuple, SAC/PPO). The
original "two harnesses" framing is honoured at the *behaviour* level, not as two types. (This supersedes an earlier
non-goal that read "not a single interface spanning both paradigms" — the interface is shared precisely because it
is minimal and bakes in neither paradigm's eval shape.)

**Non-goals:** not a multi-trial hyperparameter sweep runner; CubeDavi's bespoke eval-only modes are not modeled as
`CampaignEval` variants (they go through `TryRunStandaloneEval`).

**Single training path (M26, done 2026-06-22).** The original "stretch" idea was to have the web run the campaigns
via `CampaignRunner`. Investigation showed the web's `EnsureModel` training was **vestigial** — every checkpoint is
committed to `models/` (Git LFS) and seeded at startup, so it never ran in production. So M26 instead made the web
**load-only**: training lives solely on the dev side (Lab campaigns / Console) and is committed; the web loads and
serves. The campaigns stay `internal` to the Lab (no shared library needed — the web doesn't train). See PLAN M26.

## 15. Snake — strength via search, not more training  *(PRD §15; inserted 2026-07-10; see [SNAKE_SEARCH_PRD.md](SNAKE_SEARCH_PRD.md) + [PLAN.md](PLAN.md) M34)*

A learned Snake policy is **structurally capped at ~50 food@12** (M27's sweep: capacity, features, reward and horizon
all plateau there — a reactive net can't avoid a trap that forms several moves ahead). The lever is **planning, not
training**: a multi-ply look-ahead, single-sourced in `snake_solver.pg` so C# eval and the browser director share it
byte-for-byte, lifts play to **~81 food@12 (+60% over the plateau)** with **no retraining and no observation change**.

> **Design decision (2026-07-10).** The strength comes from an exact **flood-fill survivability search** (reuses the
> `reachableFreeSpace` the net already carries as an input). The **biggest single lever** turned out to be an
> **anti-fragmentation term** — scoring the *fraction of currently-free cells still reachable* (the reachability-ratio
> idea originally floated as a net input, but far more effective in the **search leaf score**): it lifts food@12 ~71 → ~81
> (+14%, robust across seed bases). A per-node net evaluation, by contrast, added **no** strength for ~9× the latency, so
> the trained net is kept in the loop only as a cheap **tie-breaker between equally-safe moves**. This reframes "make the
> Snake net strong": the net's reactive ceiling is real and low, so the *agent* is made strong by search while the *net*
> stays a leaf/tiebreak evaluator. The residual cap (~81 mean; single games hit ~106) is self-traps beyond the horizon —
> left to a future tail-reachability / Hamiltonian endgame.

## 16. Snake — curved-tube rendering  *(PRD §16; inserted 2026-07-10; see [SNAKE_RENDER_PRD.md](SNAKE_RENDER_PRD.md) + [PLAN.md](PLAN.md) M35)*

Cosmetic, view-layer only — no AI or logic change. The board renders as flat coloured `<div>` squares on a CSS grid
that snap one cell per tick; it reads as a blocky 1980s snake. M35 replaces the **view** with a Canvas 2D
**curved-tube** renderer: a Catmull-Rom spline through the cell centres (corners smoothly for free), a tapered tail, an
oriented head with eyes, cheap multi-pass 3-D shading, and — the biggest single win — `requestAnimationFrame`
**interpolation** between the coarse (120/150 ms) game ticks so the tube *glides* instead of teleporting.

> **Design decision (2026-07-10, 2-agent investigation).** Canvas 2D, not SVG (DOM/layout thrash, slower for a
> full-redraw loop, and its stroke can't taper any better) and not WebGL (only wins at scales this game never reaches).
> The game logic already funnels all display state through one `render(body, food, eaten)` call over a flat cell-index
> body, so the renderer swaps in behind it with **zero change** to `snake-logic.ts` / `snake-director.ts` /
> `snake_solver.ts` / the `.pg`. M34's ~81 food@12 strength and all tests are untouched — the gate is a purely visual
> in-browser before/after.

## 17. Network visualizer — see the net, and watch it learn  *(PRD §17; inserted 2026-07-12; see [NETWORK_VISUALIZER_PRD.md](NETWORK_VISUALIZER_PRD.md) + [PLAN.md](PLAN.md) M36)*

A network is otherwise only ever visible as numbers — a `.ckpt` on disk, a CSV of eval scalars. This feature makes a
net **visible**, and in particular lets you **watch it change as it trains** — the priority — and lets a newcomer to
neural nets *read* the picture. It adds a read-only **pull** seam in Core (`INetworkTelemetrySource` + `NetworkInspector`,
which describes any net from its parameter tensors), which every training campaign implements in a few lines, so
`--viz` works for **all six trainable games** with no per-trainer code. The Lab-hosted viewer shows a live node-link
graph + weight heatmaps + a loss sparkline, with **hover tooltips** explaining each neuron/connection in plain
language. Because sampling only *reads* the host-resident parameter arrays (on a background thread), a run being
watched is **bitwise-identical** to one that isn't (verified: viz vs no-viz checkpoints are SHA256-equal).

> **Design decision (2026-07-12, 4-agent investigation).** The live view is served **by the training process itself**,
> not the web app: `RLDemo.Web` is train-free and training lives in the Lab CLI where the net actually is. So the Lab
> hosts a tiny localhost `HttpListener` + **WebSocket** (`/ws`) feeding one self-contained Canvas 2D page, driven by an
> async background sample loop that pulls the campaign's telemetry source. WebSocket (over one-way SSE) is a deliberate
> call so the channel can become bidirectional — viewer→trainer control (pause/step, cadence, layer-select) — without
> swapping transport. It is **dev-only**: the socket is gated to a Development host environment and is not the web-app
> WS stack removed in M32/M33. Static `.ckpt` inspection instead belongs client-side in the Angular app (it already
> ships `.ckpt` parsers) as a follow-on `/network` page.
