# RL.NET — Product Requirements Document

**Status:** v1 implemented · 2026-06-10 — all M0–M6 gates passed; see
[PLAN.md](PLAN.md) for per-milestone results and the remaining stretch list.
**Owner:** Pieterjan
**Repo:** `C:\Repos\RL.NET` (net10.0, blank solution at start)

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

RL.NET fills that gap as an *educational-but-usable* library: every component (tensor math,
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

## 3. Non-goals (v1)

Explicitly out of scope to prevent the scope creep every research thread warned about:

- GPU execution, CNNs / Atari-scale nets (the `IComputeBackend` seam exists, nothing more)
- TorchSharp backend (optional *later* package, never a core dependency)
- Multi-agent / self-play frameworks (TicTacToe/Connect-4 deferred until single-agent API is stable)
- Distributed training, ONNX export, model-based / offline RL
- NuGet publishing, Unity/Godot adapters, netstandard multi-targeting
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
| Solution layout | `src/RL.NET.Core` (tensors, autograd, NN, spaces, agents, training), `src/RL.NET.Environments`, `src/RL.NET.Demo` (console CLI — repurposed existing project), `tests/RL.NET.Tests` | Small enough to move fast, split along the natural package seams for later. Root namespace `RLNet`. |
| Test framework | **xUnit**. Fast unit tests always; statistical solve-threshold tests (3 seeds, median) in an opt-in `[Trait("Category","Slow")]` bucket | RL bugs are statistical; single-seed pass/fail lies. |
| Logging | CSV per run + live console metrics. TensorBoard event files deferred (needs a protobuf writer) | Keep v1 dependency-free. |
| Checkpoints | JSON for tabular Q-tables; versioned little-endian binary for NN weights + optimizer state + RNG state (full resume) | Cross-version stability not promised in v1. |
| License | **MIT**, stated up front, never churned | RLMatrix's license instability is precisely the gap being filled. |
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

Rush Hour note: board logic is reimplemented inside `RL.NET.Environments` (~150 LOC) rather
than referencing `C:\Repos\Spelletjes\Rush Hour` (currently being modified by another
session); the existing app can later consume RL.NET for visualization, and its puzzle
definitions can be imported as data.

## 7. Performance targets (measured with BenchmarkDotNet on dev machine)

- Tabular: ≥ 500k env-steps/sec on GridWorld (training loop included)
- NN: ≥ 1,000 Adam steps/sec, batch 64, net 4→64→64→2, single thread
- DQN solves CartPole (≥475) in ≤ 5 min wall-clock, single core
- Training allocations: zero per-step gen0 allocations in steady state (pooled tensors/tape)
- An early **spike benchmark validates the managed-GEMM assumption before M2 is built out**

## 8. Risks

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
