# Architecture & Contributor Guide

A map of **where things live and how they fit together**, for anyone making changes to this
from-scratch C# reinforcement-learning library. It complements the other docs rather than repeating them:

- **[prd/PRD.md](prd/PRD.md)** — *why* the project exists and its design goals.
- **[prd/PLAN.md](prd/PLAN.md)** — the milestone roadmap (what was built when, and why).
- **[OPTIMIZATIONS.md](OPTIMIZATIONS.md)** — the performance/capability ledger (CPU & GPU compute, residency, training efficiency — done and not-pursued, with rationale).
- **[ADDING_A_GAME.md](ADDING_A_GAME.md)** — the end-to-end checklist for adding one new game (env → service → controller → DI → frontend).
- Feature/game PRDs under **[prd/](prd/)** (FruitCake, NoisyNets, …) — the *why* and history behind specific subsystems.

This file is the **code map**: the subsystems, the checkpoint/file formats, the wire protocol, and "where to change things" for common tasks.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Core — tensors, autograd, NN, backends, RNG](#2-core--tensors-autograd-nn-backends-rng)
3. [Environments — the game/task layer](#3-environments--the-gametask-layer)
4. [Trainers & algorithms](#4-trainers--algorithms)
5. [Checkpoints, model store & file formats](#5-checkpoints-model-store--file-formats)
6. [Live "Watch AI" WebSocket protocol](#6-live-watch-ai-websocket-protocol)
7. [Web playground & deployment](#7-web-playground--deployment)
8. [Training & comparing models (workflow)](#8-training--comparing-models-workflow)
9. [Game solving stacks — deep dives](#9-game-solving-stacks--deep-dives) (Rubik's Cube · Rush Hour · 2048)
10. [Client-side game engines (human play)](#10-client-side-game-engines-human-play)
11. [Console demo, Bench & source-generated DI](#11-console-demo-bench--source-generated-di)
12. [Tests & conventions](#12-tests--conventions)
13. [Where to change things — quick map](#13-where-to-change-things--quick-map)

---

## 1. Repository layout

Pure managed C#, `net10.0`, no Python / no libtorch / no native binaries (one exception: `AllowUnsafeBlocks`
for pointer pinning in the managed GEMM — still 100% managed). Five **published SDK libraries** ship in
lockstep; the apps and tools are unversioned.

```
MintPlayer.AI.ReinforcementLearning.sln
├── src/
│   ├── …Core/             Tensors + tape autograd, Adam, NN modules, compute-backend seam,
│   │                      seeded RNG, agents, trainers, planning, checkpoints, model store.   [SDK]
│   ├── …Environments/     The games: GridWorld, FrozenLake, CartPole, MountainCar, Snake,
│   │                      2048, RushHour, RubiksCube, FruitCake (+ inference search helpers). [SDK]
│   ├── …Hosting/          AIHost.CreateBuilder + DI (IModelStore, TimeProvider, CampaignRunner). [SDK]
│   ├── …Ilgpu/            Optional GPU backend (ILGPU → CUDA/OpenCL/CPU). Plugs under autograd. [SDK]
│   ├── …Ilgpu.Hosting/    DI glue: services.AddGpuBackend() (AdaptiveBackend CPU+GPU routing).   [SDK]
│   ├── RLDemo.Console/    CLI demo — watch agents learn & play; --save/--load checkpoints.      [app]
│   └── RLDemo.Web/        ASP.NET Core host + embedded Angular SPA playground (Watch AI/Play).  [app]
├── tests/…Tests/          xUnit: solve-threshold gates, determinism, web API contract tests.
├── tools/
│   ├── …Lab/              Long-running, resumable training campaigns + A/B & search-eval harnesses.
│   └── …Bench/            Performance benchmarking harness.
├── models/                Shipped checkpoints (Git LFS, *.ckpt) — seeds the web app & A/B baselines.
└── docs/                  PRDs, PLAN, OPTIMIZATIONS, ADDING_A_GAME, and this guide.
```

**Versioning.** `src/Directory.Build.props` holds a single `<RLNetVersion>` (e.g. `0.3.0`); the five SDK
libraries set `<Version>$(RLNetVersion)</Version>` so the whole SDK releases in lockstep. Bump it once per
release. Reference graph: `Environments`→`Core`; `Hosting`→`Core`; `Ilgpu`→`Core`; `Ilgpu.Hosting`→`Ilgpu`;
the apps/tools reference what they need and stay unversioned.

---

## 2. Core — tensors, autograd, NN, backends, RNG

A tape-based autograd `Tensor` whose math is routed through an `IComputeBackend` seam, so the same NN code
runs on a multithreaded CPU backend or an optional ILGPU/CUDA GPU backend. Determinism is preserved by a
bitwise-identical parallel row decomposition on CPU.

| Path (`src/…Core/`) | Role |
|---|---|
| [`Numerics/Tensor.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Numerics/Tensor.cs) | Tape autograd: forward ops record parents + a backward closure; `Backward()` runs them in reverse topological order, accumulating `Grad`. `GradMode.NoGrad()` (thread-local) disables recording. |
| [`Numerics/IComputeBackend.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Numerics/IComputeBackend.cs) | The seam: GEMM (+ transpose variants), elementwise `Map`/`MapBackward`, reductions (`Sum`, `LogSoftmax`, `Gather`, `HuberLoss`, `LayerNorm`). Forward ops **write**; backward ops **accumulate**. (`ManagedBackend`, the CPU impl, lives here.) |
| `Numerics/ManagedBackend` (in [`IComputeBackend.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Numerics/IComputeBackend.cs)) | Pure-managed CPU backend (`TensorPrimitives` SIMD). Large GEMMs partition output rows across `Parallel.For` workers (no reduction → bitwise-identical to sequential); `maxDegreeOfParallelism` caps it. |
| [`Nn/Adam.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/Adam.cs) | Adam with bias correction; mutable `LearningRate`; `ClipGradNorm`. |
| [`Nn/Modules.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/Modules.cs) | `IModule` (`Forward`/`Parameters`), `IValueNet` (`InputSize`/`CloneStructure`/`CopyFrom`/`GrowInput`), `Linear`, `Mlp`. |
| [`Nn/DuelingQNet.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/DuelingQNet.cs) | Shared trunk → value + advantage heads, `Q = V + (A − mean A)`; optional noisy heads; plain→noisy promotion (`ToNoisy`). |
| [`Nn/NoisyLinear.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/NoisyLinear.cs) | Factorized-Gaussian noisy layer (μ + σ⊙ε; ε is a non-grad constant resampled per step — learned σ only). |
| [`Nn/ResidualMlp.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/ResidualMlp.cs) | Deep residual value net (LayerNorm blocks) + Net2WiderNet hidden-width growth (`WidenTo`). |
| [`Nn/NetTransfer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Nn/NetTransfer.cs) | Generic, function-preserving weight transfer: exact param copy (`CopyParameters`, behind every `CopyFrom`) + input-dimension growth (`TransferGrownInput`, behind `IValueNet.GrowInput`). |
| [`Random/Xoshiro256StarStar.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Random/Xoshiro256StarStar.cs) | Version-stable PRNG (never `System.Random`); `GetState`/`SetState` for checkpointing. |
| [`Random/SeedSequence.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Random/SeedSequence.cs) | One master seed → independent streams via `RngStreams` (`Environment`, `Policy`, `Init`, `Buffer`, `Evaluation`, `Noise`, …). |
| [`src/…Ilgpu/`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu/) | GPU backend: tiled shared-memory GEMM ([`IlgpuBackend.cs`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu/IlgpuBackend.cs)); [`AdaptiveBackend.cs`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu/AdaptiveBackend.cs) routes each GEMM to CPU or GPU by a MAC threshold and falls back to CPU when no device is present. Only large nets (cube) clear the threshold. |

```csharp
// Numerics/Tensor.cs — every op records a backward closure run in reverse during Backward()
public Tensor MatMul(Tensor other) {
    Backend.Current.Gemm(Data, other.Data, data, m, k, n);                // forward (routed)
    return MakeResult(data, [m, n], [this, other], result => () => {
        if (NeedsGrad)        Backend.Current.GemmTransposeB(result.Grad!, other.Data, Grad!, m, k, n);
        if (other.NeedsGrad)  Backend.Current.GemmTransposeA(Data, result.Grad!, other.Grad!, m, k, n);
    });
}
```

`Backend.Current` is a settable global (default `ManagedBackend`); set it once at startup to switch
compute. Algorithm code never changes when you swap backends.

**Weight transfer ("keep the work").** Four function-preserving transforms let a trained net carry over instead of
retraining from scratch; the *generic* ones live in `NetTransfer`, the *structure-specific* ones on their nets:
`IValueNet.GrowInput(n)` (wider input — new features zero-init, identical output on the old ones; for when an env's
observation gains features), `ResidualMlp.WidenTo(w)` (wider hidden, Net2WiderNet), `DuelingQNet.ToNoisy()`
(plain→noisy exploration), and `CubePolicyNet.PolicyAsMlp()` (extract the policy trunk). All transfer **weights
only** — the caller rebuilds the optimizer (Adam moments are keyed to the parameter set) and, for `GrowInput`, starts
from a fresh replay buffer (stored transitions hold old-width observations).

---

## 3. Environments — the game/task layer

Every environment implements a Gymnasium-faithful contract. The **terminated-vs-truncated split** is the
key correctness detail: *terminated* = a true terminal state (bootstrap value 0); *truncated* = an external
cutoff (time limit) where the value target **must still bootstrap** from the final observation. Conflating
them is the classic silent DQN bug.

| Interface (`src/…Core/Environments/`) | Purpose |
|---|---|
| [`IEnvironment<TObs,TAct>`](../src/MintPlayer.AI.ReinforcementLearning.Core/Environments/IEnvironment.cs) | `Reset(seed)` / `Step(action)` → `StepResult` (`Observation`, `Reward`, `Terminated`, `Truncated`, `Info`); `ObservationSpace`/`ActionSpace`; `RenderString()`. |
| [`Space.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Environments/Space.cs) | `BoxSpace` (continuous ℝⁿ), `DiscreteSpace` (0..N−1). |
| [`IActionMaskProvider`](../src/MintPlayer.AI.ReinforcementLearning.Core/Environments/IActionMaskProvider.cs) | `CurrentActionMask()` → per-state legal actions; trainers filter exploration **and** the TD-target argmax. |
| [`IStatefulEnvironment`](../src/MintPlayer.AI.ReinforcementLearning.Core/Environments/IStatefulEnvironment.cs) | `SaveState()`/`RestoreState()` — opaque `byte[]` snapshot for bitwise-exact resume. |
| [`IRestartableEnvironment<TObs>`](../src/MintPlayer.AI.ReinforcementLearning.Core/Environments/IRestartableEnvironment.cs) | `IStatefulEnvironment` + `CurrentObservation()` — lets a trainer begin an episode from a restored snapshot (reverse-curriculum). |

| Game | Obs dims | Actions | Notes |
|---|---|---|---|
| CartPole | 4 | 2 | Bit-exact Gymnasium port. |
| MountainCar | 2 | 3 | Momentum-building control. |
| Snake | 177 | 4 | Egocentric 9×9 patch + scalars + **flood-fill** free-cell reachability (anti-self-trap). |
| FruitCake | 89 | 14 | Suika merge physics (**single-source solver** — one `.pg` → C# + TS, see §10); reward shaping (training-only); big-fruit position inputs; see below. |
| 2048 | 16 | 4 | Log-scaled tiles; action masking. |
| RushHour | 72 | 32 | Two 6×6 planes; puzzle-per-episode; BFS oracle for imitation. |
| RubiksCube | 324 | 12 | One-hot stickers; inverse-move masking; curriculum depth. |
| GridWorld / FrozenLake | 16 | 4 | Tabular; value-iteration ground truth for tests. |

**The static `BuildObservation` pattern.** Observation construction is a *static* method so the live
serving path feeds the net the **byte-identical** observation the policy trained on. When you change an
observation, you change it in exactly one place and bump the net's input width (the model-load guard rejects
a stale-width checkpoint). **FruitCake goes further (M32):** its `BuildObservation` — plus the net forward
pass and the depth-3 search — are single-sourced in `fruitcake_solver.pg` (see §10), so the *same* code runs
in C# and, transpiled, in the browser. `FruitCakeEnv.BuildObservation` is now a thin delegate to the
generated core.

---

## 4. Trainers & algorithms

Model-free trainers (off-policy DQN, on-policy PPO, tabular) live in `Core/Training`; planning over a
learned value net lives in `Core/Planning`; inference-time **search** lives next to each game in
`Environments`.

| Path | Role |
|---|---|
| [`Core/Training/DqnTrainer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/DqnTrainer.cs) | Double/Dueling DQN: target net, replay buffer, ε-greedy **or** NoisyNets, n-step returns, optional reverse-curriculum starts. Single-file CleanRL-style reference. |
| [`Core/Training/ReplayBuffer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/ReplayBuffer.cs) | Circular `(s,a,r,s′,terminated, next-mask)` store. Stores `terminated` **only** (never the combined done flag). |
| [`Core/Training/NStepAccumulator.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/NStepAccumulator.cs) | Folds n-step returns before buffering: reward → Σ discounted next-n; bootstrap → γⁿ. Handles terminal vs truncation at episode ends. |
| [`Core/Training/PpoTrainer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/PpoTrainer.cs) | PPO: vectorized envs, GAE(λ), clipped surrogate, orthogonal init, LR anneal. |
| [`Core/Training/Evaluator.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/Evaluator.cs) | Greedy evaluation runner (returns, lengths, success rate). |
| [`Core/Planning/ValueIterationTrainer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Planning/ValueIterationTrainer.cs) | Teacher-free deep approximate value iteration (DAVI / DeepCubeA style) over a forward model. |
| [`Core/Planning/ValueGuidedSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Planning/ValueGuidedSearch.cs), [`GreedyValuePlanner.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Planning/GreedyValuePlanner.cs) | Weighted-A\* / greedy solve over a learned cost-to-go. |
| [`Environments/FruitCake/FruitCakeSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/FruitCake/FruitCakeSearch.cs) | Depth-1→3 forward search; known current+next maximize, the 3rd ply is an **expectimax chance node** over the unknown fruit. |
| [`Environments/Game2048/Expectimax2048.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/Game2048/Expectimax2048.cs) | Expectimax over the n-tuple afterstate value (chance node = the random tile spawn). |
| [`Environments/RubiksCube/CubePolicySearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubePolicySearch.cs), [`CubeValueSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubeValueSearch.cs) | Beam / value-guided search over the cube nets. |

**`DqnOptions` knobs** (defaults in `DqnTrainer.cs`): `Hidden` `[64,64]`, `Gamma` `0.99`, `LearningRate`
`1e-3`, `BufferCapacity` `50k`, `BatchSize` `64`, `WarmupSteps` `1000`, `TrainEvery` `1`, `TargetSyncEvery`
`500`, `Epsilon` `LinearSchedule(1→0.05 over 10k)`, `MaxSteps` `100k`, `MaxGradNorm` `10`, `DoubleDqn`
`true`, `Dueling` `false`, `NoisyNets` `false` (implies Dueling), `NStep` `1`, `StartStates`/`StartStateProb`
(reverse-curriculum), `EvalEvery`/`EvalEpisodes`, `SolveThreshold`.

```csharp
// DqnTrainer.cs — the TD target (n-step aware, masked, terminated-only bootstrap)
double gammaN = Math.Pow(options.Gamma, options.NStep);
int best = /* argmax over LEGAL next actions of the online net (Double DQN) */;
double bootstrap = batch.Terminated[i] || best < 0 ? 0 : gammaN * targetQ.Data[i * actions + best];
targets[i] = (float)(batch.Rewards[i] + bootstrap);   // y = Rₙ + γⁿ·(1−terminated)·Q_target(s′,a*)
```

```csharp
// FruitCakeSearch.cs — search amplifies a (capped) reactive net WITHOUT retraining; leaf value is injected
public int ChooseColumn(FruitCakeWorld world, int current, int next) { … }   // MaxDepth 1|2|3, TopK / TopK2 pruning
```

**Inference search is the headline lever for FruitCake**: the reactive net plateaus (it can't plan), and
depth-2→3 forward search took watermelon-rate 0% → 30% → 50% with **no retraining** — the net is only the
search's leaf evaluator. See `prd/FRUITCAKE_IMPROVE_PRD.md`.

---

## 5. Checkpoints, model store & file formats

All checkpoints are versioned binary with a shared magic header; the model store keeps one *current*
checkpoint per `(environment, algorithm)` so training never restarts from scratch.

| Path (`src/…Core/Checkpoints/`) | Role |
|---|---|
| [`CheckpointFormat.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/CheckpointFormat.cs) | Magic `"RLNC"` (`0x434E4C52`) + kind string + `int32` version; primitives for floats/ints/bools/RNG; `ReadHeader` validates and returns the version for back-compat branching. |
| [`DqnTrainingState.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/DqnTrainingState.cs) | Full training state (kind `dqn-state`): nets, optimizer, replay buffer, RNGs, obs, env snapshot, n-step window. |
| [`DuelingQNetCheckpoint.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/DuelingQNetCheckpoint.cs) / [`MlpCheckpoint.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/MlpCheckpoint.cs) / [`ResidualMlpCheckpoint.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/ResidualMlpCheckpoint.cs) | Per-architecture net weights (tagged via `QNetCheckpoint`). |
| [`AdamCheckpoint.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/AdamCheckpoint.cs), [`ReplayBufferCheckpoint.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/ReplayBufferCheckpoint.cs) | Optimizer moments / algorithm-agnostic transition payload (embedded header-less in `dqn-state`). |
| [`ModelStore.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Checkpoints/ModelStore.cs) | `IModelStore` + `FileModelStore`: files named `<envId>.<algoId>.ckpt` under a root dir; atomic save (temp + rename). |

**`dqn-state` on-disk order** (after `magic+kind+version`): `Online` net → `Target` net → `Adam` →
`ReplayBuffer` → `PolicyRng` → `BufferRng` → `NoiseRng` *(v≥3)* → `CurrentObs` → `StepsCompleted` →
`LastLoss` → `LastEval` → `EnvState` (len-prefixed, −1 = none) → n-step accumulator *(v≥4)*. Version history:
**v2** type-tagged nets, **v3** + NoiseRng (NoisyNets), **v4** + n-step accumulator window.

**`dueling-q` on-disk order**: `InputSize` → hidden sizes → `Actions` → `Noisy` bool *(v≥2)* → each
parameter tensor's floats. (v1 loads as non-noisy.)

**Evolving a format safely** — bump the version constant, append new fields at the end, read them
conditionally, default sensibly for old files:

```csharp
int version = CheckpointFormat.ReadHeader(reader, Kind, Version);
// …read existing fields in order…
if (version >= 4 && reader.ReadBoolean())
    state.Accumulator = NStepAccumulator.Load(reader, …);   // older files: null → trainer makes a fresh one
```

**Git LFS.** `.gitattributes` has `*.ckpt filter=lfs diff=lfs merge=lfs -text`; shipped nets live in
`models/` (e.g. `models/fruitcake.dqn.ckpt`) and seed the web app + serve as A/B baselines.

---

## 6. Live "Watch AI" WebSocket protocol

The "Watch AI" mode is **server-authoritative**: the backend owns the env, the policy, and the clock; the
browser is a pure renderer with no game timer (this avoids client/server timer races — see
`prd/` web-interaction notes). Endpoints accept both **GET (HTTP/1.1 Upgrade)** and **CONNECT (HTTP/2
Extended CONNECT, RFC 8441)** so the socket works under HTTP/1.1 and HTTP/2.

- Snake / MountainCar use the **generic `EpisodeStreamer`** (`Reset → policy → Step → JSON frame`, paced by `tickMs`; one frame per env step).
- **FruitCake no longer uses the server at all (M32).** Its entire AI — physics, observation, net forward pass,
  and depth-3 search — is single-sourced in `fruitcake_solver.pg` and runs **in the browser** (`FruitCakeDirector`
  over the generated TS core + the shipped `wwwroot/models/fruitcake-net.ckpt`). There is **no** `FruitCakeController`,
  no `/api/fruitcake` WebSocket, and no server-side FruitCake net — per-viewer server cost is zero. See §10.

```csharp
// e.g. SnakeController — 503 until the model is loaded, then stream
[AcceptVerbs("GET", "CONNECT", Route = "live")]
public async Task Live() {
    if (!HttpContext.WebSockets.IsWebSocketRequest) { Response.StatusCode = 400; return; }
    if (model.Agent is null)                         { Response.StatusCode = 503; return; }   // still loading
    using var socket = await HttpContext.WebSockets.AcceptWebSocketAsync();
    await EpisodeStreamer.RunAsync(socket, env, act, frame, tickMs, …);
}
```

Frames are JSON (`JsonSerializerDefaults.Web`), e.g. `SnakeFrameDto(Body, Food, Action, Reward, Done, …)`,
`FruitCakeFrameDto(Fruit[], HeldTier, NextTier, Score, Danger, Done)`. The client polls
`GET /api/<game>/status` → `{status, error}` (`loading|ready|failed`) to gate the UI, then opens the socket.

```ts
// *-api.ts — plain fetch + WebSocket; no HttpClient, no environment.ts
connectLive(onFrame, onClose): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const s = new WebSocket(`${proto}://${location.host}/api/snake/live`);
  s.onmessage = e => onFrame(JSON.parse(e.data)); s.onclose = onClose; s.onerror = onClose;
  return s;
}
```

---

## 7. Web playground & deployment

`src/RLDemo.Web` is an ASP.NET Core host that **embeds and runs the Angular dev server itself** in
Development (via the SPA middleware in `Program.cs`) and serves a pre-built bundle in production. The web
process **never trains** — it loads shipped `models/*.ckpt` at startup (load-only) and serves them.

| Path | Role |
|---|---|
| [`RLDemo.Web/Program.cs`](../src/RLDemo.Web/Program.cs) | Host, SPA dev-server middleware, model-service DI, WebSocket setup, optional GPU backend. |
| [`RLDemo.Web/Services/`](../src/RLDemo.Web/Services/)`*ModelService.cs` | `IModelStartupService` — load a checkpoint at startup, **guard input width**, expose a `GreedyQAgent`, report `ModelStatus`. |
| [`RLDemo.Web/Controllers/`](../src/RLDemo.Web/Controllers/)`*.cs` | `/<game>/status`, `/<game>/live` (WebSocket), `/version` (build identity). |
| [`RLDemo.Web/ClientApp/src/app/`](../src/RLDemo.Web/ClientApp/src/app/)`<game>/` | Standalone Angular (signals) component per game: Play-Human (client loop) + Watch-AI (WebSocket); canvas render. |
| [`RLDemo.Web/ClientApp/src/app/screen-wake-lock.ts`](../src/RLDemo.Web/ClientApp/src/app/screen-wake-lock.ts) | Shared service: holds a Screen Wake Lock during Watch-AI so mobile screens don't sleep; reconnects the stream on foreground. |
| [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml) | Multi-stage build (Node build → `dotnet publish -p:EnableSpaBuilder=false` → ASP.NET runtime); VPS/Traefik deploy compose. |
| [`.github/workflows/`](../.github/workflows/) | [`pull-request.yml`](../.github/workflows/pull-request.yml) (build + `Category!=Slow` tests + `npm run build`), [`build-master.yml`](../.github/workflows/build-master.yml) (NuGet pack/push), [`playground-docker.yml`](../.github/workflows/playground-docker.yml) (image → GHCR → SSH deploy to VPS, writing `BUILD_SHA`/`IMAGE_DIGEST`/`DEPLOY_TIME`). |

```csharp
// *ModelService — load-or-fail; a stale-width checkpoint is rejected, the demo falls back to a heuristic
var net = DuelingQNetCheckpoint.Load(stream);
if (net.InputSize != SnakeEnv.ObservationSize) { Status = ModelStatus.Failed; return false; }
_agent = new GreedyQAgent(net, SnakeEnv.ActionCount); Status = ModelStatus.Ready;
```

> **Contributor note:** because the ASP.NET host runs the embedded Angular dev server, **do not run
> `ng serve` / `ng build` / `ng test` yourself** while the host is running — just `dotnet run` the web
> project and edit `ClientApp/src`; it live-reloads. The footer shows the deployed commit SHA + image digest
> (via `/api/version`) so you can confirm what's live on the VPS.

**Deploy flow:** PR (CI builds + tests + Angular build) → merge to `master` → `playground-docker` builds the
image to GHCR and SSH-deploys it to the VPS (`docker compose pull && up -d`); Traefik routes the domain.

### Gallery (submitted boards)

When a solver route succeeds (Rush Hour / Cube / 2048), the controller auto-appends the board + the AI/solver
response to a public, read-only **gallery**. `GalleryStore` (`Services/GalleryStore.cs`) writes one JSON file
per entry under `<data>/gallery/` (id = timestamp + short hash; atomic temp-write; corrupt files skipped on
read), so entries persist on the `playground-data` volume across deploys. `GalleryController` exposes
`GET /api/gallery` (newest first) and `GET /api/gallery/{id}`; the Angular `gallery/` component lists them and
links each to its game with a `?replay={id}` query param, which the game component loads to replay the solution.
No moderation — every solved board is auto-admitted; ids are validated (alphanumeric/`-`) against path traversal.

```csharp
// e.g. RushHourController after a solve — board + response captured for replay
gallery.Add("rushhour", $"{how} solved it in {trajectory.Length} moves (optimal {optimal})", board, response);
```

*Change it:* submission logic lives in each game's solve endpoint; storage format/location in `GalleryStore`;
list/replay UI in `gallery.ts` + the game component's `?replay=` handling.

---

## 8. Training & comparing models (workflow)

Training is a **campaign harness** (`ITrainingCampaign` + `CampaignRunner`) on a DI host (`AIHost`), driven
by per-game Lab CLIs. Two paradigms share one interface: **goal-reaching** (eval = solve rate; `IsComplete`
stops early) and **score-maximizing** (eval = mean return; runs to the wall-clock budget).

| Path | Role |
|---|---|
| [`Core/Training/ITrainingCampaign.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/ITrainingCampaign.cs) | `Resume` / `TrainChunk` / `Evaluate` / `Checkpoint` / `IsComplete` (+ `TryRunStandaloneEval`). |
| [`Core/Training/CampaignRunner.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/CampaignRunner.cs) | Wall-clock budget, eval/checkpoint cadence, resume lifecycle; **IO-agnostic** (calls an `OnEval` callback). |
| [`…Hosting/AIHost.cs`](../src/MintPlayer.AI.ReinforcementLearning.Hosting/AIHost.cs) | `AIHost.CreateBuilder(dataDir).Build()` → DI with `IModelStore`, `TimeProvider`, `CampaignRunner`. |
| [`…Ilgpu.Hosting/`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting/) | `services.AddGpuBackend()` registers `AdaptiveBackend` (opt-in; only large nets benefit). |
| [`tools/…Lab/Program.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/Program.cs) | `--game <name>` dispatch → per-game Lab. |
| [`tools/…Lab/FruitCakeLab.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/FruitCakeLab.cs) (+ [`…Campaign.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/FruitCakeDqnCampaign.cs)) | Flag parsing + the campaign; flags incl. `--hours`/`--steps`/`--seed`/`--lr`/`--gamma`/`--nstep`/`--shape`/`--noisy`/`--curriculum`/`--ab`/`--search-eval`. |
| [`tools/…Lab/FruitCakeAb.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/FruitCakeAb.cs) | Paired-seed A/B of two nets → mean±SD, paired Δ±SE, verdict. |
| [`tools/…Lab/FruitCakeSearchEval.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/FruitCakeSearchEval.cs) | Same net, **search vs greedy** on paired seeds → score + max-tier distribution + watermelon count. |
| [`tools/…Lab/CampaignCli.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CampaignCli.cs) | `ConsoleAndCsv(path)` — the `OnEval` IO bridge (console + CSV). |

```bash
# Train (score-maximizing, resumable): writes <data>/<env>.dqn.ckpt (deployable, save-best) + .dqn-state.ckpt (resume)
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game fruitcake --hours 2 --data ./data --seed 1 --lr 5e-4 --gamma 0.997 --nstep 3 --shape
# Resume: rerun the SAME command (auto-resumes from .dqn-state.ckpt; raises the absolute MaxSteps).

# Compare two nets (paired-seed A/B with a statistical verdict)
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game fruitcake --ab --baseline ./models --data ./candidate --ab-episodes 200

# Search vs greedy on one net (does inference lookahead help?)
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game fruitcake --search-eval --data ./data --depth 3 --topk 5 --topk2 2
```

**Resume contract:** a campaign saves a *deployable* net (`<env>.dqn.ckpt`, **save-best guarded** for noisy
DQN eval) and a *full resume state* (`<env>.dqn-state.ckpt`: optimizer + replay buffer + RNG + env snapshot).
Re-running continues bitwise-identically. The cube's self-taught DAVI campaign has its own recipe in
[`tools/…/Lab/CUBE_CAMPAIGN.md`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CUBE_CAMPAIGN.md).

---

## 9. Game solving stacks — deep dives

Beyond the high-level trainer/search map (§4), three games have substantial multi-stage solving pipelines
worth their own walkthrough. (FruitCake's is in §4 + §6; CartPole / MountainCar / Snake are single-net.)

### Rubik's Cube solving stack

Three deployable solvers: a **Kociemba two-phase** C# port (always-available oracle, ≤22 HTM, also the
imitation teacher), a **Kociemba-imitation policy net** + beam search, and a **teacher-free DAVI value net**
(EfficientCube-style) + value-guided A\*. The website serves the teacher-free policy net by default.

| Path (`Environments/RubiksCube/` unless noted) | Role |
|---|---|
| [`RubiksCubeEnv.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/RubiksCubeEnv.cs) | 324-dim one-hot sticker obs, 12 quarter-turns, inverse-move mask, curriculum scramble depth. |
| [`Kociemba/`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/Kociemba/) + [`CubeSolver.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubeSolver.cs) | Two-phase oracle (pruning tables built once, thread-safe); ≤22 HTM. |
| [`FaceletCube.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/FaceletCube.cs) | 54-sticker model + quarter-turn cycles + Kociemba-string conversion. |
| [`CubeOracle.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubeOracle.cs) | Scramble → Kociemba solve → labeled `(state, action, dist)` trajectory for imitation. |
| [`CubePolicyNet.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubePolicyNet.cs) + [`CubePolicySearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubePolicySearch.cs) | Two-headed net (move logits + distance); greedy / A\* / **beam search** (~2000 wide). |
| [`CubeQSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubeQSearch.cs) | Q-guided A\* for the masked Double-DQN net. |
| [`CubeValueSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RubiksCube/CubeValueSearch.cs) (+ [`Core/Planning/ValueGuidedSearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Planning/ValueGuidedSearch.cs)) | Batch-weighted A\* over a learned cost-to-go (~depth-15 optimal in budget). |
| [`Ilgpu/DeviceResidualMlp.cs`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu/DeviceResidualMlp.cs) + [`DeviceResidualTrainer.cs`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu/DeviceResidualTrainer.cs) | **GPU-resident** forward/train: weights stay on device, only batches cross the bus. |
| `tools/…Lab/` [`Davi`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CubeDaviCampaign.cs) / [`Efficient`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CubeEfficientCampaign.cs) / [`Imitation`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CubeImitationCampaign.cs)`Campaign.cs` | The `--game cube-davi` / `cube-policy` / `cube` campaigns. |
| [`RLDemo.Web/Controllers/CubeController.cs`](../src/RLDemo.Web/Controllers/CubeController.cs) | `/solve` (Kociemba) and `/solve-efficient` (policy beam, GPU-resident forward w/ CPU fallback). |

The cube is the only game heavy enough to use the GPU backend (the small game nets stay on CPU). Full recipe +
wall-clock: [`tools/…/Lab/CUBE_CAMPAIGN.md`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/CUBE_CAMPAIGN.md).
*Change it:* representation in `FaceletCube`/`RubiksCubeEnv`; search budgets in `Cube*Search.cs`; curriculum/LR
in the cube campaigns.

### Rush Hour solving stack

An exact **BFS oracle** labels states, an **imitation campaign** trains a two-headed policy/value net (with
DAgger on-policy mixing), and inference solves user boards with **greedy rollout → policy-guided A\*** (the value
head is the heuristic) — optimally on the boards tested.

| Path | Role |
|---|---|
| [`Environments/RushHour/RushHourEnv.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourEnv.cs) + [`RushHour.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHour.cs) | 72-dim two-plane obs, 32 actions (vehicle×dir), masking, puzzle-per-episode. |
| [`RushHourSolver.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourSolver.cs) / [`RushHourOracle.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourOracle.cs) | BFS optimal solver + full-graph labeler (forward + multi-source backward BFS → dist-to-goal + all-optimal-actions mask). |
| [`RushHourGenerator.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourGenerator.cs) | Seeded solvable-puzzle generator within a difficulty band. |
| [`RushHourPolicyNet.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourPolicyNet.cs) | Two-headed net (32 move logits + scalar distance), hidden 384. |
| [`RushHourPolicySearch.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/RushHour/RushHourPolicySearch.cs) | Greedy rollout (cycle-aware) + policy-guided A\* (`h` = value head). |
| [`tools/…Lab/RushHourImitationCampaign.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Lab/RushHourImitationCampaign.cs) | Soft-CE over the optimal-action mask + Huber distance; ~50% on-policy DAgger; resumable. |
| [`RLDemo.Web/Services/RushHourRollout.cs`](../src/RLDemo.Web/Services/RushHourRollout.cs) + [`Controllers/RushHourController.cs`](../src/RLDemo.Web/Controllers/RushHourController.cs) | `/analyze` (BFS optimal), `/solve` (greedy\|search\|dqn); returns AI + optimal trajectories for playback. |

*Change it:* obs in `RushHourBoard.WriteObservation`; net capacity in `RushHourPolicyNet`; oracle budget + DAgger
mix in the imitation campaign; A\* budget in `RushHourPolicySearch`.

### 2048 — n-tuple afterstate TD + expectimax

A classic **n-tuple TD(0)** learner (Szubert & Jaśkowski) over 17 lookup tables, paired with a test-time
**expectimax** that averages the learned value over the random tile spawn (the chance node). A generic masked
Double-DQN (`2048dqn`) is the NN baseline on the same env.

| Path (`Environments/Game2048/`) | Role |
|---|---|
| [`Game2048.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/Game2048/Game2048.cs) / [`Env2048.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/Game2048/Env2048.cs) | 4×4 exponent grid + spawn (2@0.9 / 4@0.1); 16-dim obs, 4 actions, masking. |
| [`NTuple2048Agent.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/Game2048/NTuple2048Agent.cs) | **Afterstate** TD(0): 17 tables (4 rows + 4 cols + 9 2×2, 65 536 entries each); `ChooseMove` = argmax `r + V(afterstate)`. |
| [`Expectimax2048.cs`](../src/MintPlayer.AI.ReinforcementLearning.Environments/Game2048/Expectimax2048.cs) | Search over `V`: max nodes (pick move) alternate with chance nodes (avg over spawn); depth-1 default (~1.9× greedy); transposition-memoized. |
| [`Services/Game2048ModelService.cs`](../src/RLDemo.Web/Services/Game2048ModelService.cs) + [`Controllers/Game2048Controller.cs`](../src/RLDemo.Web/Controllers/Game2048Controller.cs) | Loads the `2048/ntuple` table checkpoint; `/solve` runs expectimax on user boards. |

**Afterstate vs state value:** the agent learns `V(after)` — the board *after* the slide/merge but *before* the
random spawn — so the uncontrollable spawn is factored out; the TD target is `r + V(after)`, undiscounted (γ=1,
total-score objective). Expectimax reintroduces the spawn as an explicit chance node at search time.
*Change it:* tile patterns in `BuildTuples`; learning rate `Alpha`; depth/pruning in `Expectimax2048`; reward
form in `Env2048.Step`.

---

## 10. Client-side game engines (human play)

"Play yourself" runs **entirely in the browser** — a pure-TS engine per game ticks on `requestAnimationFrame`
(canvas games) or `setInterval` (turn-based), no backend in the loop. The browser rules mirror the C# training
env so human and AI obey identical mechanics. FruitCake goes furthest: its **physics is literally the same code
as the C# env** — one MintPlayer.Polyglot `.pg` transpiled to both (callout below) — while the others are thin
logic modules mirroring the C# envs by hand.

| Path (`RLDemo.Web/ClientApp/src/app/`) | Role |
|---|---|
| [`fruit-cake/fruitcake_solver.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruitcake_solver.ts) | **Generated (committed) — do not edit.** TS transpilation of `fruitcake_solver.pg`: `PgFruitCakeWorld` physics **plus** the whole inference path — `buildObservation`, `PgDuelingNet.forward`, `chooseColumn` (depth-3 search). Edit the `.pg` + regenerate. |
| [`fruit-cake/fruitcake-net.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruitcake-net.ts) | Parses the shipped `.ckpt` binary (mirrors the C# `DuelingQNetCheckpoint` reader) → builds `PgDuelingNet`. The one inference piece that isn't Polyglot (binary I/O). |
| [`fruit-cake/fruit-cake-director.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-director.ts) | **Client-side "watch AI"** (M32): a real-time state machine that runs the generated physics + `chooseColumn` over the loaded net locally — replaces the retired server WebSocket stream. Emits `FruitCakeFrame`s ([`fruit-cake-frame.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-frame.ts)) for `renderFrame`. |
| [`fruit-cake/fruit-cake-physics.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-physics.ts) | Thin **`FruitWorld` facade** over the generated core — **not the physics**. Re-adds host-only surface: `onMerged` (exact, from the core's merge list), `onLanded` (host-side approximation), and per-fruit `mergeBorn`/age via a side-table. Edit the `.pg`, not this. |
| `fruit-cake/` [`game`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-game.ts) · [`render`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-render.ts) · [`audio`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-audio.ts) · [`fruits`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-fruits.ts) | Rules + localStorage; Canvas 2D art + HUD + letterbox scaling; Web Audio; the 11-tier merge catalog (render/scoring; a second copy of the catalog also lives inside the `.pg`). |
| [`fruit-cake/fruit-cake.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake.ts) (component) | Fixed-timestep rAF loop, pointer aim/drop, `mode` signal (human ↔ watch), fullscreen. |
| [`snake-logic.ts`](../src/RLDemo.Web/ClientApp/src/app/snake/snake-logic.ts), [`mountaincar-logic.ts`](../src/RLDemo.Web/ClientApp/src/app/mountaincar/mountaincar-logic.ts), [`game-2048-logic.ts`](../src/RLDemo.Web/ClientApp/src/app/game-2048/game-2048-logic.ts), [`rush-hour-logic.ts`](../src/RLDemo.Web/ClientApp/src/app/rush-hour/rush-hour-logic.ts) | Browser-side rules for human play, mirroring the C# envs. |
| [`cube/cube.ts`](../src/RLDemo.Web/ClientApp/src/app/cube/cube.ts) | Three.js scene + manual turns + Kociemba/AI solver playback. |
| [`screen-wake-lock.ts`](../src/RLDemo.Web/ClientApp/src/app/screen-wake-lock.ts) | Shared service: holds a screen wake lock during watch-AI; re-acquires on foreground. |

Each component has a `mode` signal: `'human'` (browser timer) vs `'watch'` (server WebSocket stream).
Conventions: plain `fetch`/`WebSocket` (no `HttpClient`/`environment.ts`), standalone components + signals,
canvas loops run **outside** Angular's zone.
*Change it:* game rules → the `*-logic.ts` (keep the C# env in sync — see the PRD-sync comments); render/input →
`*-render.ts` or the component; new game → add a folder + route (+ server controller/service).

**Single-source FruitCake physics (MintPlayer.Polyglot).** The FruitCake solver is written **once** in
[`…/Environments/FruitCake/polyglot/fruitcake_solver.pg`](../src/MintPlayer.AI.ReinforcementLearning.Environments/FruitCake/polyglot/fruitcake_solver.pg)
and transpiled to **C#** (build-time via the `MintPlayer.Polyglot.MSBuild` PackageReference → `obj/`, wrapped by the
public `FruitCakeWorld` **facade**) and to **TypeScript** (committed `fruitcake_solver.ts` here, wrapped by the
`FruitWorld` **adapter**). Both targets are byte-identical (f64). **To change the physics, edit the `.pg` and
regenerate** — never the generated `.cs`/`.ts` or the facades' physics; the facades hold only host glue (events,
rendering hooks, RNG, state I/O). See `…/FruitCake/polyglot/README.md` and
[`prd/POLYGLOT_FRUITCAKE_PRD.md`](prd/POLYGLOT_FRUITCAKE_PRD.md).

**The `.pg` now holds the whole *inference* path too (M32).** Beyond physics, `fruitcake_solver.pg` also contains
`buildObservation` (the 89-dim vector), `PgDuelingNet.forward` (f64 dueling-Q forward pass), and `chooseColumn`
(the depth-3 expectimax search with the net-leaf inlined) — so the **entire FruitCake AI is single-sourced** and, f64
on both sides, byte-identical in C# and the browser. Consequently **watch-AI runs fully client-side**
([`fruit-cake-director.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruit-cake-director.ts) over the generated
TS core + the shipped [`wwwroot/models/fruitcake-net.ckpt`](../src/RLDemo.Web/wwwroot/models/fruitcake-net.ckpt),
parsed by [`fruitcake-net.ts`](../src/RLDemo.Web/ClientApp/src/app/fruit-cake/fruitcake-net.ts)); there is no
FruitCake server controller or net. Training stays C#/SDK-only (autograd/GEMM/GPU) and *produces* the weights; the
only per-platform inference piece is the binary `.ckpt` parser (C# `DuelingQNetCheckpoint`; TS `fruitcake-net.ts`).
See [`prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md`](prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md).

**Serving the browser weights.** The `.ckpt` files live in `src/RLDemo.Web/wwwroot/models/` (LFS) and are fetched
by the browser from `/models/*-net.ckpt`. ASP.NET's static-file middleware refuses **unknown extensions**, so
`Program.cs` registers a `.ckpt` → `application/octet-stream` mapping on `UseStaticFiles`; without it the request
falls through to the SPA `index.html` (a 200 of HTML), the parser rejects it, and the in-browser AI silently gets
no net. Any new browser-served extension needs the same mapping. (`wwwroot` is served identically in dev and prod,
so this is testable with a local `curl /models/<net>.ckpt`.)

---

## 11. Console demo, Bench & source-generated DI

| Path | Role |
|---|---|
| [`src/RLDemo.Console/Program.cs`](../src/RLDemo.Console/Program.cs) | CLI demo: arg dispatch (`grid`/`lake`/`cartpole`/`ppo`/`2048`/`2048dqn`/`rushhour`/`cube` + seed + `--save`/`--load`/`--data`); trains then animates greedy console playback. |
| [`tools/…Bench/Program.cs`](../tools/MintPlayer.AI.ReinforcementLearning.Bench/Program.cs) | Throughput harness: GEMM ops/s + GFLOP/s, full Adam train-step rate, thread-scaling sweep, GPU (ILGPU) host-span vs resident kernels; gates the CPU GEMM target. |
| [`…Core/Training/CampaignRunner.cs`](../src/MintPlayer.AI.ReinforcementLearning.Core/Training/CampaignRunner.cs) | Carries `[Register(...)]` → source-generates `AddReinforcementLearningCore()`. |
| [`…Hosting/AIHost.cs`](../src/MintPlayer.AI.ReinforcementLearning.Hosting/AIHost.cs) | `AddReinforcementLearning(dataDir)` composes the generated registration + `FileModelStore` + `TimeProvider`; `AIHost.CreateBuilder()` is the factory used by Lab/console. |
| [`…Ilgpu.Hosting/GpuBackendServiceCollectionExtensions.cs`](../src/MintPlayer.AI.ReinforcementLearning.Ilgpu.Hosting/GpuBackendServiceCollectionExtensions.cs) | Optional `AddGpuBackend()` (AdaptiveBackend). |

DI is **dogfooded via `MintPlayer.SourceGenerators`**: a `[Register(...)]` attribute on a class generates the
`Add…Core()` extension, so Core carries no DI-framework dependency while Hosting / web / Lab compose it.

```csharp
[Register(ServiceLifetime.Singleton, "ReinforcementLearningCore")]   // → generated AddReinforcementLearningCore()
public sealed class CampaignRunner(TimeProvider? timeProvider = null) { … }
```

*Change it:* add a console section to `knownSections`; bench targets in the Bench harness; a new DI service → add
`[Register(...)]` to the class (the extension regenerates).

---

## 12. Tests & conventions

- **Tests** (`tests/…Tests/`, xUnit + `Microsoft.AspNetCore.Mvc.Testing`): solve-threshold gates,
  determinism/round-trip tests, web API contract tests. Long-running tests carry `[Trait("Category","Slow")]`;
  **CI runs `Category!=Slow`** — the fast loop is `dotnet test --filter "Category!=Slow"`. `Core` (and `Ilgpu`)
  expose internals to the test project via `InternalsVisibleTo`.
- **Conventions**: from-scratch/pure-managed (no Python/ML deps); `.ckpt` via Git LFS; `net10.0` + nullable;
  RNG always `Xoshiro256StarStar` (never `System.Random`) for cross-version determinism; only the five
  libraries are NuGet-published; observation construction is a shared static method (train == serve).

---

## 13. Where to change things — quick map

| I want to… | Touch |
|---|---|
| Add an NN layer / activation | `Core/Nn/` (new `IModule`); add the op to `IComputeBackend` + `ManagedBackend` (+ ILGPU kernel) if it's new. |
| Add a backend op | Declare in `IComputeBackend`, implement in `ManagedBackend` (reference) + `Ilgpu`, route in `AdaptiveBackend`. |
| Switch to GPU / force CPU | Set `Backend.Current = new AdaptiveBackend()` (or `new ManagedBackend()`) at startup. |
| Add / tune a trainer | `Core/Training/` (`DqnOptions` knobs, or a new `…Trainer.cs`). |
| Add a new environment | Implement `IEnvironment<TObs,TAct>` (+ `IActionMaskProvider` / `IStatefulEnvironment` as needed); build obs via a static method. See [ADDING_A_GAME.md](ADDING_A_GAME.md). |
| Add inference search to a game | New `…Search.cs` in the game folder; inject the net value/policy as a delegate (keep it Core-free); wire into the serving/CLI call site. |
| Evolve a checkpoint format | Bump the version constant; append fields; read conditionally with a default (§5). |
| Add a game to the web | Env → `…ModelService` → controller (`/status` + `/live`) → DI in `Program.cs` → `<game>-api.ts` + component + route. See [ADDING_A_GAME.md](ADDING_A_GAME.md). |
| Add a training campaign | `ITrainingCampaign` + a `<game>Lab` + dispatch in `Lab/Program.cs`; optional A/B + search-eval harnesses. |
| Release the SDK | Bump `<RLNetVersion>` in `src/Directory.Build.props` (lockstep across the five libs). |
