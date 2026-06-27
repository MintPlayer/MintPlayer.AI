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
9. [Tests & conventions](#9-tests--conventions)
10. [Where to change things — quick map](#10-where-to-change-things--quick-map)

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
| `Numerics/Tensor.cs` | Tape autograd: forward ops record parents + a backward closure; `Backward()` runs them in reverse topological order, accumulating `Grad`. `GradMode.NoGrad()` (thread-local) disables recording. |
| `Numerics/IComputeBackend.cs` | The seam: GEMM (+ transpose variants), elementwise `Map`/`MapBackward`, reductions (`Sum`, `LogSoftmax`, `Gather`, `HuberLoss`, `LayerNorm`). Forward ops **write**; backward ops **accumulate**. |
| `Numerics/ManagedBackend` | Pure-managed CPU backend (`TensorPrimitives` SIMD). Large GEMMs partition output rows across `Parallel.For` workers (no reduction → bitwise-identical to sequential); `maxDegreeOfParallelism` caps it. |
| `Nn/Adam.cs` | Adam with bias correction; mutable `LearningRate`; `ClipGradNorm`. |
| `Nn/Modules.cs` | `IModule` (`Forward`/`Parameters`), `IValueNet` (`InputSize`/`CloneStructure`/`CopyFrom`), `Linear`, `Mlp`. |
| `Nn/DuelingQNet.cs` | Shared trunk → value + advantage heads, `Q = V + (A − mean A)`; optional noisy heads. |
| `Nn/NoisyLinear.cs` | Factorized-Gaussian noisy layer (μ + σ⊙ε; ε is a non-grad constant resampled per step — learned σ only). |
| `Nn/ResidualMlp.cs` | Deep residual value net (LayerNorm blocks) + Net2WiderNet growth (`WidenTo`). |
| `Random/Xoshiro256StarStar.cs` | Version-stable PRNG (never `System.Random`); `GetState`/`SetState` for checkpointing. |
| `Random/SeedSequence.cs` | One master seed → independent streams via `RngStreams` (`Environment`, `Policy`, `Init`, `Buffer`, `Evaluation`, `Noise`, …). |
| `src/…Ilgpu/` | GPU backend: tiled shared-memory GEMM; `AdaptiveBackend` routes each GEMM to CPU or GPU by a MAC threshold and falls back to CPU when no device is present. Only large nets (cube) clear the threshold. |

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

---

## 3. Environments — the game/task layer

Every environment implements a Gymnasium-faithful contract. The **terminated-vs-truncated split** is the
key correctness detail: *terminated* = a true terminal state (bootstrap value 0); *truncated* = an external
cutoff (time limit) where the value target **must still bootstrap** from the final observation. Conflating
them is the classic silent DQN bug.

| Interface (`src/…Core/Environments/`) | Purpose |
|---|---|
| `IEnvironment<TObs,TAct>` | `Reset(seed)` / `Step(action)` → `StepResult` (`Observation`, `Reward`, `Terminated`, `Truncated`, `Info`); `ObservationSpace`/`ActionSpace`; `RenderString()`. |
| `Space.cs` | `BoxSpace` (continuous ℝⁿ), `DiscreteSpace` (0..N−1). |
| `IActionMaskProvider` | `CurrentActionMask()` → per-state legal actions; trainers filter exploration **and** the TD-target argmax. |
| `IStatefulEnvironment` | `SaveState()`/`RestoreState()` — opaque `byte[]` snapshot for bitwise-exact resume. |
| `IRestartableEnvironment<TObs>` | `IStatefulEnvironment` + `CurrentObservation()` — lets a trainer begin an episode from a restored snapshot (reverse-curriculum). |

| Game | Obs dims | Actions | Notes |
|---|---|---|---|
| CartPole | 4 | 2 | Bit-exact Gymnasium port. |
| MountainCar | 2 | 3 | Momentum-building control. |
| Snake | 177 | 4 | Egocentric 9×9 patch + scalars + **flood-fill** free-cell reachability (anti-self-trap). |
| FruitCake | 83 | 14 | Suika merge physics; reward shaping (training-only); see below. |
| 2048 | 16 | 4 | Log-scaled tiles; action masking. |
| RushHour | 72 | 32 | Two 6×6 planes; puzzle-per-episode; BFS oracle for imitation. |
| RubiksCube | 324 | 12 | One-hot stickers; inverse-move masking; curriculum depth. |
| GridWorld / FrozenLake | 16 | 4 | Tabular; value-iteration ground truth for tests. |

**The static `BuildObservation` pattern.** Observation construction is a *static* method so the live
serving path feeds the net the **byte-identical** observation the policy trained on (e.g.
`FruitCakeEnv.BuildObservation(world, current, next)` is called by both `Step()` and the WebSocket frame
builder). When you change an observation, you change it in exactly one place and bump the net's input width
(the model-load guard rejects a stale-width checkpoint).

---

## 4. Trainers & algorithms

Model-free trainers (off-policy DQN, on-policy PPO, tabular) live in `Core/Training`; planning over a
learned value net lives in `Core/Planning`; inference-time **search** lives next to each game in
`Environments`.

| Path | Role |
|---|---|
| `Core/Training/DqnTrainer.cs` | Double/Dueling DQN: target net, replay buffer, ε-greedy **or** NoisyNets, n-step returns, optional reverse-curriculum starts. Single-file CleanRL-style reference. |
| `Core/Training/ReplayBuffer.cs` | Circular `(s,a,r,s′,terminated, next-mask)` store. Stores `terminated` **only** (never the combined done flag). |
| `Core/Training/NStepAccumulator.cs` | Folds n-step returns before buffering: reward → Σ discounted next-n; bootstrap → γⁿ. Handles terminal vs truncation at episode ends. |
| `Core/Training/PpoTrainer.cs` | PPO: vectorized envs, GAE(λ), clipped surrogate, orthogonal init, LR anneal. |
| `Core/Training/Evaluator.cs` | Greedy evaluation runner (returns, lengths, success rate). |
| `Core/Planning/ValueIterationTrainer.cs` | Teacher-free deep approximate value iteration (DAVI / DeepCubeA style) over a forward model. |
| `Core/Planning/ValueGuidedSearch.cs`, `GreedyValuePlanner.cs` | Weighted-A\* / greedy solve over a learned cost-to-go. |
| `Environments/FruitCake/FruitCakeSearch.cs` | Depth-1→3 forward search; known current+next maximize, the 3rd ply is an **expectimax chance node** over the unknown fruit. |
| `Environments/Game2048/Expectimax2048.cs` | Expectimax over the n-tuple afterstate value (chance node = the random tile spawn). |
| `Environments/RubiksCube/CubePolicySearch.cs`, `CubeValueSearch.cs` | Beam / value-guided search over the cube nets. |

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
| `CheckpointFormat.cs` | Magic `"RLNC"` (`0x434E4C52`) + kind string + `int32` version; primitives for floats/ints/bools/RNG; `ReadHeader` validates and returns the version for back-compat branching. |
| `DqnTrainingState.cs` | Full training state (kind `dqn-state`): nets, optimizer, replay buffer, RNGs, obs, env snapshot, n-step window. |
| `DuelingQNetCheckpoint.cs` / `MlpCheckpoint.cs` / `ResidualMlpCheckpoint.cs` | Per-architecture net weights (tagged via `QNetCheckpoint`). |
| `AdamCheckpoint.cs`, `ReplayBufferCheckpoint.cs` | Optimizer moments / algorithm-agnostic transition payload (embedded header-less in `dqn-state`). |
| `ModelStore.cs` | `IModelStore` + `FileModelStore`: files named `<envId>.<algoId>.ckpt` under a root dir; atomic save (temp + rename). |

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
- FruitCake uses a **bespoke intra-drop streamer** in `FruitCakeController` (the agent decides once per *drop*, but ~30 fps of falling/merging is streamed *between* decisions; the drop column is chosen by `FruitCakeSearch`).

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
| `RLDemo.Web/Program.cs` | Host, SPA dev-server middleware, model-service DI, WebSocket setup, optional GPU backend. |
| `RLDemo.Web/Services/*ModelService.cs` | `IModelStartupService` — load a checkpoint at startup, **guard input width**, expose a `GreedyQAgent`, report `ModelStatus`. |
| `RLDemo.Web/Controllers/*.cs` | `/<game>/status`, `/<game>/live` (WebSocket), `/version` (build identity). |
| `RLDemo.Web/ClientApp/src/app/<game>/` | Standalone Angular (signals) component per game: Play-Human (client loop) + Watch-AI (WebSocket); canvas render. |
| `RLDemo.Web/ClientApp/src/app/screen-wake-lock.ts` | Shared service: holds a Screen Wake Lock during Watch-AI so mobile screens don't sleep; reconnects the stream on foreground. |
| `Dockerfile`, `docker-compose.yml` | Multi-stage build (Node build → `dotnet publish -p:EnableSpaBuilder=false` → ASP.NET runtime); VPS/Traefik deploy compose. |
| `.github/workflows/` | `pull-request.yml` (build + `Category!=Slow` tests + `npm run build`), `build-master.yml` (NuGet pack/push), `playground-docker.yml` (image → GHCR → SSH deploy to VPS, writing `BUILD_SHA`/`IMAGE_DIGEST`/`DEPLOY_TIME`). |

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

---

## 8. Training & comparing models (workflow)

Training is a **campaign harness** (`ITrainingCampaign` + `CampaignRunner`) on a DI host (`AIHost`), driven
by per-game Lab CLIs. Two paradigms share one interface: **goal-reaching** (eval = solve rate; `IsComplete`
stops early) and **score-maximizing** (eval = mean return; runs to the wall-clock budget).

| Path | Role |
|---|---|
| `Core/Training/ITrainingCampaign.cs` | `Resume` / `TrainChunk` / `Evaluate` / `Checkpoint` / `IsComplete` (+ `TryRunStandaloneEval`). |
| `Core/Training/CampaignRunner.cs` | Wall-clock budget, eval/checkpoint cadence, resume lifecycle; **IO-agnostic** (calls an `OnEval` callback). |
| `…Hosting/AIHost.cs` | `AIHost.CreateBuilder(dataDir).Build()` → DI with `IModelStore`, `TimeProvider`, `CampaignRunner`. |
| `…Ilgpu.Hosting/…` | `services.AddGpuBackend()` registers `AdaptiveBackend` (opt-in; only large nets benefit). |
| `tools/…Lab/Program.cs` | `--game <name>` dispatch → per-game Lab. |
| `tools/…Lab/FruitCakeLab.cs` (+ `…Campaign.cs`) | Flag parsing + the campaign; flags incl. `--hours`/`--steps`/`--seed`/`--lr`/`--gamma`/`--nstep`/`--shape`/`--noisy`/`--curriculum`/`--ab`/`--search-eval`. |
| `tools/…Lab/FruitCakeAb.cs` | Paired-seed A/B of two nets → mean±SD, paired Δ±SE, verdict. |
| `tools/…Lab/FruitCakeSearchEval.cs` | Same net, **search vs greedy** on paired seeds → score + max-tier distribution + watermelon count. |
| `tools/…Lab/CampaignCli.cs` | `ConsoleAndCsv(path)` — the `OnEval` IO bridge (console + CSV). |

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

## 9. Tests & conventions

- **Tests** (`tests/…Tests/`, xUnit + `Microsoft.AspNetCore.Mvc.Testing`): solve-threshold gates,
  determinism/round-trip tests, web API contract tests. Long-running tests carry `[Trait("Category","Slow")]`;
  **CI runs `Category!=Slow`** — the fast loop is `dotnet test --filter "Category!=Slow"`. `Core` (and `Ilgpu`)
  expose internals to the test project via `InternalsVisibleTo`.
- **Conventions**: from-scratch/pure-managed (no Python/ML deps); `.ckpt` via Git LFS; `net10.0` + nullable;
  RNG always `Xoshiro256StarStar` (never `System.Random`) for cross-version determinism; only the five
  libraries are NuGet-published; observation construction is a shared static method (train == serve).

---

## 10. Where to change things — quick map

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
