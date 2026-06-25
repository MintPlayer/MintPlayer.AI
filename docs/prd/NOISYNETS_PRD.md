# NoisyNets — Product Requirements Document & Plan

> Add **NoisyNets** (learned, state-dependent exploration; Fortunato et al. 2017,
> [arXiv:1706.10295](https://arxiv.org/abs/1706.10295)) to the from-scratch
> `MintPlayer.AI.ReinforcementLearning` library as an opt-in DQN capability, with **FruitCake** as
> the first adopter.

- **Status:** Draft v1.0 · 2026-06-25 (investigation complete; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** the DQN stack (`DqnTrainer`, `DuelingQNet`, `DuelingQNetCheckpoint`) and the
  FruitCake AI work (`docs/prd/FRUITCAKE_AI_PRD.md`). Companion to [PRD.md](PRD.md) §3/§8 (algorithm
  ladder + asset portability).

---

## 1. Summary & Vision

DQN here explores with **ε-greedy** — undirected, uniform random actions that anneal to a 0.05 floor.
On a long-horizon, stochastic game (FruitCake/Suika), that dithering rarely stumbles onto a coherent
*better* multi-step strategy, and **continued/refinement training at low ε mostly polishes the
current policy rather than discovering a higher peak** (the premature-convergence concern raised in
the prior investigation). **NoisyNets** replaces ε-greedy with *learned, state-dependent* exploration:
each linear layer carries trainable noise-scale parameters (σ) alongside its means (μ); the network
explores by perturbing its own weights, and σ is learned by backprop so exploration **self-anneals**
where the policy is confident and persists where it isn't.

This is the library's first **principled exploration upgrade** beyond ε-greedy — a reusable SDK
capability (the library is the deliverable; FruitCake is the showcase), and a Rainbow building block
for later.

---

## 2. Goals & Non-Goals

### Goals
- A from-scratch **`NoisyLinear`** layer (factorized Gaussian) implementing `IModule`, learnable via
  the existing autograd.
- A **`noisy` option on `DuelingQNet`** (value/advantage heads become noisy) and a **`DqnOptions.NoisyNets`**
  training flag that resamples noise correctly and disables ε-greedy.
- **Deterministic inference** (noise off → mean weights) so serving/eval is unchanged and reproducible.
- **Backward-compatible checkpoints** — every already-shipped `*.dqn.ckpt` (snake/cube/fruitcake) keeps
  loading unchanged.
- **FruitCake as the first consumer** via a Lab `--noisy` flag, measured against the ε-greedy baseline.

### Non-Goals (v1)
- Conv layers / image observations (the library has no CNN; NoisyNets goes on the FC/Dueling heads).
- The **independent** Gaussian variant (use factorized — cheaper, standard).
- Other Rainbow components (prioritized replay, C51, n-step) — separate future work.
- Making `Mlp` / non-Dueling paths noisy (defer; FruitCake/Snake use Dueling).
- Policy-gradient entropy bonuses (a PPO/SAC tool, not value-based DQN).
- Re-shipping any model unless a noisy run **beats** the current best (keep-best gating decides).

---

## 3. Feasibility verdict (the gating question)

**Clean drop-in — zero new autograd ops required.** The factorized-noise transform
(`f(x)=sgn(x)·√|x|`) and the outer product are computed on **freshly-sampled constants in ordinary
C#**, so the autograd graph only ever sees ops that already exist: `Mul` (`TensorOps.cs:89`), `Add`
(`:35`), `MatMul` (`:9`), `AddBias` (`:49`). A constant noise tensor (`new Tensor(data, shape)`,
`RequiresGrad=false`, no parents → `NeedsGrad=false`) multiplied into a σ Parameter yields gradients
to **σ only**, never to the noise (`Mul` backward accumulates only into parents that `NeedsGrad`,
`TensorOps.cs:97-106`) — exactly NoisyNets' requirement. The only non-autograd plumbing needed is a
~3-line uniform-init helper (`Init.cs` currently has only `Orthogonal`).

---

## 4. Design

### 4.1 `NoisyLinear : IModule` (`Core/Nn/`)
Replaces `y = Wx + b` with learnable mean **and** noise-scale:

```
W = μ_w + σ_w ⊙ ε_w        b = μ_b + σ_b ⊙ ε_b
```

- **Parameters** (all `RequiresGrad=true`, yielded by `Parameters()`): `μ_w, σ_w` [in,out]; `μ_b, σ_b` [out].
  Adam updates σ with no special-casing (`Adam.cs:16-24`).
- **Factorized noise** (only `in+out` randoms): sample `ε_in∈ℝ^in`, `ε_out∈ℝ^out` ~ N(0,1);
  `f(x)=sgn(x)·√|x|`; `ε_w = f(ε_out) ⊗ f(ε_in)`, `ε_b = f(ε_out)`. Built in plain C# as **constant**
  tensors in `ResampleNoise(rng)`.
- **Init** (p = in): `μ ~ U[−1/√p, 1/√p]`, `σ = σ₀/√p` with **σ₀ = 0.5**.
- **Forward:** sampling mode → `x.MatMul(μ_w + σ_w⊙ε_w).AddBias(μ_b + σ_b⊙ε_b)`; **eval mode** (`NoiseEnabled=false`)
  → `x.MatMul(μ_w).AddBias(μ_b)` (deterministic means).
- `NoiseEnabled` **defaults false** (so any loaded net is deterministic unless training turns it on).
- Excludes the ε buffers from `Parameters()` (not learnable, not serialized).

### 4.2 `DuelingQNet` integration (`Core/Nn/DuelingQNet.cs`)
- New `bool noisy = false` ctor arg; when set, build the **value + advantage heads** as `NoisyLinear`
  (trunk noisy is optional, deferred). Retype the head fields to `IModule` (the `Forward`/`.Relu()`
  body is unchanged — both layers are `IModule`).
- Store `InputSize` as a field (today it reads `_trunk[0].Weight.Rows`, which a noisy layer wouldn't expose).
- Add `public bool Noisy { get; }`, `ResampleNoise(rng)` and `SetNoiseEnabled(bool)` that fan out to the
  `NoisyLinear` children, and **thread `noisy` through `CloneStructure()`** so the target net matches.

### 4.3 `DqnTrainer` integration (`Core/Training/DqnTrainer.cs`)
- `DqnOptions.NoisyNets` (default false); `MakeNet` passes it to `new DuelingQNet(..., noisy: NoisyNets)`.
- **Resample cadence** (per forward; online & target **independently**):
  - acting: resample the **online** net before `agent.Act` (`:172`);
  - TD update: resample **online** and **target** at the top of `TrainStep` (before `:209-211, :232`).
  - Add a dedicated `RngStreams.Noise` + `state.NoiseRng` (serialized) for reproducible resume.
- **Disable ε-greedy when noisy:** set `agent.Epsilon = 0` (skip the schedule, `:171`). The greedy
  argmax over a *noise-perturbed* online net **is** the exploration; `GreedyQAgent` is otherwise unchanged.
- **Eval determinism:** wrap eval (`:188-193`) in `SetNoiseEnabled(false) … (true)`; enable noise during
  the training step loop only.

### 4.4 Serialization & backward-compat (`Core/Checkpoints/DuelingQNetCheckpoint.cs`)
- The format already versions cleanly (`CheckpointFormat`: magic `RLNC` + kind + int version,
  `ReadHeader(reader, kind, maxSupportedVersion)`).
- **Bump `Version` 1 → 2:** write order → header → InputSize → HiddenSizes → Actions → **`bool Noisy`** → params.
  `Load`: `bool noisy = version >= 2 && reader.ReadBoolean();` → `new DuelingQNet(..., noisy)` → refill in
  `Parameters()` order (a noisy layer streams 4 tensors instead of 2).
- **v1 files (no flag) → plain net = exactly today's behavior**, so every shipped checkpoint loads unchanged.
- The embedded **resume state** (`DqnTrainingState` → `QNetCheckpoint` + `AdamCheckpoint`) keys off the
  reconstructed net's `Parameters()`, so fixing `DuelingQNetCheckpoint` **automatically** fixes the
  resume + Adam round-trip over 4-param layers.
- Serving: model services already gate on `net.InputSize` (unaffected); for inference they put the net in
  eval/means mode (`NoiseEnabled=false`, the default), so they need no change.

### 4.5 Consumer: FruitCake (`tools/…Lab/`)
- Lab `--noisy` flag → `FruitCakeDqnCampaign(noisy: true)` → `DqnOptions { NoisyNets = true }` and
  `Epsilon = LinearSchedule(0,0,1)` (ε≈0; exploration comes from σ).
- **Reuse the trained model (recommended — no full retrain).** A noisy net has more parameters than a
  plain one (4 tensors/layer at the heads vs 2), so the standard checkpoint `Load` can't ingest the
  shipped `fruitcake.dqn.ckpt` directly (shape mismatch) and the *automatic* warm-start path rejects it.
  But every noisy layer's **mean** params (μ_w, μ_b) correspond exactly to a plain `Linear`'s W/b, so a
  small **promote-plain→noisy** conversion copies the trained weights into μ (trunk copied 1:1; the
  heads' W/b → head μ) and initializes σ fresh (σ₀/√p). With noise off, the result is **behaviorally
  identical** to the current agent, so training continues from the ~741-score policy and merely adds
  learnable exploration — strictly better than cold-start. (A from-scratch noisy run is optional and
  cleaner for a pure baseline-vs-noisy comparison.) **Keep-best gating protects the shipped model**
  throughout either way.
- Other games keep `NoisyNets=false` → untouched.

### 4.6 Measurement
Compare a noisy FruitCake DQN to the ε-greedy baseline at equal env-steps: **eval mean score** and
**mean/median max-tier** over **≥20–50 episodes** (the env is stochastic — control eval variance),
across **multiple seeds**. Log **mean |σ| per noisy layer** to confirm exploration self-anneals (and
doesn't collapse to ~0 prematurely).

---

## 5. Milestone plan

- **N0 — `NoisyLinear` + unit tests.** The module + uniform-init helper. *Gate:* tests pass — σ
  gradients are nonzero after backward; two sampling forwards (with a resample between) differ;
  eval/means mode is deterministic and equals the μ-only result.
- **N1 — `DuelingQNet` noisy + checkpoint v2 + back-compat.** `noisy` ctor, `Noisy`/`ResampleNoise`/
  `SetNoiseEnabled`, `CloneStructure` threading, stored `InputSize`; checkpoint Version 2 with the noisy
  flag. *Gate:* noisy round-trip equal; resume-state (Online/Target/Adam) round-trip equal; **a v1
  (old-layout) checkpoint still loads as a plain net** (committed tiny fixture or a v1-writer helper).
- **N2 — `DqnTrainer` NoisyNets path.** Option + resample cadence (online act, online+target train) +
  `RngStreams.Noise`/`state.NoiseRng` + ε=0 + eval noise-off. *Gate:* a short noisy training smoke
  learns end-to-end (no NaN; mean score rises); a noisy `CampaignContractTests` case (fresh →
  TrainChunk → Checkpoint → fresh Resume continues, asserting `Online.Noisy`).
- **N3 — FruitCake consumer + experiment.** `--noisy` Lab flag + campaign option, **plus a
  promote-plain→noisy conversion** (load the shipped checkpoint → build the noisy net → copy means →
  fresh σ) so the noisy run **warm-starts from the current model** rather than cold-starting. Train (a
  few seeds) and compare to the ε-greedy baseline (§4.6). *Gate:* a clear, multi-seed comparison
  reported (noisy ≥ baseline, or documented if not).
- **N4 — Ship (conditional).** If a noisy net beats the current best, ship its `fruitcake.dqn.ckpt` to
  `models/` via Git LFS (serving stays deterministic — `NoiseEnabled` false). Otherwise keep NoisyNets
  as a validated opt-in capability and leave the shipped model as-is.

---

## 6. Success criteria
- **Capability:** `NoisyLinear` learns (σ gradients flow), trains stably, serializes, and serves
  deterministically — all behind an opt-in flag, with no shipped checkpoint broken.
- **Empirical:** on FruitCake, the noisy agent matches or beats the ε-greedy baseline at equal
  env-steps (multi-seed), and/or escapes the refinement plateau the ε-greedy run is stuck on.
- **SDK:** the capability is game-agnostic — any future DQN game opts in with one flag.

---

## 7. Risks
| Risk | Mitigation |
|---|---|
| **σ collapse** → exploration vanishes | Log mean \|σ\| per layer; if it collapses, raise σ₀ or reduce LR on σ. |
| **Stale / mis-resampled noise** (no exploration, biased grads) | Resampling is an explicit, tested step (online act + online&target per update); covered by the unit + smoke tests. |
| **Checkpoint back-compat regression** | Version-gated read; an explicit **v1-fixture loads-as-plain** test (N1 gate) — the hard requirement. |
| **Noisy can't warm-start the shipped net** | Expected (different shape); `--noisy` is a fresh line; keep-best protects the deployed model. |
| **Eval noise left on** → nondeterministic serving | `NoiseEnabled` defaults **false**; eval wrapped in noise-off; serving unchanged. |
| **No measured benefit on FruitCake** | It's opt-in; N3 reports honestly; the SDK capability stands regardless and helps future games. |

---

## 8. Open questions
1. **Trunk noisy too, or heads only?** Start heads-only (standard/sufficient); revisit if exploration is weak.
2. **Noise granularity** — one draw per forward (recommended, matches Rainbow/dqn_zoo) vs per-sample. Start per-forward.
3. **Should the noisy FruitCake net replace the shipped one**, or ship alongside as a labelled variant? Decide at N4 from the numbers.
4. **Generalize to `Mlp`/non-Dueling** and expose for other games (Snake) once validated on FruitCake.

**Sources:** [NoisyNets (arXiv:1706.10295)](https://arxiv.org/abs/1706.10295) ·
[Rainbow (arXiv:1710.02298)](https://arxiv.org/abs/1710.02298) ·
[google-deepmind/dqn_zoo](https://github.com/google-deepmind/dqn_zoo) ·
[thomashirtz/noisy-networks](https://github.com/thomashirtz/noisy-networks)
