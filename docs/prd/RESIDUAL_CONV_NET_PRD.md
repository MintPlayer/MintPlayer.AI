# Convolutional residual policy/value net (Core) + chess adoption — PRD

**Status:** Planned · 2026-07-13 · branch TBD (off `master`/the chess branch, stacked after M41)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M42 · **Depends on:** M41 (parallel self-play — makes the training iterations this needs affordable) · **Supersedes** the "flat MLP is the ceiling" note in [CHESS_WEB_POLYGLOT_PRD.md](CHESS_WEB_POLYGLOT_PRD.md) and [CHESS_SELFPLAY_PRD.md](CHESS_SELFPLAY_PRD.md).

## 1. Problem

Chess self-play has **plateaued at ~random play** despite 256-sim MCTS and material-shaped value targets (M40.4). The live run stalls at winRate-vs-random ~50% (drifting to 35%), material margin ~+0.1 pawns (gate needs +0.75), and policy loss barely moving (2.35→2.22 over 500 games) — **no tier ever promotes.** The honest diagnosis (recorded across the chess PRDs and the training memory): the model is the bottleneck, not search depth or reward shaping. The net is a **flat `[256,256]` ReLU MLP over a 1152-float vector** (`PolicyValueNet`) — it throws away the board's 8×8 spatial structure, so it cannot learn the local piece-interaction patterns (pins, forks, pawn chains, king safety) that chess play is built on.

**The owner's decision (2026-07-13): give chess a true AlphaZero-style convolutional residual net** — reshape the observation to `[18, 8, 8]` and run a spatial residual tower. This PRD covers building that net *reusably in Core* (so any board game can use it) and adopting it for chess, including the browser-inference twin.

## 2. Findings (from a 2-agent read-only investigation, 2026-07-13)

### 2a. What the chess net is today
`PolicyValueNet` (`src/…Core/Nn/PolicyValueNet.cs:15`) — a shared variable-depth ReLU **MLP** trunk (`Linear[]`) + two **linear** heads (policy logits `[B,4672]`, scalar value `[B,1]`); `tanh` on the value is applied by the *trainer* (`PolicyValueTraining.cs:24`), not the net. Chess builds it with trunk `[256,256]` (`ChessLab.cs:20`, `SelfPlayCampaign.cs:78/109`). It is the same net `CubePolicyNet`/`RushHourPolicyNet` wrap — all flat MLPs.

### 2b. Is a residual/conv net already reusable? Partly — and not usable as-is.
- **A residual net class IS already in Core:** `ResidualMlp` (`src/…Core/Nn/ResidualMlp.cs:21`) — pre-activation residual blocks `x → x + W₂·ReLU(LayerNorm(W₁·x))`. **But** it's single-headed **scalar** output (`IValueNet`, cost-to-go), residual **MLP** (not conv), and used only by the cube DAVI Lab campaign (`CubeDaviCampaign.cs:141`). It **cannot** serve as a chess policy/value net (no policy head, no spatial structure). So "move the residual net to the library" is a **no-op — it's already there**; the missing thing is a *two-headed convolutional* net, which does not exist.
- **No convolution exists anywhere** in Core, Environments, Lab, or the Polyglot `.pg` files. **No** Conv2D, **no** im2col/col2im, **no** pooling, **no** BatchNorm. **LayerNorm does exist** (`TensorOps.cs:272`, autograd op with learnable γ/β) and the codebase deliberately prefers it over BatchNorm (`ResidualMlp.cs:6-13`).
- **The differentiable op set** (`TensorOps.cs`): `MatMul`, `Add`, `AddBias`, `Sub`, `Mul`, `MulScalar`, `Relu/Tanh/Exp/Log/Square`, `Clamp`, `Min`, `Gather`, `Sum`, `Mean`, `SumRows`, `LogSoftmax`, `HuberLoss`, `LayerNorm`, `MseLoss`, **`Reshape`** (buffer-sharing, grad pass-through). Skip-add = plain `Add`. The backend seam (`IComputeBackend`, managed + ILGPU) exposes 3 GEMM layouts, elementwise Map/Zip, reductions, LayerNorm — **but no conv/pool/batchnorm kernels.**

### 2c. The structural blocker
`SelfPlayCampaign<TState>` and `PolicyValueTraining` reference the **concrete `PolicyValueNet` type**, not an interface (`SelfPlayCampaign.cs:45,57,102,109,282,405,436,445`; `PolicyValueTraining.cs:18`). There is **no abstraction for "a two-headed policy/value net."** A different net class can't be dropped in without introducing that interface and generalizing the campaign/trainer. This is the main refactor cost — small, but it gates everything.

### 2d. The observation is already conv-ready
Chess obs = **18 planes × 64 = 1152 floats**, laid out `plane*64 + sq` with `sq = rank*8 + file` (`chess_solver.pg:405-427`; C# mirror `ChessGame.cs:53`): planes 0–5 white P/N/B/R/Q/K, 6–11 black, 12 side-to-move, 13–16 castling, 17 en-passant. → **trivially reshapes to `[C=18, H=8, W=8]`.** Policy head target = 4672 (64×73 AlphaZero move encoding), value = 1 scalar. So the conv net's contract is unchanged: `(Logits[B,4672], Value[B,1])`.

## 3. Decision

**Build a reusable two-headed convolutional residual net in Core, gated behind a new `IPolicyValueNet` interface, and adopt it for chess** — including a conv forward in the browser-inference `.pg` twin. Keep the existing flat `PolicyValueNet` as a selectable architecture (it's the connect-4 / cube-policy / rush-hour net and the fast baseline). Do **not** remove `ResidualMlp` — it stays the cube DAVI value net.

## 4. Design

### 4a. New Core ops — Conv2D (the only genuinely new numerics)
Implement convolution as **im2col → existing GEMM → col2im**, so the heavy math reuses the already-tuned (and GPU-routed) GEMM instead of a bespoke kernel:
- **`IComputeBackend` (managed + ILGPU) + `TensorOps`:** add a differentiable `Conv2D(input, weight, bias, inC, H, W, outC, kernel, stride, pad)`.
  - **Representation:** keep tensors **rank-2** — input `[N, inC·H·W]`, output `[N, outC·outH·outW]` — with the conv op carrying `(inC,H,W,…)` explicitly. This **sidesteps the `CheckRank2` asserts entirely** (no 4-D tensor ever reaches `MatMul`/`LayerNorm`); `Reshape` is only used to re-view between conv and the flat heads.
  - **Forward:** `im2col(x)` → `[N·outH·outW, inC·k·k]`, GEMM with `weight [inC·k·k, outC]`, add bias → output. im2col/col2im are cheap index gathers (managed loops); the GEMM routes to the GPU via the adaptive backend for free.
  - **Backward:** dInput = `col2im(dOut · Wᵀ)`; dWeight = `im2col(x)ᵀ · dOut`; dBias = column-sum of dOut. All three reuse the existing GEMM layouts + `AddInto`/`BiasGradInto`.
- **Normalization:** reuse the existing **LayerNorm** (proven stable in `ResidualMlp`) rather than building spatial BatchNorm — one fewer new kernel, and consistent with the repo's deliberate LayerNorm choice. (Spatial BatchNorm is a possible later refinement, out of scope.)
- **Pooling:** not needed — AlphaZero towers are all-conv with a flatten+Linear at each head.
- **Gate:** finite-difference gradient check on `Conv2D` (dInput, dWeight, dBias) and a forward-value check vs a hand-computed small example; determinism (same output at any backend DOP, like GEMM).

### 4b. New Core net — `ConvResidualPolicyValueNet` (`Core/Nn/`)
AlphaZero-standard, parameterized by `(inputPlanes, boardH, boardW, filters, blocks, actions)`:
- **Stem:** `Conv2D 3×3 (inC→filters, pad 1)` → LayerNorm → ReLU.
- **Tower:** `blocks` × residual block `[Conv3×3 → LN → ReLU → Conv3×3 → LN → (+skip) → ReLU]` (all `filters→filters`, pad 1, so the spatial shape is preserved and the skip-add is same-shape).
- **Policy head:** `Conv2D 1×1 (filters→2)` → LN → ReLU → flatten → `Linear(2·H·W → actions)`.
- **Value head:** `Conv2D 1×1 (filters→1)` → LN → ReLU → flatten → `Linear(H·W → filters)` → ReLU → `Linear(filters → 1)` (linear; `tanh` stays in the trainer, matching `PolicyValueNet`).
- Implements **`IPolicyValueNet`** (§4c); `Save`/`Load` with a new checkpoint kind (e.g. `selfplay-pv-conv`), storing `(filters, blocks)` + all layer floats, matching `CheckpointFormat`. Provides `LayerActivations` for the `--viz` telemetry seam.

### 4c. The abstraction — `IPolicyValueNet` (the enabling refactor)
Introduce in Core:
```csharp
public interface IPolicyValueNet
{
    (Tensor Logits, Tensor Value) Forward(Tensor observations);
    IEnumerable<Tensor> Parameters();
    float[][] LayerActivations(Tensor observation);
    void Save(Stream destination, string kind);
}
```
- `PolicyValueNet` implements it verbatim (it already has every member — **zero behaviour change**, SHA-identical checkpoints).
- `SelfPlayCampaign` + `PolicyValueTraining` are generalized from the concrete type to `IPolicyValueNet` (field type + the evaluator/arena forwards). Net **construction** and **loading** go through a small architecture factory keyed on an `--arch` value (so the right `Load` runs for a given checkpoint kind).
- **Net2Net growth** (`WidenTo`/`Deepen`) stays **MLP-specific** and out of the interface — chess self-play doesn't grow the net (fixed `--hidden`); if a conv-growth path is ever wanted it's a separate optional interface. (Verify no growth call sits on the campaign path during M42.2; the investigation found none.)

### 4d. Chess adoption (Lab)
- `ChessLab`: `--arch mlp|conv` (default stays `mlp` until the conv net proves out), plus `--filters` (e.g. 64) and `--blocks` (e.g. 6). `SelfPlayCampaign` builds the chosen `IPolicyValueNet`.
- All of M40.4 (material-shaped value target, the auto-capture ladder, `--viz`) is architecture-agnostic and works unchanged.

### 4e. Browser-inference twin — conv forward in `chess_solver.pg` (do not skip)
The whole M40 browser story is single-source Polyglot inference: `PgPolicyValueNet.forward` in `chess_solver.pg` is a **flat-MLP** forward, and `chess-net.ts` parses the flat `.ckpt`. A conv net **breaks client-side play** until the `.pg` gains a **conv2d forward** (inference only — no backward, so it's straightforward: nested bounded `for` over out-channels × spatial × kernel, using `Math` from `std.math`) and the `.ckpt` parser learns the conv layout. This is transpiled to both C# and TS from the one `.pg`, and re-gated by the existing `ChessNetParityTests` (C# `Forward` vs generated net on real `.ckpt` bytes). **This is a first-class phase (M42.4), not a follow-up** — shipping stronger weights the browser can't run would regress M40.

## 5. Phases

- **M42.1 — Conv2D in Core.** `Conv2D` op (im2col+GEMM+col2im) in `IComputeBackend` (managed + ILGPU) + `TensorOps`, rank-2 representation. **Gate:** finite-difference gradient check (dInput/dWeight/dBias) + forward value check + DOP-invariant output.
- **M42.2 — `IPolicyValueNet` + the conv net, in Core.** Introduce the interface; `PolicyValueNet` implements it (SHA-identical checkpoints — proves no behaviour change). Build `ConvResidualPolicyValueNet` (Save/Load/LayerActivations). Generalize `SelfPlayCampaign`/`PolicyValueTraining` + an arch factory. **Gate:** existing self-play tests green with MLP unchanged; a train-one-batch smoke test on the conv net (loss decreases); connect-4 still trains identically.
  - **De-risking checkpoint (recommended, cheap):** before the conv tower, wire a *two-headed residual **MLP*** (add a policy head to the `ResidualMlp` pattern — **no new kernels**) behind the same interface and run a short chess train. If depth+residuals alone don't move the plateau, that's a fast signal about targets/search *before* investing in conv. Not a substitute for the conv net — a smoke test that de-risks it.
- **M42.3 — train chess with the conv net.** `--arch conv --filters 64 --blocks 6` (tune), material shaping + ladder on. **Gate:** clearly beats the flat-MLP baseline — material margin ≥ +0.75 pawns and/or winRate-vs-random well above 50% and **at least one ladder tier promotes**; report the training curve vs the MLP plateau. Reproducibility (SHA dop-1==dop-N from M41) preserved.
- **M42.4 — browser conv forward + parity.** Add a conv2d forward to `chess_solver.pg` (inference-only) + teach `chess-net.ts` the conv `.ckpt` layout; regenerate the C#/TS twin. **Gate:** `ChessNetParityTests` green (C# vs generated within f32 tol on real conv `.ckpt` bytes); the `/chess` page plays with a shipped conv tier.

## 6. Risks
1. **Conv correctness** — mitigated by the finite-difference gradient gate (M42.1) before any net is built on it.
2. **Conv net still doesn't beat the plateau** — possible if the block is targets/search, not capacity; the M42.2 residual-MLP de-risking checkpoint surfaces this cheaply. Fallback levers if so: more sims, longer training (M41 makes this affordable), or revisiting the value target.
3. **Training cost** — a conv tower is far heavier than the MLP; **this is exactly why M41 (parallel self-play) lands first.** The im2col+GEMM route keeps the heavy math on the GPU-routed GEMM.
4. **Browser perf** — a conv forward per MCTS node in JS may be slow; if so, cap browser sims per difficulty tier (the tier system already sets per-tier sims) or ship a smaller `filters/blocks` for the web than for the strongest offline tier. Flag during M42.4.
5. **Checkpoint/interface churn** — the `IPolicyValueNet` refactor touches the shared self-play path; the SHA-identical-MLP gate (M42.2) guards against a silent regression to connect-4/chess.

## 7. Verification
- M42.1: conv gradient (finite-diff) + forward-value unit tests; DOP-invariance.
- M42.2: full suite green with MLP behaviour SHA-identical; conv train-one-batch loss decrease; connect-4 unchanged.
- M42.3: chess conv run **beats the MLP baseline** on material margin / winRate with ≥1 tier promoted; determinism preserved.
- M42.4: `ChessNetParityTests` green on conv `.ckpt`; `/chess` plays a conv tier end-to-end.

See [PLAN.md](PLAN.md) M42.
