# GPU-resident training step for the conv policy/value net — PRD

**Status:** ✅ **BUILT + GPU-measured** (2026-07-14, RTX 3060). M44.1 measured → GO; M44.2 Core seam shipped; M44.3
Ilgpu resident trainer shipped — **~24× faster train step** (≈122 ms vs ≈3000 ms per 128-batch), gradient-parity vs
autograd verified. **Owner:** Pieterjan.
**Milestone:** [PLAN.md](PLAN.md) M44 · **Depends on:** M43 GPU-resident conv *forward* (`DeviceConvPolicyValueNet`,
`IPolicyValueForward`; commits `852cf31`/`b49a4c2`/`f39cf2f`) — this is its training-side sibling. **Promotes** the
"resident conv *trainer*" deferred item in [GPU_RESIDENT_CONV_PRD.md](GPU_RESIDENT_CONV_PRD.md) §Deferred.

## 1. Problem

With `--gpu`, self-play *inference* is now GPU-resident (M43, ~15× on an RTX 3060). But the **training step**
(`PolicyValueTraining.TrainStep`: `net.Forward` → CE+MSE loss → `Backward` → clip → Adam) still runs through
`Backend.Current` **host-span**: weights re-upload per GEMM and im2col/col2im run on the CPU between every conv — the
same transfer-bound pattern M43 eliminated for the forward. The cube already has the training-side answer:
`DeviceResidualTrainer` (`IResidentTrainStep`) keeps weights + gradients + Adam moments + activations on-device for the
whole fwd→bwd→clip→Adam step. This PRD is the two-headed conv analogue.

**⚠️ But measure first (see §5).** A self-play *chunk* is dominated by **generation** (MCTS to the ply-cap straggler),
not the owner-thread training step. If generation dominates, the resident *forward* + `--leaf-batch` (already shipped)
is the real lever and a resident *trainer* buys little. So M44.1 is a measurement gate before the expensive kernel work.

## 2. Findings (3-agent read-only analysis, 2026-07-14)

- **The cube trainer is a strong template, mostly reusable.** `DeviceResidualTrainer`'s `Param{W,G,M,V}` four-buffer
  model (per parameter, in `Parameters()` order, + a flat `_all` list), pre-allocated activation caches, on-device
  **grad-norm clip + Adam** (`LaunchSumSq`/`LaunchScaleInPlace`/`LaunchAdamUpdate` — fully layer-agnostic), and
  `SyncToHost` all transfer unchanged. Backward GEMM vocabulary reuses: **dW = `GemmTiled(AtB)`**, **dInput =
  `GemmTiled(ABt)`**, dBias = `LaunchBiasGrad`, plus `LaunchReluBackward`, `LaunchLayerNormParamGrad`/`InputGrad`,
  `LaunchAddInto` (skip-grad). Wiring template = `CubeDaviCampaign.BuildStack` (obtain trainer from a backend factory,
  inject as the Core seam, `SyncToHost` before checkpoint/eval).
- **The conv backward is exactly two host-loops → two new kernels**, the transpose of what M43 added to the forward:
  (a) **`Col2Im`** (dInput scatter-add — prefer **thread-per-input-element**, gather-sum every kernel tap → atomics-free),
  (b) **`GatherNCHWToMOutC`** (dOut permute `[N,outC·oH·oW]→[M,outC]`, the transpose of the forward `ScatterBias`, no bias).
  Then dW/dInput/dBias are the existing GEMM-transposes + `LaunchBiasGrad` (which maps directly onto `dOutMat[M,outC]`
  with rows=M, dim=outC).
- **The two-headed loss grads are computed on the HOST, not as kernels** (revised during M44.3). They need
  `softmax`/`tanh` (`ExpF`/`TanhF`), which the CUDA PTX backend cannot JIT without ILGPU.Algorithms/XMath — and the repo
  deliberately keeps softmax/tanh off the device (`DeviceConvPolicyValueNet` returns a linear value; the caller tanhs).
  The trainer already downloads the two heads for the loss, so it computes `dLogits[b,j] = (softmax(logits)_bj − π_bj)/B`
  and `dValueLinear[i] = valueWeight·(2/B)·(tanh(v_i) − z_i)(1 − tanh²(v_i))` on the host and uploads them (the heads are
  tiny → negligible transfer). The originally-planned `PolicyCeGrad`/`ValueTanhMseGrad` kernels were therefore dropped.
- **Everything else reuses**, incl. the LayerNorm grad kernels (dim-generic → work over the conv's whole-row
  `dim = filters·hw`) — **with one forward-side change (no new kernel):** the resident training forward must use
  `LaunchLayerNormTrain` (caches x̂, 1/σ) instead of M43's inference `LaunchLayerNorm`, and allocate per-LN
  `xhat[B·dim]`/`invStd[B]` caches, so the LN-grad kernels have their inputs.
- **The scalar `IResidentTrainStep` seam doesn't fit** (one target, one scalar loss). Need a new **two-headed** seam —
  the training dual of the already-accepted two-headed `IPolicyValueForward`.
- **Determinism:** a resident *trainer* MUTATES trained weights via non-associative GPU reductions → **not
  bitwise-reproducible**, so (unlike the inference-only forward) it **cannot** be the reference under the DOP-invariance
  SHA test. It must be **opt-in**; the **CPU autograd path stays the deterministic reference**. Guard with a
  gradient-parity test vs autograd (as the cube does), not bitwise equality.
- **Adam-resume (P.2):** the cube's resident Adam moments aren't downloaded into the campaign's Adam checkpoint, so a
  resumed `--gpu` run re-warms the optimizer (weights resume fine). Accept the same gap; document it.

## 3. Decision

Build a **two-headed GPU-resident training step** as **library** code (Core seam + Ilgpu impl + 4 new kernels), wired
into `SelfPlayCampaign` via a Core-typed factory (campaign stays Ilgpu-free). **Gate the expensive Ilgpu kernel work
behind a measurement** (M44.1). Inference-side stays M43; the CPU autograd path stays the deterministic reference.

## 4. Design (placement: generic → library)

### 4a. Core — the seam (`Core/Nn/IPolicyValueTrainStep.cs`, GPU-agnostic)
```csharp
public interface IPolicyValueTrainStep
{
    // One AlphaZero batch: resident fwd + CE/MSE backward + grad-norm clip + Adam. policyTargets row-major
    // [batch·actions] (rows sum to 1); valueTargets [batch] in [-1,1]. obsSize/actions/valueWeight/gradClip/lr are
    // CONSTRUCTION params (baked into fixed buffers + on-device Adam), not per-Step args.
    (double PolicyLoss, double ValueLoss) Step(float[] obs, float[] policyTargets, float[] valueTargets, int batch);
    void SyncToHost();   // write resident weights back into the CPU net (eval/arena/ladder/checkpoint read it)
}
```
Plus **`AutogradPolicyValueTrainStep`** (Core default): inlines the current `PolicyValueTraining.TrainStep` loss/backward
over an `IPolicyValueNet`+`Adam`; `SyncToHost` = no-op (CPU net is master). Guarantees routing through the seam is a
behaviour-preserving refactor. (Tuple return matches `TrainStep` so `_lossWindow.Add(pl, vl, 0)` is unchanged.)

### 4b. Ilgpu — `DeviceConvResidualTrainer : IPolicyValueTrainStep, IDisposable`
A **separate** object from `DeviceConvPolicyValueNet` (mirrors `DeviceResidualTrainer` vs `DeviceResidualMlp`): holds
`Param{W,G,M,V}` per parameter (Parameters() order + flat `_all`) + full activation caches (conv pre-activations,
post-ReLU maps, per-LN x̂/1/σ). Forward = M43 chain but with `LaunchLayerNormTrain`; backward per §2; clip+Adam reuse. **2 new device kernels:**
`Col2Im_Kernel`, `GatherNCHWToMOutC_Kernel` (the two loss grads are host-side — see §2). Factory:
`IlgpuBackend.CreateResidentTrainer(ConvResidualPolicyValueNet, batch, lr, clipNorm, actions, valueWeight, …)`.

### 4c. Lab — wiring (the only lab-specific part)
<!-- API note (M45): the `adaptive?.Gpu` in the snippet below became `adaptive.Gpus[0]` (multi-GPU); see MULTI_GPU_SELFPLAY_PRD.md. -->
`SelfPlayCampaign` gains a Core-typed `Func<IPolicyValueNet, Adam, IPolicyValueTrainStep>? trainStepFactory`; Resume
builds `_trainStep` (else the autograd default); `TrainChunk` calls `_trainStep.Step(obs, pi, z, _batchSize)` in the
batch loop and **`_trainStep.SyncToHost()` before `_forward.OnWeightsSynced(_net)`** so eval/arena/ladder/checkpoint see
trained weights. `ChessLab` supplies the factory (`adaptive?.Gpu is {} gpu && net is ConvResidualPolicyValueNet conv ?
gpu.CreateResidentTrainer(...) : null`) — all Ilgpu knowledge stays there.

## 5. Phases

- **M44.1 ✅ MEASURED (gate) — GO.** Instrumented `TrainChunk` with a gen-vs-train split behind env `CHESS_CHUNK_TIMING`
  (SelfPlayCampaign, off by default → no log noise). Ran `--gpu --arch conv --parallel --leaf-batch 128 --games 6
  --sims 48 --max-plies 60 --filters 64 --blocks 6` on an **RTX 3060 Laptop GPU**:

  | chunk | window | train batches | gen | train | train share |
  |---|---|---|---|---|---|
  | 1 | 360 | 2 | 13.6 s | 7.8 s (JIT-inflated) | 36.5 % |
  | 2 | 720 | 5 | 15.8 s | 14.5 s | 47.8 % |
  | 3 | 1080 | 8 | 15.0 s | 26.0 s | 63.5 % |

  **Reading:** generation is ~**constant** (~15 s: fixed by games×sims×plies, and the M43 resident forward + `--leaf-batch`
  already handle it). The **CPU/host-span training step grows linearly with the replay window** — batches/chunk =
  `epochs·⌊window/batch⌋`, and each 128-sample batch costs **~3.0 s** through `Backend.Current` host-span (weights
  re-upload per GEMM + CPU im2col/col2im — the exact transfer-bound pattern M43 fixed for *inference*). It already crosses
  50 % by window 1080; at the **default 40 000 window** (312 batches/chunk) it asymptotes to **~98 %** of chunk wall-time.
  **Decision: BUILD M44.3.** The train step is the dominant cost of any run with a non-trivial window (i.e. every serious
  / cluster run), and it's the same inefficiency M43 already proved fixable. **Caveat (honesty):** the split is
  config-dependent — it's governed by the batches-per-chunk : generation-work ratio, i.e. `(epochs·window/batch)` vs
  `(games·sims·plies)`. A tiny-window / huge-chunk run would be generation-bound and see little from a resident trainer;
  but the ~3 s/batch host-span cost is paid by *every* config with real training and is what M44.3 removes. (Measurement
  instrumentation is committed and reusable: set `CHESS_CHUNK_TIMING` to re-measure under any config.)
- **M44.2 ✅ SHIPPED — Core seam + wiring (behaviour-preserving).** `Core/Nn/IPolicyValueTrainStep.cs` = the seam +
  `AutogradPolicyValueTrainStep` (inlines the exact former `PolicyValueTraining.TrainStep` loss/backward → the CPU path
  is byte-for-byte unchanged). `SelfPlayCampaign` gained an optional `Func<IPolicyValueNet, Adam, IPolicyValueTrainStep>`
  factory (null → autograd default), builds `_trainStep` in `Resume`, routes the batch loop through `_trainStep.Step(...)`,
  and calls `_trainStep.SyncToHost()` before `_forward.OnWeightsSynced(_net)`. The now-duplicate Lab `PolicyValueTraining`
  was deleted (logic lives once in Core). `ChessLab` unchanged (factory stays null until M44.3's `CreateResidentTrainer`).
  **Gate MET:** `SelfPlayCampaignTests.ParallelGeneration_ProducesBitwiseIdenticalCheckpoint_AtAnyDop` still passes
  (checkpoint hash identical → the refactor changed nothing); all 3 SelfPlayCampaign tests green.
- **M44.3 ✅ SHIPPED — Ilgpu trainer + 2 kernels (gated on M44.1).** `DeviceConvResidualTrainer : IPolicyValueTrainStep`
  (resident forward caching x̂/σ + post-ReLU maps + im2col columns → full two-headed backward → grad-norm clip → Adam),
  reusing `LaunchLayerNormTrain` + all the M20 backward kernels + the M43 conv kernels; `CreateResidentTrainer(ConvResidual
  PolicyValueNet, …)` overload; `ChessLab` wires it for `--gpu --arch conv`. **Only TWO new device kernels** were needed
  (`Col2Im`, `GatherNCHWToMOutC` — the transposes of M43's im2col/scatter); the two loss grads (softmax−π, tanh-MSE) are
  computed on the **host**, because the repo deliberately keeps softmax/tanh off the device (no ILGPU.Algorithms/XMath —
  a CUDA PTX JIT of `ExpF`/`TanhF` fails) and the heads are tiny, so PLANNED kernels `PolicyCeGrad`/`ValueTanhMseGrad`
  were dropped (fewer kernels, no new dependency). **Gate MET:** `DeviceConvResidualTrainer_GradientsMatchAutograd`
  (device backward == autograd within f32 tol on the ILGPU CPU accelerator) + `_SyncToHost_RoundTrips`, all green.
  **On-GPU (RTX 3060, `CHESS_CHUNK_TIMING`, same config as M44.1):** the train step dropped from ~3000 ms to **~122 ms
  per 128-batch (≈24×)**; train share of a chunk fell 36→48→64 % (M44.1) to **3→5→8 %**. Generation was never the target
  (M43's resident forward owns it).

## 6. Risks
1. **Generation dominates → low ROI.** ✅ **Retired by M44.1**: at the default 40 k window the train step is ~98 % of
   chunk wall-time (not generation). Low ROI only in a tiny-window/huge-chunk regime, which no serious run uses.
2. **Conv backward kernel correctness** (col2im scatter-sum, the gather transpose, the two loss grads) — mitigated by
   the M44.3 gradient-parity test on the CPU accelerator (CI-safe, no GPU) + the exact index-math specs in §2.
3. **Determinism** — GPU training isn't bitwise-reproducible; kept opt-in, CPU autograd stays the reference (DOP test
   runs on CPU only). No new loss beyond what `--gpu` already accepts.
4. **Adam-resume (P.2)** — resident optimizer moments not checkpointed → re-warm on `--gpu` resume (weights fine).
   Accepted, documented; if lossless resume is wanted later, add m/v download to `SyncToHost` + `AdamState` for BOTH
   trainers as the shared P.2 fix.

## 7. Verification
- M44.1 ✅: measured train-vs-gen share of a `--gpu` conv chunk on an RTX 3060 (36.5 → 47.8 → 63.5 % train as the window
  filled; ~3 s/128-batch host-span; asymptotes ~98 % at the 40 k default) → GO for M44.3. Re-run with `CHESS_CHUNK_TIMING`.
- M44.2: DOP-invariance SHA test bitwise-identical (behaviour-preserving); all self-play tests green.
- M44.3 ✅: `DeviceConvResidualTrainer_GradientsMatchAutograd` (device vs autograd, ILGPU CPU accelerator, f32 tol) +
  `_SyncToHost_RoundTrips` green; on-GPU (RTX 3060) train step ≈122 ms vs ≈3000 ms per 128-batch (~24×).

## 8. Generic (library) vs lab
| Piece | Layer |
|---|---|
| `IPolicyValueTrainStep` + `AutogradPolicyValueTrainStep` | **Core** (`Core/Nn/`) |
| `DeviceConvResidualTrainer` + 2 kernels (`Col2Im`/`GatherNCHWToMOutC`) + `CreateResidentTrainer` overload | **Ilgpu** |
| `trainStepFactory` construction (`gpu.CreateResidentTrainer(...)`) | **Lab** (`ChessLab`) |
| Core-typed factory plumbing in `SelfPlayCampaign` | Lab campaign, but Ilgpu-free (Core delegate) |

See [PLAN.md](PLAN.md) M44, [GPU_RESIDENT_CONV_PRD.md](GPU_RESIDENT_CONV_PRD.md), [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) (P.2).
