# GPU-resident batched forward for the conv policy/value net — PRD

**Status:** ✅ **BUILT + measured on GPU** 2026-07-14 (M43.1 `852cf31` Core seam, M43.2 `b49a4c2` Ilgpu impl + kernels,
M43.3 `f39cf2f` Lab wiring). Verified on the ILGPU CPU accelerator AND on a real **RTX 3060 Laptop GPU**: the resident
forward is **~15× faster** than the non-resident autograd path (109.9 ms vs 1634.7 ms/forward at 64f×6b, leaf-batch 256
→ 2,329 vs 157 leaves/s), on-GPU parity to ~1e-6. **Owner:** Pieterjan.
**Milestone:** [PLAN.md](PLAN.md) M43 · **Depends on:** M42.5 batched leaf inference (`Mcts.SearchBatched` / `BatchEvaluate`,
commit `4801c98`) — the seam this plugs into. **Promotes** the deferred item in
[RESIDUAL_CONV_NET_PRD.md](RESIDUAL_CONV_NET_PRD.md) §8.1 and [OPTIMIZATIONS.md](../OPTIMIZATIONS.md) F.3.

> **API note (M45):** the `AdaptiveBackend.Gpu` used in the snippets below was later replaced by `AdaptiveBackend.Gpus`
> (the list of all CUDA GPUs) — single-device callers use `Gpus.FirstOrDefault()`. See
> [MULTI_GPU_SELFPLAY_PRD.md](MULTI_GPU_SELFPLAY_PRD.md). The snippets are kept as the as-designed record for M43.

## 1. Problem

`--gpu` + `--leaf-batch` make self-play MCTS evaluate a *wave* of leaves per `net.Forward`, but the conv net's forward
still runs through `Backend.Current` (the `AdaptiveBackend`): weights re-upload per GEMM, and activations round-trip
host↔device between each of the tower's ~14 convs (each `Conv2D` = **host im2col → GPU GEMM → host scatter**). So even
with a GPU, a conv self-play forward is transfer-bound — the opposite of the cube's value net, which runs **GPU-resident**
(`DeviceMlp`/`DeviceResidualMlp`: weights stay on-device, only the batch crosses the bus). This is the one piece that makes
the chess GPU path as efficient as the cube's and is the precondition for a cluster run to actually pay off.

## 2. Findings (3-agent read-only analysis, 2026-07-14)

- **The resident pattern is Core-seam → Ilgpu-impl → Lab-wiring.** `ITargetForward`/`IResidentTrainStep` (Core) keep Core
  GPU-agnostic; `DeviceMlp`/`DeviceResidualMlp` (Ilgpu) implement them; the cube campaign wires them
  (`AdaptiveBackend.Gpu is {} gpu ? gpu.CreateResidentForward(net) : null`). `DeviceResidualMlp.Forward` is a near-exact
  scaffold: resident weight buffers uploaded in `Parameters()` order, a **reused activation pool**, all device touches
  under `DeviceLock`, a **block-loop with skip-add** (`LaunchAddInto`).
- **Only TWO new GPU kernels are needed.** The conv forward's op chain reuses existing resident kernels — `LaunchGemmTiled`
  (register-blocked), `LaunchLayerNorm` (row-wise, **+fused ReLU**), `LaunchBiasActivation`, `LaunchAddInto`, `LaunchRelu`.
  The net's LayerNorm normalizes over the **whole flattened `C·H·W` row**, which `LaunchLayerNorm`'s one-thread-per-row
  contract already matches — so **no new normalization kernel** (one analyst's BatchNorm guess was wrong; the actual net
  uses whole-row LN). The gaps are exactly the two host loops bracketing each conv's GEMM: **device im2col** (gather NCHW →
  `cols[M, inC·k²]`, zero-padded) and **device scatter+bias** (`[M,outC]` → `[B, outC·H·W]`, folding per-channel bias).
  Both are embarrassingly parallel, one-thread-per-output-element, no shared memory, no reduction. The two 1×1 head convs
  skip im2col entirely (it's a no-op reshape at k=1/pad=0). The NCHW-as-`[B,C·H·W]` layout makes conv→flatten→head a free
  reinterpret (no permute).
- **The net interface can't carry this; a new seam is needed.** `IPolicyValueNet.Forward` is autograd-recorded and used by
  *training*; a resident forward is inference-only, flat-array, with a weight-sync lifecycle. Exactly the `ITargetForward`
  situation — but `ITargetForward` is **scalar** (cube's single head), and our net is **two-headed**. → a new two-headed
  seam parallel to `ITargetForward`.
- **Everything generic is already in Core** (`ConvResidualPolicyValueNet`, `Mcts`, `IPolicyValueNet`, `Conv2D`); only the
  campaign/CLI wiring is in the Lab. So the resident path (seam + impl) belongs in the **library**, wiring in the Lab.

## 3. Decision

Build a **two-headed GPU-resident batched forward** for the conv net, inference-only, as **library code**: a Core seam +
an Ilgpu implementation + two new device kernels, wired into the existing `BatchEvaluate` path in the Lab. Weights are
re-synced to the device on the owner thread's per-chunk cadence (not per forward). Training stays on the autograd path.

## 4. Design (placement: generic → library)

### 4a. Core — the seam (GPU-agnostic, two-headed, inference-only)
`Core/Nn/IPolicyValueForward.cs`:
```csharp
public interface IPolicyValueForward
{
    void OnWeightsSynced(IPolicyValueNet net);                                 // (re)upload resident weights
    (float[] Logits, float[] Value) Forward(float[] observations, int rows);   // raw logits [rows*actions] + linear value [rows]
}
```
- Returns **raw logits + linear (pre-tanh) value** — masked-softmax + `tanh` stay in the caller (they need each state's
  `LegalMoves`, which the device forward doesn't have). Matches `DeviceMlp.Forward` returning raw floats.
- **`AutogradPolicyValueForward`** (Core default): wraps an `IPolicyValueNet`, runs `Forward` under `NoGrad` — exactly what
  `SelfPlayCampaign.EvaluateBatch` does today, so the CPU path stays **bitwise-identical** and there's always a non-GPU
  fallback (mirrors `AutogradTargetForward`).
- Small additive change: `ConvResidualPolicyValueNet` exposes its shape (`planes/h/w/filters/blocks`) so the impl can size
  buffers (as `ResidualMlp` exposes `Width`/`Blocks`).

### 4b. Ilgpu — the implementation + two kernels
`Ilgpu/DeviceConvPolicyValueNet.cs : IPolicyValueForward, IDisposable`, beside `DeviceResidualMlp`:
- Resident weight buffers mirror `ConvResidualPolicyValueNet.Parameters()` order; uploaded once + on `OnWeightsSynced`,
  under `DeviceLock`. Reused activation pool (grow-to-max, exact-length sub-views).
- `Forward` chain on-device: **stem** `im2col → GemmTiled → scatter+bias → LayerNorm(relu)`; **N residual blocks**
  `[im2col→Gemm→scatter+bias→LN(relu) → im2col→Gemm→scatter+bias→LN → AddInto(skip) → Relu]`; **policy head**
  `1×1 conv (no im2col) → scatter+bias → LN(relu) → GemmTiled → BiasActivation`; **value head** likewise → two dense
  Linears. Reuses `LaunchGemmTiled`/`LaunchLayerNorm`/`LaunchBiasActivation`/`LaunchAddInto`/`LaunchRelu`.
- **New kernels** (register in the ctor like the rest): `LaunchIm2Col` (gather + pad zero-fill) and `LaunchScatterBias`
  (`[M,outC]`→`[B,outC·hw]` + per-channel bias broadcast — distinct from `BiasActivation`'s per-column bias).
- Factory: `IlgpuBackend.CreateResidentForward(ConvResidualPolicyValueNet)` overload.

### 4c. Lab — wiring (the only lab-specific part)
- `SelfPlayCampaign` holds an `IPolicyValueForward _forward`, built like `CubeDaviCampaign.BuildStack`:
  `(_backend as AdaptiveBackend)?.Gpu is {} gpu && _net is ConvResidualPolicyValueNet conv ? gpu.CreateResidentForward(conv)
  : new AutogradPolicyValueForward(_net)`.
- `EvaluateBatch` calls `_forward.Forward(obs, b)` instead of `_net.Forward(...)`; obs-packing + masked-softmax + tanh
  unchanged. `OnWeightsSynced(_net)` once per chunk **before** generation (generation already reads a stable net snapshot).
- No new CLI flag — `--gpu` already gates it.

## 5. Phases

- **M43.1 ✅ (`852cf31`) — Core seam.** `IPolicyValueForward` + `AutogradPolicyValueForward`; conv-net shape exposed;
  `EvaluateBatch` routed through the seam. **Gate MET:** `PolicyValueForwardTests` proves the autograd default is
  bitwise-identical to `net.Forward` (rows 1 & 5); all self-play/determinism tests green.
- **M43.2 ✅ (`b49a4c2`) — Ilgpu impl + kernels.** `DeviceConvPolicyValueNet` + the two new kernels (`Im2Col_Kernel`,
  `ScatterBias_Kernel`) + `LaunchIm2Col`/`LaunchScatterBias` + `CreateResidentForward(ConvResidualPolicyValueNet)`.
  **Gate MET:** `IlgpuBackendTests.DeviceConvForward_matches_autograd_conv_net` (rows 1 & 6) passes on the ILGPU **CPU
  accelerator** within f32 tol (runs in CI; the CUDA path runs the same kernels).
- **M43.3 ✅ (`f39cf2f`) — Lab wiring.** `SelfPlayCampaign` takes a Core-typed `forwardFactory` (keeps it Ilgpu-free);
  `ChessLab` supplies the GPU-aware factory (`--gpu` + conv → resident; else autograd) and per-chunk `OnWeightsSynced`.
  **Gate MET:** wiring green, `--gpu` safe on GPU-less machines (falls back). **On-GPU measured** (RTX 3060, `ChessLab
  --bench-forward`): resident **14.9× faster** than autograd (109.9 vs 1634.7 ms/forward at 64f×6b, leaf-batch 256),
  on-GPU parity ~1e-6.

## 6. Risks

1. **Determinism** — GPU GEMM tiling isn't bitwise-reproducible, but this is **inference-only** (no gradients → can't
   corrupt trained weights; training stays autograd), and `--gpu` already forfeits bitwise reproducibility. **No new loss.**
   The CPU autograd default stays bitwise-identical (the DOP determinism gate runs on it).
2. **im2col/scatter kernel correctness** — mitigated by the M43.2 parity test on the CPU accelerator (runs in CI without a
   GPU). Get the weight index order (`((c·k+kh)·k+kw)·outC+oc`) and per-channel bias broadcast exactly right.
3. **Single-GPU lock contention** — every device touch serializes under `DeviceLock`; the win is **batch-B per call**
   (large `--leaf-batch`, few threads), not cross-thread GPU parallelism. Documented, not a blocker.
4. **Scope creep** — WDL head + distribution stay **out of scope** (deferred, RESIDUAL_CONV_NET_PRD §8). The resident
   conv *trainer* (this forward's training-side sibling) is now designed as **M44** —
   [GPU_RESIDENT_CONV_TRAINER_PRD.md](GPU_RESIDENT_CONV_TRAINER_PRD.md).

## 7. Verification
- M43.1: self-play + DOP-determinism tests green; `EvaluateBatch` via autograd default bitwise-identical to today.
- M43.2: `DeviceConvPolicyValueNet.Forward` matches `ConvResidualPolicyValueNet.Forward` within f32 tol on the ILGPU CPU
  accelerator (no discrete GPU needed).
- M43.3: `--gpu --leaf-batch N` self-play runs; measured forward-throughput improvement reported.

See [PLAN.md](PLAN.md) M43, [RESIDUAL_CONV_NET_PRD.md](RESIDUAL_CONV_NET_PRD.md) §8, [ARCHITECTURE.md](../ARCHITECTURE.md) §4.
