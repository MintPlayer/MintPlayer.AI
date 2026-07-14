# Single-box multi-GPU self-play — PRD

**Status:** 🔜 **designed** (3-agent read-only analysis, 2026-07-14), **not built**. **Owner:** Pieterjan.
**Milestone:** [PLAN.md](PLAN.md) M45 · **Depends on:** M43 GPU-resident conv forward (`DeviceConvPolicyValueNet`,
`IPolicyValueForward`) and M44 GPU-resident conv trainer (`DeviceConvResidualTrainer`, `IPolicyValueTrainStep`) — this
partitions the *dataflow* those built across every CUDA GPU present. **Promotes** the "distributed multi-GPU" roadmap
noted in [GPU_RESIDENT_CONV_TRAINER_PRD.md](GPU_RESIDENT_CONV_TRAINER_PRD.md).

## 1. Problem

`--gpu` uses exactly **one** GPU. `IlgpuBackend.SelectDevice` (`IlgpuBackend.cs:187-194`) enumerates every device via
`context.Devices` but takes `.OfType<CudaDevice>().FirstOrDefault()` — it deliberately prefers a discrete NVIDIA card
over an Intel iGPU, but then **collapses N CUDA devices to one**. A box with 2–16 NVIDIA GPUs runs self-play on a single
one; the rest sit idle. Because a self-play chunk is **generation-bound** (M44.1: generation ≈ constant ~15 s/chunk and
dominates; the train step is now ~122 ms/batch after M44), the highest-value scaling lever is to **run self-play
generation on all available GPUs at once**.

Desired behaviour (owner's flow): **(1)** find all CUDA GPUs; **(2)** parallelize the self-play dataflow across them;
**(3)** fall back to the CPU accelerator when no GPU is present. Step (3) already exists — the CUDA↔CPU fallback is in
`SelectDevice` (final `GetPreferredDevice(preferCPU:true)`) and `AdaptiveBackend` (drops the GPU when `!IsGpu`); it just
never enumerates past the first CUDA device.

## 2. Findings (3-agent read-only analysis, 2026-07-14)

### The single-GPU seams (each is a small, well-localized assumption)
- **`IlgpuBackend.SelectDevice` (`:191`)** — `FirstOrDefault()` picks one CUDA device. The one true choke point.
  `context.Devices.OfType<CudaDevice>()` already yields the full ordered list (`DescribeDevices`, `:203-207`, prints all).
- **`IlgpuBackend` ctor (`:140-143`)** — one `Context` + one `Accelerator` + ~25 JIT'd kernels + one `DeviceLock`
  (`_lock`, `:133`) per instance. Confirmed **1 backend = 1 accelerator = 1 GPU = 1 lock**. This is exactly what we want
  *per GPU*; the ctor just needs to accept *which* `Device` instead of always calling `SelectDevice`.
- **`AdaptiveBackend` ctor (`AdaptiveBackend.cs:38`)** — hardcodes a single `new IlgpuBackend()`; exposes it as `.Gpu`.
- **`AddGpuBackend` (`GpuBackendServiceCollectionExtensions.cs:20`)** — `AddSingleton<AdaptiveBackend>()`, one instance.
- **`Backend.Current` (`IComputeBackend.cs:110-113`)** — a plain process-global compute backend that all autograd
  `Tensor` ops (and the parallel self-play worker threads) route through. Only one backend can be `Backend.Current` at a
  time, so multi-GPU must partition via the **resident** forward objects (which bypass `Backend.Current` — they run
  entirely on their own accelerator), NOT by routing generic `Tensor` ops.
- **`SelfPlayCampaign._forward` / `_trainStep` (`SelfPlayCampaign.cs:82,89`)** — single resident instances on one GPU;
  today all N generation threads call the one `_forward` and **serialize on its single `DeviceLock`**
  (`DeviceConvPolicyValueNet.Forward` holds the lock for its whole body, `DeviceConvPolicyValueNet.cs:92`). The device
  lock is the current throughput ceiling.

### The seams that already make this tractable (no rewrite)
- **Generation shards by global game-index, bitwise-invariantly.** `DeterministicParallel.Generate`
  (`DeterministicParallel.cs:40`) runs `makeItem(i, DeriveRng(baseSeed, baseIndex + i))`; the per-game RNG is keyed on
  the **global index** (`baseIndex + i`, golden-ratio stride, `:79-80`), never execution order, and the result is
  "bitwise-identical whether produced in parallel or sequentially, and at any degree of parallelism." So game index
  ranges assigned to different GPUs produce identical samples regardless of which device runs them → the clean shard axis.
- **N backends = N independent locks.** Each `IlgpuBackend` serializes only against itself, so games on GPU0 run fully
  in parallel with games on GPU1. Precisely the concurrency we need.
- **Clean device seams already exist**: `IPolicyValueForward` (inference, M43) per GPU for generation;
  `IPolicyValueTrainStep` (training, M44) on one GPU for the learner.
- **The weight lifecycle is already a broadcast seam.** Per chunk: `_trainStep.SyncToHost()` writes trained weights into
  the CPU master `_net`, then `_forward.OnWeightsSynced(_net)` re-uploads to the device (`SelfPlayCampaign.cs:232-233`).
  For multi-GPU this becomes a fan-out: `OnWeightsSynced` to **every** per-GPU forward, at the same per-chunk cadence.
- **Results already merge by index** into `_window` in ascending game order (`SelfPlayCampaign.cs:189-190`) — an
  index-range shard merges identically, no new reduction.

### What the infra layer assumes (and can keep)
- **Local model store, one campaign.** `FileModelStore` writes `<data>/chess.az.ckpt` + `chess.az-adam.ckpt`
  (`ModelStore.cs:35-36`); `CampaignRunner.Run` drives exactly one campaign to a wall-clock deadline
  (`CampaignRunner.cs:51`). **Single-box multi-GPU keeps one process, one campaign, one store** — generation just fans
  across local GPUs *inside* a chunk. No store/orchestration change needed (this is why single-box ≪ cross-machine).
- **Determinism.** Bitwise DOP-invariance (`ParallelGeneration_ProducesBitwiseIdenticalCheckpoint_AtAnyDop`,
  `SelfPlayCampaignTests.cs:97`) is a **CPU-only** guarantee already forfeited by `--gpu` (GPU float reductions aren't
  associative). Multi-GPU keeps the same stance: opt-in, CPU autograd stays the reproducible reference. Game→GPU mapping
  is by index (a given game always lands on the same GPU for a config) so assignment is reproducible even though the
  bytes aren't.

## 3. Decision

Add **single-box multi-GPU self-play generation** as **library capability + Lab wiring**: enumerate all CUDA GPUs, build
one GPU-resident forward per device, and shard self-play generation across them by game-index range; keep the training
step on the primary GPU and fan the trained weights out to every device's forward each chunk. CPU-accelerator fallback
is unchanged. **Out of scope** (explicitly, to keep the win tractable): data-parallel *training* across GPUs (gradient
all-reduce — ILGPU has no collectives, and training is not the bottleneck), and cross-machine distributed training
(actor-learner over a network — a separate systems project; see §6).

## 4. Design (placement: generic → library)

### 4a. Library — enumerate devices, make a backend device-addressable
`Ilgpu`:
- **`IlgpuBackend`**: split `SelectDevice` into `SelectDevices(context, preferCpu) → IReadOnlyList<Device>` (all CUDA
  devices in order; or the single CPU accelerator when none / `preferCpu`). Add an internal ctor
  `IlgpuBackend(Context, Device)` so a backend can be pinned to a specific device instead of always re-selecting. The
  existing `IlgpuBackend(bool preferCpu)` stays (picks the first, back-compat).
- **`AdaptiveBackend` owns N GPUs, exposes one and all.** Keep `.Gpu` = the **primary** `IlgpuBackend` (device 0) — it
  remains `Backend.Current`, the training device, and the CPU-vs-GPU GEMM router, so nothing about M43/M44 or the
  existing `--gpu` path changes. Add `.Gpus` = `IReadOnlyList<IlgpuBackend>` (all CUDA devices; a single-element list
  wrapping `.Gpu` when there's one GPU; **empty when CPU-only**). One `AdaptiveBackend` instance absorbs the multi-GPU
  complexity — callers just read `.Gpus`. Dispose all.
- Rationale for one-wrapper-owns-N (not N wrappers): keeps the DI singleton and the `Backend.Current` global intact
  (only one backend can be `Backend.Current`), and localizes multi-GPU knowledge behind a deep interface.

### 4b. Hosting — unchanged DI shape
`Ilgpu.Hosting`: `AddGpuBackend()` still registers one `AdaptiveBackend` singleton; that singleton now discovers all
CUDA devices internally. No keyed services / no N registrations needed (§4a makes the wrapper multi-GPU-aware).

### 4c. Lab — the `--gpus` flag + sharded generation (the only game-specific part)
`SelfPlayCampaign` / `ChessLab`:
- **`--gpus`** (parsed in `ChessLab.Run` beside `--gpu`, `ChessLab.cs:82`): `all` (default when set) / an integer count /
  explicit ordinals (`0,1,3`). Absent → today's single-GPU behaviour. Feeds a `SelfPlayOptions` field.
- **N resident forwards.** The campaign builds one `IPolicyValueForward` per selected GPU via a Core-typed factory (the
  Lab supplies `adaptive.Gpus[k].CreateResidentForward(conv)` per device; all Ilgpu knowledge stays in `ChessLab`, as
  M43/M44 already do). Falls back to a single autograd forward when no GPU.
- **Shard generation by index.** In `EvaluateBatch`, route a game's leaf batch to `_forwards[gpuFor(globalIndex)]` where
  `gpuFor(i) = i % nGpus` (or contiguous ranges). Because `DeterministicParallel` already keys each game on its global
  index, the game→GPU assignment is deterministic; with `--parallel` and `MaxDop ≥ nGpus`, games on different GPUs run
  concurrently (each GPU's games serialize only on that GPU's own lock). The generation closure needs the game's GPU id
  threaded through (game index is already available).
- **Training stays on the primary GPU** (`_trainStep` on `.Gpu`), after the join, as today. **Weight fan-out:** replace
  the single `_forward.OnWeightsSynced(_net)` with a loop over all per-GPU forwards, at the same per-chunk cadence.
- Eval/arena (`ArenaVsRandom`/`ArenaVsNet`) stay on the owner thread via `_net.Forward` — not sharded (small, and they
  run between chunks); a later refinement could shard them the same way.

## 5. Phases

- **M45.1 — Library: enumerate + device-addressable backend.** `SelectDevices`, `IlgpuBackend(Context, Device)` ctor,
  `AdaptiveBackend.Gpus`. **Gate:** a test asserts `Gpus` has one entry per CUDA device (on CI/CPU-only boxes: `Gpus`
  empty, `.Gpu` null, everything falls back — unchanged); `DescribeDevices` already proves enumeration works. No
  behaviour change to the existing single-GPU `--gpu` path (its tests stay green). Shippable alone.
- **M45.2 — Lab: `--gpus` + sharded generation + weight fan-out.** N resident forwards, index→GPU routing in
  `EvaluateBatch`, `OnWeightsSynced` fan-out. **Gate:** with one GPU (or CPU), output is identical to today (`--gpus 1`
  ≡ `--gpu`); the DOP-invariance SHA test still passes on the CPU path; a short `--gpus all` run on the single dev GPU
  runs correctly (exercises the plumbing at N=1).
- **M45.3 — Measure (needs a multi-GPU box).** On ≥2 NVIDIA GPUs, report generation throughput vs 1 GPU (target:
  near-linear in GPU count until the owner-thread merge / CPU MCTS bookkeeping saturates). **On the owner's single
  RTX 3060 this cannot be measured** — M45.1/2 deliver the *capability* + an N=1 correctness check; real N-GPU scaling is
  validated wherever a multi-GPU machine is available. Documented honestly, not hidden.

## 6. Risks & explicitly out of scope

1. **Can't measure real speedup on a 1-GPU dev box.** Mitigated: M45.1/2 are correctness + capability (N=1 ≡ today);
   M45.3 scaling is gated on multi-GPU hardware and reported when available. No pretending it's measured.
2. **`Backend.Current` is a single global.** Mitigated by design: generation uses per-GPU **resident** forwards (which
   don't touch `Backend.Current`); the global stays pointed at the primary GPU for training + any residual `Tensor` ops.
3. **Determinism.** Multi-GPU forfeits bitwise reproducibility further (cross-device non-associative reductions), exactly
   as `--gpu` already does. Kept opt-in; CPU autograd remains the reproducible reference; game→GPU mapping is
   index-deterministic. The CPU-only DOP-invariance SHA gate is unaffected.
4. **P.2 Adam-resume gap compounds.** The resident trainer's device m/v moments are never downloaded into the persisted
   `Adam` (`OPTIMIZATIONS.md` P.2; `GPU_RESIDENT_CONV_TRAINER_PRD.md` §Risks) — a `--gpu` resume re-warms the optimizer.
   Multi-GPU doesn't worsen it (training stays on one GPU), but the fix (download m/v in `SyncToHost`) is a shared
   prerequisite for any lossless-resume story. Cross-linked, not solved here.
5. **OUT OF SCOPE — data-parallel training across GPUs.** Replica-per-GPU + gradient all-reduce would need NCCL-style
   collectives ILGPU lacks (hand-rolled), and training isn't the bottleneck after M44. Deferred.
6. **OUT OF SCOPE — cross-machine distributed (the "16-GPU supercluster").** That needs the full actor–learner control
   plane: self-play actor processes, a cross-process/streamed replay buffer, a network sample transport + checkpoint
   distribution, a coordinator, and fault tolerance — none of which exist (the model store is local-FS only,
   `CampaignRunner` drives one campaign). Single-box multi-GPU is the tractable, in-SDK step; the seams it builds
   (per-device resident forwards, index-sharded generation, weight fan-out) are the same ones a distributed harness
   would compose. Roadmap, not this milestone.

## 7. Verification
- M45.1: `Gpus` enumerates one backend per CUDA device (or empty on CPU-only); existing `--gpu` + M43/M44 tests green.
- M45.2: `--gpus 1` ≡ `--gpu` (identical run); CPU DOP-invariance SHA test green; `--gpus all` runs correctly at N=1.
- M45.3 (multi-GPU hardware): generation throughput vs 1 GPU reported; near-linear scaling expected until the
  owner-thread merge saturates.

## 8. Generic (library) vs lab
| Piece | Layer |
|---|---|
| `SelectDevices`, `IlgpuBackend(Context, Device)` ctor, `AdaptiveBackend.Gpus` | **Ilgpu** |
| `AddGpuBackend` (unchanged shape; wrapper now multi-GPU-aware) | **Ilgpu.Hosting** |
| `--gpus` flag, N resident forwards, index→GPU routing, `OnWeightsSynced` fan-out | **Lab** (`ChessLab` + `SelfPlayCampaign`) |

See [PLAN.md](PLAN.md) M45, [GPU_RESIDENT_CONV_PRD.md](GPU_RESIDENT_CONV_PRD.md) (M43 forward),
[GPU_RESIDENT_CONV_TRAINER_PRD.md](GPU_RESIDENT_CONV_TRAINER_PRD.md) (M44 trainer + P.2), [OPTIMIZATIONS.md](../OPTIMIZATIONS.md).
