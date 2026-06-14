# Optimizations — done & planned

A running ledger of every performance / capability optimization in the SDK, with measured impact
and the milestone/commit it landed in. "Impact" numbers are measured on the dev machine
(RTX 3060 Laptop GPU + Intel Iris Xe iGPU, 8-core CPU) unless noted. Cross-references: `PLAN.md`
(milestones), `PRD.md` §4/§10 (compute seam & GPU strategy).

Legend: ✅ done · 🔜 planned · ⏳ in progress

---

## 1. CPU compute

| # | Optimization | Status | Impact |
|---|---|---|---|
| 1.1 | **SIMD GEMM** — BCL ships no matmul, so GEMM is hand-rolled `i-k-j` saxpy over `TensorPrimitives.MultiplyAdd`, all inner loops on contiguous rows; zero-valued `a[i,p]` skipped. | ✅ M2 | thousands of Adam steps/s on the classic-control net (one core) — the v1 baseline |
| 1.2 | **Multithreaded CPU GEMM** — large GEMMs partitioned across cores by **disjoint output rows** (never a reduction), so results stay **bitwise-identical** regardless of worker count (preserves determinism). Threshold-gated (≥ ~1M MACs) so small latency-bound GEMMs stay sequential. | ✅ M12a (`dafebc4`) | GEMM **3.95× on 8 cores**; full cube-1024 Adam step **2.52×** (Amdahl: only GEMM parallelizes) = 9,240 samples/s |

## 2. GPU compute (ILGPU)

| # | Optimization | Status | Impact |
|---|---|---|---|
| 2.1 | **ILGPU backend** — C# GEMM kernels JIT-compiled to PTX (CUDA), with a CPU-accelerator fallback so CI/GPU-less machines stay green. Same `IComputeBackend` seam as the managed backend. | ✅ M12c (`de362c3`) | unlocks the GPU path; host-span v1 wins only on large GEMMs |
| 2.2 | **Device selection prefers discrete CUDA** — on a laptop with both an Intel iGPU (OpenCL) and an NVIDIA card, ILGPU's `GetPreferredDevice` picked the weaker iGPU; now CUDA is selected first. | ✅ (`bcf9333`) | uses the 3060, not the iGPU |
| 2.3 | **`AdaptiveBackend`** — routes each GEMM to CPU (small/medium) vs discrete GPU (large) by a MAC-count threshold; pure CPU when no GPU. No knobs for the caller. | ✅ (`74a77e5`) | best-of-both automatically; CPU still wins < ~256M MACs (host-span transfer) |
| 2.5 | **Complete compute seam** — every autograd op (not just GEMM) now routes through `IComputeBackend` (opcode `Map`/`Zip` + reductions/LogSoftmax/Gather/Huber/LayerNorm), `ManagedBackend` the bitwise-identical reference. The general-port "Phase 1"; a backend can now run the whole fwd+bwd+Adam graph. | ✅ general port Phase 1 | architectural — no speed change; CPU stays the deterministic default |
| 2.4 | **Tiled GEMM kernel** — replaced the naive one-thread-per-output kernel with a **shared-memory tiled** kernel (load tile → `Group.Barrier` → multiply-accumulate → barrier). One generic `GemmDims`-parameterized core (rows/cols/reduction + per-operand strides + accumulate-vs-write) serves all three layouts (A·B, Aᵀ·B, A·Bᵀ) + the resident write. **Adaptive tile**: 16 on GPU, capped to the CPU accelerator's group limit, shared mem sized at compile-time max so one kernel serves every device. | ✅ M19 (`5295576`) | naive→tiled (resident operands): **1.2× @256³, 1.4× @1024³, 2.3× @2048³ → up to 620 GFLOP/s**; gain grows with size |
| 2.6 | **Register-blocked GEMM** — each thread computes a **4×4 micro-tile in explicit registers** (a `LocalMemory`/array accumulator lands in slow off-chip local memory under ILGPU and *loses* — named scalars stay in registers) over `RegBK`-deep shared tiles; a groupEdge×groupEdge group covers a (groupEdge·4)² block, groupEdge adapting to the device limit (16 on GPU). The production GEMM path (host-span + all resident paths) routes through it. | ✅ P.1/M19b | tiled→reg-blocked: **2.2× @256³, 3.2× @1024³/2048³ → ~2.0 TFLOP/s** (7.4× the naive kernel at 2048³); the throughput multiplier for the GPU-bound campaign |

## 3. GPU residency (cut host↔device transfer)

The host-span backend uploads operands + downloads results **every GEMM call**, so on the small/medium
nets transfer dominates and ~98% of the 3060 sits idle. These remove the transfers.

| # | Optimization | Status | Impact |
|---|---|---|---|
| 3.1 | **Scoped device-resident forward** — `IlgpuBackend.MlpForwardScalar` runs a whole scalar-MLP forward on-device (upload input once, GEMM→bias→ReLU chained resident, download only the scalars), killing per-layer round-trips. | ✅ M12c-perf (`a30d7a0`) | **~2× DAVI throughput** (500 iters 20s vs 40s) |
| 3.2 | **Resident weights (`DeviceMlp`) + `ITargetForward`** — weights live resident on the device and re-upload **only on the trainer's target-net sync** (not every step), via a Core-side `ITargetForward` seam (`Forward` + `OnTargetSynced`) the trainer drives. | ✅ M20 Stage 1 (`5295576`) | weight upload **per-step → per-sync (~200× fewer)**; campaign threw 30.5k→121k iters overnight (~3× more than the pre-optimization run) |
| 3.3 | **Resident residual forward (`DeviceResidualMlp`)** — the residual analogue of `DeviceMlp`: the full forward (GEMM + bias + **GPU LayerNorm** + ReLU + residual skip-add + head) chained on-device, weights resident, re-synced on target update. Needed a forward LayerNorm kernel (one thread/row) + an elementwise-add kernel. The residual net's successor eval no longer round-trips per GEMM. | ✅ M20 Stage 2 | residual DAVI **~2× iters/s** (4.5 vs 2.3 it/s, width-1024×4); the successor-eval transfer is gone — the CPU-bound autograd train step became the next bottleneck (→ 3.4) |
| 3.4 | **Fully device-resident train step (`DeviceResidualTrainer`)** — the whole DAVI step on-device: forward (caching x̂/σ + post-ReLU), full backward through the residual chain, global-norm clip, on-device Adam. Online weights mastered on the GPU; synced to the CPU net only for eval/checkpoint/target copy, via the Core-side `IResidentTrainStep` seam. New kernels: bias-grad (column sum), ReLU-backward, LayerNorm input-grad + γ/β-grad, Huber-grad, Adam-update, sum-of-squares (clip), scale, caching-LayerNorm-fwd. GEMM-transpose grads reuse the tiled kernel via `GemmDims.AtB/ABt`. | ✅ M20 Stage 3 | residual DAVI **~11 iters/s** (vs 4.5 Stage 2, 2.3 host-span) — **~4.8× end-to-end**; gradients verified to match the autograd path within tol |

## 4. Algorithmic / training efficiency

| # | Optimization | Status | Impact |
|---|---|---|---|
| 4.1 | **Batched greedy eval** — `GreedyValuePlanner` evaluates all 12 successors of a step in ONE forward instead of 12 tiny ones. | ✅ (`36d40ae`) | **~2.4× faster eval** |
| 4.2 | **DAVI raw cost-to-go (`DistanceScale=1`)** — predicting raw moves (not squashed to ~0.1) keeps targets well-separated so gradients can resolve them. | ✅ M18 (`db2027e`) | the difference between "solves only depth-1 freebies" and **≥80% teacher-free** |
| 4.3 | **Stall-fallback curriculum** — advance scramble depth at ≥0.6 solve-rate **OR** force-advance every 3000 iters; exposure to deeper states trains the value even where greedy never masters. | ✅ (`22ee724`) | unstuck the depth-7 plateau under the old 0.95 gate; reached curriculum depth 16 |
| 4.4 | **ε-loss target sync** — optional gate so the bootstrap target advances only once batch loss < ε (DeepCubeA trick): stops a target chasing a still-moving net at depth. | ✅ M21c (`ValueIterationOptions.TargetUpdateLossThreshold`) | stability lever for deep training |
| 4.5 | **Value-guided A\*** (`ValueGuidedSearch`) — weighted A* over the model, guided by the learned value; reaches states a greedy descent gets stuck on, **no retraining**. | ✅ ceiling-raiser #1 (`15f8fa3`) | on the depth-16 net: **greedy ~5% → A\* ~95% at scramble depth 11**, optimal through depth 8 |
| 4.6 | **Batch-weighted A\* (BWAS)** — expands the best N open nodes at once and scores **all their successors in one batched forward** (the value net is the dominant cost). Goal-on-pop ⇒ optimal at weight 1 under an admissible value. | ✅ M21a (`SolveWithSearchBatched`) | makes deep value-guided search usable / GPU-amortized (vs ~17s/solve per-node at depth 11) |
| 4.7 | **Residual value net + LayerNorm** — `ResidualMlp` (input proj → N residual blocks with LayerNorm, scalar head); LayerNorm (not BatchNorm) is stable under the frozen-target bootstrap. Depth-with-residuals raises the cost-to-go ceiling that width alone plateaued on (M17). | ✅ M21b | capability: pushes the depth where the value stays accurate (campaign pending) |
| 4.8 | **Full-state resumable campaigns** — Adam moments + curriculum depth + iteration count + sampler RNG persisted, so a restart continues seamlessly instead of re-warming the optimizer / re-climbing the curriculum. | ✅ (`9b79ba6`) | no lost progress across restarts |

## 5. I/O & campaign throughput

| # | Optimization | Status | Impact |
|---|---|---|---|
| 5.1 | **Parallel Kociemba data generation** — imitation solves run concurrently; the Lab uses all cores. | ✅ (`225f504`) | data generation scales with cores |
| 5.2 | **Algorithm-agnostic transition store** — `ReplayBufferCheckpoint` factored out of `DqnTrainingState` byte-identically, so the same recorded experience survives an algorithm switch. | ✅ (`6747123`) | "switch algorithm, keep the work" (PRD goal 8a) |
| 5.3 | **Parallel successor generation + bigger batch** — DAVI's ActionCount× `Apply`+featurize fan-out (pure, disjoint writes) runs under `Parallel.For` across the batch; `cube-davi --batch <n>` feeds the GPU more per step. Removed the CPU stall that left the GPU bursting-then-idle. | ✅ P.7 (`ee48bbb`) | **GPU 0% → 95–100% util** (16 W → ~85 W) at batch 512 on the residual campaign; determinism preserved |

---

## Learning-curve levers — what actually speeds time-to-solve-depth (findings 2026-06-14)

After Stages 1–3 + P.7, the residual DAVI campaign is **GPU-bound at ~620 GFLOP/s** (nvidia-smi
95–100% at batch 512). Measured comparison on the 1024×4 residual net:

| batch | it/s | samples/s | GPU |
|---|---|---|---|
| 128 | 11 | ~1,410 | idle (0%) — CPU-bound on successor gen |
| 512 | 3.2 | ~1,630 | saturated (95–100%) |

**Key conclusions:**
- **Batch size is ~throughput-neutral** (+16% samples/s 128→512) — it just moves the bottleneck from
  CPU/idle-GPU to GPU-saturated. **Not a learning-curve lever.** Worse, with an *iteration-paced*
  curriculum a bigger batch advances ~3.4× slower in wall-clock (fewer iters/s). Fix the pacing, don't
  chase batch.
- **Net width is not a lever either** (M17: diminishing; and wider = quadratically more GFLOP on a slow
  kernel). See F.1.
- **The throughput lever is the GEMM kernel itself** — we're GPU-bound. ✅ **Done (P.1/§2.6,
  register-blocked GEMM): 2.2–3.2× the tiled kernel → ~2.0 TFLOP/s.** Directly multiplies the learning
  curve for the GPU-bound campaign.
- **Cheap per-update / pacing wins (P.8, P.9)** cost almost nothing and stack on top. ✅ Done.

## Capability findings — net vs. search ceiling (measured 2026-06-14)

After the 236k-iter residual campaign (1024×4), the **live** instrumentation looked plateaued: the
greedy eval collapses ~d10–11 and the in-loop **BWAS capability probe** (8 cubes, ≤8k expansions, w=2.5)
was flat from 82k→232k iters (d12 ~7–8/8, d14 5/8, d16 3/8). That flatness was **not** the net hitting a
wall — it was the probe's tiny expansion budget capping what any heuristic can solve at depth.

A **heavy-search diagnostic** on the same 236k checkpoint (BWAS **w=1.5, ≤100k expansions**, 12 cubes/depth)
told a completely different story:

| depth | 1–15 | 16 | 17 |
|---|---|---|---|
| solved | **12/12 every depth** | 10/12 | 5/12 |
| mean QTM | **= depth (provably optimal-length)** | 16.2 | 17.0 |
| vs Kociemba QTM | ~2–2.5× shorter (e.g. d15: 15 vs 30.2) | beats on all solves | beats on all solves |

**Conclusions:**
- **The "plateau" was a search-budget artifact, not a network-capacity ceiling.** The net's value
  heuristic stays accurate (yields optimal-length solutions) out to **depth 15**, then degrades
  *gradually* (d16 83%, d17 42%) — the shape of a strong heuristic running out of reach, not a wall.
- **The in-loop instrumentation undersold capability badly** (greedy ~d10, light probe ~d14-partial vs
  the real ~d15-optimal). The live curve is the wrong yardstick for this net.
- **The cheapest, biggest near-term capability gain is eval-time search budget** — more expansions,
  weight→1, a wider batched frontier. We already have BWAS; it's a knob, not a retrain. → **P.10.**
- **A wider/deeper net is *not* the next lever.** It isn't the bottleneck through d15; it would only help
  push the d16+ frontier, and even there more search buys depth first. Reframed in the Planned table.

## Planned

| # | Optimization | Milestone | Why / expected |
|---|---|---|---|
| P.10 | **Eval-time search budget = the capability lever** — expose/tune BWAS expansions, weight (→1 for provable optimality), and frontier batch size at solve time. The 2026-06-14 diagnostic showed the *current* net solves **optimally through depth 15** with ≤100k expansions; the light probe's 8k budget hid this. Also: heavier in-loop probe so the live curve stops underselling capability. | cube-davi / web | **biggest near-term capability gain, no retraining** — converts the existing net into deeper solves. Bounded by net quality only past ~d15. |
| P.8 | **Sample/time-paced curriculum + lighter eval** — advance the scramble-depth curriculum on samples (or wall-clock), not iteration count, so batch size stops distorting pacing; and run the in-loop eval less often / fewer episodes (it's pure overhead at depth). | cube-davi | removes a real wall-clock distortion + reclaims training time |
| P.9 | **Per-update efficiency: LR scaling + ε-loss target sync** — LR was tuned for batch 128; the bigger batch supports a higher LR (linear-scaling rule). Enable `TargetUpdateLossThreshold` (DeepCubeA ε-sync — already built) so the bootstrap target only advances once loss converges. | cube-davi | faster, more stable convergence per update — near-free |
| P.2 | **Resident Adam-state checkpointing** — `DeviceResidualTrainer` keeps Adam moments on-device; they aren't yet downloaded into the campaign's Adam checkpoint, so a resumed resident run re-warms the optimizer (net weights resume fine). Add m/v download/upload to make the resident path's resume lossless. | M20 Stage 3 follow-up | lossless resume for the resident residual campaign |
| P.5 | **Recalibrate `AdaptiveBackend` threshold** — the 256M-MAC crossover was measured against the *naive* host-span kernel; re-measure now that the tiled kernel + residency change the crossover. | M19/M20 follow-up | route more work to the GPU correctly |
| P.6 | **Lab gen/train double-buffering** — overlap oracle data generation with training instead of competing for cores. | backlog | modest end-to-end gen+train speedup |

### Far future (someday / only if the workload changes)

| # | Optimization | Why it's parked |
|---|---|---|
| F.2 | **Wider / deeper residual net (1024→4096+, more blocks)** — raise the heuristic ceiling past depth 15. | **Parked behind P.10.** The 2026-06-14 diagnostic showed the 1024×4 net is *not* the bottleneck through d15 — eval-time search is. Width only helps the d16+ frontier, and even there more search buys depth first and cheaper. Revisit only after P.10 is exhausted; it's a GPU-day commitment (4096-wide ≈ 16× the per-iter GEMM, and we're GPU-bound) that the M17 "width is diminishing" result already cautions against. |
| F.1 | **Device-backed `Tensor` (general-port Phase 2)** — make `Tensor.Data/Grad` device-resident with per-op GPU dispatch (the second half of the `IComputeBackend` device-handle port; Phase 1 — the complete compute seam — is done, §2.5). | **Parked, low priority.** At our scale it's a large central-type rewrite for no measured speedup: the autograd is fine-grained + RL-sized, and the GPU *loses* to the multithreaded CPU below ~256M MACs (measured — why `AdaptiveBackend` keeps small GEMMs on CPU). Real GPU perf comes from **fusing** whole-net fwd/bwd into few kernels, which the resident paths (§3.2–3.4) already do. **Only worth it** with a fused/lazy GPU graph executor **and** large-batch (4096+) training where per-op GPU dispatch finally wins. |

---

*Keep this current: when an optimization lands, move its row from Planned to the matching section
with the measured impact and the commit hash.*
