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

### Reconfirmed on the 690M-sample net + inference-budget strategy (2026-06-16)

Re-ran the diagnostic on the current shipped 1024×4 net (`cube.value-davi-res.ckpt`, ~690M samples,
curriculum at cap d26) to check the finding still holds at far more training. It does — and more sharply:

| evidence (same net, varying ONLY the search given to it) | d14 | d15 | d16 | d17 |
|---|---|---|---|---|
| greedy (0 search) | solid | falls off | ~0 | 0 |
| BWAS 8k exp, w2.5 (in-loop probe) | 87.5% | 50% | 37.5% | — |
| BWAS 50k exp, w1.0 (heavy) | **4/4 @ 14.0qt (optimal)** | — | **2/4 @ 16.0qt (optimal)** | — |
| BWAS 100k exp, w1.0 (heavy) | 100% | 100% | 83% | 42% |

**Verdict: search-budget-bound, not capacity-bound, through ~d17.** Two independent signals: (1) solve rate
at a fixed depth climbs monotonically as the *same* net is given more search (d16: ~0 → 37.5% → 50% → 83%);
(2) every solution found is **optimal-length** — a heuristic that yields optimal solves whenever the search
reaches the goal is accurate, not misleading. Failures are the admissible search exhausting its node budget,
not the net mispredicting. (Capacity-bound would look the opposite: budget-insensitive solve rates and
long/suboptimal solutions.) The depth limit is the solver's search budget, **not the net's brain size.**

**Product goal:** reach scrambles up to god's-number (26 QTM) for as many cubes as possible, and prefer a
*solved* cube (even a few moves longer than optimal) over an honest fail. That reprioritizes the current
optimality-first stance toward solve-success, which points at two inference knobs — **bigger budget** and a
**slightly greedier weight** (>1.5 reaches the goal in fewer expansions, trading exact-optimal length for
reach). Neither needs retraining or data.

**Budget: dynamic (time-bounded) beats a fixed constant.** A constant `maxExpansions` couples latency to cube
difficulty — an unsolvable cube burns the *entire* budget before failing, so one number can't be both
"deep enough" and "fast enough." A **wall-clock deadline** instead bounds the thing the UX actually cares
about (latency) while letting every cube use as much search as fits in the time → maximal reach and
solve-rate under a fixed worst-case wait. Mechanically: thread a deadline (or `CancellationToken`) into
`ValueGuidedSearch.SolveBatched` and check it once per expansion round; the web layer passes ~15–20 s on the
resident-GPU path, a few seconds on the CPU fallback. (Alternatives considered: difficulty-scaled budget from
the net's own `V(start)` estimate — subsumed by a time cap; escalating retry-on-fail — wastes the first
attempt. The time cap is the simplest dynamic scheme that maximizes solved-cubes-per-second-of-wait.)

**Measured — deployed time-bounded solver (2026-06-16, RTX 3060, weight 1.5, 20 s deadline, 6 cubes/depth):**

| depth | solved | mean length | mean ms | worst ms |
|---|---|---|---|---|
| 10 | 6/6 | 10.0qt (optimal) | 286 | 364 |
| 12 | 6/6 | 12.0qt (optimal) | 328 | 351 |
| 14 | 6/6 | 14.0qt (optimal) | 1174 | 4954 |
| 16 | 3/6 | 16.0qt (optimal) | 10461 | 20013 |
| 18 | 1/6 | 18.0qt (optimal) | 18356 | 20033 |

Confirms the design: per-cube effort is dynamic (easy ~300 ms, hard up to the cap), worst-case latency is the
**deadline** (~20 s — the 150k expansion ceiling is never hit), solutions stay optimal-length, and the 20 s
budget now reaches d18 (the old 50k/~13 s budget could not). Caveat: d16 is heavy for interactive use (mean
~10 s, half hit the full 20 s); pushing d16 toward the offline 83% needs ~30 s+, past a comfortable web cap.
Reproduce anytime: `--game cube-davi --net residual --eval-only --time-budget 20 --weight 1.5 --max-exp 150000
--probe-depths 10,12,14,16,18 --episodes 6 --data <dir>`.

**Budget sweep → ship 15 s, not 20 s (2026-06-16, 12 cubes/depth, same seeds across budgets):**

| depth | solved 10s / 15s / 20s | mean ms 10s / 15s / 20s |
|---|---|---|
| 14 | 11/12 · **12/12** · 12/12 | 2211 · 2247 · 2292 |
| 15 | 8/12 · 8/12 · 8/12 | 4930 · 6589 · 8266 |
| 16 | 4/12 · **5/12** · 5/12 | 7407 · 10573 · 13483 |
| 17 | 6/12 · **7/12** · 7/12 | 6766 · 9086 · 11147 |

**15 s Pareto-dominates 20 s:** identical solve rate on every depth, ~2–3 s lower mean latency and 5 s lower
worst case — the extra 5 s only burns time on cubes that won't solve regardless. The 10→15 s step *does* buy a
few solves (d14 92→100%, d16/d17 +1 each), so 15 s is the knee. Shipped the GPU path at **15 s** accordingly
(`CubeController`). (d17 > d16 here is small-sample seed noise — per-cube difficulty isn't strictly monotonic
in depth.)

### Heuristic calibration — the value saturates at ~14 (2026-06-16, accuracy-bound at depth)

`--value-curve` (mean predicted `V(start)` vs scramble depth, 200 cubes/depth, current 690M net):

| depth | 2 | 6 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean V | 1.96 | 6.07 | 10.12 | 11.34 | 12.19 | 12.71 | 13.14 | 13.48 | 13.57 | 13.68 |
| V/depth | 0.98 | 1.01 | 1.01 | 0.94 | 0.87 | 0.79 | 0.73 | 0.67 | 0.62 | 0.53 |

`V` tracks depth almost exactly through ~d10, then **saturates at ~13.7 and flattens past d18** — the net can't
distinguish a d20 cube from a d26 one. This is why search collapses past d16-17 and why more budget doesn't
help: the heuristic gives no gradient to follow once it's saturated. So the deep regime is **accuracy-bound, not
search-bound** — the opposite of d≤14. Training *can* deepen the net (the value has real headroom to climb).

Root cause is a DAVI bootstrap fixed point: deep targets are `1 + V_target(s')`, capped by the saturated target
net, so the ceiling self-reinforces. Compounding it, the campaign **force-advanced the curriculum to d26 while
the value was only accurate to ~d12**, so most deep samples train on capped targets that teach nothing. The
lever is to **consolidate the accuracy frontier** (`--set-curriculum-depth ~16` + `--frontier-bias`): focus every
sample on states whose targets are still meaningful, let the value climb there, and let propagation extend the
accurate band outward — rather than spraying samples to d26. (If consolidation does NOT lift the d14-16 values,
the ceiling is capacity, and the lever becomes a wider trunk via `ResidualMlp.WidenTo` / `--grow-to`.)

### How DAVI accuracy propagates — train outward, gate on mastery (principle)

The saturation above is a symptom of *how* DAVI learns, worth stating plainly so the training recipe follows from it:

- **There is no dataset.** Training states are generated fresh each step by random scrambling — free and infinite.
  We are never short of scrambles (any depth, anytime); scramble supply is never the constraint.
- **The learnable signal is the bootstrapped *target*, not the state.** A state's target is
  `min over moves of [1 + V(neighbour)]`, computed from the net's *own* (target-net) prediction. The only hard
  anchor in the system is `V(solved) = 0`.
- **So accuracy radiates outward from the goal, one shell at a time.** 1 move out is anchored (`1 + 0`); 2 moves
  out is trustworthy only once 1-move states are learned; 20 moves out is trustworthy only once 19-move states
  are. A deep state whose neighbours the net gets wrong has a *garbage* target — neighbours guessing from
  neighbours who are also guessing. Training it can't teach anything; it just reinforces the saturated ceiling.
- **It's a continuous moving frontier, not discrete hand-offs.** You train a *window* (e.g. d1–16) and the
  accurate region grows outward on its own as inner shells firm up — overlapping waves, not "perfect d12, then
  start d14." Shells also overlap: a d16 scramble's solution path runs through d15…d1 and the one-move lookahead
  touches every neighbour, so training near the frontier exercises the inner shells too. The constraint is
  **target quality, not exposure.**
- **The advancement bar is value accuracy, not greedy solve-rate.** A shell is "ready" when `V` is accurate and
  correctly *ordered* enough that the arg-min picks the move toward the goal — modest noise is fine (the `min`
  and target-net averaging smooth it). It does **not** need ~95% greedy solves. Critically, greedy solve-rate
  plateaus ~d10–12 because greedy is myopic, so it would *never* clear a high bar deep even when `V` is fine —
  gating the curriculum on greedy stalls it far too early. Gate on a **value-accuracy / BWAS probe** instead.

**Recipe implication (the campaign's real bug).** The curriculum's **force-advance-on-stall** rule shoved the
depth to d26 whenever progress stalled — i.e. exactly when the inner shells were *not* yet mastered — so it built
deep targets on sand and the value saturated at ~14.

**Fix — IMPLEMENTED (2026-06-16).** `CubeDaviLab`'s advance rule (was `greedy ≥0.6 OR force-advance every 384k
samples`) is replaced by a **value-accuracy gate with no forced advance**: advance d→d+1 only when mean predicted
`V(d)/d ≥ --advance-ratio` (default 0.9), i.e. the value tracks true cost-to-go at the frontier so its one-step
targets for d+1 are trustworthy. Greedy solve-rate is no longer the gate (it plateaus ~d10-12 even where the value
is fine, which would freeze the curriculum). A persistent stall is now surfaced as an informational note
("needs longer training or more capacity") rather than triggering an advance — a stall is a true ceiling signal,
not something to paper over. Smoke-confirmed: a fresh net advances d2→6 on `V/d` (0.99, 1.00, 0.97, 0.94), no
forced steps. (`--set-curriculum-depth` re-pins a resumed campaign back onto the accuracy frontier so the gate can
re-earn the deeper levels honestly.)

**Auto-widen-on-plateau — IMPLEMENTED (2026-06-16): closes the loop into a capability autopilot.** The gate makes
a stall *informative* but doesn't *act* on it. `--auto-widen` adds the missing response: when the frontier loss
**plateaus** (no ≥2% improvement for `--widen-stall-samples`, default 50M) while the gate still can't advance, the
shell is **capacity-bound** (more training has stopped helping) — so the trunk auto-widens one tier (`WidenTo`,
2×, capped at `--max-width`) and training continues. The no-inaccurate-data invariant is preserved throughout:
the widen is function-preserving (accuracy unchanged), the curriculum frontier doesn't move, so the net never
trains past its mastered shell — it just gains the capacity for `V` to climb past the gate. The trigger is
**loss-plateau, not a timer**: if loss is still dropping, more training is working and it won't widen (avoids
premature widening). Smoke-confirmed: a pinned net auto-widened 16→32→64 on plateau, stopping at `--max-width`.
Together the three pieces — value-accuracy gate (advance), loss-plateau auto-widen (capacity), function-preserving
`WidenTo` (warm start) — make a run safe to leave training unattended: it deepens when accurate, grows when
capacity-bound, and never advances onto inaccurate targets.

**Honest ceiling.** Search budget is the cheap near-term lever and extends reliable reach to ~d17–18; it does
**not** get to d26 on its own. Beyond ~d17 the heuristic's accuracy degrades, and no search budget rescues an
inaccurate heuristic (DeepCubeA itself only reaches "solved, ~60% optimal" at the deep end). Pushing the
frontier toward d26 is a *combined* problem — more inference search **and** a more accurate value net (more /
better-targeted training, the F.2 / progressive-growing track). Order of operations: bank the free
inference-budget win first; treat d26 as the long game.

## Web solver backend — resident forward, not host-span (measured 2026-06-14)

The self-taught DAVI solver is wired into the web cube page (`/api/cube/solve-davi`, "Solve
(self-taught AI)" button) as a third solver beside Kociemba and the imitation DQN. It runs BWAS over
the value net. Measured the three backends for one solve on the trained 1024×4 net (weight 1.5,
≤20k expansions):

| depth | CPU (managed) | adaptive (host-span GPU) | **resident GPU** |
|---|---|---|---|
| 10 | 2558 ms | 1784 ms | **274 ms** |
| 14 | 3094 ms | 3019 ms | **468 ms** |
| 18 | 39366 ms | 40691 ms | **5267 ms** |

- **A naive `Backend.Current = AdaptiveBackend` barely helps — and is *slower* at depth 18.** The
  residual forward interleaves GPU GEMMs with CPU-delegated LayerNorm/elementwise ops, so the host-span
  path thrashes the PCIe bus. (Same lesson as the campaign: the win was never host-span.)
- **The resident forward (`DeviceResidualMlp`, weights on device) is 7–10×.** So the web solver uses the
  resident forward when a CUDA device is present (local dev → ~depth-15 reach, 50k budget, ~13 s worst
  case), and falls back to the CPU autograd forward otherwise (e.g. a GPU-less Hetzner container → 6k
  budget, a few seconds, shallower reach). `CubeValueSearch` takes the forward as an injected delegate so
  `Environments` stays free of any GPU dependency.
- **Shipping:** the 35 MB trained net (`models/cube.value-davi-res.ckpt`) is stored via **Git LFS**
  (`.gitattributes`: `*.ckpt`) and seeded into `/data` at container startup. The Dockerfile copies the
  Ilgpu csproj before restore; the GHCR build workflow checks out with `lfs: true` (the build context has
  no `.git`, so LFS must be materialized first — otherwise `COPY models/` ships pointer files).

## Investigated, not pursued — policy-guided BWAS (measured 2026-06-15)

Tested fusing the existing imitation policy net (`CubePolicyNet`) as an action prior into the value
BWAS: `f = g + weight·h + policyWeight·(−logπ(a|parent))`. **Negative result** — it does not help and
hurts at higher weight (one solve, 1024×4 value net, weight 1.5, ≤20k exp):

| depth | value-only | policy λ=1 | policy λ=2 |
|---|---|---|---|
| 14 | len 14, 4547 ms | len 14, 5060 ms | len 14, **25563 ms** |
| 16 | fail, 47 s | fail, 49 s | fail, 48 s |

- The imitation policy is **Kociemba/HTM-flavoured (~73% acc)** and pulls *against* the QTM value's optimal
  path — stronger prior = worse. It also never rescued depth 16.
- Deeper point: **value-only BWAS is already the DeepCubeA approach** (DeepCubeA uses value-only batch
  weighted A*, no policy). A *useful* policy would have to be QTM-consistent (self-distilled from the
  value's arg-min), which is its own training run — and DeepCubeA's own result says value-only suffices.
- **Conclusion:** the lever for harder cubes is **value-net capacity** (F.2 / a bigger fresh campaign),
  not a search-time policy. The prototype + the generic `SolveBatchedWithPolicy` were reverted (no dead
  code); recoverable from git history if a self-distilled policy is ever trained.

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
