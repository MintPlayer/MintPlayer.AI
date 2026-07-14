# Convolutional residual policy/value net (Core) + chess adoption — PRD

**Status:** M42.1 + M42.2 ✅ SHIPPED 2026-07-13 (merged to master via PR #31) · M42.3 🟡 **partial** — a real training pathology (draw-collapse) was diagnosed + fixed (commit `282c665`) and the self-play material regression stopped, BUT genuine playing-strength gains are **not yet demonstrated**: over a fair 40-game eval the ladder tiers don't beat the barely-trained baseline, and both available metrics saturate/are self-referential (see below). Needs a non-saturating strength eval + far more scale · M42.4 🟡 steps 1+3 done (`c1c7d8e`), steps 2+4 (browser wiring) remain. **No conv tier is shippable yet** (none proven stronger than the current MLP demo).
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M42 · **Depends on:** M41 (parallel self-play — makes the training iterations this needs affordable) · **Supersedes** the "flat MLP is the ceiling" note in [CHESS_WEB_POLYGLOT_PRD.md](CHESS_WEB_POLYGLOT_PRD.md) and [CHESS_SELFPLAY_PRD.md](CHESS_SELFPLAY_PRD.md).

## Implementation status (what actually shipped vs. this design)
- **M42.1** `Tensor.Conv2D` (commit `67806af`) — im2col → existing GEMM → col2im, in `Core/Numerics/TensorConv.cs`.
  Confirmed **no `IComputeBackend`/ILGPU change was needed** (the design's preferred route). Rank-2 `[N, C·H·W]`, LayerNorm
  reused. Gate MET: 3 finite-difference gradient checks (3×3 SAME, stride-2 valid, 1×1).
- **M42.2** `IPolicyValueNet` + `ConvResidualPolicyValueNet` + arch-agnostic campaign (commit `21b779d`). Per §4b/§4c,
  with an `IPolicyValueNetBuilder` (Mlp/Conv) as the arch factory. **The de-risking two-headed-residual-MLP checkpoint
  (§M42.2) was SKIPPED** — went straight to the conv net. Gate MET: head shapes, exact save/load round-trip, loss falls;
  MLP self-play determinism gate still byte-identical (zero behaviour change).
- **M42.3** ⏳ conv training (`--arch conv --filters 64 --blocks 6`) — runs offline via the Lab (branch
  `m42-chess-conv-net`, off master). Gate unchanged: beat the MLP baseline / ≥1 ladder tier promotes.
  - **Perf fix that unblocked it (commit `71fe44c`): parallelize eval + ladder arena.** The conv net is ~10–50× heavier
    per MCTS node than the MLP, and that surfaced a bottleneck the PRD's cost analysis (risk #3) missed: **it wasn't
    self-play generation — it was the *measurement* phase.** After each chunk, on the owner thread, `ArenaVsRandom`
    (`--eval-games` games) + `ArenaVsNet` (`--arena-games` games) ran **sequentially**, single-threaded, at full conv
    cost — and because they run between chunks, **they stalled training itself** (no gradient steps while the arena
    grinds; observed ~24 min at ~0.8 cores per cycle, one eval every ~30 min). The PRD left them sequential assuming a
    cheap MLP; false for conv. Fix: `ArenaVsRandom`/`ArenaVsNet` now run their independent, inference-only games on the
    **same `DeterministicParallel` primitive** self-play uses (one base seed per call → per-game RNG a pure function of
    `(cycle, game index)`). Inference-only ⇒ trained weights are untouched: the DOP-invariance checkpoint test still
    passes **bitwise**, and eval/arena metrics are reproducible + DOP-invariant by construction.
  - **New knob (same commit): `--max-plies`** (default 200). A chunk's wall time is bounded by its **slowest** game, so
    the ply cap — not the average game — sets self-play throughput; a weak net rarely mates, so games otherwise run to
    the cap. Lower it for a heavy net to keep evals frequent.
  - **Tuned config for the conv net's per-node cost:** `--sims 64 --games 16 --max-plies 100 --eval-games 8
    --arena-games 12 --parallel --ladder --material-weight 0.5`. (256 sims / 200 plies / 10+20 sequential eval games —
    the MLP-era defaults — made a single chunk+eval cycle take 20–30 min; this cut it to ~4–5 min/chunk with regular
    evals, all cores busy in *both* phases.)
  - **Early trend (positive — conv is learning, not plateauing like the MLP):** by ~48 self-play games, policy loss
    **8.09 → 6.78** (falling off the ~8.45 uniform), value loss **0.235 → 0.096**, and material margin vs the Level-1
    baseline **+0.50 pawns** (climbing toward the +0.75 promote gate). Contrast the MLP plateau (§1): policy barely moved
    2.35→2.22 over 500 games, margin stuck ~+0.1, no tier ever promoted. The concrete gate is a **merit tier promotion**
    (Level 2+ on material/head-to-head, not the automatic Level-1 baseline).
  - **Root-cause fix — DRAW-COLLAPSE (commit `282c665`).** Once the arena noise was removed (`--arena-games 40`), the
    trustworthy signal showed the conv net *regressing*: material vs its own baseline slid −2→−9 pawns while value loss
    fell to ~0.03 and winRate-vs-random stayed pinned at 50%. Diagnosis: a weak net that can't force mate + a short ply
    cap ⇒ **nearly all self-play games hit the cap as draws (z=0)** ⇒ the outcome signal vanished ⇒ the net trivially
    learned "value=0 always" and collapsed onto passive, shuffle-to-the-cap play, bleeding material to any real
    opponent. **My earlier throughput tuning (low sims + short plies) *caused* it** by starving the outcome signal —
    and the earlier "noisy-arena +1.00 @g160 promotion" was arena-12 noise, not real strength. Fix: **material-adjudicate
    ply-capped games** — a `GameResult.Ongoing`-at-cap position with a decisive material edge (≥1.5 pawns) trains as a
    win/loss instead of z=0, so non-mating games carry a real signal (no-op for materialless games like Connect-4, so
    the DOP-determinism test stays bitwise-green). **Result: collapse broken** — same fast config went from −9 pawns to
    **+3.78 and a merit Level-2 promotion** at the same game count.
  - **Honest limit — strength gains UNPROVEN; the metrics saturate.** The ladder promoted merit tiers, but on
    **8-game-noisy** evals + the **material** metric (which adjudication amplifies *within self-play* but doesn't
    necessarily translate to external strength). A fair **40-game winRate-vs-random** ranking of the captured tiers
    (constant sims/seed) came out ~50–59% and **did not beat the barely-trained baseline** (L1 58.8%, L3 52.5%). So the
    conv net at 64f/64-sim after ~100–200 games does **not** demonstrably play stronger chess than its own baseline.
    Two caveats keep this from being purely negative: (1) `winRate-vs-random` itself **saturates** — any net that draws
    random but can't mate it scores ~50%, so it can't cleanly rank these tiers either; (2) the draw-collapse fix is
    real (the non-adjudicated run's material slid to −9; adjudication stopped that regression). **Real gap:** there is
    **no non-saturating strength metric** — both signals are saturating (winRate) or self-referential/gameable
    (material-in-self-play). **Next steps (evidence-backed):** (a) add a non-saturating eval — play vs a simple
    material-greedy or depth-2 minimax opponent — to actually *measure* strength; (b) only then scale **training volume
    (AlphaZero needs far more than ~200 games)**, **net capacity (`--filters/--blocks`)**, and **`--sims`**. Doubling
    sims to 128 overnight did **not** lift the ceiling, consistent with capacity/volume being the real limit.
    *(Full run-by-run detail: `data/chess-conv-autorun-log.md`.)*
- **M42.4** 🟡 **steps 1+3 DONE (commit `c1c7d8e`)**: the conv forward is single-sourced in `chess_solver.pg`
  (`PgConvNet`, direct nested-loop conv + whole-row LayerNorm + residual tower + both heads) and a C# parity test
  (`ChessNetParityTests`) proves it matches `ConvResidualPolicyValueNet.Forward` on real conv `.ckpt` bytes (<2e-3).
  Dispatch is via a nullable `PgPolicyValueNet.conv` field (no interface feature in the `.pg`; filed
  [MintPlayer.Polyglot#29](https://github.com/MintPlayer/MintPlayer.Polyglot/issues/29)), so the MLP browser path is
  unchanged. **Steps 2+4 remain** (TS conv parser in `chess-net.ts` + regen `chess_solver.ts` via the CLI at
  `C:\Repos\MintPlayer.Polyglot` + wire `loadChessNet`/`chess-director.ts` + copy the chosen conv tier into
  `wwwroot/models`) — best done interactively (regenerating the committed `.ts` blind risks the live MLP `/chess` page).

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
3. **Training cost** — a conv tower is far heavier than the MLP; **this is exactly why M41 (parallel self-play) lands first.** The im2col+GEMM route keeps the heavy math on the GPU-routed GEMM. **Realized nuance (M42.3):** M41 parallelized self-play *generation*, but the conv net's per-node cost exposed a *second* sequential hot path this analysis missed — the eval + ladder **arena**, which run on the owner thread between chunks and so stall training. Both must be parallel for a heavy net; the M42.3 perf fix (commit `71fe44c`) does that. Lesson: with an expensive net, audit **every** per-chunk phase for hidden single-threaded work, not just generation.
4. **Browser perf** — a conv forward per MCTS node in JS may be slow; if so, cap browser sims per difficulty tier (the tier system already sets per-tier sims) or ship a smaller `filters/blocks` for the web than for the strongest offline tier. Flag during M42.4.
5. **Checkpoint/interface churn** — the `IPolicyValueNet` refactor touches the shared self-play path; the SHA-identical-MLP gate (M42.2) guards against a silent regression to connect-4/chess.

## 7. Verification
- M42.1: conv gradient (finite-diff) + forward-value unit tests; DOP-invariance.
- M42.2: full suite green with MLP behaviour SHA-identical; conv train-one-batch loss decrease; connect-4 unchanged.
- M42.3: chess conv run **beats the MLP baseline** on material margin / winRate with ≥1 tier promoted; determinism preserved.
- M42.4: `ChessNetParityTests` green on conv `.ckpt`; `/chess` plays a conv tier end-to-end.

## 8. Scale-out & Deferred (postponed — for a well-resourced training run)

The pipeline is a correct **single-machine** trainer. Two items were built to remove scaling ceilings; the rest are
deliberately deferred (they only pay off on a GPU cluster / long run, and shouldn't be retrofitted speculatively).

**Shipped (scale-readiness):**
- ✅ **Batched leaf inference** — `Mcts.SearchBatched` (virtual-loss; evaluates a wave of MCTS leaves in one
  `net.Forward`) + `--leaf-batch`, so self-play isn't stuck at batch-1. Opt-in; `leafBatch=1` is bitwise-identical to
  the sequential path (proven by test). Commit `4801c98`.
- ✅ **`--gpu`** wiring — installs the ILGPU `AdaptiveBackend` as `Backend.Current` (was hardcoded CPU-only for chess).
- ✅ **De-ceiling knobs via a `SelfPlayOptions` record** (commit `8c382ae`) — folded the campaign's ~20-param
  telescoping constructor into one options record and exposed the previously-hardcoded knobs a large run needs:
  `--window` (replay capacity), `--batch`, `--epochs`, `--clip`, `--temp-moves`, and the MCTS
  `--cpuct`/`--dirichlet-alpha`/`--root-noise`. Defaults reproduce prior behaviour bitwise (determinism test green).

**Deferred (postponed):**
1. **GPU-*resident* batched forward for the conv net (a conv analogue of `Ilgpu/DeviceMlp`).** Today the conv net
   routes through `Backend.Current`, which re-uploads weights per GEMM; the cube's value net keeps weights on-device
   across steps via `DeviceMlp`/`ITargetForward`. A conv `DeviceMlp` is the one piece that would (a) make the chess GPU
   path as efficient as the cube's and (b) **unify the two families' GPU inference** (see `ARCHITECTURE.md` §4, "Two
   distinct neural-search families"). **→ ✅ BUILT** (M43, 2026-07-14): a two-headed `IPolicyValueForward`
   Core seam + an Ilgpu `DeviceConvPolicyValueNet` (needed only **two new kernels** — device im2col + scatter/bias);
   correctness verified on the ILGPU CPU accelerator, on-GPU throughput pending a CUDA box. Full write-up in
   **[GPU_RESIDENT_CONV_PRD.md](GPU_RESIDENT_CONV_PRD.md)**.
2. **Distributed actor→learner topology** — many self-play workers feeding a central trainer + weight broadcast
   (the standard multi-GPU AlphaZero layout). Single-process today; would be built from scratch. **Postponed.**
3. **Quality features a strong run needs** — WDL/categorical value head, auxiliary targets (moves-left, material),
   input history planes, larger default `--filters/--blocks`. Research-backed (Leela/KataGo); **postponed** behind the
   scale decision.

See [PLAN.md](PLAN.md) M42 and [OPTIMIZATIONS.md](../OPTIMIZATIONS.md).
