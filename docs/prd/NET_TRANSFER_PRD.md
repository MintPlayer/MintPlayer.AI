# SDK — Function-Preserving Net Transfer & Input-Grow (PRD & Plan)

> The SDK's "switch algorithm, **keep the work**" thesis (PRD §8 asset portability) says learned weights should
> survive when you iterate. Today three function-preserving transfers exist but are **scattered one-offs**
> (`ResidualMlp.WidenTo`, `DuelingQNet.ToNoisy`, `CubePolicyNet.PolicyAsMlp`), the param-copy loop is **duplicated
> three times** (`CopyFrom`), and the one axis with a recurring consumer — **growing the input when an environment's
> observation gains features** — is **missing entirely**, forcing a full from-scratch retrain every time. This PRD
> adds that missing primitive and gives the transfer mechanics a single, well-structured home.

- **Status:** Draft v1.0 · 2026-06-30 (analysis complete; implementing now)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** `Core/Nn` (`IValueNet`, `Linear`, `Mlp`, `DuelingQNet`, `ResidualMlp`, `NoisyLinear`), the
  checkpoint stack (`DuelingQNetCheckpoint` et al.), and the asset-portability roadmap in `PLAN.md` /
  `PRD.md` §8. Companion to `FRUITCAKE_BIGFRUIT_INPUTS_PRD.md` (its §4.B optional warm-start path is this feature).

---

## 1. Motivation (the recurring pain, not one game)

When an environment's observation grows — a richer feature is added — the first layer's weight matrix changes shape
(`[oldIn × H] → [newIn × H]`). Every checkpoint enforces an **exact-shape** load (`DuelingQNetCheckpoint.Read`
throws on mismatch; `FruitCakeModelService` has a width guard), and there is **no weight-transfer/layer-grow
utility** — so the only option today is **retrain from scratch**. This already cost a full retrain once (FruitCake
obs 41→83) and **will recur** every time any env's observation is enriched. That is precisely the work the SDK's
"keep the work" thesis says we shouldn't throw away.

This is justified by the **recurring cross-env need**, deliberately **decoupled from the FruitCake big-fruit
experiment** (whose net is documented as saturated and may show no gain — a weak basis for an SDK feature). The
feature's value stands whether or not that experiment wins.

## 2. What already exists (and why it stays where it is)

| Transfer | Axis | Home | Keep as-is? |
|---|---|---|---|
| `ResidualMlp.WidenTo` | hidden **width** (Net2WiderNet) | `ResidualMlp` | ✅ needs block/LayerNorm structure — belongs on the net |
| `DuelingQNet.ToNoisy` | **plain→noisy** head swap | `DuelingQNet` | ✅ head-type-specific — belongs on the net |
| `CubePolicyNet.PolicyAsMlp` | **trunk extraction** | `CubePolicyNet` | ✅ head-layout-specific — belongs on the net |
| **input-dimension grow** | **first-layer in-features** | — | ❌ **MISSING — this PRD adds it** |
| `CopyFrom` (exact param-zip) | identical-shape sync | duplicated ×3 | ⚠️ **dedup into the shared home** |

Forcing the three structural transforms into a generic utility would leak each net's internals (a shallow-module
anti-pattern). The *generic* mechanics — the exact param copy and the new input-grow — are what belong in one place.

## 3. Goals & Non-Goals

### Goals
- A **function-preserving input-grow** primitive: `IValueNet.GrowInput(newInputSize)` returns a net of the larger
  input width whose output is **bitwise-identical on the original features** (new in-weights zero-init), with all
  other learned parameters carried over exactly.
- A single, documented **`NetTransfer`** home for the generic weight-transfer mechanics; the three duplicated
  `CopyFrom` bodies collapse onto it. The net-specific structural transforms get a discoverability cross-reference,
  not a forced move.
- Tests proving function-preservation + a documented transfer story (ARCHITECTURE.md).

### Non-Goals
- Auto-growing inside the trainer/checkpoint-load on a width-mismatch resume (a natural *next* consumer — left as a
  documented follow-up so this PR stays a clean, testable primitive).
- Input-grow for the policy nets (`CubePolicyNet`/`RushHourPolicyNet`) — their observation sizes are compile-time
  constants; no consumer, so no code (general-purpose *somewhat*, not speculative).
- Hidden-width/depth growth for non-`ResidualMlp` nets (no second consumer yet — generalize `WidenTo` only when one
  appears).
- Changing the strict-exact checkpoint validation (it's a deliberate safety invariant; grow happens **before** the
  shape check, on an already-loaded net).

## 4. Design

### 4.A The primitive — `IValueNet.GrowInput(int newInputSize)`
`Linear.Weight` is `[in, out]` row-major, so the old weight occupies exactly the first `oldIn × out` entries of an
`[newIn × out]` matrix. Growing the input is therefore: **construct a same-config net at `newInputSize`, copy every
parameter, and for the first (input-consuming) weight copy the old rows into the prefix and leave the new rows
zero.** Zero new in-weights ⇒ the new features contribute 0 to every first-layer pre-activation ⇒ the net computes
the **identical function on the old features** for any value of the new ones — exactly function-preserving, the same
guarantee `WidenTo` gives on the width axis.

- **Interface method** (symmetric with the existing `CloneStructure()`/`CopyFrom()` on `IValueNet`): every value net
  can grow its input; a generic caller (e.g. a future resume-with-wider-obs) needs no type switch. Each net
  implements it by building `new XNet(newInputSize, …same config…)` and delegating the copy to the shared helper —
  carrying structural config (hidden sizes, actions, noisy flag), not a pass-through.
- **Contract / invariant** (documented on the interface + asserted in the helper): a net's **first enumerated
  parameter is its input-consuming weight** `[InputSize × firstHidden]`, the second its bias. All current impls
  satisfy this. The helper fails loudly if a net ever violates it.
- **Noisy `DuelingQNet`:** the trunk's first layer is a plain `Linear` (noise lives on the heads), so grow touches
  only that weight; the head μ/σ params are carried over unchanged. Works for plain and noisy alike.

### 4.B The home — `Core/Nn/NetTransfer.cs`
A small static class documenting the **transfer story** and holding the generic mechanics:
- `CopyParameters(IValueNet dst, IValueNet src)` — the exact param-zip copy. `Mlp`/`DuelingQNet`/`ResidualMlp`
  `CopyFrom` collapse to a one-line delegation (removes the 3× duplication; behavior bitwise-unchanged, so target
  sync / shipped checkpoints are unaffected).
- `TransferGrownInput(IValueNet grown, IValueNet original)` — the input-grow copy (prefix + zero-pad). Used by each
  net's `GrowInput`.
- A doc comment that points to the net-resident transforms (`WidenTo`, `ToNoisy`, `PolicyAsMlp`) so the full set is
  discoverable from one place.

### 4.C Caller responsibilities (documented, not enforced here)
`GrowInput` transfers **weights only** — like `WidenTo`. The caller (a training campaign) must, after growing:
**rebuild the optimizer** (Adam moments are keyed to the old parameter set) and **start with a fresh replay buffer**
(stored transitions hold old-width observations and cannot feed the wider net). This makes `GrowInput` a clean
**warm-start** primitive (the FruitCake-style "load shipped net, train on the enriched env"), not a mid-run
buffer-preserving resize. Stated in the XML doc so the constraint can't be missed.

## 5. Milestone plan

- **T0 — `NetTransfer` + dedup.** New `Core/Nn/NetTransfer.cs` with `CopyParameters`; point the three `CopyFrom`
  impls at it. *Gate:* build green; existing `Mlp_CopyFrom_MakesOutputsIdentical` + `ResidualMlp` clone test still
  pass (behavior unchanged).
- **T1 — `GrowInput` primitive.** Add `IValueNet.GrowInput(int newInputSize)` + `TransferGrownInput`; implement on
  `Mlp`, `DuelingQNet`, `ResidualMlp`. Validate `newInputSize > InputSize`.
- **T2 — Tests (the proof).** For each net: (a) **function-preservation** — grown net's output on `[old ‖ arbitrary]`
  inputs equals the original's output on the old slice, to fp error; (b) the new in-weights are exactly zero; (c)
  all other params copied exactly; (d) `newInputSize ≤ InputSize` throws. Include a noisy-`DuelingQNet` case
  (noise-off determinism preserved). Mirrors the `WidenTo` test style.
- **T3 — Docs.** ARCHITECTURE.md §Nn: a short "weight transfer" paragraph (the four transforms + where each lives +
  the caller-rebuilds-optimizer/buffer note). Update `FRUITCAKE_BIGFRUIT_INPUTS_PRD.md` §4.B to reference
  `GrowInput` as the now-available optional warm-start path.
- **T4 — Trainer auto-grow on warm-start.** ✅ *Done.* `DqnTrainer.Train` now grows a `warmStart` net via
  `GrowInput` when the env's observation is wider than the net's input (full-state `resume` still guards — its
  replay buffer holds old-width obs). Integration test `Dqn_WarmStart_GrowsNetWhenObservationGained_Features`
  (warm-start a narrower net into CartPole → grows + trains). This turns "enrich obs ⇒ retrain from scratch" into
  "enrich obs ⇒ warm-continue" for every DQN campaign automatically.

## 6. Testing & verification
`dotnet test` (non-slow) stays green; new `NetTransferTests` (or extend `NnTests`) cover T2. Function-preservation
is asserted numerically (max abs diff < 1e-5 over a random batch), exactly as `WidenTo`'s preservation test does.

## 7. Risks
| Risk | Mitigation |
|---|---|
| A net's first param isn't the input weight (breaks the generic copy) | Documented interface invariant + loud assertion in `TransferGrownInput` (Rows == InputSize check); all current impls verified. |
| Caller forgets to rebuild optimizer / reuses a stale-width buffer | XML doc states it explicitly; the obs-width guard already catches a stale buffer at the env boundary. |
| Dedup of `CopyFrom` subtly changes target-net sync | `CopyParameters` is the *same* param-zip loop; covered by the existing copy tests (T0 gate). |
| Feature ships but the FruitCake consumer shows no gain | Decoupled by design (§1): justified by the cross-env recurring need, not that experiment. |

## 8. Sources
- Internal: `PLAN.md` "switch algorithm, keep the work" / `PRD.md` §8 (asset portability — the roadmap this fills);
  `ResidualMlp.WidenTo` (the function-preserving precedent + its test style); `FRUITCAKE_BIGFRUIT_INPUTS_PRD.md`
  §4.B (the optional warm-start consumer).
- Chen et al. 2016, *Net2Net* — function-preserving transforms (the basis for both `WidenTo` and zero-pad input-grow).
