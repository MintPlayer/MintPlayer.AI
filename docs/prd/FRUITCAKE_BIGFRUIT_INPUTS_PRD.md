# FruitCake AI — Big-Fruit Position Inputs (PRD & Plan)

> **Hypothesis (user's steer):** give the net the **positions of the two biggest fruits on the board**.
> The current observation is a per-column *skyline* with **no absolute fruit positions** — so "where is my
> pineapple/watermelon-in-progress" is information the net literally cannot see. This PRD specifies that input,
> answers the checkpoint-reuse question, and lays out a **cheap-to-falsify** experiment that judges the *deployed*
> (net + forward-search) system, not just the greedy net.

- **Status:** Draft v1.0 · 2026-06-30 (analysis complete; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** the FruitCake AI (`FRUITCAKE_AI_PRD.md`), the plateau work (`FRUITCAKE_IMPROVE_PRD.md` — the
  current obs/reward/search baseline), the DQN stack (`DqnTrainer`/`DuelingQNet`/`DuelingQNetCheckpoint`), and the
  A/B + search-eval harness (`tools/…Lab/FruitCakeAb.cs`, `FruitCakeSearchEval`).

---

## 1. What we confirmed (two-agent analysis, 2026-06-30)

1. **The current 83-dim observation has NO absolute positions.** It is `FruitCakeEnv.BuildObservation`
   (`…Environments/FruitCake/FruitCakeEnv.cs:144–202`): 14 surface-height + 14 top-tier + 14 danger-margin +
   14 merge-with-current + 14 adjacent-equal-pair + 5 current one-hot + 5 next one-hot + 3 globals = **83**.
   Everything is per-column structural/relational; **the (x,y) of any specific fruit is collapsed away**, and a
   *buried* big fruit need not appear in the top-tier skyline at all. ⇒ the proposed input is genuinely new info.
2. **The shipped checkpoint CANNOT be reused as-is.** Adding inputs makes the obs ≥87-dim;
   `DuelingQNetCheckpoint.Read` throws `InvalidDataException` on the first-layer parameter-length mismatch, and
   `FruitCakeModelService` (`…/Services/FruitCakeModelService.cs:43–50`) has an explicit width guard that refuses a
   stale-width net. There is **no weight-transfer / layer-grow utility** in the repo — only an exact-shape
   `warmStart` path in `DqnTrainer`. So this is a **retrain**, not a checkpoint resume (see §4.B).
3. **The reactive net is documented as saturated** (`FRUITCAKE_IMPROVE_PRD.md` §4–7, PLAN M28/M29). Every prior
   training-side lever — richer relational inputs (F2: 41→83), reward shaping (F3), n-step (F4), NoisyNets,
   reverse-curriculum, planner-distillation (F6) — either matched baseline or moved **score but not watermelon%**.
   The entire watermelon breakthrough (0% → 50%) came from **serving-side forward search**, not the net.

**Net effect:** the input idea is well-formed and adds real information, but the prior says the *likely* upside is
a marginally **better search leaf**, not a stronger greedy net. We therefore design to *falsify cheaply* and judge
the deployed net+search system. This is explicitly framed as an experiment with a real chance of a negative result
(the honest outcome for NoisyNets, curriculum, and F6).

---

## 2. Goals & Non-Goals

### Goals
- Add the **two-biggest-fruit positions** to `BuildObservation` as a small, self-contained, well-tested input block.
- Answer checkpoint reuse with a concrete decision and (optionally) the one utility that would enable warm-start.
- Run a **decisive A/B**: does the retrained net make **net+search** beat the current depth-3 bar
  (~2505 score / 50% watermelon over 100 paired games; current deployed depth-2 ≈ 30%)?
- Whatever the result, keep the **capability + measurement** (the input block, the warm-start utility if built) and
  record a clear ship / no-ship verdict — consistent with how NoisyNets/F6/curriculum were handled.

### Non-Goals
- A full 2-D occupancy grid (that's the `FRUITCAKE_IMPROVE_PRD.md` **B4** lever — a larger, separate change).
- Beating the *honest ceiling* — even SOTA Suika agents reach strong-human, not "always watermelon."
- Touching the forward-search algorithm itself (this PRD changes only the net's inputs).

---

## 3. Why this might — and might not — work

**For:** absolute positions are the one structural signal the skyline cannot represent. Knowing *where* the two
biggest fruits sit lets the policy (and, more importantly, the **search leaf** `max-Q`) reason about "drop the
next-biggest next to the buried pineapple in the left well" — exactly the long-range, cross-column placement Suika
rewards and the skyline destroys.

**Against (the prior):** F2 already added relational inputs and lifted **score +36% but watermelon stayed 0/200**
for the greedy net; the net is a saturated search leaf. Positions may improve the leaf's board-sense marginally, or
not measurably. The closest test of "make the net a better leaf" — F6 distillation — *failed* (the distilled value
head was a **worse** leaf than DQN max-Q). So the realistic expectation is **small or null**, and the plan is built
to find that out for a few hours of compute, not to assume success.

---

## 4. Design

### 4.A The new input block (`BuildObservation`)
Append a fixed-size **big-fruit block** after the existing 83 features. "Biggest" = **highest tier**; the board's
merged fruit (tiers 6–11) are the targets of interest. Selection over *all* resting bodies (not the skyline), so a
buried big fruit is captured:

- Find the top-2 bodies by **tier descending**, breaking ties deterministically by **lower on the board (larger y)
  then smaller x** (stable ⇒ the obs is reproducible; the net sees a consistent ordering).
- For each of the two, encode **3 values**: `x/W`, `y/H` (0 = top, consistent with the rest), and `tier/11`.
  - Including `tier` is cheap and meaningful: it tells the net *how* big "biggest" is and survives burial (the
    skyline's top-tier can miss a buried fruit). Drop to position-only (2 each) if the A/B shows tier hurts.
- **Define the empty cases out of existence** (per the design principles — no special-case branches in callers):
  when the board has fewer than 2 fruits, emit a neutral sentinel `(x=0.5, y=1.0, tier=0)` = "floor-center, no big
  fruit," so early-game and full-board states use the same code path. (Alternative: an explicit `present` flag per
  slot, +2 dims — only if the sentinel proves ambiguous in the A/B.)

**Dimension:** `2 × 3 = 6` → `ObservationSize` **83 → 89** (or **87** if position-only). The block is a pure-function
edit to the single static `BuildObservation`; the serving path (`FruitCakeController:74`) calls the same method, so
training and serving stay in sync automatically, and the width guard auto-rejects the stale 83-dim shipped net.

> **Open implementation note:** `BuildObservation` today takes `(FruitCakeWorld world, int current, int next)` and
> the world already exposes its resting bodies (the skyline is computed from them), so finding the top-2 by tier is
> a local scan with **no signature/DTO change** — unlike B3 next-next, which needed new preview state. Confirm the
> body list exposes position + tier (`FruitCakeWorld`) during G1; if not, that's the only added plumbing.

### 4.B Checkpoint reuse — decision
The 83-dim `models/fruitcake.dqn.ckpt` will **not** load at width 89 (strict shape check + width guard, §1.2).
Two paths:

| Path | Effort | Verdict |
|---|---|---|
| **Retrain fresh at the new width** | 0 extra code | ✅ **Recommended.** Exactly the F5 precedent (41→83 was a fresh train); the net is saturated, so warm-starting buys little, and warm-starting a *refined* net has degraded results before (the NoisyNets σ₀ finding, the short ε-greedy warm-start dip). Clean, low-risk. |
| **Weight-pad warm-start via `GrowInput`** | none (shipped) | Optional. `IValueNet.GrowInput(89)` (the `NET_TRANSFER_PRD.md` feature, now built) zero-pads the 6 new input rows and carries the rest over function-preserving, then `DqnTrainer.Train(warmStart:)` with a fresh optimizer + replay buffer. Reusable across any obs-width bump, but unlikely to change the outcome given saturation. Use only if G3-from-scratch is too slow to reach a verdict. |

**Decision:** **train fresh (G3).** Treat the warm-start utility as an optional, separately-justified SDK nicety,
not on this experiment's critical path.

---

## 5. Milestone plan

- **G0 — Lock the bar.** Record the current deployed baseline as the gate: shipped 83-dim net + forward search,
  **100-game paired** distribution via `FruitCakeSearchEval` — depth-3 (controller default `TopK=5, TopK2=2`) ≈
  **2505 score / 50% watermelon / meanTier 10.49**, depth-2 topK-10 ≈ **2278 / 30%**. *(These are the recorded
  numbers; re-run to confirm on the current build.)* This is what a new net must beat to ship.
- **G1 — Add the inputs (pure-fn edit).** ✅ *Done 2026-06-30.* `BuildObservation` appends the §4.A big-fruit block
  (`(x/W, y/H, tier/11)` of the top-2 by tier, deterministic tie-break = lower-then-leftmost, neutral
  floor-centre sentinel for empty slots); `ObservationSize` 83→89. Tests `BuildObservation_*` (correct two fruit
  incl. a buried one, tie-break, empty-board sentinel). All FruitCake tests green; the model-service width guard
  now refuses the stale 83-dim ckpt — expected, the web app falls back to heuristic until a 89-dim net ships.
- **G2 — Warm-start via `GrowInput`.** ✅ *Done 2026-06-30 — no bespoke utility needed.* The
  `NET_TRANSFER_PRD.md` feature supplies it: `FruitCakeDqnCampaign.Resume` grows the loaded 83-dim shipped net to
  89 on load (function-preserving), and `DqnTrainer` auto-grows the warm-start net (T4) as a backstop. Smoke-run
  validated: "growing the loaded net's input 83 → 89", baseline eval of the grown net succeeds (same policy on the
  old features). So G3 can train *fresh* OR warm-start from the shipped net — both paths work.
- **G3 — Train fresh at width 89.** `--game fruitcake --shape --nstep 3 --gamma 0.997 --data <fresh dir> --hours N`
  (same recipe as the shipped `fruitcake-bundle` net, so the *only* changed variable is the input block).
  Resumable (re-run the same command). Snapshot the best net by the campaign's greedy keep-best.
- **G4 — Judge & decide (the gate).** Over **≥100–200 paired games, same seeds**:
  1. **Greedy net** A/B to isolate the *input contribution* (new 89-dim greedy vs recorded 83-dim greedy 963.5 /
     meanTier 8.75 / 0 watermelon — absolute comparison, since widths can't be paired in one build, per F5).
  2. **Net + search** A/B (depth-2 *and* depth-3) — **this is the decision metric**, because search is deployed.
     Wire the new net as the leaf, run `FruitCakeSearchEval`.
  - **Ship** `models/fruitcake.dqn.ckpt` (LFS) **only if net+search beats the G0 bar** on the max-tier distribution
    (primarily watermelon%, then mean score) by a margin clearing seed-noise (judge the *distribution*, never a
    10-ep eval — PLAN M28). Otherwise **record a negative result** (here + `OPTIMIZATIONS.md`), keep the input block
    + tests, leave the shipped model unchanged.

**First session:** G0 → G1 (+ tests) → G3 (a few hours, resumable) → G4. G2 only if needed.

---

## 6. Measurement
Decision metric = the **max-tier distribution** (fraction reaching tier 9/10/**11**) of **net + forward search**
over **≥100–200 paired games, multiple seeds**, vs the G0 bar, via `FruitCakeSearchEval`/`FruitCakeAb`. Mean score
is the tiebreaker. The greedy-net A/B is diagnostic only (isolates the input's effect on the net). **Never** gate on
a single 10-episode eval — those bounced 750–971 on the *same* net (PLAN M28). Serving stays deterministic;
keep-best gating protects the shipped model.

## 7. Risks
| Risk | Mitigation |
|---|---|
| **No gain (the prior's base case)** — net saturated, positions don't move net+search. | Cheap to find out: G1 is a pure-fn edit, G3 a few hours, G4 the A/B. Negative result is acceptable & documented; capability kept. |
| Positions help greedy but not net+search (or vice-versa). | Measure **both** in G4; ship decision is net+search (deployed), greedy is diagnostic. |
| Tie-break/sentinel nondeterminism corrupts the obs. | Deterministic ordering (tier↓, y↓, x↑) + fixed sentinel; unit-tested in G1. Serving uses the same static method ⇒ no train/serve skew. |
| Width change breaks back-compat. | Width guard already refuses the stale net; the new net ships its own width (F5 precedent). Old net still loads in any 83-dim build. |
| Body list not exposed for selection. | Confirm in G1; worst case a tiny `FruitCakeWorld` accessor (no DTO/preview-state change, unlike B3). |
| Eval seed-noise misleads (M28 trap). | Gate on the ≥100–200-game paired distribution, never single evals. |

## 8. Honest verdict criteria
Given the saturation prior, the **expected** outcome is small-or-null on watermelon%. A *win* is net+search clearing
the depth-3 ~50% / ~2505 bar by a seed-noise-beating margin. A *partial* (greedy score up, net+search flat) is a
**negative for shipping** — score isn't the goal, the tier ceiling is, and search already maxes score. Treat this
like NoisyNets/F6: validate the capability, report the number honestly, ship only on a real win.

## 9. Sources
- Internal: `FRUITCAKE_IMPROVE_PRD.md` (§1 diagnosis, §4.B obs history, F2/F5/F6 results), PLAN M28 (NoisyNets +
  eval seed-noise), M29 (the three levers), `OPTIMIZATIONS.md` (curriculum + distillation negative results).
- Code: `FruitCakeEnv.BuildObservation` (`…/FruitCake/FruitCakeEnv.cs:144`), `DuelingQNetCheckpoint`
  (`…/Core/Checkpoints/DuelingQNetCheckpoint.cs`), `FruitCakeModelService` width guard
  (`…/Services/FruitCakeModelService.cs:43`), `FruitCakeSearch`/`FruitCakeSearchEval`, `FruitCakeAb`.
- Poelsma, J. — *Creating an AI that plays Suika game* (LIACS, Leiden, 2025): reactive DQN plateaus, forward-model
  search is the lever — the basis of the saturation prior.
