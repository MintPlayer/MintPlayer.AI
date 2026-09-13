# Block Dude — rebuilding the training pipeline (M60)

**Status:** proposed, 2026-09-13. Nothing here is built.

**Goal (owner):** *"achieve a good net that's capable of solving these levels"* — the 15 shipped levels, not
generated boards. The owner has stated they are willing to start over.

**Where the evidence lives.** This PRD stands on measurements taken on branch **`m59-blockdude-plateau`** and
recorded in `WEBGAMES_RETIREMENT_PRD.md` §8.4a. That branch also contains the tools this plan depends on —
`BlockDudeSearch` (net-guided weighted A*), `--value-calibration`, the no-revisit greedy tier, and the gate's
oracle filter. **It must land before this work starts**, because §3's labeller *is* `BlockDudeSearch`.

---

## 1. What is actually wrong

Four measurements, all on the same checkpoint (6.25M samples, stage 4, trunk `[1152,1152,1152]`):

| # | Finding | Rules out |
|---|---|---|
| 1 | Saturation growth climbed the whole ladder (869k → 4.0M params); **gate did not move** | capacity |
| 2 | No-revisit tie-break triples survival, solves the **same 1/15** | "it just loops" |
| 3 | Value head is **compressed**: +3.2 moves error at true distance 1–10, **−37.5 at 51+** | a usable search heuristic |
| 4 | Generator **cannot express** 53% of shipped topologies; training tops out at ~25 optimal moves vs multi-hundred | "under-trained on the right data" |

The binding constraint is **(4)**, and (3) is its symptom: the value head has no training data in the range
where the shipped levels live, so it regresses to the mean there.

**Correcting a plausible reading.** The owner's hypothesis — that the missing skill is walking *away* from the
door to fetch blocks — is not supported: that pattern appears in **37–43%** of generated boards versus **7%**
of shipped levels. It is over-represented, not absent. The hypothesis was still productive: it motivated the
calibration measurement, which found the real defect.

---

## 2. Generator — lift the topology restriction

`BlockDudeGenerator.TryBuildLayout` places the player on the LOW side of a rise and the door on the HIGH side,
with blocks only on the player's side. It can therefore only emit *stack-up-over-a-rise* puzzles (70–80%
door-above). The shipped levels are **53% door at-or-below the player**, needing descend / bridge-a-pit /
drop-blocks-in. That shape is unreachable by construction: a low door would be walk-reachable, so the
candidate is rejected as `BlocksAreDecorative`.

Needed:
- **Multi-barrier terrain** — 2–3 sequential staircases, optimal length 60–200.
- **Block piles** (stacked columns) rather than scattered singles; shipped levels average 12.7 blocks against
  the generator's 4.8.
- **Pit / descend puzzles** where the door is below and blocks must be dropped in.
- A solvability check that is not "walk-reachable ⇒ decorative", since that rule is what forbids low doors.

---

## 3. Labeller — search, because BFS cannot reach

Exact BFS caps at `OracleMaxStates`; that is *why* the curriculum stops at ~25-move boards. Boards from §2 are
unlabelable by it, so the labeller becomes **phase-2 expert iteration** (§8.1b of the M58 PRD), now unblocked
because `BlockDudeSearch` exists and solves 5/15 shipped levels unaided:

1. Generate boards from §2 that the oracle cannot label.
2. Solve what net + A* can solve.
3. Train on those trajectories; repeat as the net improves and solves more.

A solution found this way yields, for free, the true remaining distance at every state along it — precisely
the 1-to-several-hundred range the value head has never seen.

---

## 4. Value head — categorical, not scalar

Predicting an absolute scalar over a 1–200 range under Huber **regresses to the mean**, which is finding (3)
exactly. Replace with a **distribution over distance buckets** (MuZero-style categorical value): it preserves
*ranking* even when absolute accuracy is poor, and ranking is all A* needs from a heuristic.

This changes the net's output shape, so it forces a fresh run — acceptable, since §2 changes the data anyway.

---

## 5. The owner's decomposition idea — evaluated

> *"an additional AI model that tries to determine this final state from the starting state. Then another model
> can try to find a way to get from the starting state to the final state."*

**The instinct is right and it attacks the measured defect directly.** A predicted target configuration
replaces the badly-scaled scalar heuristic with a *structured* goal, which yields a naturally well-scaled
distance — "how many blocks are not yet at their final cells" — that does not compress at long horizons. It is
hierarchical planning, and it is a real answer to the horizon problem rather than a tuning knob.

Three things to get right:

- **It is one-to-many.** Many final configurations win a level, so a model trained to regress "the" final state
  will average valid answers into an invalid one. Predict an **occupancy mask** (which cells hold a block at
  the end) with the block count as a hard constraint, rather than a single blurry board.
- **The target must be reachable.** A generative model can emit configurations no legal play can produce
  (floating blocks, wrong block count). The mask formulation plus a support check keeps targets legal.
- **The training data for it should NOT be the human recordings.** Fifteen examples cannot train this. It
  should be trained on the *thousands* of optimal solutions the oracle already produces on generated boards,
  plus A* solutions on larger ones — where the final configuration is free to extract.

**What the human recordings are genuinely for** (see §6): they are the only ground-truth solutions to the real
levels that exist, since BFS cannot label boards that size. Their value is as an evaluation set and as
expert-iteration seeds, not as the training set for the target-config model.

**Sequencing.** §2–§4 are prerequisites: a target-config model trained on data that cannot express the shipped
topology inherits the same ceiling. Build it after, as the strongest candidate for §3's heuristic.

---

## 6. Human solution recording (BUILT on `m59-blockdude-plateau`)

The Block Dude page records the **whole action sequence** of any level the player completes, keyed by level
name, shortest-wins, into `localStorage` under `blockdude.solutions.v1`. Undo pops the recording so it stays
honest.

The sequence is kept rather than the final board because it is strictly more information: a per-state action
label, the final configuration, **and** a true remaining-distance label for every state along the path.

Worth its own note: the player is not expected to complete all 15. Level 11 is 42 blocks. Even a handful of
the levels A* cannot solve is more ground truth than this project currently has, which is none.

---

## 7. Risks

- **§2 is the big unknown.** "Generate a solvable multi-barrier puzzle" is materially harder than the current
  single-barrier construction, and the accept rate may collapse the way it did before (1–7% → 16–22% took four
  fixes). Budget for measuring accept rate per stage from the start.
- **Expert iteration can stall.** If A* solves nothing on the new boards, step 3 produces no data and the loop
  never starts. Mitigation: overlap the new distribution with the old so early rounds have traction.
- **The evidence base is one run.** Findings (1)–(3) come from a single checkpoint; (4) sampled 30 boards per
  stage. Strong signals, not large studies. Re-measure after §2 rather than assuming they carry over.
