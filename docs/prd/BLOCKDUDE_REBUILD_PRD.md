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

## 6a. Reverse curriculum from the demonstrations — MEASURED, and it works

The 15 human solutions (§6) are only 3,870 states, which sounds negligible against millions of training
samples. Volume is the wrong axis: they are the **only** data that exists in the distribution the net is graded
on, because §2 and §3 make every other source out-of-distribution by construction.

**The mechanism.** A suffix of a demonstrated path is a legitimate position on real shipped terrain that is only
N moves from the door. So one impossible board becomes a ladder of solvable ones, and fifteen trajectories
become thousands of distinct training tasks — on exactly the topologies and horizons the generator cannot
produce. `BlockDudeDemonstrations.ReverseCurriculumStarts(movesFromEnd)` yields them.

**Measured with `--demo-probe`** (current net, A* weight 2, ≤40k expansions, ≤6s per attempt). Numbers are the
solution length found; `-` is a miss:

| level | human | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| Level 3 | 98 | 5 | 10 | 20 | 39 | **76** |
| Level 7 | 782 | 5 | 10 | 20 | **41** | − |
| Level 8 | 494 | 5 | 10 | 20 | **40** | − |
| Level 10 | 386 | 5 | 10 | 18 | − | − |
| **Level 11** | **909** | 5 | 10 | 18 | **35** | − |
| Bonus 2 | 110 | 6 | 11 | **15** | 33 | − |

Every level is solvable 20 moves out (bar Level 5), most at 40, and **Level 11 — which A\* cannot touch from
the opening — is solved 35 moves from the end**. The frontier is therefore ~20–40 moves, which is where a
reverse curriculum starts and what training pushes outward.

**Search already beats the demonstrations locally**: Level 11 found 35 where the human took 40, Bonus 2 found
15 where the human took 20, Level 10 found 18 where the human took 20. That is the mechanism by which the
student exceeds the teacher (§8.1b of the M58 PRD) showing up before any training has happened — and it is why
`Remaining` is documented as an upper bound, never an optimum.

**Two labels, both free.** Each demonstrated state carries the action the human played AND the true moves
remaining along that path — the second landing in the 1-to-909 range where the value head was measured
under-estimating by ~37 moves with no training data at all (§8.4a). No search or oracle is needed to extract
either.

## 6b. Irreversibility — the net has never been shown a lost position (owner, 2026-09-13)

> *"in Rush Hour, a lousy move doesn't kill the entire game, whereas in Block Dude each move has to be perfect
> or the game is in a dead end"*

§8.1a of the M58 PRD records the difference. Following it into the training data finds something that section
did not: **the consequence is a hole in the data, not just a property of the game.**

The oracle already knows which states are lost. It enumerates forward from the start, then runs a **backward**
BFS from the won states, so anything the backward pass never reaches keeps `dist = -1` — reachable, but with
no path to the door. `CollectSamples` then filters `distance > 0 && mask != 0`, which discards exactly those.

**Measured** (`BlockDudeDeadEndTests`, over hold-out boards per rung):

| stage | winnable | dead | dead share |
|---|---|---|---|
| 0 | 334 | 143 | 30.0% |
| 3 | 4,121 | 1,016 | 19.8% |
| 4 | 84,717 | 31,575 | 27.2% |
| **6** | 83,366 | **99,120** | **54.3%** |
| **total** | 195,110 | 138,506 | **41.5%** |

**On the hardest rung the majority of reachable states are unwinnable, and the net has been trained on none of
them.** Every position it has ever seen was still winnable. This is free data being thrown away — the oracle
computes it and the filter drops it.

Three measured behaviours become one explanation:

- **The policy walks confidently into unrecoverable positions** (no-revisit greedy ends in `NoMove`, §8.4a
  finding 5) because it has never been shown that such positions exist.
- **The value head is compressed** (§8.4a finding 3) partly because it *cannot represent* "lost". A scalar
  regression must emit some finite number for a dead state, and at inference 41% of what it meets is dead.
- **A\* wastes its budget.** The heuristic hands dead branches a plausible finite value, so search expands
  them. At stage 6 more than half the reachable space contains no goal at all.

**This makes the categorical value head (§4) the clear next change rather than one option among several.** A
distribution over distance buckets **plus an explicit "unsolvable" bucket** fixes both defects at once and does
so natively: it represents "lost" as a class rather than as a number it has to invent, and it gives search a
pruning rule instead of a misleading estimate. A scalar head fundamentally cannot express this — which is why
patching it (say, labelling dead states with a large distance) would trade one distortion for another.

Note this is the axis on which Rush Hour and Block Dude genuinely differ, and why the Rush Hour campaign needs
none of it: with no dead ends, every state has a finite distance and the scalar regression is well-posed.

## 7. Risks

- **§2 is the big unknown.** "Generate a solvable multi-barrier puzzle" is materially harder than the current
  single-barrier construction, and the accept rate may collapse the way it did before (1–7% → 16–22% took four
  fixes). Budget for measuring accept rate per stage from the start.
- **Expert iteration can stall.** If A* solves nothing on the new boards, step 3 produces no data and the loop
  never starts. Mitigation: overlap the new distribution with the old so early rounds have traction.
- **The evidence base is one run.** Findings (1)–(3) come from a single checkpoint; (4) sampled 30 boards per
  stage. Strong signals, not large studies. Re-measure after §2 rather than assuming they carry over.
