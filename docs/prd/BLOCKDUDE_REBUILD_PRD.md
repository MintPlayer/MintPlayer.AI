# Block Dude — rebuilding the training pipeline (M60)

**Status, 2026-09-14 — partly BUILT, on branch `m59-blockdude-plateau` (M59).** The original plan had the
generator rebuild (§2) as the prerequisite for everything. Measurement changed that order: the reverse
curriculum (§6a) supplies in-distribution training data on the real levels *without* touching the generator, so
it went first and is running.

Measurement has now reordered the plan twice, in the same direction both times. §6a demoted the generator
rebuild; §6d demoted the categorical value head, after three changes to the *search* — none of them a training
change — took a frozen checkpoint from 6/15 to 10/15 shipped levels. The recurring lesson is that the cheap
measurement was worth more than the expensive rebuild it was meant to justify, so take the next item in this
plan as a hypothesis to test rather than work to schedule.

**Every "solves N/15" in this document is a TRAINING-SET score.** Phase 2 trains on the 15 shipped
levels, so they are the curriculum and the benchmark at once. Measured on held-out boards the policy
drops from 67% to 6% (§6d): this net plays *these fifteen levels*, not Block Dude. That is the reverse
curriculum working as designed and it meets the stated goal — but it is not a generalisation claim.

**Read §6d's controls before quoting any number in this document.** Uninformed search solves **6/15** shipped
levels by itself — 6/15 as best-first with `h = 0`, and 6/15 again as a uniform-prior beam. Every "solves N/15"
here is therefore a claim about the search first and the net second. Against that baseline the trained net is
worth **+2 levels in A\*** and **+4 in beam search**. Those controls did not exist until 2026-09-14, so earlier
sections overstate what they attribute to the net.

| § | What | Status |
|---|---|---|
| 2 | Generator — lift the topology restriction | **not built.** No longer blocking — but §6d's held-out measurement makes it the specific prerequisite for a net that TRANSFERS, rather than a vague want of "volume and variety" |
| 3 | Labeller — search instead of BFS | **BUILT** — `BlockDudeSearch`, net-guided weighted A* over `Core.Planning` |
| 4 | Value head — categorical, with an "unsolvable" bucket | **not built. Lower priority, stronger evidence** (§6d): the scalar head was caught *losing* Level 6 that uninformed search solves — but pairing it with the policy prior recovers that at zero training cost, while §4 forces a fresh run |
| 5 | Owner's decomposition idea | evaluated, not built; sequenced after §4 |
| 6 | Human solution recording | **BUILT** — all 15 levels recorded, validated and committed |
| 6a | Reverse curriculum + expert iteration | **BUILT** — `BlockDudeDemonstrations`, `BlockDudeExpertIterationCampaign` (`--phase 2`) |
| 6b | Dead ends the training data discards | **measured**, fix is §4 (still unbuilt — see §6d) |
| 6d | Search: batched calls, policy-as-prior, **beam search**, frontier retreat, landmark salvage | **BUILT** — 6/15 → **10/15** on frozen weights; `Core.Planning.PolicyValueSearch` + `PolicyBeamSearch` |
| 6e | Result after the overnight run | **15/15 shipped levels with NO SEARCH** — greedy, one forward pass per move — and shorter than the human demonstrations on 11 of 15 (3,644 moves vs 3,870) |

**Goal (owner):** *"achieve a good net that's capable of solving these levels"* — the 15 shipped levels, not
generated boards. The owner stated they are willing to start over.

**Where the evidence lives.** `WEBGAMES_RETIREMENT_PRD.md` §8.4a holds the measurements this plan stands on;
PLAN.md M59 holds the milestone narrative. Everything is on `m59-blockdude-plateau`, unmerged as of writing.

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

## 6c. Phase 2, first real result — the mechanism works (2026-09-13, 26 minutes of training)

Benched with `--eval-levels --search --phase 2 --resume-net` against the phase-1 baseline taken the same day:

| tier | phase 1 | phase 2, after 26 min | levels newly solved |
|---|---|---|---|
| greedy (policy alone) | **0/15** | **3/15** | Level 1, Bonus 1, Bonus 3 |
| greedy + no-revisit | 1/15 | 3/15 | — |
| net-guided A* | 5/15 | **6/15** | + Level 5 (207 moves) |

**The policy now solves levels unaided, which it had never done** — and Level 1 in 19 steps, matching the
human's 19. The curriculum reached 5/15 levels whole, frontier 19→100, in six rounds.

Worth stating plainly what this does and does not establish. It establishes that the reverse curriculum
produces a net that plays measurably better on the real levels, from a standing start of zero. It does **not**
establish the ceiling: the run used the scalar value head, which §6b shows is both compressed at long horizons
and blind to the 41.5% of states that are unwinnable. So this is a **lower bound** on the approach, and any
stall observed with this head is evidence about the heuristic rather than about the curriculum.

## 6d. The search was the bottleneck, not the value head (2026-09-14, overnight)

§6c ended by naming the scalar value head as the thing holding phase 2 back, and §4 (a categorical head with an
unsolvable bucket) as the fix. Measuring before building changed that order, exactly as §6a changed the order of
the generator rebuild. **Three changes to the search, none of them a training change, took the same frozen
checkpoint from 6/15 to 8/15** — and one of them was free.

All three numbers below are the *same weights*, benched with `--eval-levels --phase 2 --resume-net`,
weight 2, ≤200,000 expansions, ≤20s per level.

| change | shipped levels solved | what it cost |
|---|---|---|
| baseline (§6c) | 6/15 | — |
| batched net calls in search | **7/15** (+ Bonus 2) | nothing — a different call shape |
| policy head used as a search prior | **8/15** (+ Level 6) | nothing — a head that was already trained |
| policy **beam** search instead of A\* | **10/15** (+ Level 4, Bonus 4) | nothing — a different search shape |

…but read the control below before crediting those levels to the net: uninformed search alone solves
6/15, so the trained net with lookahead is worth **+2 levels over knowing nothing**, and Level 6 is a
level the value head was *losing* rather than one the policy prior won.

### What the net is actually worth: the h = 0 control

The table above says the search improved. It does **not** say the net caused it, and that question needed its
own measurement — so `--zero-h` runs the identical search with `h = 0`, which is uninformed breadth-first. The
result is chastening and worth stating first:

| tier | solved | levels |
|---|---|---|
| greedy — the policy alone, no search | 3/15 | 1, Bonus 1, Bonus 3 |
| **uninformed search, `h = 0` — the control** | **6/15** | + 2, 3, **6** |
| value head as heuristic (weight 2) | 7/15 | + 5, Bonus 2 — and **loses 6** |
| policy prior alone (weight 0, policy weight 5) | 7/15 | + Bonus 2 |
| policy prior + value head | **8/15** | + 5, 6, Bonus 2 |

**Uninformed search already solves 6 of the 15 shipped levels.** Six of the eight the best tier solves are
therefore not evidence about the net at all — they are evidence that those levels are small enough to search.
The honest claim for the trained net, with both heads and lookahead, is **+2 levels over knowing nothing**.

Two things follow, and neither was visible before the control existed.

**The value head actively misleads the search on Level 6.** Uninformed search solves it in 107 moves; the
value-guided search does not solve it at all, at any A\* weight in the sweep. That is §8.4a's compression
showing up as a concrete lost level rather than as an error bar — a heuristic that is merely uninformative
costs nothing, but one that is confidently wrong steers the frontier away from the solution. It is the
strongest argument in this document for §4, and it also means **the value head should never be used as the sole
guide**: paired with the policy prior, Level 6 comes back.

**The two heads are complementary, not redundant.** Each alone is worth exactly +1 level over the control, and
they are not the same level — the value head brings Level 5, the policy prior brings back Level 6, and using
both gets both. That is the case for `PolicyValueSearch` over picking a winner between them.

**A caveat on precision.** These are wall-clock-budgeted searches measured with a training run competing for the
same eight cores, so a single level is within noise. The differences worth believing are the ones with a
mechanism attached: the 6/15 control, and Level 6 flipping with the heuristic, both reproduce the pattern rather
than resting on one number.

**What this says about where the remaining work is.** Seven levels — 4, 7, 8, 9, 10, 11 and Bonus 4 — are solved
by *nothing*: not greedy, not blind search, not either head, not both. They are the large, long ones (Level 11
is 29×19 and 909 human moves). No amount of heuristic tuning reaches them inside a 200k-node budget, because
the budget is the wall. That is what the reverse curriculum exists to climb, and it is why the run's frontier
number, not the bench, is the thing to watch overnight.

### Beam search — the tier the long levels actually needed

The control's last line said seven levels are solved by nothing, and that the node budget is the wall. That is
a statement about the *shape* of the search, not about the heuristic, and it has a standard answer.

A\* holds an open frontier that grows with the space explored, so its reach is bounded by a node budget — and a
node budget is a wall, not a dial. Level 11 needs about 900 moves; no heuristic quality makes 900 moves
reachable inside 200,000 nodes, because the frontier alone would dwarf that. **Beam search costs
`width × depth`**, so depth is nearly free, and the long levels stop being out of range.

On the same frozen checkpoint, width 256, ≤30s per level:

| tier | solved | levels it adds |
|---|---|---|
| policy + value A\* | 8/15 | — |
| **policy beam search** | **10/15** | **Level 4 (281 moves), Bonus 4 (118)** — neither solved by any other tier at any setting |

It also finds **shorter** paths everywhere the tiers overlap: Level 5 in 129 moves against 207 / 172 / 161;
Level 2 in 73 against 87 / 82; Bonus 2 in 88 against 97 / 94.

**Its own control, because the shape changed.** `--zero-h` cannot attribute a beam result: beam and A\* are
different search shapes, so comparing beam-with-a-policy against best-first-without-a-heuristic would credit the
policy for the change of shape. So the beam tier gets a matching control — the same beam with a **uniform
prior**, where every candidate at a depth scores identically and the beam keeps whatever it enumerates first.

| | uninformed | with the net | the net is worth |
|---|---|---|---|
| best-first (A\*) | 6/15 | 8/15 | +2 |
| **beam** | **6/15** | **10/15** | **+4** |

The two uninformed baselines agree at 6/15, which is reassuring — it says the shapes are comparable and that the
6 easy levels are simply searchable. What differs is how much the net adds: **the policy head is worth twice as
much in the beam as in the A\***. The A\* tiers were understating the net, not measuring it.

That also reframes the greedy number. The policy alone solves 3/15; a *uniform* beam of 256 solves 6/15; the
policy driving that same beam solves 10/15. So the policy is far better than its greedy score suggests — what
greedy lacks is not knowledge but any capacity to survive its own single mistake, which in an irreversible game
is fatal by construction.

**Ranked by the policy head alone**, deliberately. Every candidate in a beam sits at the same depth, so
cumulative log-probabilities are directly comparable with no length normalisation — whereas comparing states at
*different* depths is precisely what a regression-trained value head is measured to be bad at. This tier
therefore sidesteps §8.4a's compression rather than working around it.

**What it gives up is completeness.** A solution pruned out of the beam is gone for good, so unlike the A\*
tiers this can fail on a problem it has ample budget for. The tiers are complements: best-first for short
awkward problems, beam for long ones. A test requires a narrow beam to *fail* on a solvable problem, so that
nobody reading "search" assumes the siblings' guarantees.

**It is in the training loop too**, as a fallback after A\*, for the same reason it exists: as a level's frontier
moves outward the suffix eventually passes what a node budget can reach. A\* is still asked first, because beam
solutions are the policy's own widened rollout and tend to be longer, and a longer path is a looser distance
label — so the better labels are preferred wherever they exist. Beam solves count as genuine solves and may
move a frontier, since they reach the door by the net's own policy. They carry their own counter in the eval
line: **if beam hits grow while A\* hits shrink, the curriculum has moved past what a node budget can reach**,
which is a fact about the run's progress worth seeing rather than averaging into one solve rate.

### Held-out: the net learned these fifteen levels, not the game

Phase 2 trains on the 15 shipped levels, so those levels are the curriculum and the benchmark **at once**.
Every "solves N/15" in this document is therefore a *training-set* score. That is legitimate — the stated goal
is these levels — but it cannot support a claim that the net learned Block Dude, and the difference is not
rhetorical. `--held-out` scores the same net on generated gate boards, the only Block Dude positions phase 2
has never seen.

Measured mid-run (172k samples, stage-4 gate boards, 64 of them, beam width 256):

| | on the 15 trained levels | on 64 held-out boards |
|---|---|---|
| greedy (policy alone) | 10/15 (67%) | **4/64 (6%)** |
| policy beam search | 12/15 (80%) | 63/64 (98%) |
| *uniform-prior beam (control)* | *6/15 (40%)* | ***63/64 (98%)*** |

**The control is the whole finding.** A uniform prior scores exactly the same 63/64 on the held-out boards, and
the net's solutions being 63/63 optimal is likewise a fact about the beam rather than about the net: these
boards are small (~25 optimal moves), so a 256-wide beam is close to exhaustive and would find the optimum with
no guidance at all. On the held-out set **the net contributes nothing measurable**. Without the control this
would have been written up as "98% of unseen boards, all optimal — it generalises", which is the opposite of
what happened.

The greedy row is the uncontaminated one, since no search can flatter it: **67% on the levels it was trained on,
6% on anything else.** The policy has fitted fifteen trajectories.

**This is the reverse curriculum working as designed, not a defect.** It is *built* to train on the shipped
levels — that is how §6a escaped the generator's distribution problem in the first place. Overfitting to the
target is the mechanism, and against the owner's goal (*"a good net that's capable of solving these levels"*)
it is a success. It just has to be labelled correctly:

- **Supported:** this net plays the 15 shipped levels well.
- **Not supported:** this net plays Block Dude. On a new level it would be roughly as good as an untrained one.

**And it is direct evidence for §2.** The generator rebuild was demoted as non-blocking, which was right — but
this is the measurement that says what it is *for*. A net that generalises needs in-distribution training data
on varied terrain, and the generator is the only thing that can produce that at volume. §2 stops being "volume
and variety" and becomes the specific prerequisite for a net that transfers.

**A cheaper partial fix, if transfer matters before §2 lands:** mix a share of oracle-labelled generated boards
back into phase-2 batches, exactly as `DemoShare` keeps an anchor in long-horizon data. Phase 2 currently trains
on shipped levels and demonstrations only, so the phase-1 skills are simply being forgotten. That is a few lines
and does not need a new generator — though it trades some of the shipped-level score for generality, which is
the owner's call rather than a default.

### Where the tiers stand

| tier | solved | what it is for |
|---|---|---|
| greedy (policy alone) | 3/15 | diagnostic: has the policy learned the game |
| uninformed best-first, `h = 0` | 6/15 | the control for the A\* tiers |
| uniform-prior beam | 6/15 | the control for the beam tier |
| value head as heuristic | 7/15 | — |
| policy prior + value head | 8/15 | short, awkward levels |
| **policy beam search** | **10/15** | **the long levels; the shipped tier** |

Five levels remain unsolved by everything: 7, 8, 9, 10 and 11 — the largest boards, up to 29×19 and 909 human
moves. Those are the curriculum's job, and the number to watch for them is the run's frontier rather than this
bench.

### The net was being called one row at a time

`BlockDudeSearch` used `ValueGuidedSearch.Solve`, which evaluates one successor per call: a 1181-wide input
through a 1152×1152×1152 trunk, as a single row. That is the entire cost of the search, and a one-row matrix
multiply wastes nearly all of it. Core already had `SolveBatched` — written for the cube, scoring a whole round
of successors in one pass — and Block Dude simply was not on it. Switching cost one new method on the net.

This matters twice, because expert iteration **generates every training sample with this same search**. In the
run, search success went 43% → 58% and levels-whole 5/15 → 7/15 within three rounds of the change landing.

### A\* weight was already saturated — which is what made the next step clear

| A\* weight | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|
| solved | 5/15 | 7/15 | 7/15 | 7/15 | 7/15 |

Everything from 1.5 up gives the same answer. There was no better setting of this knob to find, so the next
lever had to be a **different signal** rather than more of this one.

### The better-trained head was not being used at all

The search ordered its frontier by `f = g + weight·h` — the value head alone, the head §8.4a measures as
compressed at long horizons (−37.5 moves at true distance 51+). Meanwhile the policy head agrees with
demonstrated moves ~94% of the time and solves three shipped levels outright with no lookahead whatsoever. It
was discarded at search time.

`Core/Planning/PolicyValueSearch` adds it as an accumulated **surprise** term:

```
f = g + weight·h + policyWeight·Σ −log π(a)
```

A line the policy would have played costs nothing extra; one it considers absurd pays per step. The accurate
head answers the local question it is good at, and the inaccurate one is left only the coarse question.

| policy weight | 0 (value only) | 0.5 | 2 | 5 | 15 |
|---|---|---|---|---|---|
| solved | 7/15 | 7/15 | 7/15 | **8/15** | 8/15 |

5 is the default: the smallest setting that buys Level 6, a level value-guided search never solved at any A\*
weight. Paths shorten too — Level 5 went 207 → 172 → 161 moves across the three tiers.

**The prior biases order; it never prunes.** Surprise is finite and accumulated, so a disliked node is
deprioritised, not removed, and the search still reaches anything uninformed search would given budget. A hard
policy mask can make a solvable problem unsolvable, which in an irreversible game where the winning line is
often the policy's second choice would be a serious defect rather than an optimisation.

### What this does to §4

It does not refute §4 — the value head really is compressed, and §6b's 41.5% unwinnable states really are
absent from the data. It **demotes** it. The measured cost of a weak value head turned out to be recoverable by
leaning on the head that is already strong, at zero training cost, and a categorical head still requires a
fresh run because it changes the output shape. So §4 stays the plan for the next deliberate rebuild, and is no
longer the thing standing between the current net and the shipped levels.

### Two flaws found in the curriculum while doing this

**A frontier could only ever move outward.** Advancing multiplies it by 1.5, so a level can be thrown past what
the net can solve — and a level that solves nothing produces no samples, so it can never recover. The
curriculum would park it there for the rest of the run, silently, while the other levels kept the reported loss
and accuracy looking healthy. A round that solves none of its attempts now retreats (×0.8, deliberately gentler
than growth, so a level settles at the edge of its ability instead of oscillating).

**A failed search taught nothing, and 42% of them failed.** Worst on the long levels — asking A\* for a 400-move
suffix in eight seconds nearly always fails — which are precisely the levels with the most left to learn. A
failed attempt is now retried on half the budget aiming at the *human's path* as well as the door: reaching a
state the demonstration also reached is a real solution, because a winning continuation from there is known,
and the prefix the search found is new data on states the net actually visits.

Two things had to be right, and the first was wrong in the first version. It must be a **fallback, not a
combined goal**: the start state is itself on the demonstrated path, so a single search aiming at both stops at
a landmark within a few moves every time instead of pressing on to the door — dismantling the frontier
mechanism while still looking like a healthy run. And a hit is a **candidate, not a proof**: the key is a 32-bit
state hash, a collision across a 200k-node search is a few percent likely, and believing one would manufacture a
training sample asserting a win that does not exist. Every hit is confirmed by replaying the demonstrated
remainder through the engine. Only genuine door-reaching solutions move a frontier; landmark hits are reported
separately in the eval line so the two can never be read as one number.

## 6e. Result — 15/15, and shorter than the teacher on 12 of them (2026-09-14, 03:35)

Once the beam budget stopped capping the curriculum (§6d), the frontier went 494 → 735 → **909** in two rounds
and every level reached whole. Benched on the checkpoint at 368k samples, beam width 256, ≤45s per level:

| tier | at 368k samples | at 604k samples (one hour later) |
|---|---|---|
| **greedy — policy alone, no search** | 10/15 | **15/15** |
| greedy + no-revisit tie-break | 12/15 | 15/15 — the tie-break is now redundant |
| policy beam search | **15/15** | 15/15 |

Every shipped level, including Level 7 (768 moves), Level 8 (462) and Level 11 (843).

**The search became unnecessary.** An hour of training on full-length solutions for every level — the regime
that only became possible once the curriculum reached 15/15 whole — moved the policy from needing a 256-wide
beam to needing nothing at all. Pure argmax, one forward pass per move, no lookahead and no backtracking.

That overturns a standing conclusion. §8.1a of the M58 PRD argued that irreversibility *forces* net + search as
the shipped artefact, because a single wrong move is unrecoverable and a deterministic policy is trapped by the
first state it revisits. That was correct about the net it was written for and is simply no longer true of this
one: a policy that does not make the wrong move does not need to recover from it. It also makes the artefact far
cheaper to ship — a bare forward pass per move ports to the browser twin with no search to reimplement.

### Against the human demonstrations it was trained on

| level | human | net | | level | human | net | |
|---|---|---|---|---|---|---|---|
| Level 1 | 19 | 19 | = | Level 8 | 494 | **463** | −31 |
| Level 2 | 74 | **73** | −1 | Level 9 | 227 | **214** | −13 |
| Level 3 | 98 | **94** | −4 | Level 10 | 386 | **348** | −38 |
| Level 4 | 281 | 281 | = | Level 11 | 909 | **840** | −69 |
| Level 5 | 164 | **128** | −36 | Bonus 1 | 42 | **41** | −1 |
| Level 6 | 113 | **107** | −6 | Bonus 2 | 110 | **85** | −25 |
| Level 7 | 782 | **768** | −14 | Bonus 3 | 39 | 39 | = |
| | | | | Bonus 4 | 132 | **114** | −18 |

**Total 3,614 moves against the human's 3,870 — shorter on 12 levels, equal on 3, longer on none.**

**Why that number matters more than the 15/15.** A net that had memorised the demonstrations would *match*
them. Beating them on twelve levels means it is not replaying recorded keystrokes — it learned the terrain well
enough to find better lines through it. That is the student-exceeds-teacher effect §6a predicted from local
search, now visible across whole levels.

It also refines §6d's held-out finding rather than contradicting it. Both are true: the net is specialised to
these fifteen boards (6% on unseen ones), *and* on those boards it is genuinely playing rather than reciting.
Overfitting to terrain is not the same as overfitting to a path.

Greedy play beats the human on 11 levels and ties the other 4 — **3,644 moves against 3,870** — so the table
above (measured on the beam tier at 368k) understates where the net ended up. The two are close because both are
now the same underlying policy, with and without a beam in front of it.

**What this is not.** The human solutions are recorded as non-optimal, so "shorter than the human" is not
"optimal" — no optimal reference exists for boards this size, which is why the demonstrations were recorded in
the first place. Nor is it a claim about Block Dude in general: see the held-out measurement above, where the
same policy scores 6%. This net plays *these fifteen levels*, and plays them better than the person who
recorded them.

All fifteen greedy lines are in [`docs/blockdude-ai-solutions.txt`](../blockdude-ai-solutions.txt), in the
game recorder's own format, so they can be pasted back into the page and watched.

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
