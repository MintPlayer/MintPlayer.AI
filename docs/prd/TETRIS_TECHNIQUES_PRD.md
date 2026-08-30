# Tetris — techniques: tetris-aware evaluator, movement-aware placements, SRS mode, technique dial

**Status:** 🧪 SPIKES RUN 2026-08-30 — **M57.0 complete, and it re-scopes the arc.** S0 is a decisive **GO** (the evaluator was the whole story); S1/S2 are a **NO-GO on tucks**; **M57.1 is BUILT and measured (§6.S: +97% score, 33× tetrises, protocol B improved)**; **S3 is a GO on the tap budget** — lateral reach is decisive above L19 (at the kill screen DAS scores **0**, rolling **37,135**). All in §6.R. Feature work M57.1–M57.7 not started. Planned via a 4-agent investigation (repo/architecture map,
NES-technique research, Tetris-AI literature survey, training feasibility + gates — findings in §2). **No spike
has been run yet**; §6 defines four, and **S0 is decisive enough that it may cancel most of the rest**.
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M57 · branch `m57-tetris-techniques` · builds on [TETRIS_PRD.md](TETRIS_PRD.md) (M54 ship, M55 NES input).

---

## 0. Why this exists — the owner's four asks, and what the investigation found

The owner's request, verbatim in substance:

1. the trained model doesn't make **tetrises** (the highest reward);
2. it never **moves a block sideways underneath existing blocks** — the piece always goes straight down;
3. it can't do **T-spins or tuck-spins** to clear a line;
4. professional players use **DAS, hypertapping and rolling** — the AI should know about these too.

The investigation produced one finding that reorders all four:

> ### The dense regression target is anti-tetris by construction.
>
> The shipped net is trained at γ=0 against the Dellacherie basis
> (`TETRIS_PRD.md` §3.7, `TetrisDqnCampaign.cs:74`):
>
> ```
> −20·landing + 8·eroded − 20·ΔrowT − 20·ΔcolT − 40·Δholes − 20·Δwells
> ```
>
> **`−20·Δwells` penalizes the very well a tetris requires**, and `+8·eroded` pays for clearing lines *now*.
> The net is explicitly trained to flatten the stack and burn. The measured **0.01 tetrises per episode**
> (`TETRIS_PRD.md` M54.3) is not an artifact of γ=0 myopia — it is the target function doing exactly what it says.
>
> The `+8` tetris bonus in the reward cannot compensate. It pays only on the placement that clears four rows,
> and at γ=0 there is no path for that credit to reach the 10–15 placements that *built* the well. Worse, it is
> structurally marginal: `RewardTetrisBonus` is **declared at `tetris_solver.pg:63` and never read in the `.pg`**,
> is absent from the dense target, is absent from both search tiers' rollout values (`:791`, `:858`), and reaches
> the learner through one sampled action per transition against dense-term gradient mass of weight 8 — **1/9 of
> the gradient**. Raising it will not move the number.
>
> **A γ=0 agent can build wells perfectly well, provided the evaluator says wells are good.** The evaluator is the
> only channel in this architecture that can ever teach well-building. That makes ask (1) a *formula change*, not
> a horizon change — and it is the cheapest, highest-value item in this document.

This supersedes `TETRIS_PRD.md` M54.3's diagnosis ("γ=0 cannot plan multi-piece wells"), which is true but is not
the binding constraint. It also gives a better explanation for the M54.4 gate miss (net+search 59% below
Dellacherie-alone on protocol B): the net was distilled from an evaluator with **no covered-well and no
accessibility term**, and garbage rows arrive with exactly the "hole under an overhang" structure those terms price.

**A correction to the record.** `TETRIS_PRD.md` §1 and §7 both state that *"the engine seam `enumeratePlacements()`
is written so a BFS pathfinder (tucks/spins) can replace it later without touching the learner."*
**That function does not exist** — `grep -rn enumeratePlacements` over `.pg`, `.cs` and `.ts` returns zero hits. The
vertical-drop assumption is inlined at **seven sites** in `tetris_solver.pg`, each calling `dropY` directly, plus the
C# facade, the campaign's dense-target reader, and the browser pilot. There is no single swap point. This PRD plans
the work at its real size, and §9 removes the false claim from the M54 PRD.

---

## 1. Owner decisions (locked 2026-08-30)

| # | Decision | Consequence |
|---|---|---|
| **D1** | **SRS + kick tables as a SECOND MODE**, alongside the ROM-exact NRS engine — not a replacement | Real T-spins become expressible. NRS stays the default and keeps the M54 parity pin. §5 |
| **D2** | **Human input budget, technique as a dial** — DAS / hypertapping / rolling | Placement reachability becomes a function of tap rate and gravity. §4 |
| **D3** | **Full from-scratch retrain accepted** — "whatever it takes to make the model capable" | Every existing checkpoint is expendable. Cost is stated honestly in §7, not used as a constraint |
| **D4** | **Three visitor-facing radio buttons** on the Tetris page: DAS / Hypertapping / Rolling | The same dial drives human play *and* the AI's reachable set. §4.4 |

### 1.1 One decision still open — and it changes the shape of the SRS work

**Under NES scoring, a T-spin is worth essentially nothing.** This is arithmetic, not opinion:

| Clear | NES table (×(level+1)) | Guideline table (×level) |
|---|---|---|
| Double | 100 | 300 |
| **T-spin Double** | **100** — it is scored as a plain double | **1200** |
| **Tetris** | **1200** | 800 |

Under Guideline scoring a TSD (1200) *beats* a Tetris (800), and that inversion is the entire economic reason
T-spins are the backbone of modern play. Under the NES table the inversion runs the other way: **a TSD pays 1/12 of
a tetris, for two rows you could have stacked into the tetris instead.** There is no B2B, no combo, no garbage to send.

So D1 splits into two genuinely different projects, and they need different gates:

- **(A) Spins as a strength lever.** Requires SRS mode to carry **Guideline scoring** (100/300/500/800, T-spin
  800/1200/1600, B2B ×1.5). Otherwise the AI will correctly learn to almost never T-spin and the feature will look
  broken to the visitor. This makes SRS mode a *second game*, not a second rotation system — different scoring,
  different level curve (Guideline multiplies by `level`, NES by `level+1`, and the line schedules differ).
- **(B) Spins as a pathfinder side-effect + browser demo.** Ship SRS + kicks purely as the **movement model**, so
  spin placements enter the action set implicitly — which is how every guideline bot gets spins, by *not restricting
  the action space*, never by adding spin logic. Keep NES scoring. Spins then function as a **digging tool** on
  dirty/garbage boards (they fill a row under an overhang that a hard drop cannot reach, converting a would-be
  permanent hole into a clear), and as a visibly impressive thing to watch. T-spin count is a **reported statistic,
  never a gate**, and a low count is correct behaviour rather than a bug.

**Recommendation: (B), with (A) available as a later mode.** It gets the owner's ask (3) — the AI *does* T-spins,
visibly, in the browser — at near-zero marginal cost over the SRS tables D1 already commits to, without forking the
scoring table, the level curve and the whole eval protocol. This PRD is written for (B); §5.5 marks precisely what
(A) would add. **Owner call required before M57.4.**

---

## 2. Investigation findings (4 agents, 2026-08-30)

### 2.1 Repo / architecture map

- **`dropY:345` is the single line that forbids tucks.** It rejects a placement unless the column is clear from row
  0 down (`!fitsAt(..., y=0)` ⇒ −1). It is also what defines `terminated`, via `applyPlacement:481`. Relaxing it
  redefines top-out.
- **`simPlace(piece, rot, x, y)` (`:356–370`) already takes an explicit `y`** — the one placement-agnostic
  primitive. It accepts a tucked landing today. Likewise `fitsAt` (`:329–339`, `y ≥ −2` NES head-room) is a
  complete collision oracle. **A reachability search needs no new geometry.**
- The **micro API is a complete frame-level simulator** (`microSpawn/microShift/microRotate/microDropStep`,
  `gravityFrames`) with **no enumeration or search over it**. That asymmetry is the whole gap.
- **`ActionCount = 40` is a 6-way simultaneous break**: `.pg` const → obs width (`214 + 6N`) →
  `TetrisBoard.cs:24` → net head → the shipped 771 KB LFS checkpoint → the `× actions` dense rescale at
  `DqnTrainer.cs:336` (which silently rebalances dense-vs-realized gradient when N changes).
- **Both stale-checkpoint guards check `inputSize` only** (`TetrisBoard.cs:150`, `tetris-director.ts:35`). They are
  *incidentally* safe today because obs = 214+6N moves whenever N moves. An action-count change with an unchanged
  obs width would load a mismatched net and mis-index actions.
- **Three independent checkpoint parsers** must move together: `TetrisBoard.ParseDuelingQCheckpoint:162–204`,
  `tetris-net.ts:34–86`, `tools/tetris_head2head.mjs:8–30`.
- **`rotCount` is read by five enumerators** (`:344, :651, :684, :761, :850`) — see the §5 trap.
- **`simLanding`/`simEroded` are mutable instance scratch** (`:119–120`), valid only until the next `simPlace`. A
  speculative search calling `simPlace` inside the observation builder clobbers them. Same for
  `saveRows`/`restoreRows` (`:622–632`), already the hot path at 7 forwards × beam × 40.
- **`dellaSearchAction` already contains three different enumeration idioms in one function** — 40-slot at ply 1
  (`:731`), `rotCount × (W−rotW+1)` at ply 2 (`:761`), `bestDellaFor` at ply 3 (`:784`). A reachability set must be
  threaded through all three or the plies evaluate different action sets.
- **~~Polyglot 0.8.1 cannot express a real BFS queue.~~ MEASURED FALSE 2026-08-30** — see
  [polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md](polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md). The claim at
  `snake_solver.pg:266–269` (TS7022 evolving-any) **does not reproduce on 0.8.1 or on HEAD**: a real worklist BFS —
  `while`, index assignment, and `frontier.add(x)` where `x` derives from a read of `frontier` — emits
  `let frontier: number[] = []` and passes `tsc --strict`. `while`, `record`, nested `List<List<i32>>` and
  interfaces are all available too. **Every M57 construct compiles on the pinned 0.8.1.** No compiler change and
  no version bump are required; the Snake workaround is stale and is a follow-up cleanup.
- **`tetris-das.ts` is TS-only** (126 lines, no imports). `DAS_FULL`/`DAS_RESET` are **module-level `const`**, so
  **the tap rate is not settable at runtime today** — D4's radios depend on the `.pg` port in §4.3.
- **CI runs no node harness.** `tetris_parity.mjs`, `tetris_das_check.mjs`, `tetris_head2head.mjs` are all manual.

### 2.2 NES technique research

**Frame arithmetic (NTSC, 60.0988 fps, from the disassembly and tetris.wiki):**

- DAS: fresh press shifts immediately (counter := 0); held, counter → **16** then shift and subtract 6 (→10) ⇒
  sustained **one shift per 6 frames = 10.02 Hz**. Blocked tap saturates to 16 (**wall charge**); only a fresh press
  rewrites the counter, so charge survives locks.
- **Hypertapping** — Thor Aackerlund, 2011. Sustained human **10–15 Hz** (12 Hz the working figure, 13–15 for the best).
- **Rolling** — Cheez, November 2020: a thumb rests on the d-pad while other fingers drum the shell into it, so the
  rate multiplies by finger count instead of one thumb's oscillation. Sustained **20 Hz+**, top rollers 20–25 Hz.
  This is what broke the killscreen: 1.4M (2021) → 3.77M (2022) → 8.9M (2024); level-157 crash Dec 2023; rebirth Oct 2024.
- **Hardware ceiling is 30.05 Hz**, not 60 — the pad is sampled once per frame and only *newly pressed* bits count,
  so a button must be released for ≥1 sampled frame. **This is a real bug in `tetris-das.ts`** (§9).
- **Effective lock delay = one gravity period.** The drop routine backs up, increments, and on failure restores and
  locks — so a landed piece waits exactly `gravityFrames` before locking: **3 frames at L18, 2 at L19–28, 1 at L29+**.
  *This is the entire budget for every tuck and spin.*
- Frame order is **shift → rotate → drop**, and shift and rotate may occur on the same frame. So a placement's input
  cost is `max(|Δx|, rotations)`, not their sum — and even the 1-frame killscreen window admits one input.
- **Spawn is x=5**, so the **left wall is 5 taps away and the right wall 4**. This asymmetry is why the well goes on
  the right, and why StackRabbit weights `inaccessibleRight −200` against `inaccessibleLeft −100`.
- ARE is **10–18 frames** by lock height; line-clear delay a further **17–20**. DAS neither charges nor resets during
  either, so a held direction can be redirected for free — a real technique the repo cannot express (§9).

**Reachability — `max4TapHeight` / `max5TapHeight`** (the highest stack surface at which a right-wall / left-wall
placement still lands), recomputed per StackRabbit's algorithm for each timeline × gravity:

| Technique | L18 (g=3) | L19–28 (g=2) | L29+ (g=1) |
|---|---|---|---|
| **DAS 10 Hz** | 12 / 10 | **9 / 6** | **0 / −6** |
| **Hypertap 12 Hz** | 13 / 12 | 11 / 8 | **3 / −2** |
| Elite hypertap 15 Hz | 14 / 13 | 12 / 10 | 6 / 2 |
| **Rolling 20 Hz** | 15 / 14 | 14 / 12 | **9 / 6** |
| Ceiling 30 Hz | 16 / 16 | 15 / 14 | 12 / 10 |

Read this table as the justification for the whole feature:

- **L18: nothing is constrained.** Every hard drop and most tucks are reachable at any rate.
- **L19–28: DAS reaches the right wall only while the stack ≤ 9, the left only while ≤ 6.**
- **L29+ with DAS: `max5 = −6` — the left wall is unreachable at any height, and a 4-tap works only on a bare
  floor.** You cannot feed the well, so you cannot score a tetris. *That is the historical killscreen, as a number.*
- **Rolling at 20 Hz on the killscreen gives 9 / 6 — numerically identical to DAS at level 19.** One equivalence
  explains the entire rolling era: rolling converts the killscreen into "level 19 with a normal controller".

**Consequence the plan must absorb:** `TETRIS_PRD.md` §1 states *"the AI's placements have no gravity clock —
benchmark convention."* **A tap budget is meaningless without a gravity clock.** Reachability is
`f(board, piece, level, tap timeline, reaction delay)`. Dropping that convention is the real cost of D2 — bigger
than the SRS tables.

**NRS tuck/spin rules.** The rotation check at `$948B` tests **only the four target cells** — not the swept path,
not the diagonals. So NES permits exactly two families: **slides** (descend past an overhang lip in an open column,
then shift underneath within the grace window) and **spins/spin-tucks** (rotate mid-fall into a slot whose entry is
blocked for a translation, exploiting diagonal pass-through). The repo's §3.10 already models the validity rule
correctly. **T-spins as a scored move do not exist in NES Tetris** — the ROM scores by line count only.

### 2.3 Tetris-AI literature

**The academic literature optimizes LINES; the owner's objective is NES SCORE. These are different games.**
Dellacherie (660K lines), BCTS (35M), CBMPI (51M) are all uncapped line-clearing records with no scoring table;
none build wells on purpose, because clearing singles is cheapest. Transplanted into the NES table, a 35M-line agent
is a mediocre *score* agent. Every number in `TETRIS_PRD.md` §2.2 comes from that literature.

The agents that actually maximize NES score — **StackRabbit** and **BetaTetris** — are not in it, and use a
different feature set. StackRabbit's shipped weights (`params.hpp`), which are the answer to ask (1):

| Feature | Weight | |
|---|---|---|
| `hole` | −50 | standard |
| `tetris` | +50 | reward for actually scoring one |
| **`tetrisReady`** | **+6** | 4 rows complete except the well column, well height ≤ 16 |
| **`coveredWell`** | **−10** | capping your own well (scales with height-ratio **cubed**) |
| **`burn`** | **−12** | **explicit penalty per non-tetris line cleared** |
| `col9` | −3 | keeps column 9 low so the well stays reachable |
| `builtOutLeft` | +2 | |
| **`inaccessibleLeft` / `Right`** | **−100 / −200** | **the tap-speed reachability price, in the evaluator** |
| `avgHeight` | −9 | quadratic above a scare threshold |
| `death` | −3000 | |
| `surface` | +1 | base-7 surface encoding → value-iteration lookup |

...plus **mode-switched weight sets** rather than one static vector: `DIG` (burn → −1, holeWeight → −7),
`NEAR_KILLSCREEN` (**tetris → +500**), `LINEOUT` (burn → 0, builtOutLeft → +15), selected by
`getAiMode` — and critically, **`max5TapHeight < 4` ⇒ LINEOUT**, i.e. *when the left wall becomes unreachable at
this tap speed, stop building for tetrises and just survive.* That single rule is the most reusable idea in the
entire codebase for a technique dial.

**All of it is linear over the afterstate**, so it drops into the γ=0 dense target with no structural change: the
target stays `w · φ(afterstate)`, only φ and w widen.

**Measured failure modes** (practitioner journal, single-source — treat as strong anecdote):
pushing the clear-reward mapping harder toward 4-line clears made the agent *aim solely for tetrises and never
T-spin*, and **lowered** score-per-piece. The classic stack-and-camp hack (superlinear bonus + distant death
penalty ⇒ stack to ¾ height and burn for survival) is measured, not folklore. Also measured: **changing reward or γ
mid-training never worked** — a materially different reward needs a from-scratch retrain (moot here, D3).

**Action-space expansion.** StackRabbit simulates hypothetical frames (`exploreHorizontally` + a near-spawn pass +
`findTucks` against a precomputed `TUCK_SPOTS_LIST`), and prices tap-speed reachability *in the evaluator*
(`inaccessibleLeft/Right`) rather than in the action. Guideline bots use a plain SRS BFS over
{L, R, soft-drop, CW, CCW} deduped on (x, y, rot) — **tucks, slides, T-spins and wall kicks all fall out for free
with no special-case code.** BetaTetris uses a `kR × kH × kW = 4 × 20 × 10 = 800` policy head — *the lock row is
part of the action*, which is exactly what tucks require.

**No public A/B of tucks-on vs tucks-off exists.** The larger space is universally adopted by the strongest NES
agents and never ablated. §8 gate G4 is how this repo generates that number.

**Sample efficiency — a real advantage of this repo's setup.** Under γ=0 dense all-action regression you get a
supervised target for *every legal slot on every step*, so effective sample count scales **with** branching rather
than against it. A wider action space is cheap here in a way it is not for standard DQN.

**Per-setting vs conditioned model.** BetaTetris v0.1.0 shipped one model per setting; v1.0.0 replaced it with a
single model conditioned on tap speed / reaction time / aggression, and got stronger. **But this repo is a special
case that is easier:** under γ=0 dense afterstate regression the target is `w · φ(afterstate)` — a function of the
board after the piece locks and *nothing else*. It does not depend on tap speed. The budget changes only *which
placements are legal*, i.e. the mask. Hence §3's lock.

**T-spin verdict.** §1.1. Ship the pathfinder, never the reward term.

### 2.4 Training feasibility

**tet1–tet7, from `data/tet*train/logs/`** (the CLI args are recorded nowhere in the repo — reconstructed from
`TETRIS_PRD.md` §3.7 + `TetrisLab.cs` defaults):

| Run | Recipe | Steps | Wall | Best eval | Signature | Outcome |
|---|---|---|---|---|---|---|
| tet1 | γ=.995, n3, bare | 180K | 23 min | 19 / 0.5 lines | **tiny loss = starvation** | near-random |
| tet2 | + PBRS v1 (Φ sign bug) | 95K | 13 min | 2 / 0.1 | worse than unshaped | bug found |
| tet3 | + PBRS fixed | 105K | 13 min | 16 / 0.4 | Spearman 0.27 vs Dellacherie | γ 0-for-3 → pivot |
| tet4 | **γ=0 + dense, 128²** | 400K | 62 min | 16,316 / 70.5 @220K | healthy | **M54.3 ship** |
| tet5 | + `--mix-garbage` | 400K | 63 min | 18,855 @275K | late collapse −67% | measured worse |
| tet6 | **γ=0 dense, 256², dw 8, lr 5e-4** | 330K | 111 min | **83,265 / 190.3 @70K** | **falling loss + falling eval = distribution narrowing** | **M55.4 ship** |
| tet7 | warm-start refine | 300K | 76 min | 88,425 @50K | eval band healthy, loss flat | **held-out WASH** |

**The tet7 lesson is the most important protocol fact in this document.** It scored 88,425 on the *selecting*
seeds (5000+e) and was a complete wash on **held-out seeds (9000+e)** — all four tiers CI-overlapping. **≈85K is
the recipe's held-out ceiling at this scale.** Ship decisions go through `tools/tetris_head2head.mjs` on 9000+e or
they mean nothing.

**Protocol A is saturated.** Dellacherie clears 197.6 lines against a ~200 ceiling under the 500-piece cap. A is a
floor test, not a discriminator. B remains primary.

**GPU is not a lever.** `AdaptiveBackend.DefaultGpuMacThreshold` routes at 256M MACs/GEMM; the largest Tetris GEMM
is 128 × 454 × 256 = **14.9M — 17× below threshold**, and still 3× below at N=400. There is no resident dueling
trainer in the repo. "Whatever it takes" (D3) means **CPU hours**.

Measured throughput fits **rate ≈ C / params, C ≈ 9–10.5 M params·steps/s** on these 8 cores:

| N | obs (214+6N) | params | steps/s | 400K steps | state ckpt @100K buf |
|---|---|---|---|---|---|
| 40 (today) | 454 | 192K | ~55 | **2.0 h** | 0.37 GB |
| **160** | **1174** | **409K** | **~19** | **5.9 h** | **0.94 GB** |
| 200 | 1414 | 480K | ~16 | 6.9 h | 1.13 GB |
| 800 (full grid) | 5014 | 1.56M | ~4 | **28 h** | 4.0 GB |

The replay buffer is written to disk **every 10 minutes** (`tetris.dqn-state.ckpt`; tet7's is 1.107 GB at
300K × 454). At 300K × obs 1414 it is 3.4 GB. **Cap the buffer at 100–150K**, or the disk traffic dominates.

**Checkpoint reality.** An obs-width change alone survives as a *warm start* (`DuelingQNet.GrowInput`, new inputs
at zero weight) but **never as a resume** (`DqnTrainer` throws on `state.CurrentObs.Length != obsDim`). An
**action-head change forces from-scratch** — `NetTransfer` explicitly throws on non-input length mismatch, and
nothing in the repo can grow an action head. Since obs = 214+6N, N moves both. Accepted under D3; tet6 survives as
the **comparison baseline** for the no-regression gate.

---

## 3. Design — locks

### 3.1 LOCK A: widen the evaluator (the #1 lever, and it is a formula change)

Replace the 6-term Dellacherie basis with a **tetris-aware basis φ**, used *identically* in three places that
today disagree: the dense regression target, `dellaScoreFor`, and the per-action observation planes.

New terms, all linear over the afterstate, all ported from StackRabbit's shipped weights:

| Term | Meaning |
|---|---|
| `tetrisReady` | 4+ rows complete except the well column, well height ≤ 16 |
| `coveredWell` | the well column is capped (height-ratio cubed) |
| `burn` | **per non-tetris line cleared** — the term that makes singles *cost* something |
| `col9` | column 9 height above `maxSafeCol9` |
| `builtOutLeft` | far-left column above scare height when clean |
| `holeDepth`, `rowsWithHoles` | the BCTS extension (35M lines on the lines objective) |
| `inaccessibleLeft` / `inaccessibleRight` | **the tap-speed price** — derived from `max5TapHeight` / `max4TapHeight` (§4.2) |
| `avgHeight` | quadratic above a scare threshold |

**Three fixes that come with it:**

1. **The `−Δwells` sign trap.** The well you keep open for a tetris must not be penalized as a well. Split the term:
   penalize wells **outside** the well column, reward depth **in** it (bounded by `tetrisReady`'s ≤16 condition).
2. **Drop the realized-reward term from the dense target.** Once φ contains `tetris` and `burn`, the realized clear
   is already priced *inside* the evaluator; the second gradient buys nothing but the documented unit conflict
   (`dense della/10` vs `realized lines`, currently patched with `--dense-weight 8`). This removes the patch.
3. **Mode-switched weights, not one static vector** — `STANDARD` / `DIG` / `LINEOUT` / `NEAR_KILLSCREEN`, selected
   by the StackRabbit rule set, with **`max5TapHeight < 4` ⇒ LINEOUT** wiring the technique dial directly into
   strategy. This is what makes a DAS agent correctly stop building for tetrises at the killscreen.

**`RewardTetrisBonus` stays as-is and stays marginal** — it is not the lever, and §0 explains why. The engine const
must either be wired into the rollout values it claims to serve or deleted; leaving a dead const that the comment
says is load-bearing is worse than either.

### 3.2 LOCK B: movement-aware placements, replacing the vertical drop

A placement is legal iff the **frame machine**, at the selected tap timeline and the current level's
`gravityFrames`, can bring `(x, y, rot)` to the lock cell. Concretely, StackRabbit's shape:

- `exploreHorizontally` — a hypothetical frame simulation sweeping left and right for all goal rotations, applying
  a shift and a rotation on input frames and a gravity step on gravity frames;
- `explorePlacementsNearSpawn` — the blind spot: placements needing **more rotations than shifts**;
- `findTucks` — scan overhang cells against a precomputed per-piece tuck-spot list, gravity-drop, and ask whether
  any of the 8 single-frame inputs reaches it (`L R A B` plus `E I F G` = simultaneous shift+rotate, the true
  spin-tucks), validated against a y-window and checked **twice** to respect shift→rotate→drop ordering.

**Input cost as a small negative eval term** (`TUCK_COST −0.1`, `SPIN_COST −0.2`, `SPINTUCK_COST −0.3` per input) —
this is how "human-plausible" is obtained without a separate model: the AI takes the simple placement unless the
tricky one is genuinely better.

**Polyglot: no constraint.** Measured 2026-08-30 — every construct here compiles on the pinned **0.8.1**, including
a real worklist BFS (§2.1 and the feasibility handoff). **Prefer the frame simulation anyway, but for cost, not
expressibility:** a bounded forward `for` sweep is simpler and cheaper per call inside `buildObservation`, which
runs 7× per beam node. The queue remains available as the fallback if the sweep proves insufficient, and the
`(rot, x, y)` relaxation over ~880 flags indexed `rot*220 + y*10 + x` is a third option, not a workaround.
**No compiler change; no version bump in this milestone** (§11).

**The gravity clock.** `TETRIS_PRD.md` §1's "no gravity clock — benchmark convention" is **dropped**. Reachability
depends on level. Therefore **level must enter the observation** (it is not there today: 200 board + 7 + 7 + planes)
or the net sees a non-stationary action space.

### 3.3 LOCK C: action encoding — fixed superset head, mask varies

**`(rot, col, depth-below-hard-drop d)` with `d ∈ 0..3` ⇒ N = 160** — a strict superset of today's 40 (all at d=0).
Obs = 214 + 6·160 = **1174**. Measured cost 5.9 h for 400K steps.

Rejected alternatives, and why:

- **800-slot `4 × 20 × 10` (BetaTetris shape)** — 28 h and a 5,014-float MLP input. Reconsider only if S1 shows the
  depth distribution has a long tail.
- **Variable-length `V(afterstate)` scoring** — this is what *every* strong Tetris agent does, and it is the
  natural shape for a dense-regression evaluator; it also removes the action-count coupling entirely.
  `TETRIS_PRD.md` §3.3 rejected it on **trainer-machinery cost, not strength**, and that justification is now
  weaker. **It is the designed successor**, and the fallback if S1 measures a 99th-percentile tuck depth > 4.
  Not the M57 choice, because the fixed head reuses `DqnTrainer` unchanged and D3's budget is better spent on the
  evaluator than on new trainer machinery.

**The d-cap is set by S1's measurement, not by this document.**

### 3.4 LOCK D: one net, masked at inference — the technique dial costs zero extra training

> Train **one** net over the **full** reachable set. Apply the tap budget purely as an **inference-time action mask**.

This is exact, not an approximation, and it follows from the architecture: under γ=0 dense afterstate regression the
target is `w · φ(afterstate)` — a function of the board *after* the piece locks, and nothing else. Tap speed does not
enter it. The budget changes only which placements are legal.

Consequences:
- The visitor's DAS/hypertap/rolling radios cost **zero additional training runs** (contrast: six nets = 35 h).
- It generalizes to rates never trained on, because the mask is applied outside the net.
- It is the only design that can answer *"is rolling actually better than DAS"* without confounding the answer with
  two different networks.
- **Do not add the budget as an observation feature.** At γ=0 it is provably uninformative for the target, so it can
  only add variance.
- **Do** randomize the budget during rollouts — free, and it diversifies the replay distribution, which is the exact
  medicine for tet6's measured distribution-narrowing failure.

**The one exception, and it is important:** if the evaluator weights are CEM-tuned against *score* (§6, S0b), that
fitness **is a return**, so the tuned `w` *is* tap-budget-dependent. There, per-setting is both simpler and better:
run CEM three times and ship **three weight vectors (~24 floats each)**. That is the cheapest possible conditioning
and it sidesteps the question entirely.

> **Clean split: one net, masked at inference; three tuned weight vectors, one per technique.**

---

## 4. The technique dial

### 4.1 The three settings

Both serious NES bots parameterize tap speed as a cyclic **input-frame timeline** string (`X` = a frame on which an
input may be issued). Copy the abstraction verbatim.

| Setting | Timeline | Frames/shift | Hz | Justification |
|---|---|---|---|---|
| **DAS** | `X.....` | 6 | 10.02 | exact ROM auto-repeat, charge held |
| **Hypertapping** | `X....` | 5 | 12.02 | human-sustained (10–15 Hz range) |
| **Rolling** | `X..` | 3 | 20.03 | sustained rolling |
| *(ceiling, exposed but not a radio)* | `X.` | 2 | 30.05 | hardware maximum |

**DAS caveat:** the 6-frame uniform timeline is the *charged* model. A cold DAS after a direction change costs an
extra 16 − 6 = 10 frames. StackRabbit keeps the real counter in the outer layer and hands the core the flat 10 Hz
timeline, then prices the difference with `minSafeDasCharge` per lock position + a `LOSS_DAS_PENALTY`. Since the
repo **already has the real counter** in `NesInput`, keep it authoritative for human play and let the planner use
the timeline plus a charge-carry-over penalty. That penalty is what makes DAS mode wall-charge like a human.

### 4.2 Tap speed → board constraints → strategy

Port `computeYValueOfEachShift` → `max4TapHeight` / `max5TapHeight` (~20 lines), then derive, exactly as StackRabbit does:

```
scareHeight   = 0.5·(max5TapHeight − 3) + 3
maxSafeCol9   = 0.5·(max4TapHeight − 5) + 4
aiMode        = max5TapHeight < 4 ? LINEOUT : (holes ? DIG : STANDARD)
```

This is the join between §3.1 and §4: the dial does not merely filter actions, it **changes what the evaluator
wants**, which is what makes a DAS agent at level 29 correctly abandon tetris-building.

### 4.3 Porting `NesInput` into the shared `.pg`

Prerequisite for D4 — the rate is currently unsettable (§2.1). Required, concretely:

1. `DAS_FULL`/`DAS_RESET` become **instance `var` fields** (the dial writes them); `SOFT_FIRST`/`SOFT_REPEAT` become
   `const` class members (module-level `let` is unusable inside methods on 0.8.1, `.pg:65–67`).
2. **The `DasHost` interface disappears** — `.pg` has no interfaces, closures or function values. Inline against
   `this.microShift` / `this.microDropStep` / `this.gravityFrames`, all already methods on `PgTetris`.
3. `tick` loses its host parameter and gains explicit edges:
   `tick(pressL, pressR, pressDown, heldL, heldR, heldDown): bool`. The body is straight-line `if`/`++` arithmetic —
   nothing in it exceeds the 0.8.1 surface.
4. **11 new fields join serialization** (`TetrisBoard.WriteState/ReadState:223–278`), changing the
   `TetrisEnv.SaveState` byte layout and invalidating every `data/tet*train/*-state.ckpt` **resume** file — separate
   from the model checkpoints D3 already accepts losing.
5. Repoint `tools/tetris_das_check.mjs` at the generated twin (a one-line import change), and **mirror its 11 checks
   into `TetrisEngineTests`** so CI actually covers them (§2.1: no node harness runs in CI).

### 4.4 Web — the three radios

The tier selector is the template to copy, and it is small:

- **Signal + setter:** `tetris.ts:41` `tier = signal<Tier>('net')` / `setTier:109–112`. Mirror as
  `tapRate = signal<'das'|'hypertap'|'rolling'>('das')` + `setTapRate`.
- **Markup:** `tetris.html:9–26`, the `.modes` div. The radios are a **third group visible in BOTH modes** (they
  drive human play *and* the AI budget), so they go **outside** the `@if (mode() === 'human')` block at `:16–25`.
  Style hooks `.modes` / `.tier-label` in `tetris.scss:16–23`.
- **Persistence:** the `localStorage['tetris.zoom']` try/catch shape at `tetris.ts:57–68`.
- **Human play:** the component sets the rate on `game.input` in `setTapRate` and at `newGame()`. **Two carve-outs
  to preserve:** the pointer path is absolute-position and *intentionally not DAS-limited* (`tetris.ts:250–253`,
  `TETRIS_PRD.md` §3.10), and rotation stays one-per-press.
- **AI:** set the rate as state on `PgTetris` before the tier call rather than threading a parameter through all five
  (`.pg` has no overloading, so a new parameter is a breaking rename at every call site).
- **The pilot must replay, not re-plan.** `tetris-game.ts:139–164` currently re-derives its own route
  (rotate → shift → hard-drop at `PILOT_INPUT_MS = 90` ≈ 11 Hz) and bails on `stuck >= 2` (`:159`) — **silently
  substituting a different placement, with nothing testing or measuring the divergence.** Under a reachability space
  the pilot replays the input sequence the engine chose, `PILOT_INPUT_MS` is replaced by the dial, and `stuck`
  becomes a genuine bug indicator.
- **Status line** (`tetris.ts:161–171` → `tetris-render.ts:62`) shows the active rate in both modes.

**This is also the demo.** The visitor hands the AI faster inputs and watches it get measurably better — and, at the
killscreen, watches DAS fail to reach the well at all while rolling still feeds it. That is the rolling revolution,
playable, with the numbers from §2.2 behind it.

---

## 5. SRS as a second mode

### 5.1 The representation already hosts it

Two properties make this additive rather than a rewrite:

1. **`cellX`/`cellY` already use a stride of 4 rotation slots per piece** (`(p*4+r)*4+k`) — NRS merely pads the
   unused ones via `addEmptyRot():159–161`. SRS's 4 states fit the existing 28 slots with **no index change**, and
   `rotW`/`rotH`/`rotBottom` derive automatically from `:258–275`.
2. SRS is structurally *"per-state box offset + a kick list"* — exactly the `rotOffX`/`rotOffY` + `fitsAt` scheme
   already in place. `microRotate:515–516` already computes the new position as an offset delta, which is the
   arithmetic SRS uses before applying kicks.

### 5.2 What gets added

- Second offset table `srsOffX`/`srsOffY` (28 each) + `srsSpawnRot` (7), built with the same `add` loops.
- **Kick tables**: JLSTZ 8 transitions × 5 tests × 2 coords = 80 ints, I another 80, O none. Flat `List<i32>`,
  indexed `((from*4+to)*5 + test)*2`. Well inside the 0.8.1 surface — the file already builds 112-entry flat tables.
  Both tables are transcribed in the research findings and must be pinned by test, not typed from memory.
- **CCW rotation.** `microRotate()` takes no direction and `.pg` has **no method overloading**, so this is either a
  renamed `microRotateDir(dir)` — breaking `TetrisBoard.MicroRotate:94`, `tetris-game.ts:86,150` and the six
  `NesRotation_*` tests — or a second distinctly-named method. Prefer the second; the rename buys nothing.
- **T-spin detection state**: `lastMoveWasRotation: bool` and `lastKickIndex: i32` (mini-vs-full needs the kick
  index), both joining serialization (§4.3 item 4).
- **Mode field** `rotSystem: i32`, set by a **separate setter**, not by widening `reset(...)` — whose call sites
  include `TetrisBoard.cs:31`, `tetris-game.ts:60`, both node harnesses and every engine test.

### 5.3 The `rotCount` trap — the easiest silent breakage in the arc

**`rotCount` is read by five enumerators** (`dropY:344`, `bestDellaFor:651`, `buildObservationFor:684`,
`dellaSearchAction:761`, `netSearchAction:850`). SRS defines **4 states for every piece**. If SRS mode sets
`rotCount[p] = 4` for I/O/S/Z, duplicate-shape rotations become separately legal actions — the legal set grows,
`randomAction` draws differently, and `SpikeBar_*` (`TetrisEngineTests.cs:371,378,393`) plus every number in
`data/tetris-baselines-final.txt` shift — **before a single kick fires.** Canonicalize duplicate-shape states in the
enumerator, and pin that with a test.

### 5.4 T-spins with no lock delay

Sharpest open question in the mode, and the answer is more permissive than it looks: NES-style "no lock delay" is
not zero — locking is attempted only on a gravity step, so a landed piece sits for one full gravity period
(3 / 2 / 1 frames at L18 / L19–28 / L29+), and shift→rotate→drop ordering means even the 1-frame case admits an input.

- **Survives:** every T-spin rotated **from the T's natural resting position** — the standard TSD (vertical T on the
  overhang lip, downward `(0,−2)` kick) and the standard TST (1×2 kick). Downward kicks are *helped* by no lock
  delay: they move the piece where it was falling anyway.
- **Does not survive:** anything needing ≥2 sequential inputs after landing (land → shift → rotate) below L18; and
  all move-reset / infinity abuse.
- So T-spins become **frame-tight rather than impossible**, scaling with gravity exactly like NES tucks.

**This derivation is reasoned from frame ordering, not cited** — no shipped game does SRS without lock delay, so no
source discusses it. **Verify empirically in S1b before committing** (build the canonical TSD board, enumerate
reachable lock states at g = 3, 2, 1 with the kick tables and no lock delay).

### 5.5 What option (A) from §1.1 would add

Guideline scoring table (100/300/500/800; T-spin 800/1200/1600; mini 200/400; B2B ×1.5; combo 50×n×level), a
separate level curve (×`level` not ×`level+1`, different line schedule), `lineScore` widened to
`lineScore(cleared, tspinKind)` — cheap, one call site — plus its own eval protocol and baselines, because none of
the M54 numbers transfer. **Not in M57 unless the owner picks (A).**

### 5.6 Parity

The M54 pin (**472451993**) hashes action / cleared / rows / current / next and never touches `rotOffX`. **If SRS is
strictly additive and the NRS tables including `rotCount` are untouched, the pin survives** — and that is the
regression test for §5.3. A **second pin** is added for an SRS-mode run of the same protocol.

---

## 6. Spikes — run before any engine work, in this order

Each is cheap. **S0 may cancel most of the rest**, which is why it is first.

### S0 — Evaluator widening, zero training, ~15 min. *Is the low tetris rate really the target function?*

Add the §3.1 terms to `dellaScoreFor` (a formula change in the `.pg`, no action-space work, no retrain) and measure
`dellacherieAction` + `dellaSearchAction` on protocol A: **tetrises/episode, TRT = 4·tetrises/lines, NES score,
score-per-piece**. Sweep the `tetrisReady` / `coveredWell` / `burn` / `col9` weights over ~6 points.

- **GO** if some weighting reaches **≥ 2.0 tetrises/ep while keeping score ≥ 85,000** — the diagnosis in §0 is
  confirmed, ask (1) is a formula change, and LOCK A carries the arc.
- **NO-GO** if every weighting that raises tetrises drops score below ~70,000: tetris rate and score are in genuine
  conflict at this board size under the 500-piece cap, and G3's threshold must be renegotiated **before** training
  rather than after.

*This is the single most informative 15 minutes in the plan. It tests the #1 lever with no engine work at all.*

### S0b — CEM on the widened basis, ~20 min. *Does tuning pay on the right basis?*

~100-line CEM/CMA-ES loop on `DeterministicParallel` over the widened weight vector, fitness = **NES score on
protocol A** (this is `TETRIS_PRD.md`'s un-run M54.7, on a basis that can actually express tetris play). Run once per
tap setting per §3.4. Note CMA-ES on the *narrow* Dellacherie basis provably converges back to Dellacherie's hand
weights — **widen first or this is wasted**.
**GO** if a tuned vector beats hand weights on protocol B, CI-separated.

### S1 — Reachability census, ~10 min. *Does a movement-aware enumerator find anything?*

Frame-simulation enumerator in plain Node against the existing TS twin (`tetris_spike.mjs` is the precedent; the twin
exposes `fitsAt`/`dropY`). ~5,000 boards sampled from real Dellacherie play at **levels 0, 9, 18, 29**, all three tap
settings, both rotation modes. Measure: mean and 99th-percentile `|reachable|` per piece; fraction of boards with
≥1 tuck; **the distribution of `y − dropY`** (this sets the depth cap `d`, and therefore N); splits by level and rate.

- **GO** if ≥ 20% of mid-game boards expose ≥1 tuck at 10 Hz, **and** 99th-pct depth ≤ 4 (N=160 suffices),
  **and** mean `|reachable|` ≤ 60.
- **NO-GO** if tucks appear on < 10% of boards at 10 Hz — the dial is cosmetic; ship it for humans and keep the
  40-action net. Or if 99th-pct depth > 6 — N blows past 240, wall-clock doubles, **switch to `V(afterstate)`** (§3.3).

**S1b:** on the same harness, the §5.4 empirical check — canonical TSD/TST boards, reachable lock states at g = 3, 2, 1
under SRS kicks with no lock delay.

### S2 — Dellacherie over the extended set, zero training, ~5 min. *Are the extra placements worth anything?*

Run `dellacherieAction` and `dellaSearchAction` over S1's reachable set instead of the 40 hard drops, protocols A and
B, seeds 5000+e (comparable to `tetris-baselines-final.txt`). Compare against 197.6 lines / 94,636 score and
**363.8 ± 40.3** survival; report tetris rate and tuck rate.

- **GO** if protocol-B survival improves **≥ +15% CI-separated** (≥ ~420).
- **NO-GO** if flat or worse: *a perfect evaluator that cannot exploit the extra placements is decisive evidence that
  a distilled γ=0 net will not either*, and 6 h of retraining would buy a bigger action space that pays nothing.

*Run S2 immediately after S1. It can cancel the entire retrain.*

### S4 — Cost and memory probe on the real stack, ~10 min.

Once the enumerator is in the `.pg`: 5,000 throwaway training steps at the target width, measuring observed steps/s,
peak RSS, written `*-state.ckpt` size, and the env-vs-learn split (re-run at `--hidden 32,32`; the delta is learn cost).
- **GO** if projected ≥ 15 steps/s (≤ 7.5 h for 400K) and the state checkpoint ≤ 4 GB at the intended buffer.
- **NO-GO** ⇒ cap the buffer at 100–150K, or add a don't-persist-the-buffer flag (does not exist today); if env cost
  exceeds ~40% of the step, the enumerator needs incremental features **before** the long run.

**Order: S0 → S0b → S1 → S1b → S2 → engine work → S4 → the retrain.**

---


---

## 6.R Spike results (executed 2026-08-30)

All four scripts are committed under `docs/prd/tetris-spike/` and run in plain Node against the generated
TS twin — no training, no engine change, no `.pg` edit.

### S0 — evaluator widening: **GO, decisively**

`s0_evaluator.mjs`, protocol A (uniform, no garbage, 500-piece cap), 12 eps/config, seeds 5000+.

| config | score | tetris/ep | TRT | score/piece | top-out |
|---|---|---|---|---|---|
| baseline (exact Dellacherie) | 91,482 | 0.08 | 0.2% | 183.0 | 0% |
| **well-sign split only** | **104,912** | **0.92** | 1.9% | 209.8 | 0% |
| + burn −3, ready 2, tetris 8 | **136,652** | **3.83** | 8.4% | 292.5 | 8% |

**The §0 diagnosis is confirmed.** Splitting the `−Δwells` sign alone — one term, no new features —
buys **+14.7% score and 11× the tetrises**, with zero top-outs. It is a strict Pareto improvement, which
is exactly what "the target function was fighting itself" predicts.

*Methodology note, recorded because it nearly produced a false NO-GO:* the first pass copied StackRabbit's
**absolute** weights (ready +6, covered −10, burn −12) onto the Dellacherie basis. That is a scale error —
StackRabbit prices holes at −50 where Dellacherie prices them at −4 — so the tetris terms arrived ~12×
overweight, drowned the safety terms, and every config topped out 100% of the time. **Weights must be
scaled to the basis they join.**

### S0b — CEM on the widened basis: improves the mean, but not CI-separated

`s0b_cem.mjs`, pop 20 / elite 5 / 6 iters, tuned on seeds 7000+, **evaluated on held-out seeds 9000+**, 30 eps.

| policy | score | tetris/ep | TRT | score/piece | top-out |
|---|---|---|---|---|---|
| Dellacherie (baseline) | 97,983 ± 3,928 | 0.57 | 1.1% | 196.0 | 0% |
| S0 hand-widened | **156,154 ± 10,532** | 4.93 | 10.1% | 313.9 | 3% |
| CEM-tuned | 186,111 ± 34,491 | **11.03** | **28.1%** | 443.0 | **30%** |

The hand-widened vector is **CI-separated above Dellacherie (+59%)** — that is the solid result. CEM pushes
the mean to +90% and TRT to 28%, but its CI is 3× wider and **top-out rises to 30%**: fitness was raw score
with no death term, so CEM bought variance. **Do not ship the CEM vector as-is**; re-run with a death-aware
fitness (or top-out-rate constraint) and more episodes per evaluation before trusting it. The tuning/eval
seed split was disjoint by construction, per the tet7 lesson.

### S1 — reachability census: **NO-GO on tucks**

`s1_reachability.mjs`, exact frame simulation (shift→rotate→gravity, NRS pivot, no kicks), 80 boards ×
7 pieces per config. Boards with ≥1 tuck, at DAS 10 Hz / level 18:

| board population | boards with ≥1 tuck | tucks/piece | mean \|reach\| vs \|hard\| |
|---|---|---|---|
| Dellacherie (clean) | **0%** | 0.00 | 23.1 vs 23.1 |
| garbage/10 (dirty) | **1%** | 0.02 | 22.9 vs 23.0 |
| random (messy) | 19% | 1.81 | 5.7 vs 6.9 |

**A good evaluator keeps its own surface flat, so it never creates anything to tuck under.** Tucks are
plentiful only on boards a strong policy does not produce. The PRD gate needed ≥20% at 10 Hz; the
decision-relevant populations give 0–1%.

The tap dial *does* bite where tucks exist — on messy boards, rolling yields 5.3 tucks/piece against DAS's
1.8 (3×), and at the kill screen the enumerator reproduces the known physics exactly: on a hand-built ledge,
DAS and hypertapping find **0** tucks at 1 frame/row while rolling still finds 13. So the frame model is
sound; the opportunity simply is not there.

### S2 — Dellacherie over the extended set: **NO-GO**

`s2_extended_set.mjs`, same evaluator over the 40 hard drops vs the frame-simulation reachable set,
DAS 10 Hz, 10 eps, 600-piece cap, seeds 5000+.

| | protocol B survival | protocol A survival |
|---|---|---|
| hard-drop (40) | 347.4 ± 97.5 | 600.0 (cap) |
| movement-aware | 184.0 ± 52.0 | 536.3 ± 116.9 |
| *control:* reachable set restricted to hard-drop rows | *199.9 ± 74.9* | *536.3 ± 116.9* |

Gate needed **+15% CI-separated**; measured **negative**. Tucks were chosen only 0.8–1.4 times per episode.

> **Honest caveat on the magnitude.** The control — the identical code path with tucked placements removed —
> does **not** reproduce the hard-drop baseline on protocol B (199.9 vs 347.4), even after matching the
> engine's rotation-major/first-strict-improvement tie-break. So the **−47% headline is confounded and must
> not be quoted as "tucks cost 47%"**. What *is* robust: tucks are chosen rarely, they do not help, and the
> gate fails — which agrees independently with S1. The residual control gap is unexplained; the likely
> candidate is that on late, messy protocol-B boards the DAS-budgeted reachable set is genuinely *smaller*
> than the 40 hard-drop placements (S1's random population shows exactly that: 5.7 reachable vs 6.9 hard).
> If so it is a real effect and a further argument against the feature, but it is **not measured**, and
> anyone reopening this should close that gap first.

### S3 — lateral reach and the tap budget at PINNED gravity: **GO, and it corrects S1/S2's scope**

`s3_lateral_reach.mjs`. **Owner hypothesis (2026-08-30):** *the model may prefer a flat field because it
cannot get pieces over to the side.* S0–S2 could not test this — **they all started at level 0 (48
frames/row), where input speed binds on nothing.** Gate G7 pre-registered precisely this failure mode and
was not honoured. This spike pins gravity instead.

**Part 1 — lateral reach.** Greatest flat-stack height at which the piece can still reach each wall
(spawn x=5, so column 9 is 4 taps and column 0 is 5):

| | DAS 10Hz | hyper 12Hz | hyper 15Hz | rolling 20Hz | 30Hz cap |
|---|---|---|---|---|---|
| L9 (6f/row) | 16 | 16 | 16 | 16 | 16 |
| L18 (3f/row) | 15 | 16 | 16 | 16 | 16 |
| L19 (2f/row) | **13** | 14 | 15 | **16** | 16 |
| L29 (1f/row) | **7** | 9 | 11 | **13** | 15 |

**Part 2 — strength, widened evaluator, tap-constrained action set, gravity pinned** (8 eps, 400-piece cap,
seeds 5000+):

| | DAS | hyper 12 | hyper 15 | rolling 20 | 30Hz |
|---|---|---|---|---|---|
| **L18** score | 60,178 | 64,323 | 87,398 | 84,575 | 92,440 |
| **L19** score | 37,135 | 39,068 | 60,178 | **79,910** (+115% vs DAS) | 84,575 |
| **L29** score | **0** | 15 | 780 | **37,135** | 60,178 |
| L29 pieces survived | 21 | 23 | 52 | **224** | 302 |
| L29 well-column touches/ep | **2.0** | 3.1 | 7.5 | **34.6** | 49.1 |

**At the kill screen DAS scores zero** — 21 pieces, two well-column touches in a whole episode. Rolling
scores 37,135 and survives 224. This reproduces the real rolling revolution from first principles: the kill
screen was unscoreable with DAS and hypertapping until rolling was invented in 2020, and the engine
reproduces that without being told.

*Caveat:* 8 episodes, CIs ±25–31k, so orderings within a few thousand points at L18/L19 are **not**
separated. The L29 result (0 vs 37,135) is not a CI question.

**The synthesis — there are TWO independent causes of flatness, and S0 found only one:**

1. **The `−Δwells` sign trap** — wrong at *every* level, fixed by S0, worth +59%.
2. **Genuine unreachability at high gravity** — *correct* behaviour for DAS (a stack above
   `max5TapHeight` really cannot feed the well), *wrong* for rolling. This is exactly what StackRabbit's
   `max5TapHeight < 4 ⇒ LINEOUT` mode switch encodes.

The current model cannot tell these regimes apart, because it has **no input model at all** — so it flattens
uniformly, which is right for DAS at L29 and leaves most of the score on the table for rolling.

**This corrects §6.R's earlier conclusion.** The technique dial is **not** a cosmetic/demo feature; it is a
first-order strength factor above level 19. What S1/S2 correctly rule out is the **tuck** half of the
movement-aware action space. What they did *not* test — and S3 now shows is decisive — is **lateral reach**.
Those are different axes, and only the second one pays.

**Revised consequences:**
- **M57.1 (evaluator widening) still leads**, and gains two terms it did not have: `inaccessibleLeft` /
  `inaccessibleRight` derived from `max4/max5TapHeight`, plus the LINEOUT-style mode switch. Without them the
  agent cannot know which regime it is in.
- **M57.3 is re-scoped, not cancelled.** Drop tucks (S1/S2). Keep the **tap-budgeted legality mask** — the
  frame simulator earns its place by deciding *which columns are reachable at this level and rate*, which is
  a mask over the existing 40 actions, **not** an action-space expansion. N stays 40; no retrain forced by
  action-count change.
- **G7 (a high-gravity protocol) is now mandatory, not optional.** Every gate measured at level 0 is blind to
  the effect that dominates real play. Protocol A's level-0 start is why the shipped net looks acceptable.
- **The three radios (M57.6) are a genuine strength control**, and the visitor can watch DAS fail at the kill
  screen while rolling keeps scoring.

### S1b — not run

The SRS-without-lock-delay reachability check (§5.4) was not executed. With S1/S2 removing the case for the
movement-aware action space, it is only decision-relevant if SRS mode is pursued for the browser demo.

### What the spikes change

1. **M57.1 (evaluator widening) is promoted to the whole arc.** It is a formula change, it is measured at
   **+59% score CI-separated** with **9× the tetrises**, and it needs no action-space work, no frame model
   and no retrain to demonstrate. The dense target, `dellaScoreFor` and the observation planes all read the
   same basis, so one change lifts the net *and* both search tiers.
2. **M57.3 (movement-aware enumeration) loses its strength justification.** Per §6's own NO-GO clause: ship
   the dial for humans, keep the 40-action net. The frame simulator is written and validated — it belongs in
   the browser pilot (so watch-mode replays a real input sequence at the selected rate) rather than in the
   action space.
3. **M57.5 (the retrain) is not yet justified at N=160.** A retrain on the *widened evaluator* at the
   existing N=40 is the cheap, high-value run — and it is the one the measurements support.
4. **The tap dial (M57.6, the three radios) survives as an authenticity/demo feature**, which is what G6
   pre-registered as a shippable honest result. It makes the AI more human, not stronger.

---

## 6.S M57.1 BUILT — the widened evaluator, measured in the engine (2026-08-30)

Implemented in `tetris_solver.pg` (`dellaScoreFor`), so it lifts **both** scripted tiers and the search tier
at once. **The observation planes, the net and the checkpoint are deliberately untouched** — this milestone
changes what the evaluator *wants*, not what the network *sees*, so `tetris.dqn.ckpt` stays valid and no
retrain is forced yet.

### What was added
- `wellSumExceptWell()` — wells penalised everywhere **except** the well column. *The* sign fix.
- `tetrisReady()`, `coveredWell()`, `colHeight()` — the board shape the Dellacherie basis cannot express.
- `maxTapHeight(taps)` + `tapFramesPerShift` + `setTapRate()` — the tap budget from S3, and the
  `inaccessibleLeft/Right` penalties derived from it.
- **Two mode switches**, both load-bearing and both measured:
  - **LINEOUT** (`maxTapHeight(5) < 4`) — the left wall is unreachable at this level and tap rate, so
    tetris-building is futile; stop paying for the well.
  - **DIG** (`holes > 0`) — on a holed board *burning is how you survive*. Without this switch the widened
    evaluator scored +30% on protocol A but **lost 52% of protocol-B survival**: it refused to clear the
    singles that dig a garbage board out. This was measured, not anticipated.

### Weights
CEM-tuned (`s5_tune_widened.mjs`) under a **constrained** fitness —
`(A_score/100k + 0.6·A_tetrises/4) × min(1, B_survival/364)`. The multiplicative term means survival below
the M54 baseline scales the whole objective down and **cannot be bought back with score**, which is the
lesson from S0b (raw-score CEM bought 30% top-outs). Tuned on seeds 7000+, evaluated on held-out 9000+.

### Results — 30 episodes, seeds 5000+, through the TS twin (the browser's exact code)

| | protocol A score | lines | tetrises/ep | TRT | protocol B survival |
|---|---|---|---|---|---|
| M54 dellacherie | 94,636 | 197.6 | 0.26 | 0.5% | 363.8 ± 40.3 |
| **M57.1 dellacherie** | **186,179 ± 18,961 (+97%)** | 190.3 | **8.50 (33×)** | **17.9%** | **430.2 ± 66.6 (+18%)** |
| M54 della-search | 93,678 | — | — | — | 1480 (right-censored) |
| **M57.1 della-search** | **218,560 ± 56,045 (+133%)** | 141.7 | **15.60** | **44.0%** | 1413 ± 88 |

**Gate G1 (no regression) PASSES on both protocols** — protocol B *improved* for the scripted tier and is
at baseline for the search tier (whose 1480 was right-censored at the 1500 cap, so the two overlap).
**Gate G3 (tetris rate)**: TRT 44% for della-search against a ≥50% target — close, and 88× the shipped net's
0.5%. Competitive human maxout pace is 60%.

The tap dial shows no effect on protocol A (186,179 / 185,922 / 184,525 for DAS / hyper / rolling), which is
**correct and expected**: protocol A starts at level 0 and rarely reaches the gravity where reachability
binds. S3 is where the dial pays, and G7's high-gravity protocol is how it will be gated.

### Deliberate re-pins
- **Parity checksum 472451993 → 765594964.** The protocol drives `dellacherieAction`/`dellaSearchAction`, so
  a change of evaluator moves it by construction. The *rules* are untouched — piece stream, lock path,
  clears, garbage and scoring all unchanged. TS twin re-verified: `node tools/tetris_parity.mjs` agrees.
- **`SpikeBar_DellacherieClearsNearMaximalLines` → `SpikeBar_DellacherieBuildsForTetrisesTradingLinesForScore`.**
  The old test asserted 197.4 lines and **zero** top-outs — the signature of exactly the flatten-and-burn
  behaviour this milestone removes. The new test pins the new contract (score ≥ 110k, tetrises ≥ 2.0,
  lines ≥ 165) and keeps a stack-and-camp watchdog (≤ 6 top-outs in 20 episodes).

**529/529 fast bucket green; CrazyFruits parity and the 11 DAS frame checks unaffected.**

### Still open after M57.1
- **The net has not been retrained**, so the *shipped browser net* still plays the old way. The evaluator
  lives in the scripted + search tiers only. M57.5 (retrain against the widened dense target) is where the
  net inherits this — and it now has a much better teacher to distil from.
- The dense target in `TetrisDqnCampaign.cs:74` is still the OLD narrow basis, and the observation planes
  still carry the old six features. Widening those is what forces the retrain.
- G7's high-gravity protocol is not yet implemented, so the tap-dial terms are unexercised by any gate.

---

## 6.T M57.5 + G7/G6 — the net's basis, and the dial measured on the real engine (2026-08-30)

### The target-scale bug (recorded because it cost a training run)

M57.5 widened the observation to 15 per-action planes and switched them from **deltas** to **absolute**
afterstate values, so the dense target could reconstruct the evaluator exactly instead of up to a per-state
constant. That was right for exactness and **wrong for scale**: measured over 869 legal actions on real
boards, the widened evaluator's raw values sit at **mean −92.1, sd 28.7**, so a bare `/10` produced targets
centred at **−9.2 with sd 2.9** — against M54's roughly zero-centred, sd≈1.

The first run degraded accordingly, and the signature was misleading:

| steps | score | lines | loss |
|---|---|---|---|
| 15K | 120 | 2.9 | 26.8 |
| 40K | 89 | 2.2 | 19.7 |
| 60K | **29** | **0.7** | 14.1 |

Falling loss with falling eval is the PRD's *distribution-narrowing* signature — but here it was a
**target-scale failure**, and the giveaway was the absolute loss level (14–29 against tet6's ~1.3). The old
delta-based planes existed precisely to keep the target centred; switching to absolute values removed that
without replacing it.

**Fix: centre the target per state** on the mean over legal actions, then scale. This is free — a dueling
V head absorbs any per-state constant by construction, and what the advantage head must learn is the
*ranking*, which centring leaves untouched. It is also more robust than deltas, since it standardises the
scale regardless of how the basis is later extended. Immediate effect at 10K steps:

| | score | lines | pieces | loss |
|---|---|---|---|---|
| uncentred (broken) | 50 | 1.2 | 36.9 | 29.3 |
| **centred (fixed)** | **222** | **5.1** | **50.4** | **1.59** |

The anti-drift test was updated to pin the *centred* invariant (`target == (engineValue − meanLegal)/10`)
rather than the absolute one.

### M57.5 training journal (2026-08-30)

| run | change | outcome |
|---|---|---|
| `tet8train` | first attempt, uncentred target | **abandoned** — score 120→89→29 over 60K while loss fell 26.8→14.1. Not distribution narrowing: the absolute loss level (14–29 vs tet6's ~1.3) identified a target-SCALE failure. |
| `tet9train` | centred target | healthy — loss 1.59 at 10K, and **0.20 tetrises/ep by 35K**, which the shipped net only reached after 400K. Peaked **14,970 at 60K**, then decayed 5,601 (85K) → 3,423 (115K) with loss falling 1.29→0.95. **This time it IS distribution narrowing** — same shape tet6 showed at 70K. |
| `tet10train` | + `--eps-end 0.12 --buffer 150000` | the PRD's documented anti-narrowing remedy. Running. |

**The diagnostic lesson worth keeping:** *falling eval + falling loss* is ambiguous on its own. Distribution
narrowing and a target-scale failure produce the same shape. **The absolute loss level separates them** — a
healthy Tetris run here sits near 1.0–1.5, so 14–29 means the targets are mis-scaled, not the replay
distribution. Check the level before reaching for the narrowing remedy.

Buffer note: at obs 814 a 300K buffer would write a ~3.9 GB state checkpoint every 10 minutes, so it is
capped at 150K (~2.0 GB) per the feasibility estimate.

### G7 — high-gravity protocol, and G6 — the technique dial, on the shipped engine

`s6_g7_protocol.mjs`, 16 eps, 400-piece cap, seeds 5000+, NES start levels via the new `setStartLevel`.

**dellacherie (widened):**

| start | DAS 10Hz | hyper 12Hz | rolling 20Hz |
|---|---|---|---|
| L0 | 116,171 | 116,388 | 119,850 |
| L18 | 243,565 | 266,151 | 273,028 |
| **L19** | **168,404** (TRT 4.2%) | 249,808 (14.3%) | **288,556** (15.5%) |
| L29 | 283,350 | 293,925 | 250,800 |

**della-search(8,5):**

| start | DAS 10Hz | hyper 12Hz | rolling 20Hz |
|---|---|---|---|
| L0 | 156,594 (TRT 40.5%) | 166,826 (43.1%) | 167,530 (42.4%) |
| L18 | 416,266 (40.3%) | 382,789 (43.5%) | **437,504 (49.0%)** |
| **L19** | **148,128 (TRT 0.6%, 0.25 tetrises)** | 428,169 (41.0%) | **451,675 (TRT 52.2%, 16.19 tetrises)** |
| L29 | 243,525 (2.1%) | 243,525 (2.1%) | 219,975 (0.6%) |

**G6 paired per-seed deltas vs DAS** (exact pairing — the piece stream is seed-determined and
setting-independent):

| | hyper 12Hz | rolling 20Hz |
|---|---|---|
| L0 | +216 ± 379 — **ns** | +3,679 ± 8,172 — **ns** |
| L18 | +22,586 ± 29,189 — ns | **+29,463 ± 27,474 — SIGNIFICANT** |
| **L19** | **+81,404 ± 52,386 — SIGNIFICANT** | **+120,153 ± 44,089 — SIGNIFICANT** |
| L29 | +10,575 ± 13,149 — ns | −32,550 ± 41,548 — ns |

**G6 PASSES**, and it passes in exactly the pattern the physics predicts: **no effect at L0** (48 frames/row
constrains nothing — a good negative control), significant at L18/L19, and the headline is L19, where the
search tier under DAS collapses to **TRT 0.6%** while rolling sustains **TRT 52.2%** — clearing the G3
target of 50%. The AI cannot feed the well under DAS at that gravity, which is precisely the owner's
hypothesis, now measured on the shipped engine rather than a spike harness.

*L29 is noisy and non-significant here*: the level multiplier is large enough that even a low-TRT policy
scores well, and 16 episodes at a 400-piece cap is too coarse to separate the settings. Report L29 with more
episodes before drawing any conclusion from it.

### Engine addition: NES start-level selection

`setStartLevel(lvl)` + `levelForLines()` implement the ROM A-TYPE curve — first level-up after
`min(start*10+10, max(100, start*10−50))` lines, then every 10 — so an 18-start reaches 19 at 130 lines and
the kill screen at 230. Default 0, and `reset` clears it, so every pre-M57 protocol is byte-identical.
Pinned by `StartLevel_FollowsTheNesTransitionCurveAndDrivesGravity`.

## 7. Milestones

- **M57.0 — Spikes S0/S0b/S1/S1b/S2.** Gates as above. **This milestone can end the arc** with a measured
  "the evaluator was the whole story" or "the dial is cosmetic" — both shippable results.
- **M57.1 — Evaluator widening (LOCK A).** Widened φ in the `.pg`, mirrored into `dellaScoreFor`, the observation
  planes and `DenseTargetsFromObservation`; the `−Δwells` split; realized-reward term dropped; mode-switched weights.
  **Gates:** the three copies of φ agree by test (this is the M54 hazard — `TetrisDqnCampaign.cs:74` is a *hand-written
  inverse* of `.pg:689–694`, with a magic `PlaneBase = 214`); S0's tetris rate reproduced in C# within CI.
- **M57.2 — Frame model into the `.pg` (§4.3).** `NesInput` ported, serialization widened, 11 DAS checks mirrored
  into `TetrisEngineTests`. **Gates:** the 11 frame-exact checks green in **CI**; parity pin re-established.
- **M57.3 — Movement-aware enumeration (LOCK B) + tap dial (§4.2).** Frame-simulation enumerator, tuck spots, input
  costs, `max4/max5TapHeight` → scare/col9/mode. Level enters the observation. **Gates:** `micro == macro` — replaying
  a placement's input sequence through `microShift`/`microRotate`/`microDropStep` reproduces the enumerated afterstate
  **for every reachable placement** on hand-drawn boards (**the highest-risk item in the arc**); new parity pin;
  S1's census reproduced in C#.
- **M57.4 — SRS second mode (§5).** Kick tables, CCW, T-spin detection, mode setter, duplicate-state canonicalization.
  **Gates:** NRS pin **472451993 unchanged** (proves additivity); new SRS pin; kick tables pinned by test against the
  transcribed tables; §5.3 regression test.
- **M57.5 — Retrain (LOCK C/D).** From scratch, N per S1, 400K steps, randomized tap budget in rollouts, buffer
  capped per S4. **Gates:** §8, on held-out seeds 9000+e.
- **M57.6 — Web (D4).** Three radios in both modes, pilot replays the engine's input sequence, status line, stale-guard
  hardening. **Gates:** live Playwright desktop + phone; 0 console errors; 0 `/api/tetris*`; each radio measurably
  changes the AI's legal set; browser ≤ 50 ms/move (**at risk** — see G6).
- **M57.7 — Ship.** `ARCHITECTURE.md`, PLAN sync, PRD results, ckpt to LFS, §9 corrections. One PR.

---

## 8. Gates

Held-out seeds **9000+e** via `tools/tetris_head2head.mjs` for every ship decision (the tet7 lesson). Baselines:
random 4.4 / 22.5 · Dellacherie 94,636 (197.6 lines) / 363.8 ± 40.3 · della-search 93,678 / 1480 (censored) ·
**shipped tet6 held-out: A 85,199 ± 3,519, B 176.2**.

| | Gate |
|---|---|
| **G1 no-regression** | A ≥ 80,000 **and** B ≥ 165, neither CI-separated below tet6. *Deliberately non-regression, not improvement — an improvement gate on survival re-incentivizes camping.* |
| **G2 survival** | Protocol B ≥ **200** pieces, CI-separated above 176.2 (gap-share 52% vs the shipped net's 45%) |
| **G3 tetris rate** | **TRT = 4·tetrises/lines ≥ 50%** (current ≈ 0.05%), **and** ≥ 2.0 tetrises/ep on A, CI-separated. Threshold renegotiable **only** on an S0 NO-GO, before training. *The one metric that cannot be gamed by surviving longer.* |
| **G3b efficiency** | **Score per piece** CI-separated above tet6. Directly counters "it just lived longer" |
| **G4 tuck attribution** | The **same trained net**, evaluated twice — full reachable set vs non-hard-drop placements masked. The delta **is** the measured value of tucks. Pass = ≥ 5% on B, CI-separated, **plus** tuck rate ≥ 3%. *Without the ablation a nonzero tuck rate proves nothing.* **No public A/B of this exists — this number is a contribution.** |
| **G5 spin attribution** | Identical, spin-only placements masked. T-spin count/game **reported, not gated** (§1.1). Expect a small delta concentrated on dirty/garbage boards |
| **G6 technique sweep** | All three settings reported. **Paired per-seed design** — the piece stream is seed-determined and setting-independent, so pairing is exact; bootstrap CI on the mean delta, not two independent means (A's independent CI is ±3,519, which would hide a real 3% effect). Pass = Δ(rolling − DAS) on B > 0, paired CI excluding 0. **The slowest setting must not regress vs tet6.** *A null result is shippable and honest: the dial is then a human-authenticity feature, not a strength lever.* |
| **G7 high-gravity** | A **start-level-18/19 protocol** is added. Without it the tap budget is unconstrained (§2.2 L18 row) and G4–G6 measure noise |
| **G8 structure** | Tetris-ready fraction; covered-well events/game; holes per 100 pieces; burn rate. Reported |
| **G9 stack-and-camp watchdog** | Top-out cause histogram + mean board height at top-out. A camping agent shows high mean height, late top-outs, low TRT |
| **G10 browser budget** | ≤ 50 ms/move. **At real risk:** `dellaSearchAction(8,5)` costs ≈ 289·N placement evaluations — 11.6K at N=40 (measured ~30 ms), **46K at N=160 ⇒ ~120 ms**, and net-search rebuilds a 1174-float observation per node. Re-tune beams (8,5 → 4,3 gives 88·N ≈ 14K) and re-measure |
| **G11 engine** | NRS pin unchanged; new SRS + reachability pins; `micro == macro` over every reachable placement; 11 DAS checks in CI |

**Reporting discipline.** Every result carries its format string — board, randomizer, start level, level cap, piece
cap, **tap rate, reaction delay**, garbage period, seed set — as BetaTetris does. An NES-score number without the tap
rate attached is meaningless.

---

## 9. Corrections to land in the same PR

Found during the investigation; all are current-state defects, not new work.

1. **`TETRIS_PRD.md` §1 and §7 claim an `enumeratePlacements()` seam that does not exist.** Remove the claim.
2. **`TetrisLab.cs` still defaults to `--gamma 0.995 --nstep 3 --hidden 128,128` with PBRS on** — re-running "the
   default" reproduces the *failed* tet1, not the shipped tet6. And `TetrisDqnCampaign`/`TetrisDqnOptions` xmldoc
   still assert that "the γ=0 dense recipe structurally does not transfer", the opposite of what shipped.
3. **The training CLI args are recorded nowhere in the repo.** `data/tet*train-console.txt` logs start times and
   keep-best events but not the flags. Persist the invocation into the run directory.
4. **`tetris-das.ts` hypertap ceiling is wrong.** The comment says "≤1 shift/frame ≈ 30 Hz"; one shift per *frame* is
   60 Hz. The NES samples the pad once per frame and counts only *newly-pressed* bits, so a button must be observed
   released for ≥1 frame — the true ceiling is **one shift per 2 frames = 30.05 Hz**. On a 240 Hz display the current
   code permits 60 shifts/s, double the hardware maximum. Fix: require an observed release before a latch counts as fresh.
5. **No ARE / line-clear delay.** NES has 10–18 frames of entry delay (by lock height) and 17–20 of line-clear delay,
   during which DAS neither charges nor resets — so a held direction can be **redirected for free**, a real technique.
   Adding ARE also gives the AI a natural per-piece thinking budget and fixes the felt pacing at high levels.
6. **Rotation ordering must be shift → rotate → drop within a frame**, with both permitted on the same frame.
   `NesInput.tick` handles shift and drop; rotation is applied elsewhere in the component. If a rotation issued on
   frame *n* is not applied after that frame's shift and before its gravity step, the **1-frame killscreen tuck
   window behaves differently from the ROM** — and that window is exactly what §4 is measuring.
7. **Stale-checkpoint guards check `inputSize` only.** Add `net.actions !== ActionCount` to both
   (`TetrisBoard.cs:150`, `tetris-director.ts:35`).
8. **`RewardTetrisBonus` is declared at `tetris_solver.pg:63` and never read there**, while its comment claims
   `netSearchAction`'s rollout must share its units. Wire it or delete it.
9. **The browser pilot silently substitutes placements** (`tetris-game.ts:159`, `stuck >= 2`) with nothing testing or
   measuring the divergence. M57.3 makes it replay; add the metric regardless.
10. **CI runs no node harness.** Mirror the DAS checks into `TetrisEngineTests`, and consider a CI job for the parity twins.

---

## 10. Risks

1. **S0 confirms the evaluator diagnosis and the rest of the arc becomes optional.** *Not a risk — the best outcome.*
   Sequence the spikes so this is discovered in 15 minutes rather than after 6 hours of training.
2. **`micro == macro` fails for some reachable placement** (M57.3). The highest-risk item: two step APIs, now over a
   frame-accurate path rather than a vertical drop, under two rotation systems. *Mitigated:* exhaustive test over
   every reachable placement on hand-drawn boards, not sampled.
3. **The `rotCount` trap silently shifts every baseline** (§5.3). *Mitigated:* the NRS pin is the regression test —
   it must be **unchanged** after M57.4.
4. **Browser exceeds 50 ms/move at N=160** (G10). *Mitigated:* re-tuned beams; the M53 worker protocol exists if
   search grows past the frame budget.
5. ~~**Polyglot 0.8.1 cannot express the enumerator.**~~ **RETIRED 2026-08-30 — measured false.** All five M57
   constructs (kick tables with negative literals, the ported `NesInput` class, the frame sweep, record-based tuck
   spots, and a real BFS queue) `check` clean on the pinned 0.8.1 and the emitted TS passes `tsc --strict`. See
   [polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md](polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md).
6. **Distribution narrowing repeats** (tet6's measured falling-loss/falling-eval). *Mitigated:* the randomized tap
   budget in rollouts is itself a diversifier; `--eps-end` / `--buffer` knobs already exist. Note ≈85K was the
   recipe's held-out ceiling — expect the *evaluator*, not the sampling, to lift it.
7. **G3's 50% TRT is unreachable under the 500-piece cap.** *Mitigated:* S0 measures this before training; the gate is
   renegotiable **only** on an S0 NO-GO, and only before the retrain.

---

## 11. Out of scope

Guideline scoring / SRS-as-a-second-game (§1.1 option A — owner call, not deferred-for-size); hold piece; multi-piece
preview > 1; versus/two-board garbage battles; conv Q-net + resident GPU trainer (**GPU is measurably not a lever
here** — §2.4); a misdrop model; the expectimax tablebase; `V(afterstate)` variable-length scoring (§3.3 — the
designed successor, and the S1 NO-GO fallback); PAL rules.

**The Polyglot version bump** (0.8.1 → HEAD) is explicitly out of scope and is **not** a dependency of this
milestone — M57 runs on 0.8.1 as pinned. It is worth doing on its own small arc: cost is one
`sed -E 's/^([[:space:]]*)init\(/\1constructor(/'` over the **22 `init(` sites in the 7 solvers** (after which all
seven `check` clean at HEAD), plus re-validating every game's parity pin — HEAD's `List.clear()` lowering changed
from rebinding (`this.bag = []`) to in-place mutation (`this.bag.length = 0`), which is a *fix* that makes C# and TS
agree under aliasing, but is observable. Details in
[polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md](polyglot-pilot/POLYGLOT_M57_FEASIBILITY.md) §2.4–§2.5.

---

## 12. References

**Primary (NES).** [meatfighter, *Applying AI to Nintendo Tetris*](https://meatfighter.com/nintendotetrisai/) — the
disassembly: gravity table ($898E), spawn (5,0), DAS 16/6, rotation validity ($948B), lock-delay = drop-delay, slides
and spins, per-frame input ordering ·
[tetris.wiki *Tetris (NES)*](https://tetris.wiki/Tetris_(NES)) — 60.0988 fps, wall charge, **ARE 10–18**, line-clear
delay 17–20, level transition, PAL ·
[*Lock delay*](https://tetris.wiki/Lock_delay) · [*Nintendo Rotation System*](https://tetris.wiki/Nintendo_Rotation_System) ·
[*Super Rotation System*](https://tetris.wiki/Super_Rotation_System) — both 5-candidate kick tables ·
[*T-Spin*](https://tetris.wiki/T-Spin) — 3-corner rule, mini vs proper, 1×2-kick promotion ·
[*Scoring*](https://tetris.wiki/Scoring) · [*Maxout*](https://tetris.wiki/Maxout) · [CTWC rules](https://thectwc.com/rules/).

**Movement-aware agents.** [StackRabbit](https://github.com/GregoryCannon/StackRabbit) —
`move_search.cpp`, `piece_ranges.cpp`, `eval.cpp`, `params.hpp`, `config.hpp`, `precompute.ts`, `stackrabbit.lua`;
[TetrisTrainer](https://gregorycannon.github.io/TetrisTrainer/); 102,252,920 points / level 237 ·
[BetaTetris](https://github.com/BetaTetris/betatetris-tablebase) — `TapTable`, tap-speed/reaction parameterisation,
2,426,105 average score; [older](https://github.com/adrien1018/beta-tetris);
[playground](https://fractal161.github.io/btpg/) ·
[Cold Clear](https://github.com/MinusKelvin/cold-clear) / [2](https://github.com/MinusKelvin/cold-clear-2) ·
[smartspot2/tetris-ai](https://github.com/smartspot2/tetris-ai) (SRS BFS pathfinder).

**Learning.** [Algorta & Şimşek, *The Game of Tetris in Machine Learning*, arXiv:1905.01652](https://ar5iv.labs.arxiv.org/html/1905.01652) ·
[Thiery & Scherrer, BCTS](https://inria.hal.science/inria-00418930/document) ·
[codemyroad, *Tetris AI — The (Near) Perfect Player*](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/) ·
[nuno-faria/tetris-ai](https://github.com/nuno-faria/tetris-ai) ·
[Rex L, *RL on Tetris*](https://rex-l.medium.com/reinforcement-learning-on-tetris-707f75716c37) /
[part 2](https://rex-l.medium.com/reinforcement-learning-on-tetris-2-f12f74f70788) (single-source practitioner journal).

**Community.** [Tetris Interest, *The Rolling Revolution*](https://tetrisinterest.com/the-rolling-revolution-the-never-ending-story-of-nes-tetris/) ·
[Kotaku on rolling](https://kotaku.com/nes-tetris-players-call-it-rolling-and-theyre-setting-1846767518) ·
[ejona86/taus](https://github.com/ejona86/taus) (TRT ROM hack).

**Repo.** [TETRIS_PRD.md](TETRIS_PRD.md) (M54/M55) · [SNAKE_SEARCH_PRD.md](SNAKE_SEARCH_PRD.md) (strength = search) ·
[CRAZY_FRUITS_RANKING_PRD.md](CRAZY_FRUITS_RANKING_PRD.md) (dense targets, per-action planes) ·
`docs/ARCHITECTURE.md` · `docs/ADDING_A_GAME.md`.

**Do not cite:** a "910,000 lines" figure that search summaries attribute to *Applying DQN to Tetris using high-level
state spaces* — the paper could not be retrieved and the number appears conflated with BCTS.
