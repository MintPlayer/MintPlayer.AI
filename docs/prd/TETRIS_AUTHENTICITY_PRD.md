# Tetris NES authenticity + the tetris rate, and a Block Dude layout fix

**Status:** 🔵 PLANNED — not started. Written 2026-09-16 from a 4-agent investigation (gravity/rotation,
tetris-rate, technique research, Block Dude layout). Findings in §2; **three of the five asks are already
designed in [TETRIS_TECHNIQUES_PRD.md](TETRIS_TECHNIQUES_PRD.md) §4–§5 and were never built** (M57.2, M57.3,
M57.4, M57.6). This PRD supersedes those milestones with a re-scoped, re-measured plan.
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M62 · branch `m62-tetris-authenticity` · builds on
[TETRIS_PRD.md](TETRIS_PRD.md) (M54 ship, M55 NES input) and [TETRIS_TECHNIQUES_PRD.md](TETRIS_TECHNIQUES_PRD.md)
(M57 evaluator widening + tet15 net).

**One PR for the arc** — including the Block Dude item, which is unrelated in subject but ships in the same
unit of work per the repo's one-PR rule.

---

## 0. The owner's five asks, verbatim in substance

1. **Drop speed above level 29.** Watching CTWC, the speed appears to change at "level 30, 130 and 230", and
   not to increase after 230.
2. **The AI still does not try to make tetrises.**
3. **Rotation is clockwise-only.** NES allows CCW as well.
4. **The AI does not simulate hypertapping or rolling.** Research the tap frequencies for both, and add a
   second button group on the Tetris page — normal / hypertapping / rolling — active when "Watch AI" is on.
5. **(Block Dude, not Tetris.)** Move the Previous / Level / Next buttons above the playfield so they stop
   shifting as the field size changes.

The investigation changes the shape of two of them, and confirms three as straightforward work.

---

## 1. What the investigation changes

### 1.1 Ask (1) is a mix-up of line counts with speed changes — and our table is already correct

`gravityFrames` (`tetris_solver.pg:461-469`, mirrored `TetrisBoard.cs:129-130`) was verified value-by-value
against the ROM table at `$898E`:

```
L0-8: 48-5·lvl (48,43,38,33,28,23,18,13,8) · L9: 6 · L10-12: 5 · L13-15: 4 · L16-18: 3 · L19-28: 2 · L29+: 1
```

That is **exactly authentic NES NTSC**. `levelForLines` (`.pg:317-326`) likewise implements the real ROM rule
`first level-up at min(start·10+10, max(100, start·10−50))` lines, then every 10.

**130 and 230 are line thresholds, not levels.** On the standard 18-start, that formula puts **level 19 at 130
lines** and **level 29 at 230 lines**. And the owner's final observation is exactly right for the wrong reason:
**level 29 is the last speed change in the game** — NTSC gravity is a flat 1 frame/row from 29 to 255. Nothing
increases after it. The "30" is most likely the early level-ups (20/30 lines) or the visible behaviour change on
crossing into killscreen territory.

So there is **no bug to fix**, and adding speed steps at 30/130/230 would make the engine *wrong*.

What genuinely exists past 29 is a **ROM hack used by CTWC/CTM Masters**, not the original game: the
**level-39 "super killscreen" (2xks)** patch doubles gravity to **2 rows/frame** at level 39, to bound match
length. Lower divisions use "level 39 halt" instead. That is the authentic way to deliver what the ask is
reaching for — as an explicit opt-in variant, never by editing the base table.

Three other post-29 facts, recorded so they are not mistaken for gravity later: the **level 138+ palette
glitch** (`level − 10` stored signed; 138−10 = 128 → −128 → near-black pieces), the **level display** breaking
well before that, and the true **game crash at ~155–157** (Blue Scuti, Dec 2023).

**Bonus find:** `setStartLevel` exists in the engine (`.pg:329` → `TetrisBoard.cs:127`) but **the frontend never
calls it** — every browser game starts at level 0. A CTWC-style start-level picker is a pure frontend change on
top of shipped engine API, and it is the highest-value authenticity win in this ask.

### 1.2 Ask (2): the teacher already makes tetrises — the net loses them in distillation

**The diagnosis recorded in `TETRIS_TECHNIQUES_PRD.md` §0 is stale and must not be re-applied.** All three of
its premises are refuted by current master (`ff36180`):

| Prior claim | Status today |
|---|---|
| γ = 0, no bootstrap path | **REFUTED** — γ = 0.995 (`TetrisLab.cs:30`), n-step = 3 (`TetrisLab.cs:45`) |
| dense target has `−20·Δwells`, penalizing the tetris well | **REFUTED** — well column *excluded*, weight now **−0.847** (`.pg:79`, `:700 wellSumExceptWell`); `+7.047` tetris term and `+3.402` tetris-ready-row term added |
| a tetris is worth less than 4 singles | **REFUTED** — see below |

The arithmetic now runs the right way:

| | 4 singles | one tetris |
|---|---|---|
| NES score | 160·(lvl+1) | **1200·(lvl+1)** — 7.5× |
| Realized reward (`TetrisEnv.cs:166`) | 4 | **12** — 3× |
| Dense target | `4 × (−3.700)` burn = −14.80 | `+7.047`, plus up to `+13.6` paid *during the build* |

Measured effect of the M57.1 fix (PRD §6.S, 30 eps): `dellacherie` 94,636 → **186,179** (+97%), tetrises/ep
0.26 → **8.50** (33×), tetris-rate 0.5% → **17.9%**; `della-search(8,5)` reaches **15.60/ep, 44%**.

**So why does the owner still see no tetrises?** Because the browser's default tier is the plain net, and the
net is the lossy step:

> Same target function. Exact argmax → **17.9–44%** tetris-rate. The trained net's approximate argmax →
> **1.4%** (0.67 tetrises/ep, held-out). Capacity and fit are ruled out — the target is *exactly linear* in the
> observation, and measured R² is 0.868–0.893.
>
> **A tetris needs ~10 consecutive correct argmaxes to hold column 9 open. One error burns the well.**
> Score and survival tolerate argmax noise; a tetris does not.

That reframes ask (2) completely. It is **not** a horizon problem, **not** a reward-shaping problem, and **not**
a capacity problem. It is argmax fidelity — or, far cheaper, putting the exact evaluator back in the loop at
inference by changing which tier the browser defaults to.

### 1.3 Ask (3) is cheap, and checkpoint-safe

`microRotate()` (`.pg:566-577`) is clockwise-only: `nr = (activeRot + 1) % rotCount[current]`. The comment at
`.pg:564` is correct — that cycle *is* the NES A-button order. Authoritative mapping: **A = CW, B = CCW**.

`rotCount` is per-piece (`.pg:205-241`): I/S/Z = 2, O = 1, J/L/T = 4. For a 2-state piece CW and CCW are
identical and for O it is a no-op, so **only J, L and T are actually affected**.

The critical property: **the RL action space is an afterstate space**, not per-frame inputs —
`ActionCount = 40` (4 rotations × 10 columns, hard-masked, `TetrisBoard.cs:21`). The agent picks an *absolute
target rotation index*; it never chooses a direction. Therefore adding CCW:

- does **not** change `ActionCount` (40) or `ObservationSize` (854),
- invalidates **no checkpoints**, and clears the `net.actions != ActionCount` guard at `TetrisBoard.cs:170`,
- keeps every parity test green **provided the CW path stays byte-identical** (checksum `765594964`).

Rotation direction is purely a human-input / micro-path concern.

### 1.4 Ask (4): the mechanism already exists, but on a fabricated cadence — and there is a design fork

**Tetris does not use the repo's server-authoritative WebSocket watch-AI convention.** It is fully client-side
("Pattern C", `tetris-director.ts:1-6`); `grep -rn tetris --include=*.cs src/RLDemo.Web/` returns zero hits.
There is no server path to insert into.

The AI does **not** teleport. It picks a macro placement, then a "pilot" plays it with simulated presses —
`PILOT_INPUT_MS = 90` (`tetris-game.ts:20`) ≈ **11.1 Hz ≈ 5.4 frames/input**. That constant is the entire
current input model, and it is fiction: it sits almost exactly at hypertapping speed, so **today's "normal" AI
is already hypertapping without saying so.**

A frame-exact DAS state machine already exists — `NesInput` (`tetris-das.ts`, `DAS_FULL = 16`,
`DAS_RESET = 10` ⇒ 6-frame repeat, wall charge, spawn carry-over) — but it is wired to the **human path only**
(`tetris-game.ts:104-111`, gated on `human === true`).

**Researched constants** (NTSC 60.0988 Hz, 1 frame ≈ 16.639 ms):

| Technique | Model | Frames/input | Effective shifts/s | vs DAS |
|---|---|---|---|---|
| **DAS** (hold) | 16-frame charge, then 6-frame repeat | 16 → 6 | **10.02** | 1.0× |
| **Hypertapping** | no charge, every shift immediate | ~5 (12 Hz) | **~12** (10–15) | ~1.2× |
| **Rolling** (Cheez, 2020) | no charge | ~3 (20 Hz) — 2 at 30 Hz | **~20–30** | **~2–3×** |
| *(ceiling, not a radio)* | 1 shift/frame, NMI-limited | 1 | 60 | — |

Sources: [meatfighter NES Tetris AI](https://meatfighter.com/nintendotetrisai/) ·
[tetris.wiki](https://tetris.wiki/Tetris_(NES,_Nintendo)) · [Hard Drop — DAS](https://harddrop.com/wiki/DAS) ·
[Kotaku — rolling](https://kotaku.com/nes-tetris-players-call-it-rolling-and-theyre-setting-1846767518) ·
[Engadget](https://www.engadget.com/teens-tetris-and-rolling-130002013.html) ·
[CTM rules](https://ctm.gg/rules/) · [CTWC rules](https://thectwc.com/rules/)

The qualitative difference that matters: **DAS pays a 16-frame charge on every direction change** unless
pre-charged against a wall. Hypertapping and rolling pay none. That is why rolling made killscreen play
survivable, and it is the thing worth showing a visitor.

M57.0's spike S3 already measured the payoff at pinned gravity: **at the kill screen DAS scores 0 and rolling
scores 37,135.**

**The fork (needs an owner call — see D2).** The technique dial can live in one of two places:

- **(i) Client-side only.** The pilot synthesises press/release edges into the existing `NesInput`; the C#
  training env and every parity test are untouched. Cheap, honest as a *demo*, and it makes the killscreen
  difference visible. The AI's macro decision is unchanged — only whether it can physically reach the column
  before gravity locks the piece.
- **(ii) Into the `.pg` as a legality mask** (`TETRIS_TECHNIQUES_PRD.md` §4.2/§4.3). The tap budget filters the
  action set and feeds `scareHeight` / `maxSafeCol9`, so the dial **changes what the evaluator wants** — a DAS
  agent at L29 correctly abandons tetris-building. A genuine strength control. Costs a retrain, requires
  porting `NesInput` into the `.pg`, adds 11 fields to serialization, and **invalidates every
  `data/tet*train/*-state.ckpt` resume file** (model checkpoints survive; resume files do not).

### 1.5 Ask (5) is a pure DOM move

The Block Dude page is a plain block-flow stack — **no page-level flex or grid** in `block-dude.scss` (unlike
`snake.scss:1`). Every top-level element is a normal block sibling, so reordering is a DOM move with no CSS
restructuring. Current template order (`block-dude.html`): h1 → intro → **playfield** `.bd-stage` :8-12 →
status :14 → D-pad `.controls` :16-23 → **`.actions` Prev/Level/Next :25-34** → second `.actions` :36-47 → …

The ask is well-founded: `.bd-stage` has **no fixed aspect ratio** — it is bound per level from the grid
(`block-dude.ts:67-70`, levels 19×8 → 29×19), so the entire lower stack really does shift vertically on every
level change. Moving the row above the board makes its position stable.

Two side-effects to accept deliberately: **margin collapse** (`.bd-stage` 0.75rem vs `.actions` 0.5rem) changes
the intro→stage gap slightly, and **tab order** follows DOM order, so Prev/Next moves ahead of the D-pad.

---

## 2. Design — locks

### LOCK A — gravity: variants, never a table edit
`gravityFrames` stays byte-identical for levels 0–29. Post-29 behaviour becomes a flag on the engine:
`0 = authentic` (default, 1 frame/row forever), `1 = CTM "39 halt"`, `2 = CTWC 2xks` (2 rows/frame from 39).
Default path must be bit-identical, so every checkpoint and the parity checksum survive.

**Structural note:** 2xks needs sub-1-frame gravity, which the current `i32` frames-per-row signature cannot
express. Cleanest is a `rowsPerStep` companion rather than switching to a float. This is the only structural
change gravity work requires.

### LOCK B — CCW: add a sibling, never modify the CW path
Add `microRotateCcw()` with the identical body and
`nr = (activeRot + rotCount[current] − 1) % rotCount[current]`. `microRotate()` is untouched. Re-transpile via
`dotnet build` (pgconfig routes both twins) — **never hand-edit `tetris_solver.ts`**. Then
`TetrisBoard.MicroRotateCcw()`, `tetris-game.ts rotateCcw()`, and keys **Z = B = CCW, X = A = CW** (already half
in place at `tetris.ts:209`).

### LOCK C — the dial is a pilot property, and the pilot must replay
Whichever fork D2 picks, the browser pilot **replays the input sequence the engine chose** rather than
re-deriving its own route. Today `tetris-game.ts:139-164` re-plans and silently bails to a hard drop on
`stuck >= 2` (`:159`) — substituting a different placement with nothing measuring the divergence. Under a
technique budget, `stuck` becomes a genuine bug indicator and must be asserted, not swallowed.

### LOCK D — express techniques in frames, not milliseconds
Gravity is already frames-based (`board.gravityFrames(level)`, `NES_FRAME_MS` at `tetris-das.ts:18`). Keeping
the tap model in frames makes the two commensurable, which is the whole point of the feature.

### LOCK E — one input machine
Watch mode reuses the human path's `frameAcc` fixed-timestep loop and drives the existing `NesInput`, instead
of keeping its own ms accumulator (`tetris-game.ts:116-135`). DAS becomes "hold the direction"; hypertap and
rolling become "press/release every N frames". Re-implementing DAS a second time inside the pilot is
explicitly rejected.

---

## 3. Spikes — run before any engine work, in this order

### S1 — Measure the search tier's tetris rate. ✅ RUN 2026-09-16 — and it refuted the cheap option.

`--baselines 30`, eval seeds 5000+e (the standard baseline protocol; the draft said 9000+, which is not what
`RunProtocol` uses — corrected here rather than in the harness). Protocol A, TRT = 4·tetrises/lines:

| tier | A score | lines | tetrises/ep | **TRT** | B survival |
|---|---|---|---|---|---|
| random | 2.7 | 0.1 | 0.00 | 0.0% | 22.2 |
| dellacherie | 175,217 ± 26,427 | 184.6 | 8.33 | **18.1%** | 343.1 |
| **della-search(8,5)** | **232,479 ± 66,780** | 140.1 | **18.00** | **51.4%** | **1412.0 ± 94.2** |
| net *(browser default)* | 103,327 ± 8,180 | 192.5 | 1.30 | **2.7%** | 220.1 |
| **net-search(8)** | 97,470 ± 3,960 | 198.2 | **0.65** | **1.3%** | 343.7 |

> ### NO-GO on net-search — and the reason matters.
> **Wrapping the net in search makes tetrises RARER, not commoner** (1.3% vs the plain net's 2.7%), and its
> A-score is *lower* than the plain net's. Search amplifies whatever the value function already wants; the net
> does not want wells, so searching harder over it just finds better ways not to build one.
>
> **The tetrises come from the Dellacherie evaluator, not from search.** `della-search` — the same beam, the
> same depth, but scoring afterstates with the widened *scripted* evaluator — reaches **51.4% TRT and 18.0
> tetrises/ep**, and dominates protocol B by **+311%** over plain Dellacherie (1412 vs 343).
>
> This is the same mechanism as §1.2 seen from the other side: the evaluator knows how to tetris and the net
> does not, so every tier that routes through the net inherits the blindness.

**Consequence for M62.4:** the cheap win is real but it is a different change than planned — the browser's best
tier must become **della-search(8,5)**, which drops the net out of the recommended path entirely. Under D3's
revised gate that tier passes (51.4% ≥ 40%). Remaining check is **latency** (G6, ≤ 50 ms/move): della-search
costs ~7K board sims per move and is currently a selectable tier, not the default, so its in-browser cost is
unmeasured.

**Honest note:** della-search trades lines for tetrises on protocol A (140.1 lines vs the net's 192.5) and
tops out more often (11/20 vs 2/30). It scores far higher because a tetris pays 1200 vs 40 — but a visitor
watching will see a shorter, denser game. That is the correct trade for "make tetrises", and it should be
stated rather than hidden.

### S2 — Pilot divergence census. ~20 min.
Instrument `tetris-game.ts:159` and count, over ~30 watch episodes at levels 0/19/29, how often the pilot's
realized placement differs from the director's chosen one. This is currently unmeasured and silently wrong.
It also establishes the baseline the technique dial will be judged against.

### S3 — Reachability at the dial's three rates. 🟡 IN PROGRESS.

Measured through `--baselines --tap <frames> --start-level <n>` (both options added in M62.3a) rather than by
a separate census, so the number reported is the thing we actually care about — score and tetrises under a
given pair of hands — not a proxy.

> **Method note that nearly produced a false negative.** `maxTapHeight` divides the tap budget by
> `gravityFrames(level)`, so **at level 0 (48 frames/row) every technique reaches every column and the dial
> is mathematically a no-op.** An A/B from the default start level would have reported "the dial does
> nothing" and been wrong. The dial only bites from ~L19 (2 frames/row) — which is exactly where real
> players switch technique, and why CTWC starts at 18/19 rather than 0.

**Level 0, 12 eps** (dilute by construction, recorded for completeness): rolling vs the DAS default moves
`della-search` 18.00 → **20.83** tetrises/ep and `net-search` 0.65 → **0.83**. Right direction, small, and
exactly as small as the gravity argument predicts.

### S3.R — results (16 eps, matched seeds, only `--tap` differing)

> **⚠️ PRE-MASK NUMBERS — not comparable to anything measured after M62.3b.** These were taken while the tap
> rate affected only *preference* (`evalAfterstate` + observation planes). Once D7 lands and `legalMask`
> consults the budget, `--tap` changes which placements are legal, and every figure below describes an
> engine that no longer exists. Re-measure rather than compare.

**L19 — the dial is a genuine strength control, with no retrain:**

| tier | tetrises/ep DAS → Roll | TRT | A-score DAS → Roll |
|---|---|---|---|
| dellacherie | 1.69 → **9.88** (5.8×) | 3.4% → **21.6%** | 229,500 → **401,785** (+75%) |
| **della-search** | 0.25 → **20.56** (**82×**) | 0.5% → **47.3%** | 191,762 → **605,237** (+216%) |
| net | 1.69 → 0.94 | 3.5% → 2.0% | 224,686 → 194,229 |
| net-search | 1.19 → 0.75 | — | 221,974 → 198,362 |

**An exact identity worth keeping.** L29+rolling reproduces L19+DAS *cell for cell* — same lines (196.4),
same tetrises (1.69), same top-outs; only the score differs, via the level multiplier. That is arithmetic,
not coincidence: `maxTapHeight` divides `taps · tapFrames · rowsPerStep` by `gravityFrames`, and
`6/2 = 3/1`. **Rolling buys exactly one gravity doubling** — it makes the kill screen play like L19 did with
DAS, which is precisely the real-world claim about the technique.

### M62.3b RESULTS — reachability enforced, and the kill-screen claim finally reproduces

With `legalMask` consulting the tap budget (opt-in `--reach`), 12 eps, matched seeds:

**L29, the kill screen — the headline:**

| tier | DAS score (lines) | Rolling score (lines) |
|---|---|---|
| dellacherie | **1,200** (0.9) | **193,400** (113.4) |
| della-search | **1,500** (1.2) | **275,550** (198.3) |
| net | 200 (0.2) | 202,700 (127.4) |
| net-search | **0** (0.0) | 280,700 (171.4) |

**This is M57.0's "at the kill screen DAS scores 0, rolling scores 37,135" — reproduced on the shipped
engine for the first time.** It never reproduced before because nothing stopped DAS from reaching column 9;
now the mask does. The 6/2 = 3/1 identity survives enforcement exactly: L29+rolling gives dellacherie 113.4
lines, identical to L19+DAS.

**Latency (G6):** `della-search` **p50 ≈ 11 ms, p99 ≈ 24 ms** — comfortably inside the 50 ms budget, so D12's
default tier is affordable even paying for root reachability. `net-search` is the expensive one (p50 ≈ 40 ms,
p99 ≈ 78 ms) but it is not the default.

**Cost of honesty:** at L19/DAS, enforcing reachability takes dellacherie from 196.2 → 113.4 lines. That is
the real price of making the hands physical, and it should be quoted rather than hidden.

### C1 was not theoretical — it cost 170 lines a game

The interference audit predicted the root would misbehave with legality on truth and valuation on the proxy.
**Measured:** at L19 with rolling, `della-search` collapsed to **27.8 lines** while the same tier at L29 with
rolling scored 198.3. Mechanism exactly as predicted — the proxy said the well was serviceable, so the tier
committed to building one; the true mask then refused every placement that could feed it, and it stranded. At
L29 the proxy said "lineout", it played flat, and survived.

**Resolved in M62.3b**: `refreshReachSpan()` computes the reachable column span ONCE per decision (the root
already pays 40 simulations for the mask) and caches it; `evalAfterstate` reads the cached bounds, so the
thousands of calls inside a search stay cheap. Under enforcement `lineout` becomes *"the well column is
unreachable"* — a fact, not a height heuristic — and the accessibility penalties key off the span. With
enforcement off every formula reduces to the proxy exactly, which is what keeps the default bit-identical.

**A second defect this exposed:** running out of *reachable* placements returned −1 and the caller simply
stopped the episode, recorded as **not a top-out** — an episode ending for no stated reason.
`hasLegalPlacement` now follows the mask the policy actually plays against.

### ⚠️ The finding that re-scoped M62.3b: the dial changes PREFERENCES, not LEGALITY (pre-fix)

`legalMask` (`.pg:515`) **does not consult the tap budget** — verified by reading it. The tap rate reaches
only `evalAfterstate` (via `lineout`/`overLeft`/`overRight`) and the observation planes. So the AI still
*can* place a piece anywhere; it merely values placements differently.

Consequences, stated plainly because two of them contradict earlier claims in this repo:

1. **M57.0's "at the kill screen DAS scores 0" is not reproducible on the shipped engine.** We measure DAS at
   L29 scoring **344,550** with 185 lines. Nothing prevents DAS from reaching column 9, because macro
   placements are applied instantly (`TetrisEnv.cs:9`: the AI's placement decisions have no gravity clock).
   The spike measured a *hypothetical* constrained agent; the engine never became one.
2. **M62.3b (the legality mask) is REQUIRED, not optional.** Without it the dial is honest about strategy but
   dishonest about physics — a visitor selecting DAS at the kill screen watches an AI that still slides
   pieces to the wall it could not possibly reach. This reverses the draft's "ship the dial, gate the mask".
3. **The retrain is justified — for a different reason than D2 assumed.** Not "the mask changes behaviour"
   but: **the net was trained at a single tap rate (6) and cannot exploit the dial at all** — it gets
   *worse* with rolling (0.94 vs 1.69) while the evaluator tiers get 5.8–82× better. The fix is to
   **randomize the tap rate across training episodes** so the net learns a policy conditioned on its hands.

**Also measured — the net cannot survive the kill screen.** At L29 with DAS: 112.9 lines, **14/16 top-outs**
on protocol A, and 101 pieces on B vs `della-search`'s 1,387. `net-search` goes **−70.8%** vs dellacherie
there. The shipped browser default is the weakest tier at exactly the setting a visitor is most likely to
find impressive.

### S4 — Argmax fidelity probe. ~30 min, no training.
Over held-out rollouts, log how often the net's top-1 equals the exact evaluator's top-1, split by whether
column 9 is currently open. Confirms or refutes the "~10 consecutive argmaxes" mechanism directly, and tells
us whether option #2/#3 in §4 is worth building.

---

## 4. Ranked options for ask (2)

Metric: **TRT = 4·tetrises / lines**, plus absolute tetrises/ep, protocol A, 30 eps, held-out seeds 9000+.
Current: net **1.4% / 0.67**; della-search **44% / 15.60**.

| # | Option | Effort | Likelihood | Gate |
|---|---|---|---|---|
| 1 | **Promote the search tier** (S1). Browser default becomes net-search / della-search. | ~0 | very high | TRT ≥ 40%, ≥ 8/ep, ≤ 50 ms/move |
| 2 | **Shared per-candidate scorer.** Reshape the 16×40 candidate block to `[B·40,16]`, apply a shared MLP → `[B,40]`. Fits `IValueNet` unchanged, ~40× fewer params, 40 labelled examples per state. Attacks argmax fidelity directly. | medium | high | plain-net TRT ≥ 20%, ≥ 4/ep, A-score not CI-below 91,891 |
| 3 | **Argmax-aware loss.** Huber optimizes MSE, not ranking; add a margin / top-k term so the top-1 gap is what is fitted. | low–med | med-high | as #2, cheaper to falsify |
| 4 | **Re-run CEM with TRT in the fitness.** Lifts the teacher, which the net then distils. | low | medium | teacher TRT ≥ 55% at A ≥ 150k, B not CI-below 430 |
| 5 | **Fix PBRS Φ** to use `wellSumExceptWell` (`TetrisEnv.cs:126`) — the last place the tetris well is still penalized (~0.04 reward/cell). | trivial | low | no-regression only |
| 6 | M57.3 movement-aware enumeration / M57.4 SRS. | high | low for TRT | — |

**G3 must be renegotiated.** The standing 50% TRT gate is likely unreachable for a plain net under the
500-piece cap — della-search, playing the evaluator's *exact* argmax, only reaches 44%. This is already logged
as risk 7 (`TETRIS_TECHNIQUES_PRD.md:1219`). Proposed replacement: **plain net TRT ≥ 20% and ≥ 4 tetrises/ep;
search tier ≥ 40%.** Settle this before spending on #2–#4.

---

## 5. Milestones

- **M62.0 — Spikes.** ✅ **S1 + S3 RUN** (`0a0d051`), and both changed the plan: S1 killed the "promote
  net-search" option (1.3% TRT, *worse* than the plain net), S3 proved the dial is a real strength control
  but only over *preference*, which is what forced D7. **S2 is absorbed** — D9's replay makes the divergence
  census moot. **S4 (argmax fidelity) not run**, and is now optional: D12 routes around the net rather than
  through it.
- **M62.1 — CCW rotation (LOCK B).** ✅ SHIPPED (`8afc107`). `.pg` sibling fn, C# facade, `rotateCcw()`,
  Z/X keys, three help strings, 8 new test cases. CW path byte-identical; `ActionCount` 40 and
  `ObservationSize` 854 unchanged — **though C2 will now widen the observation, so the no-invalidation
  property ends at M62.4c.** Touch CCW deferred (D4 still open).
- **M62.2 — Gravity variants + start-level picker (LOCK A).** ✅ SHIPPED (`dc4621d`). `killscreenMode` 0/1/2
  with the `rowsPerStep` companion, `forceGameOver`, web gravity loops honouring both, start-level picker
  (0/9/15/18/19/29) wired to the previously-uncalled `setStartLevel`. Authentic table pinned across
  levels 0–255 and the 18-start line thresholds (130/230) pinned by test.
- **M62.3 — Technique dial (LOCK C/D/E).** 🟡 **M62.3a SHIPPED** (`02bcb35`): `Technique` + frames table in
  `tetris-das.ts`, three buttons, pilot cadence driven by the dial with DAS paying its 16-frame charge,
  status line and hint carrying the real rates, `--tap`/`--start-level` in the Lab.
  **The milestone shrank on discovery:** M57.1 already built the engine half (`tapFramesPerShift`,
  `setTapRate`, `maxTapHeight`, and `evalAfterstate` already consuming the tap budget via
  `lineout = m5 < 4` and `overLeft`/`overRight`) — but **`SetTapRate` was never called by any production
  code**, so only DAS has ever been in effect. Fork (ii) was mostly dead code to wire, not code to write.
  Also corrected: the pilot ran at `PILOT_INPUT_MS = 90` ≈ 11.1 Hz ≈ 5.4 frames — **the "normal" AI was
  already hypertapping**, unlabelled.
  **M62.3b — reachability mask + timeline (D7–D11).** ⬜ Next. Factor the inlined enumeration into ONE seam
  (finally making `TETRIS_PRD.md`'s long-false claim true), have it run a true shift-while-falling simulation
  honouring `rowsPerStep` and both rotation directions, return legality **and** the input timeline, and feed
  the root evaluator's reach terms from the same computation (C1). Rollouts keep the proxy. Re-pin the parity
  checksum; instrument latency + root/search disagreement (D11).
- **M62.4 — Tetris rate + the default tier.** ⬜ Split in three, in order: **(a)** point the browser default at
  `della-search` (D12, one line) and add the hint sentence; **(b)** baseline the existing net *under the mask*
  — free, since the run is needed anyway to show the mask regressed nothing, and it doubles as the
  pre-training baseline; **(c)** retrain with randomized tap rate plus an explicit tap-rate input (D13 + C2),
  accepting the observation widening and using `GrowInput` to transplant the shipped net.
- **M62.5 — Block Dude control row.** ✅ SHIPPED (`8afc107`). Pure DOM move above `.bd-stage`; margin-collapse
  and tab-order changes accepted deliberately.
- **M62.6 — Doc/code defect corrections** (§7). ✅ SHIPPED (`8afc107`). Four verified before editing; one
  (`RewardTetrisBonus`) turned out to be a live inconsistency, not a dead declaration — deferred to M62.4.
- **M62.7 — Ship.** ⬜ One PR, `ARCHITECTURE.md` + PLAN sync.

---

## 6. Gates

| # | Gate |
|---|---|
| **G1** | ✅ Default gravity path **bit-identical** — 46/46 tests green including the parity checksum |
| **G2** | 🟡 CW rotation path byte-identical and `ActionCount` 40 — **permanent**. `ObservationSize` 854 holds through M62.3b, then **deliberately breaks at M62.4c** (C2 adds the tap-rate input); the shipped `.ckpt` is transplanted via `GrowInput`, not loaded as-is |
| **G1b** | **The parity checksum is re-pinned, not abandoned.** M62.3b moves it (the mask changes behaviour); the gate is that both twins agree on the NEW value over a 1000-move seeded episode. A checksum that cannot be reproduced in TS is a failed gate, not a new baseline |
| **G9** | **Root/search disagreement is reported** in the Lab baselines (D11). First reading is calibration; a value is recorded before any threshold is set |
| **G10** | **Pilot divergence is zero** — under D9 the pilot replays the engine's timeline, so `stuck` firing is a hard error, not a fallback |
| **G3** | **(settled by D3)** search tier TRT ≥ 40% — `della-search` **51.4% PASS** · plain net TRT ≥ 20% and ≥ 4 tetrises/ep — **2.7% FAIL**, the target of M62.4 |
| **G4** | ✅ **PASSED pre-mask** — rolling vs DAS at L19: dellacherie 1.69 → 9.88 tetrises/ep, della-search 0.25 → **20.56 (82×)**, score +216%. **Must be re-measured post-mask** (C3) |
| **G5** | ⛔ Superseded by **G10** — D9 makes divergence structurally impossible rather than merely bounded |
| **G6** | Browser: ≤ 50 ms/move for the default tier — now **`della-search`** (D12), which is also the tier D8/D9 make most expensive. **If missed, make della-search fit; do not drop it** (D11) |
| **G7** | Block Dude: the Prev/Level/Next row does not move vertically when switching between the shortest (19×8) and tallest (29×19) levels |
| **G8** | 🟡 `dotnet build` clean; `tsc --noEmit -p tsconfig.app.json` clean; `dotnet test` green — **all three green as of `0a0d051`**, re-checked per milestone |

### The concentrated risk in D7–D13

Worth stating separately because it is not visible in any single gate: **D8 + D9 put a per-placement
simulation and a variable-length timeline on the hot path, and D12 makes the tier that hits that path
hardest (`della-search`, ~7K afterstates/move) the browser default.** The fidelity risk and the latency risk
and the default-tier risk are now *the same risk*. G6 is therefore the gate most likely to fail, and D11
already fixes the response (make it fit, don't drop it) so that the failure does not silently re-open D12.

Second-order: **D10 is provisional.** If the disagreement metric (G9) comes back high, the fix is truth
everywhere, which is the one outcome that genuinely forces `della-search` out of the browser — the thing D12
and the owner's stated preference both depend on. That is the single seam in this design most likely to
require rework.

---

## 6.D Why the search declines tetrises — measured, and it was not what I expected

Built `--decline-census` to count the owner's report ("gets a long bar, could grab a tetris, puts it
somewhere else — about 1 in 4 times") rather than reason about it. 10 episodes, seeds 5000+e:

| tier | tetris on the table | **took it** | declines on a HOLED board |
|---|---|---|---|
| dellacherie (1-ply) | 90 steps | **88 (97.8%)** | 0 of 2 |
| della-search (2-ply) | 400 steps | **192 (48.0%)** | **1 of 208 (0.5%)** |

Two findings, the first of which killed my own hypothesis:

1. **It is not the DIG gate.** 99.5% of declines happen on CLEAN boards, so "a single hole blinds the
   evaluator to the well" — plausible, and consistent with the owner's other observation that the agent
   digs well — is simply wrong as an explanation for this.
2. **It is the SEARCH.** 1-ply Dellacherie takes 97.8% of available tetrises; adding one ply of look-ahead
   halves it.

**The mechanism is a double-count.** `tetrisReady` is a *state* bonus paying `EvalReady` per ready row, so a
multi-ply search counts the same four rows again at every ply it declines to cash them, while taking the
tetris banks `eroded + EvalTetris` **once** and leaves a board worth zero:

```
decline   ≈ 13.6 (ply 1) + 13.6 (ply 2)   = 27.2
take      ≈ 16 eroded + 7.05 EvalTetris   = 23.1
```

Holding beats cashing by ~4 — but only when there is a second ply to double-count in, which is exactly why
the 1-ply tier does not show it. **The AI is not undervaluing tetrises; the evaluator pays rent on a well
you never cash, and the search found the exploit.**

`tetrisPayback` returns the consumed rows on cashing. Monotonic on take-rate: **0.0 → 48.0%, 0.5 → 59.4%,
1.0 → 70.0%**. Take-rate is a proxy, so tetrises/episode and score are being measured before any default
changes; `0.0` reproduces the shipped behaviour exactly. Note the baseline already cashes **17.25
tetrises/episode** despite declining half the offers — offers are frequent — so the fix must raise *that*,
not merely the ratio.

## 6.E The shipped net's training recipe, recovered from its own checkpoint

`TETRIS_TECHNIQUES_PRD.md` lists as a live defect that **"the training CLI args are recorded nowhere in the
repo"**, and it bit immediately: three attempts to fine-tune the shipped net failed on argument mismatches
before one started. What the failures recovered, which is now the only record:

| fact | value | how it was recovered |
|---|---|---|
| hidden layers | **256,256** | checkpoint size 1,180,907 B ≈ 283k params × 4 for 854 inputs; `[128,128]` would be ~0.5 MB |
| n-step | **1** | resume state refused `--nstep 3`: *"Resume state n-step 1 != options 3"* |
| γ | 0.995 | accepted by the resume |
| `--dense` | **must NOT be passed** | *"DenseTargets requires Gamma == 0"* — the campaign injects its per-action dense targets through its own path, so this flag is actively wrong for Tetris |
| resume point | 195,000 placements, baseline 99,598 | run log, matching the M57.5 record |

**Two corrections to the record follow.** The PRD and the Lab default both say **n-step 3**; the net that
actually shipped used **n-step 1**. And a bare `--game tetris --dense` — the intuitive reading of "dense
all-action regression" — **throws**, so the documented recipe could not have been the one used.

**Fixed going forward:** `LabHost` now appends the full invocation to `<dataDir>/invocation.txt` on every
training start. A checkpoint that cannot say how it was made cannot be reproduced, compared against, or
resumed — which is exactly the position the shipped net was in.

## 7. Corrections to land in the same PR

Found during the investigation, all currently wrong in the repo:

1. `TETRIS_PRD.md:16, :67, :75, :184, :188` assert an `enumeratePlacements()` seam that **does not exist**
   (grep hits `docs/` only). Enumeration is inlined at **7 `.pg` sites** (`:515`, `:808`, `:900`, `:909`,
   `:923/:953`, `:1009`, `:850`). Already logged at `TETRIS_TECHNIQUES_PRD.md:1174`; fix the source docs.
2. `TetrisEnv.cs:23` and `TetrisBoard.cs:27` say the observation is **814**; it is **854**.
3. `.pg:819` comment still says 454 / six planes.
4. `RewardTetrisBonus` (`.pg:63`) is declared and never read in the `.pg`; only the C# copy
   (`TetrisBoard.cs:44`) is live, via `TetrisEnv.cs:166`. **This is more than a dead declaration.** Its own
   comment asserted that *"netSearchAction's rollout reward must use the same units as training"* — but
   `netSearchAction` scores afterstates through `evalAfterstate` and never adds the bonus, so **the search
   tier's rollout reward is not in training units**: a 4-line clear is worth 4 to the rollout and 12 to the
   learner. M62.6 corrects the comment to state this truthfully rather than silently changing rollout
   semantics, because fixing it moves the parity checksum and needs its own before/after measurement.
   **Deferred to M62.4**, where the search tier is already under measurement — a plausible (untested)
   contributor to the search tier under-valuing tetrises.
5. `TETRIS_TECHNIQUES_PRD.md` §0's headline diagnosis (γ=0, `−20·Δwells`, tetris < 4 singles) is **stale** and
   actively misleading — it was fixed by M57.1/M57.5. Mark it superseded, pointing at §1.2 here.

---

## 8. Owner decisions

**D1, D2 and D3 were decided 2026-09-16.** D4–D6 remain open.

| # | Decision | Resolution |
|---|---|---|
| **D1** | Gravity variants | ✅ **Add both behind a flag.** `killscreenMode`: `0 = authentic` (default, bit-identical), `1 = CTM "39 halt"`, `2 = CTWC 2xks` (2 rows/frame from L39). Needs the `rowsPerStep` companion |
| **D2** | Where the technique dial lives | ✅ **Fork (ii): into the `.pg` as a legality mask + retrain.** The dial becomes a real strength control feeding `scareHeight`/`maxSafeCol9`, not a rendering effect. Accepts the retrain cost and the loss of every `data/tet*train/*-state.ckpt` **resume** file (model checkpoints survive). S3 no longer gates it — it now informs the mask's shape instead |
| **D3** | The tetris-rate gate | ✅ **Gate the tier, not the number.** G3 becomes: **search tier TRT ≥ 40%** (della-search measured **51.4% — PASS**) and **plain net TRT ≥ 20% + ≥ 4 tetrises/ep** (measured 2.7% — FAIL, and the target of M62.4's training work) |

### D7–D13 — the reachability design (decided 2026-09-16, by interview)

| # | Decision | Resolution |
|---|---|---|
| **D7** | Where the tap budget becomes a constraint | **In the engine.** `legalMask` consults it, so training, the Lab, all five tiers and the browser share one rule. Rejected: masking only in the browser director (would leave two different AIs — the benchmarked one and the watched one) |
| **D8** | What the mask computes | **True per-placement reachability** — simulate the piece shifting while falling against the real stack profile. Rejected: the `maxTapHeight` proxy, which ignores the stack between spawn and target |
| **D9** | Does the sim emit the input sequence | **Yes — and the pilot replays it.** Divergence becomes zero by construction; `stuck` becomes an assertion rather than a swallowed substitution. Closes G5 and makes S2 moot. Flat `List<i32>` with a stride convention (the engine has no nested lists) |
| **D10** | Does the search pay for truth too | **Truth at the root, proxy inside rollouts** — beam and expectimax are already approximations, so a cheaper legality model there is consistent with what search is. ⚠️ **PROVISIONAL**, falsified by D11 |
| **D11** | What overturns D10 | Instrument **latency** and **root/search disagreement**. Overturn if > 50 ms/move or disagreement is judged excessive. **The 5% figure is CALIBRATION on first read, not a gate** — an uncalibrated gate on a new metric would read as failure and cost the search tier. **If latency misses, make `della-search` fit** (timeline only for the chosen action, narrower beam, per-move cache) rather than dropping it |
| **D12** | Browser default tier | **`della-search`**, with all five tiers still selectable and one factual line in the hint noting the hand-tuned evaluator currently out-tetrises the trained net |
| **D13** | Retrain | **Yes, with tap rate randomized per episode**, sequenced AFTER the mask — the signal it must learn does not exist until `legalMask` consults the budget, so training now would reproduce the net we already have |

### Interference audit — three conflicts found when compiling D7–D13

Recorded because each was invisible while the decisions were taken one at a time.

1. **The root would have been internally inconsistent (C1).** D8 puts *legality* on truth, but `evalAfterstate`
   derives `lineout = m5 < 4`, `overLeft` and `overRight` from the **proxy** — so the same root decision would
   mask on one model and decide *whether to want tetrises at all* on another, firing the LINEOUT/DIG/STANDARD
   switch at the wrong time. D10 framed this as root-vs-rollout; it is actually inside the root.
   **Resolved:** compute reach ONCE at the root, feed both the mask and the evaluator's reach terms from it.
   Rollouts keep the proxy for both. **One model per altitude, not one per consumer.**
2. **D13's randomization needs the net to know its own hands (C2).** With the rate varying per episode the net
   must be able to tell which technique is live; today it can only infer this implicitly from plane 11
   (`rowsAboveTapReachableHeight`). If that signal is too weak it learns one averaged policy and the
   randomization **silently buys nothing** — indistinguishable from success until the dial fails to move it.
   **Resolved:** add an explicit tap-rate input. This widens the observation past 854 and **invalidates the
   "no checkpoint invalidation" property M62.1 enjoyed** — acceptable because D13 already accepts a retrain,
   and M57.5 precedent (`DuelingQNet.GrowInput`) transplants the old net function-preservingly.
3. **The S3 table stops being reproducible (C3).** Once `--tap` affects legality rather than only preference,
   every S3.R figure describes a different engine. They are labelled **pre-mask** below. This repo has twice
   been misled by comparing across exactly such a boundary (the stale γ=0 diagnosis, the phantom
   `enumeratePlacements` seam) — the label is the cheap defence.

**Two couplings, no conflict:** the truth sim must honour `rowsPerStep` so 2xks halves *real* reach and not
only proxy reach; and it must consider **both rotation directions** now that M62.1 shipped CCW — which is what
**absorbs D5** (the pilot no longer chooses a rotation direction; the simulation does, and the pilot replays).

### Still open

| # | Decision | Why it matters |
|---|---|---|
| **D4** | CCW on touch — second on-screen button, or two-zone tap? | Two-zone tap overloads an existing gesture |
| **D5** | Should the pilot pick the *shorter* rotation direction once CCW exists? | Cosmetic at low levels, but it **changes watch-mode outcomes at killscreen speeds** — a deliberate call, not a freebie |
| **D6** | **Carried over from `TETRIS_TECHNIQUES_PRD.md` §1.1, still unanswered:** SRS mode as a strength lever (Guideline scoring) or as a movement model (NES scoring)? | Under NES scoring a T-spin double pays **100** vs a tetris's **1200**, so spins are correctly near-worthless. Only relevant if M57.4 is ever revived; that PRD recommends the movement-model reading |

---

## 9. Out of scope

- **SRS / kick tables / T-spins (M57.4).** Not being done here. §8 D6 records the open decision; the economics
  under NES scoring make it a low-value strength lever.
- **Movement-aware enumeration for tucks (M57.3).** Ended by measurement in M57.0 ("the evaluator was the whole
  story"); S1/S2 there were a NO-GO on tucks. The tap-budget *mask* survives, and is D2 fork (ii) above.
- **Porting the whole `NesInput` into the `.pg`** unless D2 picks fork (ii).
- **Reproducing the level-138 palette glitch or the ~155 crash.** Recorded in §1.1 as facts, not as work.
