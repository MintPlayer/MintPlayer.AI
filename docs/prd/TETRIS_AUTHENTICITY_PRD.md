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

### S1 — Measure the search tier's tetris rate. ~0 effort, and it may resolve ask (2) outright.
`net-search(8)` tetris-rate is **unmeasured** since M57.1. One `--baselines 30` run on held-out seeds 9000+.
If it lands near della-search's 44%, then **changing the browser's default tier delivers tetrises today**, and
M62.4 shrinks to a default change plus a latency check.
**GO/NO-GO:** GO to tier-default if TRT ≥ 40% and ≥ 8 tetrises/ep at ≤ 50 ms/move in the browser.

### S2 — Pilot divergence census. ~20 min.
Instrument `tetris-game.ts:159` and count, over ~30 watch episodes at levels 0/19/29, how often the pilot's
realized placement differs from the director's chosen one. This is currently unmeasured and silently wrong.
It also establishes the baseline the technique dial will be judged against.

### S3 — Reachability at the dial's three rates. ~30 min.
For each technique, at levels 18 / 19 / 29, compute what fraction of the 40 macro placements is physically
reachable before lock. Reproduces S3's DAS-scores-0 result on the *shipped* engine and tells us whether fork
(ii) is buying real strategy change or only a visual.
**GO/NO-GO on fork (ii):** only worth its retrain if reachable-set size differs by ≥ 15% between DAS and
rolling at L19.

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

- **M62.0 — Spikes S1–S4.** ⬜ Gate the rest of the arc. S1 may collapse M62.4 to a one-line default change.
- **M62.1 — CCW rotation (LOCK B).** ⬜ `.pg` sibling fn, C# facade, `rotateCcw()`, Z/X keys, help strings,
  a second touch button, CCW round-trip tests (4× CCW = identity for J/L/T; no-op for O; equals CW for I/S/Z).
- **M62.2 — Gravity variants + start-level picker (LOCK A).** ⬜ Post-29 flag with `rowsPerStep`; frontend
  start-level picker (0 / 9 / 15 / 18 / 19 / 29) wired to the existing `setStartLevel`.
- **M62.3 — Technique dial (LOCK C/D/E).** ⬜ `Technique = 'das' | 'hypertap' | 'roll'` with a frames table in
  `tetris-das.ts`; watch mode moved onto the frame clock; three radios mirroring the tier row
  (`tetris.html:18-25`, `tetris.ts:41/109-112`); localStorage persistence; status line shows the active rate;
  hint sentence carries the 10 / 12 / 20 Hz numbers. Scope depends on **D2**.
- **M62.4 — Tetris rate.** ⬜ Option #1 always; #2/#3 only if S1/S4 say the plain net is worth fixing.
- **M62.5 — Block Dude control row.** ⬜ Move `block-dude.html:25-34` above `.bd-stage`; accept the margin and
  tab-order changes; check the two-`.actions`-siblings split breaks nothing (no `+`/`~` selectors exist).
- **M62.6 — Doc/code defect corrections** (§7). ⬜
- **M62.7 — Ship.** ⬜ One PR, `ARCHITECTURE.md` + PLAN sync.

---

## 6. Gates

| # | Gate |
|---|---|
| **G1** | Default gravity path **bit-identical**: parity checksum `765594964` unchanged, same seed ⇒ same trajectory |
| **G2** | CW rotation path byte-identical; `ActionCount` 40 and `ObservationSize` 854 unchanged; `wwwroot/models/tetris.dqn.ckpt` still loads |
| **G3** | **(renegotiate, §4)** plain net TRT ≥ 20% and ≥ 4 tetrises/ep; search tier ≥ 40% |
| **G4** | Technique dial measurably changes outcomes: rolling CI-above DAS on protocol A at L19 start |
| **G5** | Pilot divergence (S2) does not regress; `stuck >= 2` bail-outs are asserted, not silent |
| **G6** | Browser: ≤ 50 ms/move for whichever tier is default; no long tasks over ≥ 30 s of watching |
| **G7** | Block Dude: the Prev/Level/Next row does not move vertically when switching between the shortest (19×8) and tallest (29×19) levels |
| **G8** | `dotnet build` clean; `tsc --noEmit -p tsconfig.app.json` clean; `dotnet test` green (`Category=Slow` for gates) |

---

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

## 8. Open owner decisions

| # | Decision | Why it matters |
|---|---|---|
| **D1** | Gravity variants: authentic-only, or add "39 halt" and/or CTWC 2xks? | Determines whether M62.2 needs the `rowsPerStep` structural change at all |
| **D2** | Technique dial: **(i)** client-side demo, or **(ii)** `.pg` legality mask + retrain? | (ii) makes the dial a real strength control but costs a retrain and invalidates every `*-state.ckpt` resume file. S3 gates it |
| **D3** | Accept the G3 renegotiation (50% → net 20% / search 40%)? | The standing gate is probably unreachable; spending on options #2–#4 against it would be spending against an impossible target |
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
