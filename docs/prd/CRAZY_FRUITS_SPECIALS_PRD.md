# Crazy Fruits specials (striped / wrapped / sugar bomb) — PRD

**Status:** ✅ shipped 2026-07-24 (M50.3 closed via stop-loss — gates 2/3 honestly missed). **Superseded on
the net front by M51 ([CRAZY_FRUITS_RANKING_PRD.md](CRAZY_FRUITS_RANKING_PRD.md), 2026-07-25):** `cf9train`
(dense all-action regression + shaped obs plane, 928→1040) passes ALL the §5 gates — +117.2% over random,
gap-share 91%, created 9.57/fired 10.39 — and replaced `cf5train` as the shipped checkpoint.
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M50 · extends [CRAZY_FRUITS_PRD.md](CRAZY_FRUITS_PRD.md) (M49, shipped — specials were its §7 out-of-scope item) · branch `m49-crazy-fruits` (owner decision: ONE branch/PR — #38 — for the whole Crazy Fruits arc)

## 1. Goal

Candy-Crush-style special pieces on the shipped M49 match-3: **striped fruit** (match-4 → clears a full
row/column), **wrapped fruit** (L/T/+ intersection match → 3×3 double explosion), and the **sugar bomb**
(match-5 straight → swap with any fruit clears all of that type), including all special+special combo swaps —
in the single-source engine so human play, the scripted tiers, and the trained net all get them, and the
retrained primitive net demonstrably *uses* them.

## 2. Rules lock (M50.0 — every rule deterministic, ZERO RNG draws)

Planning (`deterministicValue`) and C#↔TS parity both require rules that consume no randomness; every
ambiguity found in the sources is locked here.

**Creation** (from any qualifying match, swap-made or cascade-made; priority **bomb > wrapped > striped**,
resolved per shape, rows-then-columns ascending as the tiebreak):
- **Striped** ← 4 in a straight line. **Blast direction = PERPENDICULAR to the creating match** (horizontal
  4-match → a vertically-striped fruit that clears its COLUMN; stripes are painted along the blast).
  *Owner correction 2026-07-24: the research agent's "blast ∥ match" resolution was wrong — the real Candy
  Crush rule is blast ⊥ match, with the stripe paint showing the blast direction.*
- **Wrapped** ← a horizontal ≥3 run and a vertical ≥3 run sharing a cell (L, T, +; any larger intersection).
  Spawns at the intersection cell.
- **Sugar bomb** ← 5+ in a straight line. Colorless (fruit type 0).
- **Spawn cell:** the swapped cell when the player's swap made the shape (each swapped cell spawns its own
  special if both sides match); cascade-created line matches spawn at the run's **lowest cell index**
  (deterministic convention — the canonical game leaves this undocumented).
- **Spawn-cell collision (owner bug report 2026-07-24):** a spawn cell that already holds a special is never
  overwritten — that special stays match-marked and **fires**, and the creation **relocates to the nearest
  plain cell of the shape** (along the run; outward along both arms from a wrapped pivot; ties toward the
  lower flat index). A shape with **no plain cell creates nothing** — every special in it simply fires.
  Matches real Candy Crush: dragging a striped into a 4-line fires it AND paints another fruit in the line.
- **Relocated creations are SHIELDED (owner report round 2, M50.6):** the relocated special is immune to
  **blasts** (striped beams, wrapped boxes, armed re-explosions, bomb zaps, combo areas) for the **rest of
  the move** — the fired special must not consume its own replacement (a wrapped's 3×3 always covered the
  nearest run cell, so the player otherwise never kept the promised special; the armed second explosion
  covered it again a step later). **Matches** in later cascade steps still consume it normally, and
  creations placed on their preferred (plain) spawn cell keep the form-then-trigger chain rule below —
  the shield applies ONLY to relocations. `stageSwap` clears the shield, so it never outlives the move.
- The creation cell is scored like its matched neighbours but receives the special instead of clearing.

**Activation** (specials fire when cleared by anything — matches, blasts, combos; chains are unbounded but
each cell activates at most once per step). **Ordering (owner rule 2026-07-24): specials FORM before the
step's activations run** — a fresh special lands on its spawn cell first, and if any blast/combo of that same
step reaches it, it fires immediately (an untouched fresh special survives the step unmarked):
- **Striped:** clears its entire row (stripedH) or column (stripedV).
- **Wrapped:** explodes its 3×3, becomes **armed**, falls with gravity, then explodes its 3×3 AGAIN at its
  landing cell on the next cascade step (the canonical double explosion — the settle gap is observable).
- **Bomb, swapped with a plain fruit:** clears every fruit of that type (+ itself). **Bomb hit passively**
  (caught in a blast): clears every fruit of the board's **most frequent type** (ties → lowest type index) —
  deterministic stand-in for the canonical "random color" (RNG is banned by this lock).
- Blast-cleared cells score exactly like match-cleared cells: **10·(k+1)** at cascade step k — the M49
  proportional-scoring lock (§3.5 there) extends unchanged; no exponential jackpots.

**Combo swaps** (always legal; the swap itself is the activation):
| Swap | Effect |
|---|---|
| bomb + plain | clear all of that fruit type (+ the bomb) |
| bomb + striped | every fruit of the striped's type becomes striped (orientation `(r+c)%2` — deterministic) and all fire |
| bomb + wrapped | every fruit of the wrapped's type detonates as a 3×3 blast |
| bomb + bomb | clear the whole board |
| striped + striped | one full row AND one full column through the combo centre |
| striped + wrapped | 3 full rows AND 3 full columns around the combo centre |
| wrapped + wrapped | one 5×5 blast at the combo centre, then the armed double-explosion fires again after the settle |

**Combo centre (owner rule 2026-07-24): the gesture's LAST-SELECTED cell** — the second tap of a tap-tap, the
dragged-to cell of a drag. The action space can't carry gesture direction, so `stageSwap(action, targetCell)`
takes it from the host; **AI/baseline moves deterministically default to the action's bottom/right cell.**

**Legality (replaces "swap must produce a match"):** a swap is legal iff it produces a 3+ typewise line
through a swapped cell, OR either cell is a bomb, OR both cells are specials. Striped/wrapped + plain with no
line stays illegal. **The 112 adjacent-swap action space is unchanged** — a bomb still only swaps with a
NEIGHBOUR.

**Creation scores NOTHING (owner decision 2026-07-24): specials yield points only when they FIRE.** The
in-game score is therefore pure fire-only; the γ=0 creation signal moves to a TRAINING-ONLY shaping term
(§3.5): the train env adds **+40 striped · +60 wrapped · +100 bomb** per creation to the *reward*, while the
eval env and every gate measure the bare game score. Combos earn no extra flat bonus — they already clear
more cells. Uniform wrapped rule (implementation lock): a wrapped's first detonation ALWAYS arms it — via
match, chain, or combo — so it always explodes twice; the wrapped fruit scores at first detonation, the
armed shell's second blast scores its victims plus the shell.

## 3. Design

### 3.1 Cell encoding — packed i32, base 16
`grid` value = `kind·16 + type`; kinds `0 none · 1 stripedH · 2 stripedV · 3 wrapped · 4 bomb · 5 wrappedArmed`
(internal-only: never on a stable board). Decoders `fruitOf(v) = v % 16`, `kindOf(v) = divExact(v, 16)` — the
blessed exact ops; max value 86. **Plain fruit keep their current values 1..6**, so every mutation site
(`swapCells`, gravity, Fisher–Yates reshuffle, state serialization, animation snapshots, directed-test
boards) stays correct untouched, and the parity checksum hashes kinds for free.
**Rejected:** a parallel `kinds` list — doubles every mutation/serialization/snapshot site.

### 3.2 Detection — scanMatches → scanRuns + creation resolver
The existing row/column scans additionally emit `PgCfRun {horizontal, start, len}` records; a second bounded
pass resolves creations by the §2 priority (runs consumed by a higher-priority shape don't create again).
Typewise comparisons via `fruitOf` with a `type != 0` guard everywhere (`wouldFormRun`, `cellInMatch`,
`anyMatchOnBoard`, run scans, `swapProducesMatch`'s same-value early-out) — a bomb (type 0) never joins a run.

### 3.3 Activation — bounded worklist; the armed wrapped rides the grid
No recursion: after marks + creations, a `pending` worklist (index pointer, `for _ in 0..Cells` bound — each
cell activates once) expands striped/wrapped/bomb effects; newly hit specials join the list. The wrapped's
first blast leaves `wrappedArmed` in the grid: it falls through `collapseColumns` like any fruit and
self-fires at the start of the next `clearStep` — which therefore returns > 0, so **both drivers**
(`applySwap`'s drain loop and the web host's animation k-loop) keep stepping until it resolves, **with zero
API-shape changes**. Invariant (tested): a stable board never contains kind 5.

### 3.4 Moves — `stageSwap` + `swapIsLegal`
`stageSwap(action)` becomes the single swap entry for both hosts: swaps, and records a staged combo (bomb or
special+special) that `clearStep(0, …)` executes into the same worklist. `swapIsLegal` (the §2 rule) replaces
`swapProducesMatch` inside `legalMask`, `hasLegalSwap`, the would-match plane, `randomAction`,
`expectimaxAction`, `netAction`, the facade, and the host's revert check. For the animating host the engine
exposes per-step fields `lastClearedBy` (match / striped / wrapped / bomb per cell) and `lastCreated` —
`applySwap` ignores them.

### 3.5 Observation + reward — 928 floats, recalibrated normalizer, the SAME lever as M49
- **+4 kind planes** (stripedH, stripedV, wrapped, bomb) = (6+4+1)·64 + 2·112 = **928** floats. A colored
  special sets BOTH its fruit plane and its kind plane (the net can reason about what a bomb swap clears);
  no plane for `wrappedArmed` (never visible on a stable board).
- **The per-action deterministic-value feature prices FIRING:** activations, chains and combos flow through
  `resolveCascades(false)` automatically (still RNG-free — pinned by a purity test). Under the fire-only
  scoring lock (§2) it deliberately does NOT price creation — that signal is the training env's
  creation-shaping term (`ShapeCreationRewards`: +40/+60/+100 per creation added to the REWARD only, fed by
  the engine's per-move creation counters). The net therefore learns creation preference from the shaped
  reward + the kind planes, and firing value from the feature. **Rejected for v1:** a raw "creates-special"
  flag plane; potential-based shaping at γ=0 (**mathematical no-op**: the shaping term `γΦ(s′)−Φ(s)` loses
  its `γΦ(s′)` half at γ=0 and cannot change any argmax).
- **Normalizer recalibration:** per-action planes ÷100 → **÷300** and reward `points/30` → `points/K` with K
  re-picked in M50.2 so a typical move ≈ O(1) and a bomb+bomb board-clear isn't a 50σ TD target (heavy-tail
  watch: per-move reward histogram).

### 3.6 AI — γ=0 v1, ONE pre-registered escalation
**v1 keeps the locked γ=0 recipe** (M49 finding: bootstrapping diverges on refill noise; the deterministic
feature is what gates). Known blind spot, accepted and *measured* rather than guessed: γ=0 fires specials
when immediately worth it and never HOLDS one for a combo. **Escalation — only if the combo gate (§5 M50.3)
fires:** γ=0.5 + 3-step returns (CANDYRL's setting on real Candy Crush) + potential-based shaping
Φ = Σ option-value of on-board specials (meaningful once γ>0). One escalation, then stop-loss.

### 3.7 Baselines — two new tiers, all engine-free
Random / greedy / expectimax-1 inherit specials through `immediateScore`/`deterministicValue`. New:
**expectimax-2** (2-ply deterministic — the one baseline that can see create→fire and combos; the
`expectimax-2 − expectimax-1` gap *quantifies the combo option-value* and triggers the §3.6 escalation) and
**specials-greedy** (immediate score + fixed creation preference — the "does the net do more than big
matches" sanity bar). All five re-measured on the new env; the M49 bars (2259.7/2387.0/4270.9) are obsolete.

### 3.8 Game-over condition — 30-move rounds + an endless escape hatch (owner Q 2026-07-24)
Human play changes from M49's endless mode to **rounds of 30 moves**, ending on a round-over screen (score,
best, "play again"). Deadlocks still **reshuffle mid-round and never end the game** — "game over when no
move exists" was considered and rejected on measurement: the engine logged **zero deadlocks across every
seeded run** (1,000-move parity episode, all training/eval episodes), so that condition would effectively
never fire, and a board holding a bomb cannot deadlock at all (a bomb always has a legal swap). **30 moves
(not 20) because it is the trained episode framing:** the player's round score becomes directly comparable
to the measured tiers — the round-over screen can honestly say "random averages ~X · the AI ~Y · beat it?".
**Endless-mode toggle (owner addition):** a button lets the player keep playing past 30 moves — enabling it
also dismisses an already-shown round-over screen and resumes the same board. The moment endless touches a
game, that game's score is permanently exempt from the "best" indicator (rounds and best both resume with
the next normal game), so the 30-move leaderboard framing stays honest.

### 3.9 Frontend
Renderer decodes packed values: FruitCake clipart by `fruitOf`, overlays by `kindOf` (stripe lines along the
blast axis, wrapper ring/glow, a colorless dark sugar-bomb sphere with sprinkles). The `pop` animation step
is ENRICHED, not multiplied: it carries `clearedBy` (row/column beam, 3×3 shockwave ring, zap-lines for a
bomb) and `created` (creation cells sparkle in instead of fading). `fall` and `reshuffle` unchanged
(FallMove already carries packed values). Host swaps `swapCells` → `stageSwap`; the k-loop is untouched.
Director/net-loader/ckpt-parser: zero changes (dims come from the ckpt).

## 4. Locked constants (do not re-derive)
Encoding **kind·16+type** · kinds **0..5** (5 internal) · actions **112 (unchanged)** · observation **928**
(6 fruit + 4 kind + would-act planes + 2×112 per-action ÷300) · creation bonuses **+40/+60/+100** · blast
cells **10·(k+1)** · passive bomb **most-frequent type, ties → lowest** · bomb+striped orientations
**`(r+c)%2`** · wrapped+wrapped **5×5, double** · striped blast **⊥ match** (owner-corrected) · cascade
spawn **lowest run cell** · spawn-cell collision **nearest plain cell, ties → lower index; none → no
creation** (M50.5) · relocated creations **blast-shielded for the move, match-consumable** (M50.6) · eval
protocol **500 held-out episodes (seeds 5000+e), mean ± 95% CI** · net **928→256→256→dueling→112, γ=0** (v1).

## 5. Milestones & gates (falsifiable, in order)

- **M50.0 — Rules lock.** ✅ 2026-07-24. §2 of this PRD reviewed against the M49 proportional-scoring lock.
  **Gate: every rule stated deterministically with zero RNG draws** — plus the owner's fire-only scoring
  amendment folded in before any engine code depended on the old creation-bonus lock.
- **M50.1 — Engine.** ✅ SHIPPED 2026-07-24. Packed encoding + decoders, run-recording scan + creation
  resolver, activation worklist + armed wrapped, `stageSwap`/`swapIsLegal`, all combos,
  `lastClearedBy`/`lastCreated` + per-move creation/fired telemetry, extended
  `immediateScore`/`deterministicValue` (run a REAL step-0 clearStep / full deterministic cascade and
  restore everything). **Gate: directed tests for every creation shape (incl. priority + cascade spawn),
  every activation (incl. the two-step wrapped timeline + stable-board-never-armed invariant), every combo,
  chain reactions; the 20-seed × 30-move invariant sweep; a planning-purity test (no RNG, byte-identical
  state) on a specials-rich board; C#↔TS parity checksum RE-PINNED via the node harness — committed to
  `tools/cf_parity.mjs`.** No training before this gate.
  *Gate result: 49 CrazyFruits tests green (final count — incl. the three owner corrections below, the
  striped/wrapped→bomb chain-removal scenarios, and the stepwise-host-protocol ≡ applySwap equivalence
  test; the host layer's round/endless/best rules have their own committed harness `tools/cf_host_tests.mjs`).
  Hand-computed exact scores: striped row-fire 80 · chain striped→striped 150 · wrapped first-fire 100
  (then >100 with the armed re-fire) · bomb+plain (n+1)·10 · bomb+bomb 640 · striped+striped 150 ·
  striped+wrapped 390 · wrapped+wrapped 250 · bomb+striped conversion 220 · passive bomb = row +
  most-frequent-type, computed dynamically. **Three owner corrections landed during the milestone, each
  re-verified end-to-end:** (1) striped creation flipped to the real blast-⊥-match rule; (2) combo blasts
  centre on the gesture's last-selected cell (`stageSwap` grew the target-cell parameter; a directed test
  pins that the same swap staged toward each end fires the matching column); (3) specials FORM before the
  step's activations, so a fresh special blasted in the same step fires immediately (directed 190-point
  double-match test: a fresh striped triggered by a wrapped's box within its own creation step). Parity pin
  history 533753109 → 801202210 → **995400597** (score 95550; the seeded episode exercises creation, chains
  and combos byte-identically). Random's 1,000-move score ~96k vs 71k pre-specials: the predicted auto-fire
  floor-raise.*
- **M50.2 — Env/obs + baselines.** ✅ SHIPPED 2026-07-24. 928-float observation + labels + plane tests;
  normalizer K=100 (random means ~86 pts/move) and ÷300 planes; expectimax-2 (beam-8) + specials-greedy
  tiers, single-sourced in the `.pg`; `--baselines 500` table + per-tier specials usage stats.
  **Gates: (a) ordering expectimax-2 ≥ expectimax-1 ≥ greedy > random, CI-separated where claimed; (b) a
  directed board where greedy provably picks the bomb swap; (c) PRE-TRAINING ENV VALIDATION — if
  random-with-specials ≥ 0.70 × expectimax-2, specials are too self-firing to be skill-differentiating: fix
  scoring before any training** (the flat-landscape guard).
  *Gate result — ALL PASS, and specials dramatically widened the skill landscape (final numbers on the
  fully owner-corrected rules; the two interim tables were within a few % on every row): random 2598.7±72.4
  (3.7 created / 2.8 fired per ep — the auto-fire floor) · greedy 3497.9±96.2 (**+35%** — was +6% without
  specials) · specials-greedy 3867.1±112.8 (+49%) · expectimax-1 5931.4±162.4 (+128%) · expectimax-2
  8135.0±172.0 (**+213%**). Env validation: random = **32%** of expectimax-2 (< 70% ✓). Expectimax-2 gap
  over expectimax-1 = **+37.2%** (≫ 10%): plan-ahead/hold-for-combo value is real — the M50.3 escalation
  trigger is ARMED. Directed tier tests: greedy provably takes the bomb swap; specials-greedy provably
  builds a striped where plain greedy fires the row.*
- **M50.3 — Retrain (from scratch — input width changed).** ✅ CLOSED 2026-07-24 (stop-loss invoked;
  best net shipped, misses reported honestly). **Gates (bars on the FINAL M50.6 shield rules; 500 episodes,
  seeds 5000+e): (1) ≥ +30% over random-with-specials (bar 3392.0), CI-separated; (2) no-regression — ≥ 64%
  of the random→expectimax-1 gap (bar 4747.8, the M49 ratio); (3) specials-exploitation — created/ep ≥ 7.3
  AND fired/ep ≥ 5.6 (2× random 3.67/2.81).** Final-rules baselines: random 2609.2±72.9 · greedy
  3510.3±96.0 · specials-greedy 3903.3±111.8 · expectimax-1 5950.8±161.1 · expectimax-2 8097.7±169.5; env
  validation 32% ✓; e2 gap +36.1% armed the escalation.
  *Verdict — γ=0 won again. Attempt 1 (γ=0 + creation shaping, 400k, `cf5train`) evaluated on the FINAL
  rules: **4040.4±128.9 = +54.9% over random, CI-SEPARATED — gate 1 PASS**, +15.1% over greedy; gap share
  **43% — gate 2 MISS**; created **5.81 — gate 3(created) MISS**, fired 5.75 ≥ 5.6 pass. The pre-registered
  escalation (γ=0.5 + 3-step + PBRS Φ over on-board specials; two runs voided by the in-flight rule fixes —
  `cf6train` killed at 210k by M50.5, `cf7train` finished without the shield at only +33.4% under final
  rules) got its clean from-scratch run on the final rules (`cf8train`, 400k): **3408.2±103.4 = +30.6% —
  gate 1 by a hair, gap share 24%, created 4.66/fired 4.12 — WORSE than γ=0 on every gate.** Bootstrapping
  loses to γ=0 on match-3 refill noise even with PBRS — the M49 γ-lesson reconfirmed at n=2. Stop-loss:
  `cf5train` SHIPPED to `wwwroot/models/crazyfruits.dqn.ckpt` (LFS; the per-action feature planes are
  computed by the live engine at inference, which is why the pre-shield net transfers unchanged — 4051 →
  4040). Gates 2/3(created) remain honestly missed: the reactive net exploits specials it stumbles into
  (fired ≈ created ≈ 5.8) but does not hold-for-combo the way expectimax-2 proves is possible; closing that
  gap likely needs a search-guided policy (the M34 snake lesson), not another reward schedule.* Round-over
  screen gained the net's bar (~4 000).
- **M50.4 — Web.** ✅ SHIPPED 2026-07-24. Overlay art (owner-requested SQUARE candy wrapper with folded
  corner tabs + gloss, fruit visible inside; outlined stripes along the blast axis; colorless sprinkled
  sugar-bomb sphere) + enriched pop step (striped beams, blast rings, bomb zap glows, creation sparkles) +
  `stageSwap` gesture wiring + six watch tiers + the 30-move round framing (§3.8: round-over screen with
  score / best / the measured tier bars; deadlock still reshuffles mid-round). **Gate: tsc clean; headless
  node smoke; live Playwright desktop + touch; zero console errors; NetParity green.**
  *Gate result: tsc clean; headless smoke — 60 greedy moves through the real game layer land on
  engine-exact grids with specials firing (score 8130 vs ~4360 pre-specials). LIVE: watch tiers
  specials-greedy + expectimax-2 exercised (screenshots show the square-wrapped pineapple, striped fruit,
  two sugar bombs on board, expectimax-2 at 6080 by move 20); a REAL 30-move human round (164 pixel-diff
  probed swaps) ended on the round-over screen — "Round over! 2530 · best 3000 · random ~2 600 ·
  expectimax-2 ~8 000 · tap to play again" — and tapping started a fresh round; zero console errors
  throughout. `CrazyFruitsNetParityTests` green (dims come from the ckpt, so the retrained net drops in).*
- **M50.5 — Creation-collision fix (owner bug report 2026-07-24: drag a striped into a line of 4).** ✅
  SHIPPED 2026-07-24. `placeCreation` used to overwrite-and-unmark whatever sat on the spawn cell, so a
  special dragged into its own match neither fired nor yielded a relocated creation — on all four creation
  paths (striped swap-cell, wrapped pivot, bomb match-5, cascade lowest-cell). Fix per the §2 spawn-cell
  collision lock: `spawnCellFor`/`creationCellForWrapped` return the nearest *plain* cell (ties → lower flat
  index; −1 = no creation, counters untouched); a colliding special stays match-marked and fires through the
  unchanged form-then-trigger seed pass. **Gate: 8 directed tests (the reported drag exact-scored at 180 —
  row + relocated fresh striped chain-fired; wrapped variant 160 + armed refire; regression guard for the
  plain-spawn path; all-special run → 240, zero creations; equidistant tie → lower index; match-5 bomb
  relocation 170 with bomb-over-striped priority intact; wrapped-pivot relocation 110; cascade relocation) —
  all green, full suite 57/57.** Parity re-pinned **563660409** (score 86340; C# = TS verified via
  `tools/cf_parity.mjs`), host harness green. Baselines re-measured (rules changed → M50.3 bars updated):
  random 2613.9±72.9 · greedy 3499.2±95.6 · specials-greedy 3906.1±112.8 · expectimax-1 5974.6±161.9 ·
  expectimax-2 8139.1±169.3; env validation 32% ✓; e2 gap +36.2% (escalation stays armed). The in-flight
  escalation run was killed at 210k (buggy-rules data) and restarted on the fixed rules (`cf7train`).
- **M50.6 — Shielded relocations (owner bug report round 2, 2026-07-24: wrapped drag never left the new
  special standing).** ✅ SHIPPED 2026-07-24. Two-agent verdict: engine and web layer were functionally
  correct after M50.5 (the dragged wrapped fired and the striped was created in all 13 audited geometries) —
  but the wrapped's own 3×3 always covers the nearest-plain relocation cell, so the relocated striped
  chain-fired in the same step (form-then-trigger) and the armed refire re-covered it a step later: the
  player NEVER kept the promised special. A dragged striped only spared it when the blast axis missed —
  exactly the owner's observation that striped "worked". Fix per the §2 shield lock: `shielded[]` cells
  (set only for RELOCATED creations, cleared by `stageSwap`) are skipped by `markCell` — all blast/combo
  marking flows through it, match marking doesn't, so later-step matches still consume the special and
  in-place creations keep the 190-point chain rule. **Gate: collision tests updated to shield semantics
  (reported striped drag now exact-scores 100 with the fresh striped SURVIVING on the grid; wrapped drag 90
  surviving BOTH explosions with `moveSpecialsFired == 2`; wrapped-pivot 80 keeping an unarmed wrapped;
  FreshSpecial_FiresImmediately untouched) — 57/57 green.** Parity re-pinned **481681208** (score 95950 —
  up from 86340: surviving specials fire later, the shield is score-positive even under random play); host
  harness green. The completed-but-void `cf7train` run (trained without the shield) is discarded; training
  restarts from scratch on the final rules (`cf8train`).

Effort: engine+tests ≈ 1.5× M49.1 (~25–35 directed tests); env/campaign delta small; frontend ≈ 0.5× M49.4;
training wall-clock ≈ 2–3× M49.3. Roughly 2–3 focused sessions + one training run.

Locked constants addendum (§3.8): human play = **30-move rounds** (same framing as the trained episodes;
round-over screen shows the measured tier bars) · deadlock **reshuffles, never ends the game** (measured:
zero deadlocks in every seeded run; a bomb makes deadlock impossible).

Locked constants addendum (owner, fire-only scoring): game score has **NO creation bonuses**; the
**+40/+60/+100** values live in the training env's `ShapeCreationRewards` term (reward-only, train env
only); per-move engine telemetry `moveCreatedStriped/Wrapped/Bombs` + `moveSpecialsFired` feeds shaping and
the usage gates.

## 6. Risks
1. **Rules under-specification breeding parity bugs** — killed first by M50.0: every ambiguity the sources
   left open is locked deterministically in §2 before any code.
2. **γ=0 blindness to hold-for-combo value** — bounded, and *measured* instead of feared: expectimax-2
   quantifies it; the single pre-registered escalation (γ=0.5 + 3-step + PBRS) exists for exactly this;
   raw creation-bonus hoarding is impossible (bonuses pay at creation, which requires making matches).
3. **Variance explosion from board-clears** — heavy-tailed TD targets; mitigated by the recalibrated
   normalizer, Huber loss, and the deterministic feature carrying the ranking signal.
4. **Specials compress the skill gap** (auto-firing raises random's floor) — the M50.2 pre-training
   env-validation gate catches a flat landscape before any training spend.
5. **Per-action plane saturation** (bomb values ≫ ÷100 scale) — ÷300 rescale picked from measured
   histograms in M50.2, before training.

## 7. Out of scope
Jelly/frosting/blockers and level goals · timed modes · a "hold for combo" search tier deeper than
expectimax-2 · canonical random passive-bomb color (deterministically replaced, §2) · Sugar-Crush end-of-game
sweeps · sound.

## 8. References
Team reports 2026-07-24 (mechanics: Candy Crush Fandom wiki [Striped/Wrapped/Colour-Bomb/Special Candy],
King zendesk "Creating and combining Special Candies", withoutthesarcasm.com guides, candycrush-cheats.com,
BlueStacks guide — striped ∥-blast resolution; engine: line-referenced analysis of `crazyfruits_solver.pg` +
hosts; AI: CANDYRL (Karimi et al., King/KTH — γ=0.5 on real Candy Crush, creation/usage rewards),
Kristensen & Burelli arXiv:2007.01542 (power-piece planes; hard-mask collapse is PPO-specific — does not
apply to masked off-policy DQN), Kamaldinov & Makarov IEEE CoG 2019, Gudmundsson et al. IEEE CIG 2018
(creation-rule encodings)) · [CRAZY_FRUITS_PRD.md](CRAZY_FRUITS_PRD.md) (all M49 locks this PRD extends).
