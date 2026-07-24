# Crazy Fruits specials (striped / wrapped / sugar bomb) — PRD

**Status:** 🔜 planned 2026-07-24 (3-agent investigation: canonical Candy Crush mechanics, engine impact, AI/training impact)
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
- **Striped** ← 4 in a straight line. **Blast direction = PARALLEL to the creating match** (horizontal
  4-match → clears its ROW). *Resolution note: wiki texts calling the stripes "perpendicular" describe the
  sprite paint, not the blast — all sources agree on the directional effect; implement blast ∥ match.*
- **Wrapped** ← a horizontal ≥3 run and a vertical ≥3 run sharing a cell (L, T, +; any larger intersection).
  Spawns at the intersection cell.
- **Sugar bomb** ← 5+ in a straight line. Colorless (fruit type 0).
- **Spawn cell:** the swapped cell when the player's swap made the shape (each swapped cell spawns its own
  special if both sides match); cascade-created line matches spawn at the run's **lowest cell index**
  (deterministic convention — the canonical game leaves this undocumented).
- The creation cell is scored like its matched neighbours but receives the special instead of clearing.

**Activation** (specials fire when cleared by anything — matches, blasts, combos; chains are unbounded but
each cell activates at most once per step):
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
| striped + striped | one full row AND one full column through the swap point (directions overridden) |
| striped + wrapped | 3 full rows AND 3 full columns centered on the swap |
| wrapped + wrapped | one 5×5 blast, then the armed double-explosion fires again after the settle |

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

### 3.8 Game-over condition — 30-move rounds (owner Q 2026-07-24)
Human play changes from M49's endless mode to **rounds of 30 moves**, ending on a round-over screen (score,
best, "play again"). Deadlocks still **reshuffle mid-round and never end the game** — "game over when no
move exists" was considered and rejected on measurement: the engine logged **zero deadlocks across every
seeded run** (1,000-move parity episode, all training/eval episodes), so that condition would effectively
never fire, and a board holding a bomb cannot deadlock at all (a bomb always has a legal swap). **30 moves
(not 20) because it is the trained episode framing:** the player's round score becomes directly comparable
to the measured tiers — the round-over screen can honestly say "random averages ~X · the AI ~Y · beat it?".

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
**`(r+c)%2`** · wrapped+wrapped **5×5, double** · striped blast **∥ match** · cascade spawn **lowest run
cell** · eval protocol **500 held-out episodes (seeds 5000+e), mean ± 95% CI** · net **928→256→256→dueling→112,
γ=0** (v1).

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
  *Gate result: 42 CrazyFruits tests green. Hand-computed exact scores: striped row-fire 80 · chain
  striped→striped 150 · wrapped first-fire 100 (then >100 with the armed re-fire) · bomb+plain (n+1)·10 ·
  bomb+bomb 640 · striped+striped 150 · striped+wrapped 390 · wrapped+wrapped 250 · bomb+striped conversion
  220 · passive bomb = row + most-frequent-type, computed dynamically. Parity re-pin 533753109 (score
  85650, and the seeded episode exercises creation, chains, combos AND a mid-episode reshuffle — the first
  ever observed — byte-identically). Random's 1,000-move score rose 70990 → 85650 (+21%): the predicted
  auto-fire floor-raise, to be re-measured properly in M50.2.*
- **M50.2 — Env/obs + baselines.** 928-float observation + labels + plane tests; normalizer K and ÷300
  picked from measured per-move histograms; expectimax-2 + specials-greedy tiers; `--baselines 500` table.
  **Gates: (a) ordering expectimax-2 ≥ expectimax-1 ≥ greedy > random, CI-separated where claimed; (b) a
  directed board where greedy provably picks the bomb swap; (c) PRE-TRAINING ENV VALIDATION — if
  random-with-specials ≥ 0.70 × expectimax-2, specials are too self-firing to be skill-differentiating: fix
  scoring before any training** (the flat-landscape guard).
- **M50.3 — Retrain (from scratch — input width changed).** γ=0, ~300–500k moves (2–3× M49; same
  architecture — capacity was never the lever). **Gates: (1) ≥ +30% over random-with-specials, 500 episodes,
  CI-separated; (2) no-regression — the net captures ≥ 64% of the random→expectimax-1 gap (M49's ratio);
  (3) specials-exploitation — specials created/ep AND fired/ep ≥ 2× the random rate (makes "it uses
  specials" falsifiable, not vibes).** *Combo gate (escalation trigger, not a fail): if net combo-rate ≈
  random's AND expectimax-2 > 1.10 × expectimax-1 → run the §3.6 escalation ONCE.* Stop-loss: gates still
  failing after the escalation ⇒ stop and write up. Ships the new `crazyfruits.dqn.ckpt` (LFS).
- **M50.4 — Web.** Overlay art + enriched pop step (beams/ring/zap/sparkle) + `stageSwap` wiring + the
  30-move round framing (§3.8: round-over screen with score / best / the measured tier bars as challenge
  lines; deadlock still reshuffles mid-round). **Gate: tsc clean; headless node smoke incl. a combo swap;
  live Playwright desktop + emulated touch showing a striped creation, a row blast, a wrapped double
  explosion, a bomb swap, AND the round-over screen at move 30; zero console errors; NetParity green on the
  shipped ckpt.**

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
