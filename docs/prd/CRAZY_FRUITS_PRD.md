# Crazy Fruits (match-3) — PRD

**Status:** ✅ SHIPPED 2026-07-24 on `m49-crazy-fruits` (PR #38) — engine/env/campaign/net/web all shipped,
**every falsifiable gate green** incl. the live in-browser device check (net **+57.2%** over random,
CI-separated, on the 500-episode protocol). Planned same day (4-agent investigation: game provenance, repo
integration map, PRD/input conventions, match-3 RL prior art).
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M49 · branch `m49-crazy-fruits` · the playground's first **match-3** game (env id `crazyfruits`, type prefix `CrazyFruits`)

## 1. The game — what's confirmed, what's assumed

**Crazy Fruits** was a Flash advergame on the Belgian Belgacom kids portal **kidcity.be** (~2007–2009,
Wayback-confirmed as its own KidCity "house" in NL and FR). The portal shell and a menu thumbnail survive; **the
gameplay SWF was never archived**, so the original's exact grid, fruit count and scoring are unrecoverable. What
IS confirmed is the identity: a bright cartoon **fruit market-stall** theme — a big strawberry wordmark
"CRAZY FRUITS", bunting flags, saturated kid-friendly art. We keep the confirmed theme and build against the
canonical Bejeweled-style ruleset (explicitly **assumed, not verified** against the original):

- Rectangular grid of fruits; the player swaps two **orthogonally adjacent** fruits.
- A swap is only valid if it creates a **line of 3+ equal fruits** (row or column); an invalid swap animates and
  **reverts**, costing nothing.
- Matched fruits are removed, fruits above **fall down**, new random fruits **refill from the top**; any matches
  formed by settling resolve automatically as **cascades** until the board is stable.
- Scoring rewards bigger lines and cascades. Human play is **endless**: when no legal swap exists the board
  **reshuffles** (kids-portal-gentle — no game over, score just keeps accumulating; a session is "play as long
  as you like").

## 2. Scope decision — primitive net first

The deliverable is the **game working properly on desktop (mouse) and smartphone (touch)** plus a
**primitively trained net** that demonstrably beats random play. Serious training strength is explicitly
**future work** (a later milestone can resume the same campaign). This shapes everything below: smallest
credible net, shortest training run, gates about correctness and "beats random" — not about strength.

## 3. Design

### 3.1 Engine — `crazyfruits_solver.pg`, single-source from day one
Author the whole rules engine in MintPlayer.Polyglot
(`src/MintPlayer.AI.ReinforcementLearning.Environments/CrazyFruits/polyglot/crazyfruits_solver.pg`, one
`pgconfig.json` include routing the TS twin to `ClientApp/src/app/crazy-fruits/crazyfruits_solver`), the
draughts/snake/FruitCake pattern: board generation, match detection, swap legality, gravity + refill, cascade
resolution, scoring, the legal-move mask, `buildObservation`, the net `forward`, and the scripted baselines
(§3.7) all live in the `.pg`, wrapped by a thin C# facade. Browser play (M49.4/M49.5) is then a pure-frontend
milestone by construction, and C#↔TS parity is testable with the existing `PolyglotSolverParityTests` pattern.

### 3.2 Deterministic refill RNG — f64-exact LCG (the cross-language lock)
Refill randomness must be **bit-identical in C# and TS** or parity tests and the browser director diverge.
Lock: a **Lehmer/Park–Miller LCG** (minstd: `state = state * 48271 % 2147483647`) implemented inside the `.pg` —
all intermediates < 2^53, so it is **exact in f64 arithmetic** on both sides with no integer/bitwise ops needed.
Seed is a parameter of `reset`. **Rejected:** xoshiro (the C# env family's usual RNG) inside the `.pg` — needs
64-bit bitwise ops that don't survive an f64-only emission; xoshiro stays on the C# host side for things that
never cross the boundary (e.g. training exploration).

### 3.3 Board, action space, masking
**Lock: 8×8 grid, 6 fruit types** — the classic Bejeweled dimensions, matching the (assumed) original and giving
the game its authentic feel. One config for training AND play; no separate "small AI board".
**Rejected:** 6×6/5-fruits training board (prior art's "minimal" pick) — two configs would double the parity
surface and the 8×8 flat MLP is still tiny.

- Actions = adjacent swaps, canonically indexed: 8·7 horizontal + 7·8 vertical = **112 discrete actions**.
- **Hard action mask = match-producing swaps only** (simulate each candidate, check for a 3-line). Masking is
  mandatory, not an optimization: published match-3 results show unmasked DQN/PPO scoring *worse than random*,
  and King measured ~8× faster learning with the mask exposed. The env implements `IActionMaskProvider`;
  training filters both exploration and the TD-target argmax (the existing masked-DQN machinery).
- Board invariants, enforced in-engine: the initial board has **no pre-existing matches** and **≥1 legal swap**;
  after every step the board is stable and has **≥1 legal swap** (§3.5). The mask therefore never goes all-false
  — the −∞-mask collapse class of bugs is defined out of existence.

### 3.4 Observation — one-hot planes + a would-match plane + per-action feature planes
`buildObservation` (shared verbatim between training and serving) emits **6 one-hot fruit planes +
1 "would-match" plane** (both endpoints of every legal swap). **Rejected:** integer-coded grid (imposes a
false ordinal relation between fruit types — a known killer in match-3 RL).
**Amended 2026-07-24 (Risk-1 mitigation exercised, §6):** two **per-action feature planes** were added after
the plain 448-float observation measurably failed the M49.3 gate twice (+1.9% at γ=0.99, +7.8% at γ=0): per
action, the **immediate step-0 score** and the **deterministic cascade value** (gravity-only — the expectimax
tier's signal, worth +89% scripted), both ÷100. Total **672 floats**. The net ranks these against board
context and learns the refill-EV residual — with them it gates at **+57.2%**.

### 3.5 Scoring, episodes, reward
- **Scoring (lock, proportional — no exponential cascade jackpots to reward-hack):** each fruit cleared in
  cascade step *k* (0-based; the swap's own match is step 0) scores **10·(k+1)**; flat line bonuses **+20** for
  a 4-line, **+50** for a 5+-line. A plain 3-match = 30 points.
- **AI episode = fixed budget of 30 moves** (score maximization, the `DqnScoreCampaign` paradigm). Human play is
  endless (§1); the move budget exists only for training/eval framing.
- **Reward = moveScore / 30** (a plain 3-match ⇒ 1.0). No step penalty — every masked action is productive by
  construction, and prior art (King) found step penalties suppress objective-seeking.
- **Deadlock defined out of existence:** if the stable board after a move has no legal swap, the engine
  **reshuffles in-place** (re-deal the existing multiset of fruits until ≥1 legal swap and no instant match)
  *inside* `step`/`reset`. The agent never sees a zero-legal-action state; the human player sees a brief
  "reshuffling!" animation instead of a game over.

### 3.6 Environment + campaign — configuration, not new machinery
`CrazyFruitsEnv : IEnvironment<float[], int>` + `IActionMaskProvider` + `IStatefulEnvironment`, a thin adapter
over the `.pg` core (the Snake/FruitCake shape). Training reuses the **M46 `DqnScoreCampaign` spine as-is**
(env injected via ctor, `DqnScoreOptions`): `AddCrazyFruitsCampaign()` in the Campaigns lib + a
`CrazyFruitsLab` and a `--game crazyfruits` case in the Lab. Checkpoint key `crazyfruits.dqn`, shipped at
`src/RLDemo.Web/wwwroot/models/crazyfruits.dqn.ckpt` (Git LFS — Pattern C serves it as a static file).

### 3.7 Net + algorithm — masked dueling DQN (the Snake recipe)
**Lock: masked dueling DQN over a flat MLP** — `DuelingQNet`, 448 → 256 → 256 → dueling heads → 112, invalid
actions masked to −∞. Exactly the proven Snake/FruitCake stack: existing trainer, existing checkpoint format,
existing browser-side `PgDuelingNet` parser pattern. Refill randomness is plain environment stochasticity —
off-policy replay absorbs it.
**Rejected:** AlphaZero/MCTS self-play (the chess/draughts path) — three independent misfits: refill is
stochastic *and hidden* (chance nodes explode the tree), there is no opponent (nothing for self-play to attach
to), and no adversary means no natural curriculum. **Rejected:** PPO — viable per King's paper, but their result
is a catalogue of brittleness fixes (soft mask channel, entropy retuning, forced step-resets); wrong choice for
a "primitive first" goal when the masked-DQN plumbing already exists.

### 3.8 Scripted baselines = sanity gates = difficulty tiers
Three policies implemented in the `.pg` (they need the same engine internals and must run in the browser):
1. **Random** — uniform over the legal mask. The floor every trained net must clearly beat.
2. **Greedy** — legal swap maximizing immediate match points (deterministic, cheap). The reference bar.
3. **Expectimax-1** — greedy plus the deterministic part of the cascade, refill treated as noise. The "hard" tier.

They triple as (a) pre-training sanity checks that scoring/masking are correct (greedy MUST beat random by a
wide margin — if not, the env is broken), (b) the eval opponents the net is judged against, and (c) the watch-AI
difficulty tiers on the site.

### 3.9 Web architecture — Pattern C, fully client-side
No controller, no server net (the modern default: snake/mountaincar/fruitcake/chess/draughts). Frontend folder
`ClientApp/src/app/crazy-fruits/`:
- `crazyfruits_solver.ts` — generated, gitignored Polyglot twin (engine + baselines + net forward).
- `crazyfruits-net.ts` — `.ckpt` parser building the TS `PgDuelingNet` (clone of `fruitcake-net.ts`).
- `crazy-fruits-director.ts` — client-side watch-AI state machine (tier: random/greedy/expectimax/net).
- `crazy-fruits.ts/.html/.scss` + `crazy-fruits-render.ts` — component (signals, `mode` = human/watch) and
  Canvas 2D renderer, rAF outside Angular's zone; fruit-stall theme (bunting, strawberry accent, big readable
  cartoon fruits — original art, no KidCity/Belgacom branding or logo reuse).
Registration: route in `app.routes.ts`, nav link in `app.html`, card in `home/home.ts`.

### 3.10 Input — one pointer-events path for smartphones AND desktops
Requirement: the game must work properly on smartphones (touch) and desktops (mouse). **Lock: unified
`PointerEvent`s** — `(pointerdown)/(pointermove)/(pointerup)` bound on the canvas — the repo idiom proven by
FruitCake. Pointer events are the DOM's unification of exactly the two requested trios: on touch devices
`pointerdown/move/up` fire for `touchstart/touchmove/touchend`, on desktops for `mousedown/mousemove/mouseup`
— one code path, no double-fire, pen support free.
**Rejected:** separate `touchstart/touchmove/touchend` + `mousedown/mousemove/mouseup` listener pairs — touch
browsers synthesize mouse events after touch (double-fire hazards) and two gesture codepaths must be kept in
sync forever. (2048's raw-touch swipe detector predates the FruitCake pattern and detects a *directional
gesture on the whole board*; Crazy Fruits drags a *specific cell*, which pointer events express directly.)

Two gestures, both supported:
- **Drag-swap:** `pointerdown` on a cell + `pointermove` beyond a half-cell threshold toward a neighbour ⇒
  attempt that swap; `pointerup` before threshold ⇒ falls back to select.
- **Tap-tap:** `pointerdown`+`up` on a cell selects it (highlight); next tap on an orthogonal neighbour attempts
  the swap; any other tap moves the selection.

Mechanics copied from FruitCake/2048: `touch-action: none` on the canvas (we own dragging — no page pan/zoom),
coordinates via `getBoundingClientRect()` → cell = rect-relative fraction × 8 (bounds-checked),
`preventDefault()` in handlers, `setPointerCapture` on drag start so the gesture survives leaving the canvas.
Animations (swap glide, invalid-swap shake-and-revert, pop, fall) run on the rAF loop.

## 4. Locked constants (do not re-derive)
Board **8×8** · **6** fruit types · actions **112** (8·7 + 7·8) · observation **672** floats (6 one-hot planes +
would-match plane + 2×112 per-action features, §3.4 amendment) · net **672→256→256→dueling→112** · **γ = 0**
(the M49.3 finding: match-3's controllable signal is immediate; γ=0.99 bootstrap noise destabilized training) ·
episode **30 moves** · reward **moveScore/30** · scoring **10·(k+1)/fruit at cascade step k, +20/+50 line
bonuses** · RNG **minstd LCG (48271 / 2^31−1) via Schrage, exact in i32 on both sides** · ckpt key
**`crazyfruits.dqn`** · eval protocol **500 held-out seeded episodes (seeds 5000+e), mean ± 95% CI** ·
measured bars: random **2259.7±49.9** · greedy **2387.0±49.3** · expectimax-1 **4270.9±98.3** · net **3552.5±83.3**.

## 5. Milestones & gates (falsifiable, in order)

- **M49.1 — Engine.** ✅ SHIPPED 2026-07-24. `crazyfruits_solver.pg` (board gen, match/swap/gravity/refill/cascade/scoring, legal
  mask, reshuffle-on-deadlock, minstd LCG, `buildObservation`, random/greedy/expectimax-1 baselines) + C#
  facade + `pgconfig.json` include. **Gate: engine unit tests** (init board: no matches, ≥1 legal swap; after
  every step: stable board, ≥1 legal swap; mask equals brute-force match-producing set on random boards;
  hand-computed scores on directed 3/4/5-line and 2-step-cascade positions; reshuffle preserves the fruit
  multiset) **+ C#↔TS parity: a seeded 1,000-move random-policy episode produces byte-identical
  board states and scores in C# and the emitted TS.** No training before this gate.
  *Gate result: 16 tests green first run (incl. the minstd 10,000th-draw reference 399268537); parity
  checksum over every move's action/points/full grid identical C#↔TS (78377593, score 70990) under node
  type-stripping — and re-verified unchanged after the M49.4 stepwise-API refactor and the §3.4 observation
  amendment. Design note: Schrage's decomposition (i32-exact both sides) replaced the planned raw-f64 LCG —
  same minstd generator, stronger exactness argument.*
- **M49.2 — Env + campaign + Lab.** ✅ SHIPPED 2026-07-24. `CrazyFruitsEnv` (+ mask/stateful interfaces), `AddCrazyFruitsDqnCampaign()`
  over the `DqnScoreCampaign` spine, `--game crazyfruits` (`--baselines N` prints the gate table). **Gate:
  baseline ordering over 500 seeded episodes — greedy beats random and expectimax-1 ≥ greedy, non-overlapping
  95% CIs** (proves scoring+masking end-to-end before any training) **+ campaign contract test
  (fresh→TrainChunk→Checkpoint→Resume) + one end-to-end training chunk.**
  *Gate result: random 2259.7±49.9 · greedy 2387.0±49.3 (CI-separated, +6%) · expectimax-1 4270.9±98.3
  (+89%). Finding that shaped M49.3: cascade PLANNING is where the skill is — line size alone is nearly
  irrelevant (+6%) while the deterministic-cascade signal is worth +89%. Contract test + 3k-step end-to-end
  chunk green.*
- **M49.3 — Primitive training run.** ✅ SHIPPED 2026-07-24. **Gate: mean episode score ≥ +30% over random-legal on the 500-episode
  held-out eval, non-overlapping 95% CIs.** The vs-greedy number is *reported* honestly but not gated.
  Checkpoint committed to `wwwroot/models/crazyfruits.dqn.ckpt` (LFS).
  *Gate result — three runs, one lever each (no-thrash): (1) γ=0.99 Snake-recipe verbatim: loss exploded
  0.62→8.7, eval dipped below random mid-run, 500-episode result **+1.9% OVERLAPPING — FAIL** (the
  Kamaldinov trap, exactly as §6 Risk 1 warned). (2) **γ=0** (Q = expected immediate points incl. stochastic
  cascades — bounded targets): stable loss ~0.18, **+7.8% CI-separated** and above greedy, still under the
  bar — FAIL. (3) γ=0 + the §6 Risk-1 pre-registered feature mitigation (§3.4 amendment, obs 448→672):
  **3552.5±83.3 = +57.2% over random, CI-SEPARATED — PASS**, +48.8% over greedy; expectimax-1 (4270.9) is
  the scripted ceiling = future-training headroom.*
- **M49.4 — Web game (human play).** ✅ SHIPPED 2026-07-24. Component + canvas
  renderer (fruit-stall theme) + pointer input (both gestures, §3.10) + animations + registration (route,
  nav, home card). Design addition: the engine grew a **stepwise move API** (`clearStep`/`finishMove`;
  `applySwap` drains the same loop) so the animating host and training are the same rules by construction.
  **Gate: playable end-to-end on desktop mouse AND smartphone touch — drag-swap, tap-tap, invalid-swap
  revert, cascade + reshuffle animations all function; no page scroll/zoom during play; browser smoke against
  the running host.**
  *Gate result: `tsc --noEmit` clean; headless node smoke of the REAL game layer (reverts consume nothing,
  60 greedy moves land on engine-exact grids). LIVE Playwright run against the host, desktop (960px, mouse)
  AND emulated smartphone (390px, `hasTouch`, real touch events): tap-tap select ring lands on the tapped
  cell on both; a human tap-tap swap cleared a 3-line (score 30 · move 1, screenshot); mouse drag and CDP
  touch-drag both fired the gesture (illegal picks reverted correctly, move not consumed); **window.scrollY
  unchanged through the touch drag** (`touch-action: none` holds); zero console/page errors. Screenshot-
  driven UX fix during the pass: a failed tap-tap swap now clears the selection like a successful one.*
- **M49.5 — Watch AI + difficulty tiers.** ✅ SHIPPED 2026-07-24. `crazyfruits-net.ts`
  parser + director + tier picker (Random / Greedy / Expectimax / Trained net). **Gate: TS net forward parity
  with C# (the `*NetParityTests` pattern, real ckpt bytes) + a full 30-move watch episode runs in the browser
  on every tier.**
  *Gate result: `CrazyFruitsNetParityTests` green (byte-level parse mirrors the TS parser line-for-line;
  f64-over-float32 forward matches the SDK net within f32 tolerance; masked netAction legal). Node simulation
  of the exact browser path (real shipped ckpt → TS parser → generated net): all four tiers play full legal
  30-move episodes; ordering reproduces (random 2379 · greedy 2520 · net 3393 · expectimax 4507 over 20 eps).
  LIVE: watch mode caught mid-cascade at move 7/30 with the trained net (score 330, "+30" pop on-screen);
  all four tier buttons exercised in-browser on desktop and mobile; zero errors.*

## 6. Risks
1. **The net barely beats random** — the documented match-3 trap (Kamaldinov: naive DQN/PPO lost to random;
   King needed heavy engineering). Mitigations already designed in: hard mask (§3.3), one-hot + would-match
   observation (§3.4), baselines verified *before* training (M49.2 gate), and a v1 gate that only demands
   beating random. If it still fails: add the per-action immediate-score feature before touching the net.
2. **C#↔TS refill divergence** — any RNG or float drift desyncs the browser director. Mitigated by the
   f64-exact LCG lock (§3.2) and the byte-identical-episode gate in M49.1.
3. **Cascade reward hacking** — net farms lucky cascade patterns. Mitigated by proportional (linear-in-k)
   cascade scoring and held-out eval seeds.
4. **Original-game fidelity** — the real rules are unrecoverable (SWF never archived); we ship assumed-standard
   match-3 and the confirmed theme. Accepted; noted honestly on the PRD and (if asked) the site.

## 7. Out of scope
Serious/long training and strength tuning (future milestone — the campaign resumes from the shipped ckpt) ·
special pieces (striped/wrapped/sugar bomb) — **now planned as M50, see
[CRAZY_FRUITS_SPECIALS_PRD.md](CRAZY_FRUITS_SPECIALS_PRD.md)** · levels, goals, timers · sound · gallery
submissions · server-side serving (Pattern A) · decompiling the original SWF.

## 8. References
Wayback CDX for kidcity.be (`page.php?house=crazyfruits`, NL 2007-10-30 / FR 2009-02-11; menu thumbnail
`img/crazyfruits/th_list_nl.gif`) · Kamaldinov & Makarov, *Deep RL in Match-3 Game*, IEEE CoG 2019
(ieee-cog.org/2019/papers/paper_152.pdf — DQN/PPO below random, A3C won) · *Strategies for Using PPO in Mobile
Puzzle Games* (King), arXiv:2007.01542 (mask ⇒ ~8× speedup; step-penalty retro) · *Testing Match-3 Video Games
with Deep RL*, arXiv:2007.01137 · CandyRL / *Generalized RL for Gameplay* (KTH) · repo:
`docs/ADDING_A_GAME.md`, `SNAKE_SEARCH_PRD.md` (masked dueling DQN), `fruit-cake.ts` (pointer input),
`DRAUGHTS_SELFPLAY_PRD.md` (.pg-first engine discipline).
